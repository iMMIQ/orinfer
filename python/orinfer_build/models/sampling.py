"""Build the shared GPU sampler into a current native execution package.

The CLI atomically republishes a model with unchanged weights and frontend.
Both model builders call append() to publish the same six TileLang bodies.
"""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import os
from orinfer_build.compiler.compile import primitive, instantiate, compile_function
from orinfer_build.packaging.publication import atomic_model, clone_model, load_model
from orinfer_build.common.files import write_json

FUNCTIONS = {
    "initialize": ("Counts", "Bias"),
    "history": ("History", "BiasIds", "BiasValues", "Control", "Counts", "Bias"),
    "scores": (
        "Logits",
        "Counts",
        "Bias",
        "Mask",
        "Control",
        "Parameters",
        "Scores",
        "Keys",
        "Maximum",
        "Invalid",
    ),
    "masses": ("Scores", "Keys", "Maximum", "Invalid", "Parameters", "Mass", "Output"),
    "radix_histogram": ("Keys", "Mass", "State", "Control", "Parameters", "Hist"),
    "radix_choose": ("Hist", "State", "Control", "Parameters", "Output"),
}


def geometry(vocab, context):
    blocks = (vocab + 1023) // 1024
    return {
        "Counts": ("i32", [vocab]),
        "Bias": ("f64", [vocab]),
        "History": ("i32", [context]),
        "BiasIds": ("i32", [vocab]),
        "BiasValues": ("f64", [vocab]),
        "Control": ("i32", [4]),
        "Parameters": ("f64", [6]),
        "Mask": ("u32", [(vocab + 31) // 32]),
        "Scores": ("f64", [vocab]),
        "Keys": ("u64", [vocab]),
        "Maximum": ("f64", [blocks]),
        "Invalid": ("i32", [blocks]),
        "Mass": ("u64", [vocab]),
        "State": ("u64", [7]),
        "Hist": ("u64", [blocks, 256]),
        "Output": ("i32", [2]),
    }


def batch_layout(shapes):
    import math

    widths = {"i32": 4, "u32": 4, "f64": 8, "u64": 8}
    offsets, stride = {}, 0
    for name, (dtype, shape) in sorted(shapes.items()):
        if name == "Output":
            continue
        offsets[name] = stride
        stride += (math.prod(shape) * widths[dtype] + 255) // 256 * 256
    return offsets, stride


def append(data, package, root, output):
    from kernels.cuda.model import sampling
    from kernels.cuda.model.independent_rows import independent_rows
    from orinfer_build.compiler.host_compiler import Compiler

    metadata = data["metadata"]
    vocab, context = metadata["vocab"], metadata["max_context"]
    logits = next(b for b in metadata["buffers"] if b["name"] == metadata["logits"])
    if logits["dtype"] != "f32" or logits["shape"] != [vocab]:
        raise ValueError("Shared GPU sampling requires contiguous FP32 vocabulary logits")
    shapes = geometry(vocab, context)
    offsets, stride = batch_layout(shapes)
    previous = {b["name"]: b for b in metadata["buffers"]}

    def buffer(name, dtype, shape):
        buffer = dict(
            name=name,
            dtype=dtype,
            shape=shape,
            layout="native_contiguous",
            alignment=256,
            access="read_write",
        )
        if name in previous:
            if {k: v for k, v in previous[name].items() if k != "data"} != buffer:
                raise ValueError("Inconsistent shared sampling buffer: " + name)
        else:
            metadata["buffers"].append(buffer)
            package["buffer_contracts"].append(buffer.copy())
            data["buffer_scopes"][name] = "workspace"

    for name, (dtype, shape) in shapes.items():
        buffer("Sampling" + name, dtype, shape)
    buffer("SamplingRowsPointers", "u64", [128, 2])
    buffer("SamplingRowsScratch", "u8", [128, stride])
    buffer("SamplingRowsOutput", "i32", [128, 2])
    buffers = {b["name"]: b for b in metadata["buffers"]}
    compiler = Compiler(root, root, output, buffers)

    def sampler(name):
        return name.startswith(("sampling/", "sampling_rows/"))

    package["kernels"] = [k for k in package["kernels"] if not sampler(k["name"])]
    package["dynamic_batch_kernels"] = [
        k for k in package.get("dynamic_batch_kernels", []) if not sampler(k["name"])
    ]
    for function, names in FUNCTIONS.items():
        built = primitive(
            getattr(sampling, function), vocab, *([context] if function == "history" else [])
        )
        bindings = {
            name: dict(
                kind="buffer", name=metadata["logits"] if name == "Logits" else "Sampling" + name
            )
            for name in names
        }
        variants = (
            [(f"sampling/{function}", {})]
            if not function.startswith("radix_")
            else [
                (
                    f"sampling/{mode}/{digit}/"
                    + ("hist" if function == "radix_histogram" else "choose"),
                    dict(digit=digit, mode=mode),
                )
                for mode in range(3)
                for digit in range(12)
            ]
        )
        for name, constants in variants:
            binding, _ = compiler.emit(
                dict(name=name),
                name,
                bindings,
                1,
                key=f"sampling-{vocab}-{context}-{function}",
                symbolic=False,
                kernel=built,
                constants=constants,
            )
            package["kernels"].append(binding)
        fn = instantiate(
            getattr(sampling, function), vocab, *([context] if function == "history" else [])
        )
        modes = {name: ("private", "Logits" if name == "Logits" else "Scratch") for name in names}
        if "Output" in names:
            modes["Output"] = ("row", "Output")
        fn, inputs = independent_rows(
            fn,
            128,
            modes,
            {"Logits": 0, "Scratch": 1},
            dynamic=True,
            offsets={name: offsets[name] for name in names if name in offsets},
        )
        batched = compile_function(fn)
        batched_bindings = {"Pointers": dict(kind="buffer", name="SamplingRowsPointers")}
        if "Output" in inputs:
            batched_bindings["Output"] = dict(kind="buffer", name="SamplingRowsOutput")
        for name, constants in variants:
            name = name.replace("sampling/", "sampling_rows/", 1)
            binding, launch = compiler.emit(
                dict(name=name),
                name,
                batched_bindings,
                128,
                key=f"sampling-rows-{vocab}-{context}-{function}",
                kernel=batched,
                constants=constants,
            )
            package["kernels"].append(binding)
            package["dynamic_batch_kernels"].append(launch)
    # Native registration needs these final program contracts even when the
    # online create request omits build-only operator definitions.
    package["plan_kernel_contracts"] = sorted(
        set(package.get("plan_kernel_contracts", []))
        | {k["name"] for k in package["kernels"] if sampler(k["name"])}
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--model-output", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    data, origin, package = load_model(a.model)
    from orinfer_build.packaging.package import config_signature

    if data["architecture"] == "qwen3_5":
        package["config_signature"] = config_signature(
            json.loads((a.model / "config.json").read_text())
        )
    from orinfer_build.packaging.compact import compact_package

    with atomic_model(a.model_output) as staging:
        root = clone_model(a.model, staging, origin)
        append(data, package, root, a.output)
        digest = compact_package(root, package, staging / "cache/packages")
        shutil.rmtree(root)
        data["execution_package"] = digest
        write_json(staging / "cache/model.json", data)
        from orinfer_build.packaging.publication import ROOT

        engine = os.environ.get("ORINFER_ENGINE", str(ROOT / "target/release/orinfer"))
        completed = subprocess.run(
            [engine, "plan-model", str(staging)], capture_output=True, text=True
        )
        if completed.returncode:
            raise ValueError("Native sampling plan validation failed: " + completed.stderr.strip())
        report = json.loads(completed.stdout)["manifest"]
        if (
            not report.get("sampling", {}).get("batch")
            or len(report["programs"].get("sampling", [])) != 76
            or len(report["programs"].get("sampling_rows", [])) != 76
        ):
            raise ValueError("Native library omitted the shared sampling program")
    write_json(a.output / "result.json", dict(model=str(a.model_output), execution_package=digest))
    print("COMPLETE", digest, flush=True)


if __name__ == "__main__":
    main()
