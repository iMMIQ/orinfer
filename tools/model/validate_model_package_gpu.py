"""Native C package -> Rust loader -> real CUDA stateful graph integration.

Uses a four-token fixture, not model-quality or performance evidence. Checks
all graph modes, changed prompts, teacher-forced inputs and full logits.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import torch
from safetensors.torch import save_file

from kernels.fixtures.model_package import transition
from tools.model.publication import file_hash, write_json
from tools.operators.abi import evaluate, parse_host
from tools.operators.common import ROOT, configure, export_kernel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine", type=Path, default=ROOT / "target/release/orinfer")
    args = parser.parse_args()
    configure()
    torch.manual_seed(20261002)
    model = args.output / "model"
    weights = model / "cache/weights"
    package = model / "cache/packages/.building"
    weights.mkdir(parents=True)
    (package / "lib").mkdir(parents=True)
    subprocess.run(
        [
            "cc",
            "-shared",
            "-fPIC",
            "-Werror",
            "-Wall",
            "-Wextra",
            "-I",
            str(ROOT / "crates/orinfer-model-sdk/include"),
            str(ROOT / "crates/orinfer-engine/tests/fixtures/model_package.c"),
            "-o",
            str(package / "lib/model.so"),
        ],
        check=True,
    )
    weight = torch.rand((4, 4), dtype=torch.float32) - 2
    weight[torch.arange(4), (torch.arange(4) + 1) % 4] = 3
    save_file(
        {"Weight": weight},
        weights / "one.safetensors",
        metadata={"orin.layout.Weight": "row_major"},
    )
    write_json(
        weights / "model.safetensors.index.json", {"weight_map": {"Weight": "one.safetensors"}}
    )
    buffers = []
    for name, dtype, shape, access in [
        ("Weight", "f32", [4, 4], "read"),
        ("Input", "i32", [1], "read_write"),
        ("Token", "i32", [1], "read_write"),
        ("Status", "i32", [1], "read_write"),
        ("Position", "i32", [1], "read_write"),
        ("Logits", "f32", [4], "read_write"),
        ("Scratch", "i32", [1], "read_write"),
    ]:
        buffers.append(
            dict(
                name=name,
                dtype=dtype,
                shape=shape,
                access=access,
                layout="row_major",
                alignment=256,
                data=dict(
                    tensor="Weight", sha256=hashlib.sha256(weight.numpy().tobytes()).hexdigest()
                )
                if name == "Weight"
                else None,
            )
        )
    kernels = []
    for prefill in (True, False):
        name = "prefill" if prefill else "decode"
        directory = package / "kernels" / name
        export_kernel(transition(prefill), directory)
        (launch,) = parse_host((directory / "host.txt").read_text())
        config = launch["launch_expressions"]

        def identity(file):
            return dict(file=str(file.relative_to(package)), sha256=file_hash(file))

        ordered = launch["ordered_arguments"]
        assert all(
            a["ctype"] == "ctypes.c_void_p"
            and a["value"].endswith(".data_ptr()")
            and a["value"].removesuffix(".data_ptr()") in {b["name"] for b in buffers}
            for a in ordered
        )
        kernels.append(
            dict(
                name=name,
                module=identity(directory / "kernel.cubin"),
                source=identity(directory / "kernel.cu"),
                host_abi=identity(directory / "host.txt"),
                symbol=launch["symbol"],
                grid=[evaluate(config["gridDim" + axis], {}) for axis in "XYZ"],
                block=[evaluate(config["blockDim" + axis], {}) for axis in "XYZ"],
                shared_memory_bytes=evaluate(config["sharedMemBytes"], {}),
                cooperative=False,
                args=[
                    dict(kind="buffer", name=a["value"].removesuffix(".data_ptr()"))
                    for a in ordered
                ],
            )
        )
    manifest = dict(
        schema_version=1,
        runtime_abi=1,
        target="sm_87",
        architecture="test_family",
        compute_policy="test_policy",
        config_signature=None,
        prefill_profiles=[],
        buffer_contracts=[{k: v for k, v in b.items() if k != "data"} for b in buffers],
        kernels=kernels,
        toolchain={"tilelang": "0.1.15"},
        execution=dict(
            abi_version=1,
            library=dict(file="lib/model.so", sha256=file_hash(package / "lib/model.so")),
            package="test-model",
            version="1",
        ),
    )
    write_json(package / "package.json", manifest)
    digest = file_hash(package / "package.json")
    package.rename(package.with_name(digest))
    metadata = dict(
        schema_version=2,
        target="sm_87",
        model="native-package-fixture",
        chunk_tokens=1,
        max_context=8,
        vocab=4,
        buffers=buffers,
        toolchain={"tilelang": "0.1.15"},
        reset_buffers=["Input", "Token", "Status", "Position"],
        input="Input",
        token="Token",
        status="Status",
        position="Position",
        logits="Logits",
        weight_bytes=64,
        weight_parameters=16,
        weight_scope="integration-fixture",
    )
    assets = {}
    for name in ("config.json", "tokenizer.json", "chat_template.jinja", "generation_config.json"):
        (model / name).write_text("{}")
        if name != "config.json":
            assets[name] = file_hash(model / name)
    write_json(
        model / "cache/model.json",
        dict(
            schema_version=1,
            architecture="test_family",
            compute_policy="test_policy",
            execution_package=digest,
            frontend_assets=assets,
            metadata=metadata,
            buffer_scopes={
                b["name"]: "weights"
                if b["name"] == "Weight"
                else "workspace"
                if b["name"] in ("Scratch", "Logits")
                else "sequence"
                for b in buffers
            },
        ),
    )
    requests = [
        dict(id="first", input_tokens=[0], max_new_tokens=5, logits_steps=list(range(5))),
        dict(id="changed", input_tokens=[2], max_new_tokens=5, logits_steps=list(range(5))),
        dict(
            id="forced",
            input_tokens=[1],
            max_new_tokens=5,
            forced_tokens=[3, 0, 2, 1],
            logits_steps=list(range(5)),
        ),
    ]
    request_file = args.output / "requests.json"
    write_json(request_file, dict(requests=requests))
    subprocess.run(
        [str(args.engine), "validate-model", str(model)], check=True, capture_output=True, text=True
    )
    report = dict(complete=False, scope=__doc__, modes=[])
    for mode in ("off", "decode_only", "full"):
        write_json(
            request_file, dict(requests=requests, logits_output=str(args.output / f"logits-{mode}"))
        )
        result = subprocess.run(
            [str(args.engine), "run-model", str(model), str(request_file), "--cuda-graph", mode],
            check=True,
            capture_output=True,
            text=True,
        )
        result = json.loads(result.stdout)
        write_json(args.output / f"{mode}.json", result)
        for request, actual in zip(requests, result["requests"]):
            token = request["input_tokens"][-1]
            expected = []
            for step in range(5):
                if step:
                    token = (
                        request.get("forced_tokens", [])[step - 1]
                        if request.get("forced_tokens")
                        else expected[-1]
                    )
                expected.append((token + 1) % 4)
                data = torch.frombuffer(
                    bytearray(Path(actual["logits_files"][step]).read_bytes()), dtype=torch.float32
                )
                assert torch.equal(data, weight[token]), (mode, request["id"], step, "logits")
            assert actual["output_tokens"] == expected, (mode, request["id"], "tokens")
        report["modes"].append(
            dict(
                mode=mode,
                requests=len(requests),
                tokens_exact=True,
                logits_exact=True,
                captured_programs=result["captured_programs"],
            )
        )
        write_json(args.output / "results.json", report)
    report["complete"] = True
    write_json(args.output / "results.json", report)
    print("All native-package GPU graph modes and changed-input checks passed.", flush=True)


if __name__ == "__main__":
    main()
