"""Upgrade a prepared batch package with exact GPU greedy history processors.

Weights and existing operator assets are immutable hardlinks. Publication uses
a fresh directory and checks the Rust-registered plan before an atomic rename.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.model.publication import (
    atomic_model,
    clone_model,
    commit_package,
    file_hash,
    load_model,
    write_json,
)
from tools.model.upgrade_batching import bind
from tools.operators.abi import parse_host


def upgrade(model, destination, report, specialize):
    from kernels.model.greedy_sampling import history_counts, penalized_partials, penalized_merge
    from tools.operators.common import configure, export_kernel

    configure()
    data, origin, package = load_model(model)
    metadata = data["metadata"]
    if not package.get("batch_profiles") or package.get("greedy_sampling"):
        raise ValueError("Expected batch package without GPU greedy processors")
    operator = clone_model(model, destination, origin)
    vocab, context = metadata["vocab"], metadata["max_context"]
    blocks = (vocab + 4095) // 4096
    for name, dtype, shape in [
        ("SamplingHistory", "i32", [context]),
        ("SamplingLength", "i32", [1]),
        ("SamplingCounts", "i32", [vocab]),
        ("SamplingParameters", "u64", [3]),
        ("SamplingKeys", "u64", [blocks]),
        ("SamplingIDs", "i32", [blocks]),
        ("SamplingBad", "i32", [blocks]),
    ]:
        if any(b["name"] == name for b in metadata["buffers"]):
            raise ValueError("Duplicate sampling buffer")
        metadata["buffers"].append(
            dict(
                name=name,
                dtype=dtype,
                shape=shape,
                layout="contiguous",
                alignment=256,
                access="read_write",
                data=None,
            )
        )
        data["buffer_scopes"][name] = "sequence"
    factories = [
        (
            "count",
            lambda: history_counts(vocab, context),
            dict(History="SamplingHistory", Length="SamplingLength", Counts="SamplingCounts"),
        ),
        (
            "partials",
            lambda: penalized_partials(vocab),
            dict(
                X=metadata["logits"],
                Counts="SamplingCounts",
                Parameters="SamplingParameters",
                Keys="SamplingKeys",
                IDs="SamplingIDs",
                Bad="SamplingBad",
            ),
        ),
        (
            "merge",
            lambda: penalized_merge(vocab),
            dict(
                Keys="SamplingKeys",
                IDs="SamplingIDs",
                Bad="SamplingBad",
                Token=metadata["token"],
                Status=metadata["status"],
            ),
        ),
    ]

    def compile_kernel(name, factory):
        print("compile", name, flush=True)
        out = operator / "decode-aot" / name
        export_kernel(factory(), out)
        host = parse_host((out / "host.txt").read_text())
        if len(host) != 1:
            raise ValueError("Expected one kernel export")

        def identity(filename):
            path = out / filename
            return dict(file=str(path.relative_to(operator)), sha256=file_hash(path))

        return dict(
            **host[0],
            module=identity("kernel.cubin"),
            source=identity("kernel.cu"),
            host_abi=identity("host.txt"),
        )

    for name, factory, pointers in factories:
        export = compile_kernel("greedy-" + name, factory)
        package["kernels"].append(bind(export, "greedy_sampling/" + name, pointers, {}))
    if specialize:
        from kernels.model.w4_small_m import w4_small_m

        config = json.loads((model / "config.json").read_text())
        text = config.get("text_config", config)
        h, f = text["hidden_size"], text["intermediate_size"]
        if (h, f) != (5120, 17408):
            raise ValueError("Projection tuning currently validated for 27B dimensions")
        kernels = {k["name"]: k for k in package["kernels"]}
        buffers = {b["name"]: b for b in metadata["buffers"]}
        for family, rows in [("GateUp", 4), ("GateUp", 8), ("Down", 2), ("Down", 4), ("Down", 8)]:
            if rows not in package["batch_profiles"]:
                continue
            gate = family == "GateUp"
            layout = buffers["L0_" + family + "_P"]["layout"]
            if layout not in ("u4_warp_n64_k128_mma_i8", "u4_warp_n64_k128_mma_f16"):
                raise ValueError("Unsupported projection layout")
            export = compile_kernel(
                f"{family}-m{rows}",
                lambda: w4_small_m(
                    rows,
                    2 * f if gate else h,
                    h if gate else f,
                    1 if gate else 8,
                    "float16" if gate else "float32",
                    TILE_N=128 if gate else 64,
                    weight_layout="i8" if layout.endswith("mma_i8") else "f16",
                    byte_permute=True,
                    vector_words=4,
                ),
            )
            for layer, kind in enumerate(text["layer_types"]):
                if buffers[f"L{layer}_{family}_P"]["layout"] != layout:
                    raise ValueError("Projection layouts differ across layers")
                slot = (8 if gate else 10) if kind == "linear_attention" else (5 if gate else 7)
                name = f"batch_m{rows}/layer{layer}/k{slot}"
                pointers = dict(
                    A="Norm" if gate else "Activated",
                    PP=f"L{layer}_{family}_P",
                    S=f"L{layer}_{family}_S",
                    Z=f"L{layer}_{family}_Z",
                    O="GateUp" if gate else "Partial",
                )
                if name not in kernels:
                    raise ValueError("Missing registered batch projection")
                kernels[name] = bind(export, name, pointers, {})
        package["kernels"] = list(kernels.values())
    package["greedy_sampling"] = True
    package["buffer_contracts"] = [
        {k: v for k, v in b.items() if k != "data"} for b in metadata["buffers"]
    ]
    digest = commit_package(destination, operator, data, package)
    write_json(
        report / "upgrade.json",
        dict(
            execution_package=digest,
            weight_bytes=metadata["weight_bytes"],
            specialized_projections=specialize,
            sampling_scratch_bytes=context * 4 + vocab * 4 + blocks * 16 + 28,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--specialize-projections", action="store_true")
    args = parser.parse_args()
    destination = args.model_output.absolute()
    if destination.exists():
        parser.error("Destination already exists")
    with atomic_model(destination) as staging:
        upgrade(args.model.resolve(strict=True), staging, args.output, args.specialize_projections)
    print("DECODE PACKAGE READY", destination, flush=True)


if __name__ == "__main__":
    main()
