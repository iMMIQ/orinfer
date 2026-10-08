"""Assemble compatible fixed-M manifests into one shared-weight Rust model.

Maximum-M workspace owns the allocation; each graph retains its compiled
logical strides and bounds. Only immutable initializers with different values
receive private buffers; mutable model states keep identical reset contracts.
No quantization, kernel recompilation or persistent weight duplication.
"""

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path


SIZES = {"u8": 1, "i8": 1, "f16": 2, "f32": 4, "i32": 4}


def size(buffer):
    return math.prod(buffer["shape"]) * SIZES[buffer["dtype"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("models", nargs="+", type=Path)
    args = ap.parse_args()
    assert not args.output.exists(), "Refuse existing output"
    sources = [(p, json.loads(p.read_text())) for p in args.models]
    sources.sort(key=lambda x: x[1]["chunk_tokens"])
    assert len({m["chunk_tokens"] for _, m in sources}) == len(sources)
    assert len(sources) >= 2
    primary_path, primary = sources[-1]
    assert not any(m.get("prefill_plans") for _, m in sources), (
        "Only standalone source manifests accepted"
    )
    merged = copy.deepcopy(primary)
    merged["prefill_plans"] = []
    merged["kernels"] = []
    merged["programs"] = {}
    proof = []
    linked = {}
    args.output.mkdir(parents=True)

    def link(identity, root, prefix):
        original = root / identity["file"]
        relative = Path(prefix) / identity["file"]
        destination = args.output / relative
        if relative not in linked:
            assert original.is_file()
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(original, destination)
            linked[relative] = identity["sha256"]
        else:
            assert linked[relative] == identity["sha256"]
        return dict(file=str(relative), sha256=identity["sha256"])

    # The maximum-M artifact's weights and allocations are the one source of
    # residency. Secondary scalar constants are read_write, not model weights.
    for b in merged["buffers"]:
        if b["data"]:
            b["data"] = link(b["data"], primary_path.parent, "shared")
    for path, model in sources:
        c = model["chunk_tokens"]
        prefix = f"m{c}"
        rename = {}
        checks = []
        for field in (
            "schema_version",
            "target",
            "model",
            "max_context",
            "vocab",
            "weight_bytes",
            "weight_parameters",
            "weight_scope",
            "input",
            "token",
            "status",
            "logits",
            "position",
            "reset_buffers",
        ):
            assert model[field] == primary[field], (c, field)
        assert {k: v for k, v in model["toolchain"].items() if k != "output_root"} == {
            k: v for k, v in primary["toolchain"].items() if k != "output_root"
        }
        assert len(model["buffers"]) == len(primary["buffers"])
        original_primary = {b["name"]: b for b in primary["buffers"]}
        for b in model["buffers"]:
            other = original_primary[b["name"]]
            assert all(b[k] == other[k] for k in ("dtype", "layout", "alignment", "access"))
            assert size(b) <= size(other), (c, b["name"])
            if b["access"] == "read":
                assert b == other, (c, b["name"], "weight mismatch")
            elif b["data"] and b["data"] != other["data"]:
                assert other["data"] is not None
                small = (path.parent / b["data"]["file"]).read_bytes()
                large = (primary_path.parent / other["data"]["file"]).read_bytes()
                if small != large[: len(small)]:
                    # The existing builder's only chunk-dependent scalar
                    # initializers. Reject an unknown state/layout semantic.
                    assert b["name"] in ("LengthPrefill", "IndexPrefill") and size(b) == 4
                    assert b["name"] not in model["reset_buffers"]
                    name = f"{b['name']}_{prefix}"
                    rename[b["name"]] = name
                    item = copy.deepcopy(b)
                    item["name"] = name
                    item["data"] = link(item["data"], path.parent, prefix)
                    merged["buffers"].append(item)
                checks.append(
                    dict(
                        buffer=b["name"],
                        private=b["name"] in rename,
                        prefix_initializer_equal=small == large[: len(small)],
                    )
                )
        chosen = (
            ("prefill", "head", "decode") if c == primary["chunk_tokens"] else ("prefill", "head")
        )
        used = {
            op["name"]
            for phase in chosen
            for op in model["programs"][phase]
            if op["kind"] == "kernel"
        }
        for kernel in model["kernels"]:
            if kernel["name"] not in used:
                continue
            item = copy.deepcopy(kernel)
            item["name"] = prefix + "_" + kernel["name"]
            for kind in ("module", "source", "host_abi"):
                item[kind] = link(item[kind], path.parent, prefix)
            for argument in item["args"]:
                if argument["kind"] == "buffer":
                    argument["name"] = rename.get(argument["name"], argument["name"])
            merged["kernels"].append(item)
        for phase in chosen:
            program = copy.deepcopy(model["programs"][phase])
            name = "decode" if phase == "decode" else phase + "_" + prefix
            for op in program:
                if op["kind"] == "kernel":
                    op["name"] = prefix + "_" + op["name"]
                else:
                    for field in ("source", "destination"):
                        if field in op:
                            op[field] = rename.get(op[field], op[field])
            merged["programs"][name] = program
        merged["prefill_plans"].append(
            dict(chunk_tokens=c, prefill_program="prefill_" + prefix, head_program="head_" + prefix)
        )
        proof.append(
            dict(
                chunk_tokens=c,
                source_model=str(path),
                source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                initializer_checks=checks,
                weights_identical=True,
            )
        )
    # Legacy names remain valid schema defaults; selection uses explicit plans.
    merged["programs"]["prefill"] = merged["programs"][f"prefill_m{primary['chunk_tokens']}"]
    merged["programs"]["head"] = merged["programs"][f"head_m{primary['chunk_tokens']}"]
    manifest = args.output / "model.json"
    manifest.write_text(json.dumps(merged, indent=2) + "\n")
    report = dict(
        status="assembled",
        source_models=proof,
        model_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        weight_bytes=merged["weight_bytes"],
        buffer_bytes=sum(size(b) for b in merged["buffers"]),
        kernel_count=len(merged["kernels"]),
        prefill_plans=merged["prefill_plans"],
        scope="Single weight set; maximum workspace; native execution/numeric/performance verification pending",
    )
    (args.output / "assembly-report.json").write_text(json.dumps(report, indent=2) + "\n")
    source_dir = args.output / "measurement-source"
    source_dir.mkdir()
    (source_dir / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
