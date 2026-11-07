"""Attach one dynamic device categorical program to a prepared A2 package."""

import argparse
import json
from pathlib import Path

from tools.build.files import atomic_json
from tools.build.shared_json import decode


def attach(model, base):
    from kernels.model.rows import row_count
    from tools.operators.ascend import sampling
    from tools.operators.ascend.build import compile_kernel, operand

    capacity = model["batch"].get("output_capacity") or model["batch"]["owners"]
    vocab = model["ranks"][1]["packets"][0]["vocab_first"]
    if not 2 <= capacity <= 4096 or not 0 < vocab < 1 << 24:
        raise ValueError("Unsupported sampling capacity/vocabulary")
    for rank in model["ranks"]:
        for packet in rank["packets"]:
            if (
                packet.get("role", "target") == "target"
                and packet["vocab_first"] != rank["rank"] * vocab
            ):
                raise ValueError("Device sampling requires contiguous equal TP vocabulary shards")
    parts = (vocab + 1023) // 1024
    rows = row_count(capacity, True)
    result = dict(capacity=capacity, vocab=vocab, rng="philox2x32-10-v1")
    for field, name, arguments, ports, parameters in [
        (
            "local",
            "categorical",
            dict(rows=rows, vocab=vocab),
            [
                ("logits", "f32", (capacity, vocab), "read"),
                ("controls", "i32", (capacity, 4), "read"),
                ("active", "i32", (1,), "read"),
                ("partial", "f32", (capacity, parts, 2), "write"),
            ],
            dict(vocab=vocab),
        ),
        (
            "merge",
            "merge",
            dict(rows=rows, parts=parts),
            [
                ("partial", "f32", (capacity, parts, 2), "read"),
                ("active", "i32", (1,), "read"),
                ("output", "f32", (capacity, 2), "write"),
            ],
            dict(parts=parts),
        ),
    ]:
        path = compile_kernel(
            getattr(sampling, name)(**arguments),
            "sampling." + name,
            [operand(*p) for p in ports],
            parameters,
            base / "cache",
            row_capacity=capacity,
        )
        result[field] = str(path.relative_to(base))
    model["sampling"] = result
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile-workers", type=int, default=1)
    args = parser.parse_args()
    if (
        args.plan.parent.resolve() != args.output.parent.resolve()
        or args.plan.resolve() == args.output.resolve()
    ):
        parser.error("Preserve the input and keep its asset directory")
    model = decode(json.loads(args.plan.read_text()))
    print(json.dumps(attach(model, args.plan.parent)), flush=True)
    atomic_json(args.output, model)


if __name__ == "__main__":
    main()
