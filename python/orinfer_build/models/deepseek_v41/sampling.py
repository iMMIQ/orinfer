"""Attach one dynamic device categorical program to a prepared A2 package."""

import argparse
import json
from pathlib import Path

from orinfer_build.common.files import atomic_json
from orinfer_build.common.shared_json import decode


def attach(model, base):
    from kernels.shared.rows import row_count
    from kernels.ascend import sampling
    from orinfer_build.compiler.ascend.build import compile_kernel, operand

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


def couple_draft(model, base):
    """Reuse the Markov chain with the target's per-position Gumbel draws."""
    from kernels.shared.rows import row_count
    from kernels.ascend import sampling
    from orinfer_build.compiler.ascend.build import compile_kernel, operand

    if not model.get("sampling"):
        raise ValueError("Coupled drafts require target device sampling")
    owners, vocab = model["batch"]["owners"], model["sampling"]["vocab"]
    compiled = {}
    packets = 0
    for rank in model["ranks"]:
        name = "sampling.markov.controls"
        strides = set()
        for packet in rank["packets"]:
            if packet.get("role") != "draft":
                continue
            if packet.get("sampling_controls"):
                raise ValueError("Draft already has coupled sampling controls")
            stride = packet.get("draft_stride")
            if not stride or packet["rows"] != owners * stride:
                raise ValueError("Coupled draft requires the shared physical Markov stride")
            strides.add(stride)
            steps = 0
            for instruction in packet["body"]:
                if instruction["op"] != "kernel":
                    continue
                path = instruction["manifest"]
                spec = json.loads((base / path).read_text())
                if spec["op"] != "topk.add_bias_argmax":
                    continue
                if (
                    spec["parameters"]["block_size"] != stride
                    or spec["parameters"]["vocab"] != vocab
                    or len(instruction["buffers"]) != 5
                    or len(spec["operands"]) != 5
                ):
                    raise ValueError("Markov head differs from the coupled sampling contract")
                if stride not in compiled:
                    ports = spec["operands"] + [
                        dict(
                            name="controls", dtype="i32", shape=[owners * stride, 4], access="read"
                        )
                    ]
                    result = compile_kernel(
                        sampling.markov(row_count(owners, True), vocab, stride),
                        "sampling.markov",
                        [operand(p["name"], p["dtype"], p["shape"], p["access"]) for p in ports],
                        dict(vocab=vocab, block_size=stride),
                        base / "cache",
                        row_capacity=owners,
                    )
                    compiled[stride] = str(result.relative_to(base))
                instruction["manifest"] = compiled[stride]
                instruction["buffers"].append(name)
                steps += 1
            if steps != packet["coordinates"]["draft_block"]:
                raise ValueError("Incomplete coupled Markov chain")
            packet["sampling_controls"] = name
            packets += 1
        if len(strides) != 1:
            raise ValueError("Publish one physical Markov stride per rank")
        rank["buffers"].append(dict(name=name, bytes=owners * next(iter(strides)) * 16))
    if packets == 0:
        raise ValueError("No DSpark draft packets")
    model["sampling"]["coupled_draft"] = True
    return dict(packets=packets, kernels=compiled)


def attach_processed(model, base):
    """Extend the existing sampler with masks and bounded probability summaries."""
    from kernels.ascend import sampling_processed as implementation
    from kernels.shared.rows import row_count
    from orinfer_build.compiler.ascend.build import compile_kernel, operand

    result = model.get("sampling")
    if not result:
        raise ValueError("Processed sampling requires the categorical sampler")
    capacity, vocab = result["capacity"], result["vocab"]
    parts = (vocab + 1023) // 1024
    rows = row_count(capacity, True)
    for field, factory, arguments, ports in [
        (
            "processed_local",
            implementation.local,
            dict(rows=rows, vocab=vocab),
            [
                ("logits", "f32", (capacity, vocab), "read"),
                ("controls", "i32", (capacity, 4), "read"),
                ("active", "i32", (1,), "read"),
                ("masks", "u8", (capacity, ((vocab + 255) // 256) * 32), "read"),
                ("options", "i32", (capacity, 2), "read"),
                ("partial", "f32", (capacity, parts, implementation.WIDTH), "write"),
            ],
        ),
        (
            "processed_merge",
            implementation.merge,
            dict(rows=rows, parts=parts),
            [
                ("partial", "f32", (capacity, parts, implementation.WIDTH), "read"),
                ("active", "i32", (1,), "read"),
                ("options", "i32", (capacity, 2), "read"),
                ("output", "f32", (capacity, implementation.WIDTH), "write"),
            ],
        ),
    ]:
        path = compile_kernel(
            factory(**arguments),
            "sampling_processed." + field.removeprefix("processed_"),
            [operand(*p) for p in ports],
            {key: value for key, value in arguments.items() if key != "rows"},
            base / "cache",
            row_capacity=capacity,
        )
        result[field] = str(path.relative_to(base))
    return result


def attach_rejection(model, base):
    """Reuse the Markov proposal bank and categorical merge for TP rejection."""
    from kernels.ascend import sampling_rejection as implementation
    from kernels.shared.rows import row_count
    from orinfer_build.compiler.ascend.build import compile_kernel, operand

    sampler = model.get("sampling")
    if not sampler or not sampler.get("coupled_draft"):
        raise ValueError("Rejection sampling requires the shared stochastic Markov head")
    strides = {
        p.get("draft_stride")
        for r in model["ranks"]
        for p in r["packets"]
        if p.get("role") == "draft"
    }
    if len(strides) != 1 or None in strides:
        raise ValueError("Rejection sampling requires one physical Markov stride")
    stride = next(iter(strides))
    capacity, vocab = sampler["capacity"], sampler["vocab"]
    pool = model["batch"]["owners"] * stride
    ranks, parts = len(model["ranks"]), (vocab + 1023) // 1024
    result = dict(pool_rows=pool, stride=stride, ranks=ranks)
    f32, i32 = "f32", "i32"
    logits = ("logits", f32, (capacity, vocab), "read")
    draft = ("draft", f32, (pool, vocab), "read")
    controls = ("controls", i32, (capacity, 4), "read")
    proposals = ("proposals", i32, (capacity, 4), "read")
    active = ("active", i32, (1,), "read")
    statistics = ("partial", f32, (capacity, parts, 4), "read")
    for name, arguments, ports in [
        (
            "statistics",
            dict(vocab=vocab, pool_rows=pool),
            [
                logits,
                draft,
                controls,
                proposals,
                active,
                ("partial", f32, (capacity, parts, 4), "write"),
            ],
        ),
        (
            "merge_statistics",
            dict(vocab=vocab, pool_rows=pool),
            [
                statistics,
                logits,
                draft,
                controls,
                proposals,
                active,
                ("out", f32, (capacity, 8), "write"),
            ],
        ),
        (
            "normalize",
            dict(ranks=ranks),
            [
                ("gathered", f32, (ranks, capacity, 8), "read"),
                proposals,
                active,
                ("out", f32, (capacity, 4), "write"),
            ],
        ),
        (
            "select",
            dict(vocab=vocab, pool_rows=pool),
            [
                logits,
                draft,
                controls,
                proposals,
                ("normalizers", f32, (capacity, 4), "read"),
                active,
                ("partial", f32, (capacity, parts, 2), "write"),
            ],
        ),
    ]:
        path = compile_kernel(
            getattr(implementation, name)(rows=row_count(capacity, True), **arguments),
            "sampling_rejection." + name,
            [operand(*p) for p in ports],
            arguments,
            base / "cache",
            row_capacity=capacity,
        )
        result[name] = str(path.relative_to(base))
    sampler["rejection"] = result
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile-workers", type=int, default=1)
    parser.add_argument(
        "--couple-draft", action="store_true", help="Extend existing target sampling"
    )
    parser.add_argument(
        "--processed", action="store_true", help="Attach masks and probability summaries"
    )
    parser.add_argument("--rejection", action="store_true", help="Attach TP probability rejection")
    args = parser.parse_args()
    if (
        args.plan.parent.resolve() != args.output.parent.resolve()
        or args.plan.resolve() == args.output.resolve()
    ):
        parser.error("Preserve the input and keep its asset directory")
    model = decode(json.loads(args.plan.read_text()))
    if sum((args.processed, args.couple_draft, args.rejection)) > 1:
        parser.error("Select one sampler extension")
    transform = (
        attach_rejection
        if args.rejection
        else attach_processed
        if args.processed
        else couple_draft
        if args.couple_draft
        else attach
    )
    print(json.dumps(transform(model, args.plan.parent)), flush=True)
    atomic_json(args.output, model)


if __name__ == "__main__":
    main()
