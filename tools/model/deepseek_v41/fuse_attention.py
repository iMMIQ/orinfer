"""Replace DSv4.1 prefill QK/softmax/PV chains with a shared tiled A2 kernel."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
import multiprocessing
from pathlib import Path

from tools.build.files import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tile-m", type=int, choices=[16, 32, 64], default=32)
    parser.add_argument("--tile-k", type=int, choices=[128, 256], default=128)
    parser.add_argument("--reuse-kv", action="store_true")
    parser.add_argument(
        "--owner-min-rows",
        type=int,
        default=0,
        help="Keep the original chain below this owner-local query length",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=2,
        help="Select packets whose declared minimum row count is at least this value",
    )
    parser.add_argument("--compile-workers", type=int, default=3)
    args = parser.parse_args()
    if args.plan.parent.resolve() != args.output.parent.resolve():
        parser.error("Input and output must share the asset directory")
    if args.min_rows < 2:
        parser.error("Fused prefill requires a minimum of two rows")
    from tools.model.deepseek_v41.build import compile_task
    from tools.operators.ascend.build import operand

    base = args.plan.parent
    model = json.loads(args.plan.read_text())
    tasks, descriptors = {}, {}

    def descriptor(step):
        path = step["manifest"]
        if path not in descriptors:
            descriptors[path] = json.loads((base / path).read_text())
        return descriptors[path]

    for rank in model["ranks"]:
        definitions = {b["name"]: b for b in rank["buffers"]}
        converted = 0
        for packet in rank["packets"]:
            if packet.get("min_rows", 0) < args.min_rows:
                continue
            body, old, index, count = [], packet["body"], 0, 0
            capacity = packet["rows"]
            while index < len(old):
                step = old[index]
                if step["op"] != "kernel" or descriptor(step)["op"] != "attention_prefill.qk":
                    body.append(step)
                    index += 1
                    continue
                qk = descriptor(step)
                if qk["parameters"].get("_row_capacity") != capacity or step.get("fixed_rows"):
                    raise ValueError("Expected the original dynamic prefill attention chain")
                softmax = []
                end = index + 1
                while (
                    end < len(old)
                    and old[end]["op"] == "kernel"
                    and descriptor(old[end])["op"] == "attention_prefill.softmax"
                ):
                    softmax.append(old[end])
                    end += 1
                if (
                    not softmax
                    or end == len(old)
                    or old[end]["op"] != "kernel"
                    or descriptor(old[end])["op"] != "attention_prefill.pv"
                ):
                    raise ValueError("Incomplete QK/softmax/PV chain")
                pv = old[end]
                if any(s["buffers"] != softmax[0]["buffers"] for s in softmax):
                    raise ValueError("Softmax buckets disagree on their operands")
                if (
                    step["buffers"][4] != softmax[0]["buffers"][0]
                    or pv["buffers"][0] != softmax[0]["buffers"][4]
                ):
                    raise ValueError("Attention intermediates have unexpected consumers")
                parameters = {
                    k: qk["parameters"][k] for k in ("rows", "capacity", "ratio", "compressed")
                }
                parameters["tile_m"] = args.tile_m
                parameters["tile_k"] = (
                    args.tile_k
                    if (
                        not parameters["compressed"]
                        or (parameters["capacity"] // parameters["ratio"]) % args.tile_k == 0
                    )
                    else 128
                )
                parameters["reuse_kv"] = args.reuse_kv
                c, m = parameters["capacity"], capacity
                long_length, blocks = (
                    c // parameters["ratio"],
                    (m * 8 + args.tile_m - 1) // args.tile_m,
                )
                scratch, tile_k = [], parameters["tile_k"]
                for name, dtype, shape, size in (
                    (
                        "scores",
                        "f32",
                        [blocks, args.tile_m, tile_k],
                        blocks * args.tile_m * tile_k * 4,
                    ),
                    (
                        "probabilities",
                        "bf16",
                        [blocks, args.tile_m, tile_k],
                        blocks * args.tile_m * tile_k * 2,
                    ),
                    ("output", "f32", [blocks, args.tile_m, 512], blocks * args.tile_m * 512 * 4),
                ):
                    name = f"attention.fused.{m}.{args.tile_m}.{tile_k}.{int(args.reuse_kv)}.{name}"
                    definition = dict(name=name, bytes=size)
                    if name not in definitions:
                        rank["buffers"].append(definition)
                        definitions[name] = definition
                    elif definitions[name] != definition:
                        raise ValueError("Conflicting fused attention scratch")
                    scratch.append((name, dtype, shape))
                specs = [
                    operand("query", "bf16", [m * 8, 512]),
                    operand("swa", "bf16", [c, 512]),
                    operand("long_kv", "bf16", [long_length, 512]),
                    operand("positions", "i32", [m]),
                    operand("selected", "i32", [m, long_length]),
                    operand("sink", "f32", [8]),
                    operand("output", "bf16", [m * 8, 512], "write"),
                ] + [
                    operand(name, dtype, shape, "scratch")
                    for name, (_, dtype, shape) in zip(
                        ("score_tile", "probability_tile", "output_tile"), scratch, strict=True
                    )
                ]
                key = json.dumps(parameters, sort_keys=True)
                if key not in tasks:
                    tasks[key] = dict(
                        module="tools.operators.ascend.attention_fused",
                        name="prefill",
                        arguments=parameters,
                        specs=specs,
                        row_capacity=m,
                        dynamic_arguments={"rows": 1},
                        placeholder=f"@fused_attention{len(tasks)}",
                    )
                if args.owner_min_rows:
                    if not 2 <= args.owner_min_rows <= capacity:
                        raise ValueError("Invalid owner attention selection boundary")
                    for original in old[index : end + 1]:
                        original = deepcopy(original)
                        original["owner_range"] = [1, args.owner_min_rows - 1]
                        body.append(original)
                fused = dict(
                    op="kernel",
                    manifest=tasks[key]["placeholder"],
                    buffers=step["buffers"][:4]
                    + softmax[0]["buffers"][2:4]
                    + [pv["buffers"][4]]
                    + [s[0] for s in scratch],
                )
                if args.owner_min_rows:
                    fused["owner_range"] = [args.owner_min_rows, capacity]
                body.append(fused)
                index, count = end + 1, count + 1
            if count not in (0, 40):
                raise ValueError("The prefill packet does not cover 40 attention layers")
            converted += count
            packet["body"] = body
        if not converted:
            raise ValueError("No original prefill attention packets match the selected minimum")
        # Remove the obsolete full score/probability views and their banks.
        used = {
            name
            for p in rank["packets"]
            for s in p["body"] + p["head"]
            if s["op"] == "kernel"
            for name in s["buffers"]
        }
        pending = list(used)
        while pending:
            definition = definitions[pending.pop()]
            if "view" in definition:
                parent = definition["view"]["parent"]
                if parent not in used:
                    used.add(parent)
                    pending.append(parent)
        for definition in list(rank["buffers"]):
            name = definition["name"]
            if name not in used and any(
                marker in name
                for marker in (
                    "attention.packed.score",
                    "attention.packed.prob",
                    "attention.packed.workspace",
                )
            ):
                rank["buffers"].remove(definition)
    manifests = {}
    with ProcessPoolExecutor(
        max_workers=args.compile_workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        for placeholder, path in pool.map(compile_task, [(t, str(base)) for t in tasks.values()]):
            manifests[placeholder] = path
            print(dict(compiled=len(manifests), total=len(tasks)), flush=True)
    for rank in model["ranks"]:
        for packet in rank["packets"]:
            for step in packet["body"]:
                if step["op"] == "kernel":
                    step["manifest"] = manifests.get(step["manifest"], step["manifest"])
    atomic_json(args.output, model)


if __name__ == "__main__":
    main()
