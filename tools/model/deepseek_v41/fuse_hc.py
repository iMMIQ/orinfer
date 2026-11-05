"""Fuse the shared eight-split normalization and mHC Sinkhorn boundary offline."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

from tools.build.files import atomic_json
from tools.model.deepseek_v41.assembly import (
    check_intermediate_liveness,
    publish_package,
    validate_rows,
)
from tools.model.deepseek_v41.batch import compile_task


def chain(normalize, sinkhorn, describe):
    """Only combine a complete, same-row producer and consumer contract."""
    n, s = describe(normalize), describe(sinkhorn)
    if not n or n["op"] != "hc_prefill.reduce":
        return None
    fields = {"op", "buffers", "manifest", "fixed_rows", "row_expression"}
    if (
        not s
        or s["op"] != "vector.mhc_split_sinkhorn"
        or set(normalize) - fields
        or set(sinkhorn) - fields
        or normalize.get("fixed_rows") != sinkhorn.get("fixed_rows")
        or normalize.get("row_expression") != sinkhorn.get("row_expression")
        or normalize["buffers"][1] != sinkhorn["buffers"][0]
        or n["parameters"]["rows"] != s["parameters"]["rows"]
        or n["parameters"].get("_row_capacity") != s["parameters"].get("_row_capacity")
        or n["parameters"].get("splits", 8) != 8
        or n["parameters"].get("eps", 1e-20) != 1e-20
        or s["parameters"].get("tile_rows", 1) != 32
    ):
        raise ValueError("mHC fusion requires the matching eight-split and Sinkhorn contract")
    return n, s


def build(plan, output, model_dir=None, *, tile_rows=32, capacity_tiles=None):
    if tile_rows not in (2, 4, 8, 16, 32):
        raise ValueError("Unsupported mHC row tile")
    capacity_tiles = capacity_tiles or {}
    if any(
        capacity < 1 or tile not in (2, 4, 8, 16, 32) for capacity, tile in capacity_tiles.items()
    ):
        raise ValueError("Invalid mHC physical capacity tuning")
    if output.exists():
        raise ValueError("Require a new mHC build directory")
    model = json.loads(plan.read_text())
    base = output / "source"
    publish_package(model, plan.parent, base)
    descriptions, tasks, changed = {}, {}, 0

    def describe(step):
        if step.get("op") != "kernel":
            return None
        name = step["manifest"]
        if name not in descriptions:
            descriptions[name] = json.loads((base / name).read_text())
        return descriptions[name]

    for rank in model["ranks"]:
        definitions = {d["name"]: d for d in rank["buffers"]}
        for packet in rank["packets"]:
            # The same fused implementation remains available to other roles.
            # Keep the target interval and physical row expressions, including
            # sequence-parallel shards and the decoder suffix domain.
            if (
                packet.get("role", "target") != "target"
                or packet.get("batch_decode")
                or packet.get("min_rows", 1) < 32
            ):
                continue
            program, result, index = packet["body"], [], 0
            while index < len(program):
                match = (
                    chain(program[index], program[index + 1], describe)
                    if index + 1 < len(program)
                    else None
                )
                if match is None:
                    result.append(program[index])
                    index += 1
                    continue
                n, s = match
                first, second = program[index : index + 2]
                check_intermediate_liveness(
                    program[index + 2 :],
                    [first["buffers"][1]],
                    describe,
                    model["batch"],
                    definitions,
                )
                params = dict(
                    rows=n["parameters"]["rows"],
                    width=n["parameters"]["width"],
                    iterations=s["parameters"].get("iterations", 20),
                    eps=s["parameters"].get("eps", 1e-6),
                    tile_rows=capacity_tiles.get(n["parameters"]["_row_capacity"], tile_rows),
                )
                key = "@hc:" + json.dumps(params, sort_keys=True)
                if key not in tasks:
                    specs = deepcopy([n["operands"][0], *s["operands"][1:]])
                    specs[0]["name"] = "mixes"
                    tasks[key] = dict(
                        key=key,
                        module="tools.operators.ascend.vector",
                        name="mhc_normalized_sinkhorn",
                        arguments=params,
                        capacity=n["parameters"]["_row_capacity"],
                        dynamic=dict(rows=1),
                        specs=specs,
                    )
                result.append(
                    dict(
                        op="kernel",
                        manifest=key,
                        buffers=[first["buffers"][0], *second["buffers"][1:]],
                        **{k: first[k] for k in ("fixed_rows", "row_expression") if k in first},
                    )
                )
                changed += 1
                index += 2
            packet["body"] = result
    if not changed:
        raise ValueError("No eligible mHC boundaries")
    compiled = dict(compile_task((task, base)) for task in tasks.values())
    for rank in model["ranks"]:
        for packet in rank["packets"]:
            for step in packet["body"]:
                if step.get("manifest") in compiled:
                    step["manifest"] = compiled[step["manifest"]]
    validate_rows(model, base, model_dir=model_dir)
    publish_package(model, base, output / "package")
    atomic_json(output / "result.json", dict(changed=changed, kernels=len(tasks)))
    return output / "package" / "model.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--tile-rows", type=int, choices=(2, 4, 8, 16, 32), default=32)
    parser.add_argument(
        "--capacity-tiles",
        type=Path,
        help="Offline measured row tile by physical capacity; one dynamic kernel per capacity",
    )
    args = parser.parse_args()
    capacity_tiles = (
        {int(k): v for k, v in json.loads(args.capacity_tiles.read_text()).items()}
        if args.capacity_tiles
        else None
    )
    print(
        build(
            args.plan,
            args.output,
            args.model_dir,
            tile_rows=args.tile_rows,
            capacity_tiles=capacity_tiles,
        )
    )


if __name__ == "__main__":
    main()
