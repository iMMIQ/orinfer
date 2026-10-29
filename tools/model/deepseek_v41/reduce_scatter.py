"""Build a ReduceScatter candidate from an already sequence-sharded A2 plan."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path

from tools.build.files import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile-workers", type=int, default=2)
    args = parser.parse_args()
    if args.plan.parent.resolve() != args.output.parent.resolve():
        parser.error("Input and output must share the asset directory")
    from tools.model.deepseek_v41.build import compile_task
    from tools.operators.ascend.build import operand

    model = json.loads(args.plan.read_text())
    if len(model["ranks"]) != 8:
        raise ValueError("ReduceScatter expects the native TP8 row partition")
    tasks = {}
    for rank in model["ranks"]:
        partitions = {}
        for definition in rank["buffers"]:
            if "view" in definition:
                partitions.setdefault(definition["view"]["parent"], []).append(definition)
        changed = 0
        for packet in rank["packets"]:
            if packet.get("min_rows", 0) < 512:
                continue
            capacity = packet["rows"]
            if capacity % 8:
                raise ValueError("Sequence capacity is not divisible by TP8")
            body, active = [], True
            for step in packet["body"]:
                if step["op"] == "limit_rows":
                    active = False
                if not active or step["op"] != "reduce":
                    body.append(step)
                    continue
                if (
                    step.get("scatter")
                    or step.get("fixed_rows")
                    or step["elements"] != capacity * 5120
                ):
                    raise ValueError("Unexpected unsharded collective geometry")
                dtype, size = ("f32", 4) if step["fp32"] else ("bf16", 2)
                destination = step["destination"]
                views = partitions.get(destination, [])
                if len(views) != 1:
                    raise ValueError("ReduceScatter requires one explicit local result view")
                view = views[0]
                count = capacity // 8 * 5120
                if view["bytes"] != count * size or (
                    "view" in view and view["view"]["offset"] != rank["rank"] * count * size
                ):
                    raise ValueError("The consumer view differs from its rank's contiguous rows")
                for consumer in packet["body"] + packet["head"]:
                    if consumer["op"] == "kernel" and destination in consumer["buffers"]:
                        raise ValueError("A full collective result still has a direct consumer")
                view.pop("view", None)
                key = json.dumps([capacity, dtype])
                if key not in tasks:
                    tasks[key] = dict(
                        module="tools.operators.ascend.memory",
                        name="zero_row_tail",
                        arguments=dict(
                            rows=capacity,
                            capacity=capacity,
                            width=5120,
                            dtype="float32" if step["fp32"] else "bfloat16",
                        ),
                        specs=[
                            operand("positions", "i32", [capacity]),
                            operand("out", dtype, [capacity, 5120], "read_write"),
                        ],
                        row_capacity=capacity,
                        dynamic_arguments={"rows": 1},
                        placeholder=f"@reduce_scatter{len(tasks)}",
                    )
                body.append(
                    dict(
                        op="kernel",
                        manifest=tasks[key]["placeholder"],
                        buffers=[packet["coordinates"]["slots"], step["source"]],
                    )
                )
                body.append(
                    dict(
                        step,
                        destination=view["name"],
                        elements=count,
                        scatter=True,
                        fixed_rows=True,
                    )
                )
                changed += 1
            packet["body"] = body
        if changed == 0:
            raise ValueError("No sequence-parallel collectives were selected")
        print(dict(rank=rank["rank"], reduce_scatter=changed), flush=True)
    replacements = {}
    with ProcessPoolExecutor(
        max_workers=args.compile_workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        for placeholder, path in pool.map(
            compile_task, [(t, str(args.plan.parent)) for t in tasks.values()]
        ):
            replacements[placeholder] = path
    for rank in model["ranks"]:
        for packet in rank["packets"]:
            for step in packet["body"]:
                if step["op"] == "kernel":
                    step["manifest"] = replacements.get(step["manifest"], step["manifest"])
    atomic_json(args.output, model)


if __name__ == "__main__":
    main()
