"""Share exact FP32 router columns across decode ranks using existing operators.

The full projection's K splits and FP32 reduction order stay fixed. Aligned
column spans overlap at the tail, and AllGather restores the 384 logical logits
before the original TopK. Expert placement, weights and request state stay fixed.
"""

from copy import deepcopy
import json
import math

from orinfer_build.models.deepseek_v41.transform import (
    compile_and_replace,
    define_buffer,
    finalize_plan,
    plan_parser,
    stage_plan,
)
from orinfer_build.contracts.row_contract import expression


def column_spans(output, ranks, alignment=32):
    if min(output, ranks, alignment) < 1 or output % ranks:
        raise ValueError("Router columns must divide the rank count")
    logical = output // ranks
    physical = math.ceil(logical / alignment) * alignment
    if physical > output:
        raise ValueError("Router matrix is smaller than its physical shard")
    return physical, [min(rank * logical, output - physical) for rank in range(ranks)]


def build(plan, output, workers=4):
    from orinfer_build.models.deepseek_v41.build import compile_task
    from orinfer_build.compiler.ascend.build import operand

    model, base, describe = stage_plan(plan, output, workers=workers)
    peers = len(model["ranks"])
    if [rank["rank"] for rank in model["ranks"]] != list(range(peers)):
        raise ValueError("Router shards require ordered ranks")
    tasks = {}

    def kernel(op, arguments, specs, capacity):
        key = json.dumps([op, arguments, specs, capacity], sort_keys=True)
        if key not in tasks:
            module, name = op.split(".")
            tasks[key] = dict(
                module="kernels.ascend." + module,
                name=name,
                arguments=arguments,
                specs=specs,
                row_capacity=capacity,
                dynamic_arguments=dict(rows=1),
                placeholder=f"@router{len(tasks)}",
            )
        return tasks[key]["placeholder"]

    changes = 0
    for rank in model["ranks"]:
        definitions = {b["name"]: b for b in rank["buffers"]}

        def buffer(name, size, **kwargs):
            return define_buffer(rank, definitions, name, size, label="router shard", **kwargs)

        for packet in rank["packets"]:
            if not packet.get("batch_decode"):
                continue
            capacity, body = packet["rows"], []
            for step in packet["body"]:
                route = describe(step)
                if route is None or route["op"] != "routing.route":
                    body.append(step)
                    continue
                if len(body) < 2:
                    raise ValueError("Missing router projection")
                part, reduction = body[-2:]
                p, q = describe(part), describe(reduction)
                if (
                    p is None
                    or q is None
                    or p["op"] != "dense.fp32_partials"
                    or q["op"] != "dense.fp32_reduce"
                    or part["buffers"][-1] != reduction["buffers"][0]
                    or reduction["buffers"][-1] != step["buffers"][0]
                    or p["parameters"].get("activation_dtype") != "float32"
                    or p["parameters"]["_row_capacity"] != capacity
                ):
                    raise ValueError("Router requires the original FP32 projection/reduction pair")
                columns, k = p["parameters"]["output"], p["parameters"]["reduction"]
                physical, starts = column_spans(columns, peers)
                splits = p["parameters"].get("splits", 8)
                if q["parameters"].get("splits", 8) != splits:
                    raise ValueError("Router projection split order differs")
                weight = part["buffers"][1]
                if definitions[weight]["bytes"] != columns * k * 4:
                    raise ValueError("Router FP32 weight extent differs")
                view = buffer(
                    weight + ".router_shard",
                    physical * k * 4,
                    view=dict(parent=weight, offset=starts[rank["rank"]] * k * 4),
                )
                prefix = f"batch.router.shard.m{capacity}."
                scratch = buffer(prefix + "partials", capacity * physical * splits * 4)
                send = buffer(prefix + "send", capacity * physical * 4)
                receive = buffer(prefix + "receive", peers * capacity * physical * 4)
                args = {
                    key: value for key, value in p["parameters"].items() if not key.startswith("_")
                }
                args.update(rows=capacity, output=physical)
                a = deepcopy(part)
                a["manifest"] = kernel(
                    "dense.fp32_partials",
                    args,
                    [
                        operand("activation", "f32", [capacity, k]),
                        operand("weight", "f32", [physical, k]),
                        operand("out", "f32", [capacity, physical // 32, splits, 32], "write"),
                    ],
                    capacity,
                )
                a["buffers"] = [part["buffers"][0], view, scratch]
                args = {
                    key: value for key, value in q["parameters"].items() if not key.startswith("_")
                }
                args.update(rows=capacity, output=physical)
                b = deepcopy(reduction)
                b["manifest"] = kernel(
                    "dense.fp32_reduce",
                    args,
                    [
                        operand("partials", "f32", [capacity, physical // 32, splits, 32]),
                        operand("out", "f32", [capacity, physical], "write"),
                    ],
                    capacity,
                )
                b["buffers"] = [scratch, send]
                restore = deepcopy(step)
                restore["manifest"] = kernel(
                    "memory.gather_columns",
                    dict(
                        rows=capacity,
                        width=physical * peers,
                        ranks=peers,
                        dtype="float32",
                        output_width=columns,
                    ),
                    [
                        operand("gathered", "f32", [peers, capacity, physical]),
                        operand("out", "f32", [capacity, columns], "write"),
                    ],
                    capacity,
                )
                restore["buffers"] = [receive, step["buffers"][0]]
                body[-2:] = [
                    a,
                    b,
                    dict(
                        op="gather",
                        source=send,
                        destination=receive,
                        elements=capacity * physical,
                        fp32=True,
                        elements_expression=expression(f"m * {physical}"),
                    ),
                    restore,
                ]
                body.append(step)
                changes += 1
            packet["body"] = body
    if not changes:
        raise ValueError("No decode router projections found")
    compile_and_replace(model, tasks, base, workers, compiler=compile_task)
    finalize_plan(
        model,
        base,
        output,
        save_base=True,
        report_name="router-sharding.json",
        report=dict(projections=changes),
    )
    return output / "package/model.json"


def main():
    args = plan_parser(__doc__, default_workers=4, model_dir=False).parse_args()
    print(build(args.plan, args.output, args.compile_workers))


if __name__ == "__main__":
    main()
