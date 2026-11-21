"""Project only this TP rank's indexer query rows with existing A2 operators."""

from importlib import import_module

from copy import deepcopy
import json
import math

from orinfer_build.models.deepseek_v41.batch import SIZES, compile_task
from orinfer_build.models.deepseek_v41.assembly import (
    check_intermediate_liveness as check_query_liveness,
    selected_target_packet,
)
from orinfer_build.models.deepseek_v41.transform import (
    compile_and_replace,
    finalize_plan,
    plan_parser,
    stage_plan,
)
from orinfer_build.contracts.row_contract import expression


def query_chain(steps, index, describe):
    """Recognize the complete projection/RoPE/quantization/partition boundary."""
    first = steps[index]
    if first["op"] != "kernel":
        return None
    desc = describe(first)
    if desc["op"] != "dense.projection" or not any(
        ".indexer.wq_b." in name for name in first["buffers"]
    ):
        return None
    chain = steps[index : index + 7]
    expected = [
        "dense.projection",
        "model_ops.partial_rope",
        "vector.quantize",
        "dense.projection",
        "memory.partition_rows",
        "memory.partition_rows",
        "memory.partition_rows",
    ]
    if (
        len(chain) != 7
        or any(s["op"] != "kernel" for s in chain)
        or [describe(s)["op"] for s in chain] != expected
    ):
        raise ValueError("Require the complete indexer query producer and its three partitions")
    q, rope, quant, weights, *parts = chain
    if not (
        rope["buffers"][0] == q["buffers"][4]
        and quant["buffers"][0] == rope["buffers"][3]
        and [s["buffers"][0] for s in parts] == [*quant["buffers"][1:], weights["buffers"][4]]
        and any(".indexer.weights_proj." in n for n in weights["buffers"])
    ):
        raise ValueError("Indexer query intermediate buffers do not match")
    return chain


def build(plan, output, workers=4, minimum=512, query_tile_m=128, maximum=None, model_dir=None):
    from kernels.shared.rows import row_count
    from orinfer_build.compiler.ascend.build import operand
    from orinfer_build.contracts.ascend.geometry import physical_shapes

    model, base, describe = stage_plan(
        plan,
        output,
        workers=workers,
        minimum=minimum,
        maximum=maximum,
        validate_source=True,
        model_dir=model_dir,
    )
    peers = len(model["ranks"])
    if peers != 8 or not model.get("batch", {}).get("partitioned_tables"):
        raise ValueError("Require the TP8 owner-row indexer partitions")
    tasks, counts = {}, []

    def register(module, name, arguments, specs, capacity, factor=1):
        task = dict(
            module=module,
            name=name,
            arguments=arguments,
            specs=specs,
            capacity=capacity,
            dynamic=dict(rows=factor),
        )
        key = json.dumps(task, sort_keys=True)
        if key not in tasks:
            tasks[key] = dict(task, key=f"@local_indexer{len(tasks)}")
        return tasks[key]["key"]

    for rank in model["ranks"]:
        definitions = {b["name"]: b for b in rank["buffers"]}

        def define(name, size, parent=None):
            value = dict(name=name, bytes=size)
            if parent is not None:
                if definitions[parent]["bytes"] < size:
                    raise ValueError("Local projection view exceeds its shared allocation")
                value["view"] = dict(parent=parent, offset=0)
            if name in definitions:
                if definitions[name] != value:
                    raise ValueError("Conflicting local query allocation: " + name)
            else:
                rank["buffers"].append(value)
                definitions[name] = value
            return name

        for packet in rank["packets"]:
            if not selected_target_packet(packet, minimum, maximum):
                continue
            capacity = packet["rows"]
            if capacity % peers:
                raise ValueError("Indexer row capacity must divide into equal TP partitions")
            count = capacity // peers
            prefix = f"batch.indexer.local.m{capacity}."

            def partition(source, label, width, dtype):
                target = define(prefix + label, count * width * SIZES[dtype])
                arguments = dict(
                    rows=capacity,
                    count=count,
                    width=width,
                    first=rank["rank"],
                    stride=peers,
                    dtype=dict(i8="int8", f32="float32", bf16="bfloat16")[dtype],
                    dynamic_count=True,
                )
                manifest = register(
                    "kernels.ascend.memory",
                    "partition_rows",
                    arguments,
                    [
                        operand("source", dtype, [capacity, width]),
                        operand("out", dtype, [count, width], "write"),
                    ],
                    capacity,
                )
                return target, dict(op="kernel", manifest=manifest, buffers=[source, target])

            def local(step, bindings, factor=1):
                desc = describe(step)
                module, name = desc["op"].split(".")
                args = {k: v for k, v in desc["parameters"].items() if not k.startswith("_")}
                if args["rows"] != capacity * factor:
                    raise ValueError("Query row multiplier differs from its original geometry")
                args["rows"] = count * factor
                if module == "dense" and args.get("integer"):
                    args["tile_m"] = query_tile_m
                factory = getattr(import_module("kernels.ascend." + module), name)
                shapes = physical_shapes(
                    factory(**dict(args, rows=row_count(count, True) * factor)), count
                )
                specs, buffers = deepcopy(desc["operands"]), []
                for i, (old, spec, shape) in enumerate(
                    zip(step["buffers"], specs, shapes, strict=True)
                ):
                    spec["shape"] = shape
                    size = math.prod(shape) * SIZES[spec["dtype"]]
                    target = bindings.get(i, old)
                    if i not in bindings and spec["access"] == "scratch":
                        target = define(
                            prefix + f"workspace.{module}.{args.get('output')}", size, old
                        )
                    elif (
                        i not in bindings
                        and old.startswith("ones")
                        and size < definitions[old]["bytes"]
                    ):
                        target = define(prefix + f"view.{old}.{size}", size, old)
                    if definitions[target]["bytes"] < size:
                        raise ValueError("Local query operand is smaller than its AOT descriptor")
                    buffers.append(target)
                manifest = register("kernels.ascend." + module, name, args, specs, count, factor)
                return dict(
                    op="kernel",
                    manifest=manifest,
                    buffers=buffers,
                    row_expression=expression(f"(m + {peers - 1}) // {peers}"),
                )

            old, body, index, changed = packet["body"], [], 0, 0
            while index < len(old):
                chain = query_chain(old, index, describe)
                if chain is None:
                    body.append(old[index])
                    index += 1
                    continue
                q, rope, quant, weights, *parts = chain
                check_query_liveness(
                    old[index + len(chain) :] + packet["head"],
                    [
                        q["buffers"][4],
                        rope["buffers"][3],
                        *quant["buffers"][1:],
                        weights["buffers"][4],
                    ],
                    describe,
                    model["batch"],
                    definitions,
                )
                for part in parts:
                    p = describe(part)["parameters"]
                    if (p["stride"], p["first"], p["rows"]) != (peers, rank["rank"], capacity):
                        raise ValueError("Indexer output partitions differ from the query rank")
                inputs = []
                for source, label, width, dtype in [
                    (q["buffers"][0], "activation", 1280, "i8"),
                    (q["buffers"][2], "scale", 1, "f32"),
                    (weights["buffers"][0], "hidden", 5120, "bf16"),
                    (rope["buffers"][1], "cosine", 32, "f32"),
                    (rope["buffers"][2], "sine", 32, "f32"),
                ]:
                    target, launch = partition(source, label, width, dtype)
                    inputs.append(target)
                    body.append(launch)
                query = define(prefix + "query", count * 4096 * 2)
                rotated = define(prefix + "rotated", count * 4096 * 2)
                body += [
                    local(q, {0: inputs[0], 2: inputs[1], 4: query}),
                    local(rope, {0: query, 1: inputs[3], 2: inputs[4], 3: rotated}),
                    local(
                        quant,
                        {0: rotated, 1: parts[0]["buffers"][1], 2: parts[1]["buffers"][1]},
                        factor=32,
                    ),
                    local(weights, {0: inputs[2], 4: parts[2]["buffers"][1]}),
                ]
                changed += 1
                index += len(chain)
            packet["body"] = body
            counts.append(dict(rank=rank["rank"], family=packet["min_rows"], queries=changed))
    compile_and_replace(model, tasks, base, workers, compiler=compile_task)
    return finalize_plan(
        model,
        base,
        output,
        owner_tables=True,
        model_dir=model_dir,
        report_name="local-indexer.json",
        report=dict(
            minimum=minimum,
            query_tile_m=query_tile_m,
            kernels=len(tasks),
            families=counts,
            additional_owner_bytes=0,
        ),
    )


def main():
    parser = plan_parser(__doc__, default_workers=4, min_rows=512, max_rows=True)
    parser.add_argument("--query-tile-m", type=int, default=128)
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.plan,
                args.output,
                args.compile_workers,
                args.min_rows,
                args.query_tile_m,
                args.max_rows,
                args.model_dir,
            )
        )
    )


if __name__ == "__main__":
    main()
