"""Upload disjoint BF16 embedding columns and fuse their gather with HC expansion."""

import json

from orinfer_build.models.deepseek_v41.assembly import check_intermediate_liveness
from orinfer_build.models.deepseek_v41.transform import (
    compile_and_replace,
    define_buffer,
    finalize_plan,
    plan_parser,
    stage_plan,
)
from orinfer_build.contracts.row_contract import expression


def transform(model, describe, kernel):
    width = 5120
    peers = len(model["ranks"])
    if (
        peers < 1
        or width % peers
        or (width // peers * 2) % 32
        or [r["rank"] for r in model["ranks"]] != list(range(peers))
    ):
        raise ValueError("Embedding requires ordered, aligned column shards")
    columns = width // peers
    changes = 0
    for rank in model["ranks"]:
        definitions = {b["name"]: b for b in rank["buffers"]}

        def buffer(name, size, **kwargs):
            return define_buffer(rank, definitions, name, size, label="device embedding", **kwargs)

        def emit(name, arguments, ports, capacity):
            specs = [
                dict(name=role, dtype=dtype, shape=shape, access=access)
                for role, _, dtype, shape, access in ports
            ]
            return dict(
                op="kernel",
                manifest=kernel(name, arguments, specs, capacity),
                buffers=[p[1] for p in ports],
            )

        for packet in rank["packets"]:
            original, body = packet["body"], []
            index = 0
            while index < len(original):
                step = original[index]
                if step["op"] != "embed":
                    body.append(step)
                    index += 1
                    continue
                source = step["destination"]
                capacity = packet["rows"]
                if definitions[source]["bytes"] != capacity * width * 2:
                    raise ValueError("Embedding input capacity differs")
                index += 1
                if index >= len(original):
                    raise ValueError("Embedding must feed the original HC expansion")
                tail = original[index]
                spec = describe(tail)
                if spec and spec["op"] == "memory.zero_row_tail":
                    p = spec["parameters"]
                    if (
                        tail["buffers"][-1] != source
                        or p["width"] != width
                        or p["dtype"] != "bfloat16"
                        or p.get("partitions") != peers
                        or "first" in p
                    ):
                        raise ValueError("Embedding tail clear differs")
                    index += 1
                if index >= len(original):
                    raise ValueError("Embedding must feed the original HC expansion")
                expand = original[index]
                spec = describe(expand)
                if spec is None or spec["op"] != "vector.mhc_expand":
                    raise ValueError("Embedding must feed the original HC expansion")
                p = spec["parameters"]
                local = p["rows"]
                if p["width"] != width or len(expand["buffers"]) != 2:
                    raise ValueError("Embedding HC geometry differs")
                incoming = expand["buffers"][0]
                partitions, first = 1, 0
                if incoming != source:
                    view = definitions[incoming].get("view", {})
                    partitions, first = peers, rank["rank"]
                    expected = expression(f"((m + {peers - 1}) // {peers}) * {first * width * 2}")
                    if (
                        view.get("parent") != source
                        or view.get("row_capacity") != capacity
                        or view.get("offset") != local * first * width * 2
                        or view.get("offset_expression") != expected
                        or local * peers != capacity
                        or expand.get("row_expression")
                        != expression(f"(m + {peers - 1}) // {peers}")
                    ):
                        raise ValueError("Embedding partition must preserve its row coordinates")
                elif local != capacity or "row_expression" in expand:
                    raise ValueError("Unpartitioned embedding row capacity differs")
                send = buffer(source + ".columns", capacity * columns * 2)
                body.append(
                    dict(
                        op="embed",
                        destination=send,
                        first_column=rank["rank"] * columns,
                        columns=columns,
                    )
                )
                gathered = source
                collective = dict(
                    op="gather",
                    source=send,
                    destination=gathered,
                    elements=capacity * columns,
                    fp32=False,
                    elements_expression=expression(f"m * {columns}"),
                )
                if partitions > 1:
                    gathered = buffer(
                        source + ".received",
                        peers * local * columns * 2,
                        view=dict(parent=source, offset=0),
                    )
                    body.append(
                        emit(
                            "zero_row_tail",
                            dict(
                                rows=capacity,
                                capacity=capacity,
                                width=columns,
                                dtype="bfloat16",
                                partitions=peers,
                            ),
                            [
                                (
                                    "positions",
                                    packet["coordinates"]["slots"],
                                    "i32",
                                    [capacity],
                                    "read",
                                ),
                                ("out", send, "bf16", [capacity, columns], "read_write"),
                            ],
                            capacity,
                        )
                    )
                    collective = dict(
                        op="exchange",
                        source=send,
                        destination=gathered,
                        elements=local * columns,
                        fp32=False,
                        elements_expression=expression(
                            f"((m + {peers - 1}) // {peers}) * {columns}"
                        ),
                    )
                body.append(collective)
                restore = emit(
                    "gather_columns",
                    dict(rows=local, width=width, ranks=peers, dtype="bfloat16", paths=4),
                    [
                        ("gathered", gathered, "bf16", [peers, local, columns], "read"),
                        ("out", expand["buffers"][1], "bf16", [local, 4, width], "write"),
                    ],
                    local,
                )
                if partitions > 1:
                    restore["row_expression"] = expand["row_expression"]
                body.append(restore)
                index += 1
                changes += 1
                check_intermediate_liveness(
                    original[index:], [source], describe, model.get("batch", {}), definitions
                )
            packet["body"] = body
    if not changes:
        raise ValueError("No CPU embedding inputs found")
    return changes


def build(plan, output, workers=4):
    from orinfer_build.models.deepseek_v41.build import compile_task

    model, base, describe = stage_plan(plan, output, workers=workers, allow_resume=True)
    tasks = {}

    def kernel(name, arguments, specs, capacity):
        key = json.dumps([name, arguments, specs, capacity], sort_keys=True)
        if key not in tasks:
            tasks[key] = dict(
                module="kernels.ascend.memory",
                name=name,
                arguments=arguments,
                specs=specs,
                row_capacity=capacity,
                dynamic_arguments=dict(rows=1),
                placeholder=f"@embedding{len(tasks)}",
            )
        return tasks[key]["placeholder"]

    changes = transform(model, describe, kernel)
    compile_and_replace(model, tasks, base, workers, compiler=compile_task)
    finalize_plan(
        model,
        base,
        output,
        save_base=True,
        report_name="embedding-columns.json",
        report=dict(packets=changes, kernels=len(tasks), columns=5120 // len(model["ranks"])),
    )
    return output / "package/model.json"


def main():
    args = plan_parser(__doc__, default_workers=4, model_dir=False).parse_args()
    print(build(args.plan, args.output, args.compile_workers))


if __name__ == "__main__":
    main()
