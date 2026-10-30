"""Exchange query/head layouts around the shared private Cube attention factories."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import hashlib
import inspect
import json
import math
import multiprocessing
from pathlib import Path

from tools.build.files import file_hash
from tools.model.row_contract import expression


def build(plan, output, workers=4, minimum=512, pv_tile=256, qk_head_tile=16, pv_head_tile=16):
    from kernels.model.rows import row_count
    from tools.model.deepseek_v41.batch import compile_task, DTYPES, SIZES
    from tools.model.deepseek_v41.converge import compact_buffers, publish_package, validate_rows
    from tools.operators.ascend import attention_cube, kv_int8, memory
    from tools.operators.ascend.build import operand
    from tools.operators.ascend.geometry import physical_shapes
    from tools.operators.ascend.private_rows import indirect

    model = json.loads(plan.read_text())
    if output.exists() or not model.get("batch", {}).get("partitioned_tables"):
        raise ValueError("Require a query-partitioned private plan and a new output directory")
    validate_rows(model, plan.parent)
    base = output / "source"
    publish_package(model, plan.parent, base)
    contract = model["batch"]
    peers = len(model["ranks"])
    if peers != 8 or not 1 <= minimum <= 2048 or pv_tile not in [32, 64, 128, 256]:
        raise ValueError("Require TP8 and a legal row threshold/PV tile")
    descriptions, tasks = {}, {}
    column_ids = {json.dumps(c, sort_keys=True): i for i, c in enumerate(contract["columns"])}

    def describe(path):
        if path not in descriptions:
            descriptions[path] = json.loads((base / path).read_text())
        return descriptions[path]

    def register(module, name, arguments, specs, capacity, **extra):
        task = dict(
            module=module.__name__,
            name=name,
            arguments=arguments,
            specs=specs,
            capacity=capacity,
            dynamic=dict(rows=1),
            **extra,
        )
        identity = json.dumps(task, sort_keys=True)
        if identity not in tasks:
            tasks[identity] = dict(task, key=f"@query_attention{len(tasks)}")
        return tasks[identity]["key"]

    changes = 0
    for rank in model["ranks"]:
        definitions = {b["name"]: b for b in rank["buffers"]}

        def define(name, size, weight=None):
            if name not in definitions:
                b = dict(name=name, bytes=size)
                if weight is not None:
                    b["weight"] = weight
                rank["buffers"].append(b)
                definitions[name] = b
            elif definitions[name]["bytes"] < size:
                raise ValueError("Shared attention buffer is too small: " + name)
            return name

        def movement(name, arguments, specs, buffers, capacity, rows=None):
            step = dict(
                op="kernel",
                manifest=register(memory, name, arguments, specs, capacity),
                buffers=buffers,
            )
            if rows is not None:
                step["row_expression"] = expression(rows)
            return step

        def private(step, params, columns, external, capacity):
            old = describe(step["manifest"])
            name = old["op"].removeprefix("private_rows.")
            module, function_name = (
                (kv_int8, "gather_attention")
                if name == "kv_int8_gather_attention"
                else (attention_cube, name.removeprefix("attention_cube_"))
            )
            factory = getattr(module, function_name)
            function = factory(**dict(params, rows=row_count(capacity, True)))
            tensors = [function.buffer_map[p] for p in function.params if p in function.buffer_map]
            owned = {
                t.name: i
                for i, t in enumerate(
                    t for t in tensors if t.name + "_owner" in old["parameters"]["_owner_aliases"]
                )
            }
            aliases = {t.name + "_owner": str(t.dtype) for t in tensors if t.name in owned}
            if aliases != old["parameters"]["_owner_aliases"] or len(owned) != len(columns):
                raise ValueError("Attention primitive changed its private tensor ABI")
            shapes = physical_shapes(function, capacity)
            for tensor, shape in zip(tensors, shapes, strict=True):
                if tensor.name not in owned:
                    continue
                column = columns[owned[tensor.name]]
                size = math.prod(shape) * SIZES[DTYPES[str(tensor.dtype)]]
                if column["row_bytes"] is not None:
                    column.update(row_bytes=size // capacity, row_partition=peers)
                    if size % capacity:
                        raise ValueError("Nonuniform query column")
                if size > definitions[column["buffer"]]["bytes"]:
                    raise ValueError(
                        "Partition exceeds shared/private allocation: " + column["buffer"]
                    )
                if (
                    column["row_bytes"] is not None
                    and size < definitions[column["buffer"]]["bytes"]
                ):
                    parent = column["buffer"]
                    view = parent + ".query_attention.view"
                    if view not in definitions:
                        definition = dict(name=view, bytes=size, view=dict(parent=parent, offset=0))
                        rank["buffers"].append(definition)
                        definitions[view] = definition
                    elif definitions[view] != dict(
                        name=view, bytes=size, view=dict(parent=parent, offset=0)
                    ):
                        raise ValueError("Inconsistent shared query view")
                    column["buffer"] = view
            _, identities, grid = indirect(
                function, owned, contract["owners"], capacity, table_columns=len(columns)
            )
            if [x for x in identities[3:] if x is not None] != [
                t.name for t in tensors if t.name not in owned
            ]:
                raise ValueError("Unexpected external attention operands")
            metadata = {k: v for k, v in old["parameters"].items() if k.startswith("_owner_")}
            metadata.update(
                _owner_source_sha256=file_hash(Path(inspect.getfile(factory))),
                _owner_token_capacity=capacity,
            )
            specs = deepcopy(old["operands"][:3])
            specs += [
                operand(t.name, DTYPES[str(t.dtype)], shape)
                for t, shape in zip(tensors, shapes, strict=True)
                if t.name not in owned
            ]
            key = register(
                module,
                function_name,
                dict(params, rows=capacity),
                specs,
                capacity,
                columns=owned,
                owners=contract["owners"],
                table_columns=len(columns),
                metadata=metadata,
            )
            ids = []
            for c in columns:
                signature = json.dumps(c, sort_keys=True)
                if signature not in column_ids:
                    column_ids[signature] = len(contract["columns"])
                    contract["columns"].append(c)
                ids.append(column_ids[signature])
            digest = hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]
            pointer = "batch.query_attention.pointers." + digest
            work = (
                "batch.query_attention.work."
                + hashlib.sha256(json.dumps([grid, peers], sort_keys=True).encode()).hexdigest()[
                    :16
                ]
            )
            contract["pointers"][pointer] = ids
            contract["work"][work] = grid
            contract.setdefault("partitioned_tables", {}).update({pointer: peers, work: peers})
            define(pointer, contract["owners"] * (len(columns) + 1) * 8)
            define(work, 8192 * 2 * 4)
            return dict(
                op="kernel",
                manifest=key,
                owner_work=grid,
                buffers=[work, pointer, contract["active"]] + external,
            )

        def cols(step):
            return [
                deepcopy(contract["columns"][i]) for i in contract["pointers"][step["buffers"][1]]
            ]

        def is_op(step, name):
            return (
                step["op"] == "kernel"
                and describe(step["manifest"])["op"] == "private_rows." + name
            )

        packets = []
        for original in rank["packets"]:
            if original.get("batch_decode") or original.get("max_rows", original["rows"]) < minimum:
                packets.append(original)
                continue
            packet = deepcopy(original)
            if original.get("min_rows", 1) < minimum:
                fallback = deepcopy(original)
                fallback["max_rows"] = minimum - 1
                packets.append(fallback)
                packet["min_rows"] = minimum
            capacity = packet["rows"]
            if capacity % peers:
                raise ValueError("Query capacity must divide the peer count")
            local = capacity // peers
            send = define("batch.query_attention.send", capacity * 4096 * 2)
            receive = define("batch.query_attention.receive", capacity * 4096 * 2)
            swa = define("batch.query_attention.swa_indices", local * 128 * 4)
            dummy = define("batch.query_attention.unused_indices", local * 512 * 4)
            packed_swa = False
            local_indices = {}
            steps, body, index, layer_count = packet["body"], [], 0, 0

            def exchange():
                return dict(
                    op="exchange",
                    source=send,
                    destination=receive,
                    elements=local * 4096,
                    fp32=False,
                    elements_expression=expression(f"((m + {peers - 1}) // {peers}) * 4096"),
                )

            def pack(source, padded=False):
                return movement(
                    "partition_rows",
                    dict(
                        rows=capacity,
                        count=local,
                        width=4096,
                        first=0,
                        stride=peers,
                        dtype="bfloat16",
                        dynamic_count=True,
                        all_partitions=True,
                    ),
                    [
                        operand("source", "bf16", [capacity, 4096]),
                        operand("out", "bf16", [capacity, 4096], "write"),
                    ],
                    [source, send],
                    capacity,
                    f"((m + {peers - 1}) // {peers}) * {peers}" if padded else None,
                )

            while index < len(steps):
                step = steps[index]
                if step["op"] == "gather" and step.get("int32") and index + 1 < len(steps):
                    following = steps[index + 1]
                    if (
                        following["op"] == "kernel"
                        and describe(following["manifest"])["op"] == "memory.interleave_rows"
                    ):
                        local_indices[following["buffers"][1]] = step["source"]
                if not is_op(step, "kv_int8_gather_attention"):
                    body.append(step)
                    index += 1
                    continue
                gather_columns = cols(step)
                gather_params = {
                    k: v
                    for k, v in describe(step["manifest"])["parameters"].items()
                    if not k.startswith("_")
                }
                qk_step, softmax_step = steps[index + 1 : index + 3]
                if not is_op(qk_step, "attention_cube_qk") or not is_op(
                    softmax_step, "attention_cube_softmax"
                ):
                    raise ValueError("Require the shared gather/QK/softmax/PV sequence")
                qk_columns = cols(qk_step)
                query = qk_columns[0]["buffer"]
                if not packed_swa:
                    body.append(
                        movement(
                            "partition_rows",
                            dict(
                                rows=capacity,
                                count=local,
                                width=128,
                                first=rank["rank"],
                                stride=peers,
                                dtype="int32",
                                dynamic_count=True,
                            ),
                            [
                                operand("source", "i32", [capacity, 128]),
                                operand("out", "i32", [local, 128], "write"),
                            ],
                            [gather_columns[4]["buffer"], swa],
                            capacity,
                        )
                    )
                    packed_swa = True
                body += [
                    pack(query),
                    exchange(),
                    movement(
                        "gather_columns",
                        dict(rows=local, width=32768, ranks=peers),
                        [
                            operand("gathered", "bf16", [peers, local, 4096]),
                            operand("out", "bf16", [local, 32768], "write"),
                        ],
                        [receive, query],
                        local,
                        f"(m + {peers - 1}) // {peers}",
                    ),
                ]
                gather_columns[4]["buffer"] = swa
                selected = gather_columns[5]["buffer"]
                if gather_params["compressed"] and selected not in local_indices:
                    raise ValueError("Missing matching private TopK query partition")
                gather_columns[5]["buffer"] = (
                    local_indices[selected] if gather_params["compressed"] else dummy
                )
                body.append(private(step, gather_params, gather_columns, [], local))
                for op_step in [qk_step, softmax_step]:
                    parameters = {
                        k: v
                        for k, v in describe(op_step["manifest"])["parameters"].items()
                        if not k.startswith("_")
                    }
                    if parameters["heads"] != 8:
                        raise ValueError("Expected original eight local heads")
                    parameters["heads"] = 64
                    if is_op(op_step, "attention_cube_qk"):
                        parameters["head_tile"] = qk_head_tile
                    extra = []
                    if is_op(op_step, "attention_cube_softmax"):
                        old_sink = definitions[op_step["buffers"][3]]
                        weight = deepcopy(old_sink["weight"])
                        if (
                            weight["kind"] != "matrix"
                            or weight["rows"] != 8
                            or weight["columns"] != 1
                        ):
                            raise ValueError("Unsupported attention sink storage")
                        weight.update(first_row=0, rows=64)
                        extra = [
                            define(weight["name"] + ".query_attention.full_sink", 64 * 4, weight)
                        ]
                    body.append(private(op_step, parameters, cols(op_step), extra, local))
                index += 3
                variants = []
                while index < len(steps) and is_op(steps[index], "attention_cube_pv"):
                    variants.append(steps[index])
                    index += 1
                if not variants or len({v["buffers"][1] for v in variants}) != 1:
                    raise ValueError("Require one shared PV pointer table")
                ranges = sorted(
                    (
                        contract.get("work_ranges", {}).get(
                            v["buffers"][0], dict(min_rows=1, max_rows=capacity)
                        )
                        for v in variants
                    ),
                    key=lambda x: x["min_rows"],
                )
                cursor = 1
                for extent in ranges:
                    if extent["min_rows"] != cursor:
                        raise ValueError("PV variants must cover the owner domain without overlap")
                    cursor = extent["max_rows"] + 1
                if cursor != capacity + 1:
                    raise ValueError("Incomplete PV owner domain")
                pv = variants[0]
                parameters = {
                    k: v
                    for k, v in describe(pv["manifest"])["parameters"].items()
                    if not k.startswith("_")
                }
                parameters.update(heads=64, tile_d=pv_tile, head_tile=pv_head_tile)
                output_name = cols(pv)[2]["buffer"]
                body.append(private(pv, parameters, cols(pv), [], local))
                body.append(
                    movement(
                        "zero_row_tail",
                        dict(
                            rows=capacity,
                            capacity=local,
                            width=32768,
                            dtype="bfloat16",
                            partitions=peers,
                            first=rank["rank"],
                        ),
                        [
                            operand("positions", "i32", [capacity]),
                            operand("out", "bf16", [local, 32768], "read_write"),
                        ],
                        [packet["coordinates"]["slots"], output_name],
                        capacity,
                    )
                )
                body += [
                    pack(output_name, True),
                    exchange(),
                    movement(
                        "interleave_rows",
                        dict(
                            rows=capacity,
                            count=local,
                            width=4096,
                            ranks=peers,
                            dtype="bfloat16",
                            dynamic_count=True,
                        ),
                        [
                            operand("source", "bf16", [capacity, 4096]),
                            operand("out", "bf16", [capacity, 4096], "write"),
                        ],
                        [receive, output_name],
                        capacity,
                    ),
                ]
                layer_count += 1
            if layer_count != 40:
                raise ValueError("Attention exchange must cover all40 layers")
            packet["body"] = body
            packets.append(packet)
            changes += layer_count
        rank["packets"] = packets
    jobs = [(task, base) for task in tasks.values()]
    if workers == 1:
        compiled = dict(compile_task(job) for job in jobs)
    else:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            compiled = dict(pool.map(compile_task, jobs))
    for rank in model["ranks"]:
        for packet in rank["packets"]:
            for step in packet["body"] + packet["head"]:
                if step["op"] == "kernel":
                    step["manifest"] = compiled.get(step["manifest"], step["manifest"])
    compact_buffers(model, owner_tables=True)
    validate_rows(model, base)
    publish_package(model, base, output / "package")
    return dict(
        changed_layers=changes, compiled=len(tasks), model=str(output / "package/model.json")
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--compile-workers", type=int, default=4)
    p.add_argument("--min-rows", type=int, default=512)
    p.add_argument("--pv-tile-d", type=int, default=256, choices=[32, 64, 128, 256])
    p.add_argument("--qk-head-tile", type=int, choices=[16, 32, 64], default=16)
    p.add_argument("--pv-head-tile", type=int, choices=[16, 32, 64], default=16)
    args = p.parse_args()
    print(
        json.dumps(
            build(
                args.plan,
                args.output,
                args.compile_workers,
                args.min_rows,
                args.pv_tile_d,
                args.qk_head_tile,
                args.pv_head_tile,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
