"""Fuse private causal index scores and local TopK using the existing owner ABI."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import inspect
import json
import math
import multiprocessing
from pathlib import Path

from tools.build.files import atomic_json, file_hash
from tools.model.deepseek_v41.batch import compile_task
from tools.model.deepseek_v41.assembly import compact_buffers, publish_package, validate_rows


def build(plan, output, workers=4, minimum=512, tile_rows=8):
    from kernels.model.rows import row_count
    from tools.operators.ascend.build import operand
    from tools.operators.ascend.geometry import physical_shapes
    from tools.operators.ascend.indexer_fused import causal_topk
    from tools.operators.ascend.private_rows import indirect

    if output.exists():
        raise ValueError("Require a new output directory")
    model = json.loads(plan.read_text())
    base = output / "source"
    publish_package(model, plan.parent, base)
    contract, peers = model["batch"], len(model["ranks"])
    descriptions, tasks, prototype = {}, {}, {}
    private_names = ("batch.prefill.indexer_fused.cube", "batch.prefill.indexer_fused.scores")
    for name in private_names:
        if name not in contract["private"]:
            contract["private"].append(name)

    def describe(step):
        if step.get("op") != "kernel":
            return None
        path = step["manifest"]
        if path not in descriptions:
            descriptions[path] = json.loads((base / path).read_text())
        return descriptions[path]

    def column(value):
        if value not in contract["columns"]:
            contract["columns"].append(value)
        return contract["columns"].index(value)

    changed = 0
    for rank in model["ranks"]:
        definitions = {v["name"]: v for v in rank["buffers"]}

        def buffer(name, size):
            if name in definitions:
                if definitions[name]["bytes"] != size:
                    raise ValueError("Fused buffer extent differs")
            else:
                item = dict(name=name, bytes=size)
                rank["buffers"].append(item)
                definitions[name] = item

        for packet in rank["packets"]:
            if packet.get("batch_decode") or packet.get("min_rows", 1) < minimum:
                continue
            old, body, index, count = packet["body"], [], 0, 0
            while index < len(old):
                desc = describe(old[index])
                if desc is None or desc["op"] != "private_rows.indexer_cube_causal_logits":
                    body.append(old[index])
                    index += 1
                    continue
                step = old[index]
                following = old[index + 1]
                next_desc = describe(following)
                if next_desc is None or next_desc["op"] != "topk.local":
                    raise ValueError("Require adjacent original index scores and local TopK")
                p, tp = desc["parameters"], next_desc["parameters"]
                if not all(
                    p.get(k) is True for k in ("length_input", "vector_scales", "direct_sum")
                ):
                    raise ValueError("Fused indexer requires the original causal FP32 head sum")
                capacity, length = p["_owner_token_capacity"], p["length"]
                ids = contract["pointers"][step["buffers"][1]]
                if (
                    len(ids) != 8
                    or tp["length"] != length
                    or tp.get("selected", 512) != 512
                    or following["buffers"][:2]
                    != [contract["columns"][i]["buffer"] for i in (ids[6], ids[5])]
                    or contract.get("partitioned_tables", {}).get(step["buffers"][0]) != peers
                    or contract.get("partitioned_tables", {}).get(step["buffers"][1]) != peers
                    or contract.get("work_ranges", {}).get(step["buffers"][0])
                    not in (None, dict(min_rows=1, max_rows=capacity))
                ):
                    raise ValueError("Fused indexer original row/owner views differ")
                identity = (capacity, length)
                if identity not in prototype:
                    arguments = dict(rows=capacity, length=length, tile_rows=tile_rows)
                    function = causal_topk(**dict(arguments, rows=row_count(capacity, True)))
                    tensors = [
                        function.buffer_map[v] for v in function.params if v in function.buffer_map
                    ]
                    columns = {v.name: i for i, v in enumerate(tensors)}
                    _, identities, grid = indirect(function, columns, contract["owners"], capacity)
                    if len(identities) != 3 or grid != step["owner_work"]:
                        raise ValueError("Fused indexer changes the bounded owner work schedule")
                    writes = {"candidates", "cube_workspace", "score_workspace"}
                    task = dict(
                        key=f"@indexer_fusion{len(tasks)}",
                        module="tools.operators.ascend.indexer_fused",
                        name="causal_topk",
                        arguments=arguments,
                        capacity=capacity,
                        dynamic=dict(rows=1),
                        columns=columns,
                        owners=contract["owners"],
                        table_columns=len(columns),
                        metadata=dict(
                            _owner_token_capacity=capacity,
                            _owner_source_sha256=file_hash(Path(inspect.getfile(causal_topk))),
                            _owner_aliases={v.name + "_owner": str(v.dtype) for v in tensors},
                            _owner_access=[
                                dict(column=i, write=v.name in writes)
                                for i, v in enumerate(tensors)
                            ],
                            _owner_writes=sorted(writes),
                        ),
                        specs=[
                            operand("OwnerWork", "i32", [8192, 2]),
                            operand("OwnerPointers", "u64", [contract["owners"], len(columns) + 1]),
                            operand("PacketActive", "i32", [1]),
                        ],
                    )
                    shapes = physical_shapes(function, capacity)
                    sizes = [
                        math.prod(shape)
                        * {"int8": 1, "float16": 2, "bfloat16": 2, "int32": 4, "float32": 4}[
                            str(t.dtype)
                        ]
                        for t, shape in zip(tensors, shapes, strict=True)
                    ]
                    tasks[identity] = task
                    prototype[identity] = task["key"], sizes
                key, sizes = prototype[identity]
                for i in range(6):
                    c = contract["columns"][ids[i]]
                    bound = (
                        c["row_bytes"] * capacity
                        if c["row_bytes"] is not None
                        else definitions[c["buffer"]]["bytes"]
                    )
                    if sizes[i] != bound:
                        raise ValueError("Fused index input changes physical geometry")
                destination = following["buffers"][2]
                if sizes[6] != definitions[destination]["bytes"]:
                    raise ValueError("Fused local TopK output changes geometry")
                new_ids = ids[:6] + [
                    column(
                        dict(
                            buffer=destination,
                            row_bytes=sizes[6] // capacity,
                            row_alignment=1,
                            decode=False,
                            row_partition=peers,
                        )
                    )
                ]
                for name, size in zip(private_names, sizes[7:], strict=True):
                    buffer(name, size)
                    new_ids.append(
                        column(dict(buffer=name, row_bytes=None, row_alignment=1, decode=None))
                    )
                digest = hashlib.sha256(json.dumps(new_ids).encode()).hexdigest()[:16]
                pointer = "batch.indexer_fused.pointers." + digest
                contract["pointers"][pointer] = new_ids
                contract["partitioned_tables"][pointer] = peers
                buffer(pointer, contract["owners"] * (len(new_ids) + 1) * 8)
                body.append(
                    dict(
                        step,
                        manifest=key,
                        buffers=[step["buffers"][0], pointer, contract["active"]],
                    )
                )
                changed += 1
                count += 1
                index += 2
            if count not in (0, 8):
                raise ValueError("Fused indexer packet must cover all eight original indexers")
            packet["body"] = body
    if not changed:
        raise ValueError("No causal indexer/TopK chains matched")
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        replacements = dict(pool.map(compile_task, [(task, str(base)) for task in tasks.values()]))
    for rank in model["ranks"]:
        for packet in rank["packets"]:
            for step in packet["body"] + packet["head"]:
                if step.get("manifest") in replacements:
                    step["manifest"] = replacements[step["manifest"]]
    compact_buffers(model, owner_tables=True)
    validate_rows(model, base)
    assets = publish_package(model, base, output / "package")
    report = dict(
        changed_chains=changed,
        kernels=len(tasks),
        assets=assets,
        model=str(output / "package/model.json"),
        tile_rows=tile_rows,
    )
    atomic_json(output / "indexer-fusion.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tile-rows", type=int, choices=(1, 2, 4, 8), default=8)
    parser.add_argument("--min-rows", type=int, default=512)
    parser.add_argument("--compile-workers", type=int, default=4)
    args = parser.parse_args()
    print(
        json.dumps(
            build(args.plan, args.output, args.compile_workers, args.min_rows, args.tile_rows)
        )
    )


if __name__ == "__main__":
    main()
