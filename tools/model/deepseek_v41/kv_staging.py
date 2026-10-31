"""Stage postquant SWA values once for all selected queries in a native packet."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib
import inspect
import json
import math
import multiprocessing
from pathlib import Path

from tools.build.files import atomic_json, file_hash
from tools.model.deepseek_v41.batch import compile_task, DTYPES, SIZES
from tools.model.deepseek_v41.assembly import compact_buffers, publish_package, validate_rows


def build(plan, output, workers=4, minimum=512):
    from kernels.model.rows import row_count
    from tools.operators.ascend.build import operand
    from tools.operators.ascend.geometry import physical_shapes
    from tools.operators.ascend.private_rows import indirect

    if output.exists():
        raise ValueError("Require a new output directory")
    model = json.loads(plan.read_text())
    base = output / "source"
    publish_package(model, plan.parent, base)
    contract = model["batch"]
    descriptions = {}
    tasks = {}
    prototypes = {}
    stage_name = "batch.prefill.kv_staging.swa"
    if stage_name in contract["private"]:
        raise ValueError("SWA staging is already present")
    scratch_name = "batch.prefill.indexer_fused.cube"
    if scratch_name not in contract["private"]:
        raise ValueError("SWA staging requires existing owner-private indexer scratch")
    contract["private"].append(stage_name)
    contract["columns"].append(
        dict(buffer=stage_name, row_bytes=None, row_alignment=1, decode=None)
    )
    stage_column = len(contract["columns"]) - 1

    def describe(step):
        if step["op"] != "kernel":
            return None
        path = step["manifest"]
        if path not in descriptions:
            descriptions[path] = json.loads((base / path).read_text())
        return descriptions[path]

    counts = []
    for rank in model["ranks"]:
        definitions = {b["name"]: b for b in rank["buffers"]}

        def buffer(name, size):
            if name not in definitions:
                v = dict(name=name, bytes=size)
                definitions[name] = v
                rank["buffers"].append(v)
            elif definitions[name]["bytes"] != size:
                raise ValueError("SWA staging extent differs")

        def bind(module, name, params, capacity, ids, writes, partition, external=()):
            identity = json.dumps([module, name, params, capacity], sort_keys=True)
            if identity not in prototypes:
                factory = getattr(importlib.import_module("tools.operators.ascend." + module), name)
                function = factory(**dict(params, rows=row_count(capacity, True)))
                tensors = [
                    function.buffer_map[v] for v in function.params if v in function.buffer_map
                ]
                owned = [t for t in tensors if t.name != "sink"]
                columns = {t.name: i for i, t in enumerate(owned)}
                _, identities, grid = indirect(function, columns, contract["owners"], capacity)
                if [v for v in identities[3:] if v is not None] != (["sink"] if external else []):
                    raise ValueError("SWA staging external ABI differs")
                shapes = physical_shapes(function, capacity)
                sizes = [
                    math.prod(shape) * SIZES[DTYPES[str(t.dtype)]]
                    for t, shape in zip(tensors, shapes, strict=True)
                    if t.name in columns
                ]
                metadata = dict(
                    _owner_token_capacity=capacity,
                    _owner_source_sha256=file_hash(Path(inspect.getfile(factory))),
                    _owner_aliases={t.name + "_owner": str(t.dtype) for t in owned},
                    _owner_access=[
                        dict(column=i, write=t.name in writes) for i, t in enumerate(owned)
                    ],
                    _owner_writes=sorted(writes),
                )
                specs = [
                    operand("OwnerWork", "i32", [8192, 2]),
                    operand("OwnerPointers", "u64", [contract["owners"], len(columns) + 1]),
                    operand("PacketActive", "i32", [1]),
                ]
                specs += [
                    operand(t.name, DTYPES[str(t.dtype)], shape)
                    for t, shape in zip(tensors, shapes, strict=True)
                    if t.name not in columns
                ]
                task = dict(
                    key=f"@kv_staging{len(tasks)}",
                    module="tools.operators.ascend." + module,
                    name=name,
                    arguments=dict(params, rows=capacity),
                    capacity=capacity,
                    dynamic=dict(rows=1),
                    columns=columns,
                    owners=contract["owners"],
                    table_columns=len(columns),
                    metadata=metadata,
                    specs=specs,
                )
                tasks[identity] = task
                prototypes[identity] = (task["key"], grid, sizes)
            key, grid, sizes = prototypes[identity]
            if len(ids) != len(sizes):
                raise ValueError("SWA staging pointer count differs")
            for column_index, size in zip(ids, sizes, strict=True):
                col = contract["columns"][column_index]
                bound = (
                    capacity * col["row_bytes"]
                    if col["row_bytes"] is not None
                    else definitions[col["buffer"]]["bytes"]
                )
                if size != bound:
                    raise ValueError("SWA staging changes physical geometry: " + col["buffer"])
            digest = hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]
            pointer = "batch.kv_staging.pointers." + digest
            contract["pointers"][pointer] = ids
            work = (
                "batch.kv_staging.work."
                + hashlib.sha256(
                    json.dumps([grid, partition], sort_keys=True).encode()
                ).hexdigest()[:16]
            )
            contract["work"][work] = grid
            if partition > 1:
                contract.setdefault("partitioned_tables", {}).update(
                    {work: partition, pointer: partition}
                )
            buffer(pointer, contract["owners"] * (len(ids) + 1) * 8)
            buffer(work, 8192 * 2 * 4)
            return dict(
                op="kernel",
                manifest=key,
                owner_work=grid,
                buffers=[work, pointer, contract["active"]] + list(external),
            )

        for packet in rank["packets"]:
            if packet.get("batch_decode") or packet.get("min_rows", 1) < minimum:
                continue
            if definitions[scratch_name]["bytes"] < 4096 * 512 * 2:
                raise ValueError("Indexer scratch cannot cover the SWA ring")
            stage_view = dict(
                name=stage_name, bytes=4096 * 512 * 2, view=dict(parent=scratch_name, offset=0)
            )
            rank["buffers"].append(stage_view)
            definitions[stage_name] = stage_view

            def root(name):
                while name in definitions and "view" in definitions[name]:
                    name = definitions[name]["view"]["parent"]
                return name

            def touches_scratch(step):
                if step["op"] not in ("kernel", "gather", "exchange"):
                    return True
                names = list(step.get("buffers", []))
                names.extend(step[k] for k in ("source", "destination") if k in step)
                for name in list(names):
                    if name in contract["pointers"]:
                        names.extend(
                            contract["columns"][i]["buffer"] for i in contract["pointers"][name]
                        )
                return any(root(name) == scratch_name for name in names)

            body = []
            pending = None
            scatters = consumers = fused = 0
            for step in packet["body"]:
                desc = describe(step)
                if desc is not None and desc["op"] == "private_rows.kv_int8_scatter":
                    p = desc["parameters"]
                    ids = contract["pointers"][step["buffers"][1]]
                    cache_name = contract["columns"][ids[2]]["buffer"]
                    if cache_name.startswith("cache.swa."):
                        if (
                            pending is not None
                            or p["table_rows"] != 4096
                            or p.get("reference_bf16")
                            or not p.get("logical_capacity")
                            or p["_owner_token_capacity"] != packet["rows"]
                            or p["_owner_token_capacity"] + 127 > p["table_rows"]
                        ):
                            raise ValueError("Require one complete SWA producer before attention")
                        pending = dict(index=len(body), parameters=p, ids=ids, touched=False)
                        scatters += 1
                    elif pending is not None:
                        pending["touched"] |= touches_scratch(step)
                    body.append(step)
                elif (
                    desc is not None and desc["op"] == "private_rows.attention_fused_selected_int8"
                ):
                    p = desc["parameters"]
                    ids = contract["pointers"][step["buffers"][1]]
                    if (
                        pending is None
                        or pending["ids"][2:4] != ids[1:3]
                        or p.get("swa_storage_rows") != 4096
                    ):
                        raise ValueError("Selected attention has no matching SWA producer")
                    if pending["touched"]:
                        body.append(step)
                    else:
                        producer = pending["parameters"]
                        body[pending["index"]] = bind(
                            "kv_int8",
                            "scatter_staged",
                            dict(table_rows=4096, logical_capacity=producer["logical_capacity"]),
                            producer["_owner_token_capacity"],
                            pending["ids"] + [stage_column],
                            set(producer["_owner_writes"]) | {"staged"},
                            1,
                        )
                        params = {
                            k: v for k, v in p.items() if not k.startswith("_") and k != "rows"
                        }
                        params["staged_swa"] = True
                        body.append(
                            bind(
                                "attention_fused",
                                "selected_int8",
                                params,
                                p["_owner_token_capacity"],
                                ids + [stage_column],
                                set(p["_owner_writes"]),
                                len(model["ranks"]),
                                step["buffers"][3:],
                            )
                        )
                        fused += 1
                    pending = None
                    consumers += 1
                else:
                    if pending is not None:
                        pending["touched"] |= touches_scratch(step)
                    body.append(step)
            if pending is not None or scatters != 40 or consumers != 40 or fused == 0:
                raise ValueError("SWA fusion must resolve every layer producer and consumer")
            packet["body"] = body
            counts.append(fused)
    if len(counts) != len(model["ranks"]):
        raise ValueError("Incomplete rank staging transform")
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        replacements = dict(pool.map(compile_task, [(v, str(base)) for v in tasks.values()]))
    for rank in model["ranks"]:
        for packet in rank["packets"]:
            for step in packet["body"] + packet["head"]:
                if step.get("manifest") in replacements:
                    step["manifest"] = replacements[step["manifest"]]
    compact_buffers(model, owner_tables=True)
    validate_rows(model, base)
    assets = publish_package(model, base, output / "package")
    report = dict(
        producers=sum(counts),
        consumers=sum(counts),
        kernels=len(tasks),
        assets=assets,
        model=str(output / "package/model.json"),
        stage_view_bytes=4096 * 512 * 2,
        additional_owner_bytes=0,
        scratch_parent=scratch_name,
        selection="scratch untouched between producer and consumer",
    )
    atomic_json(output / "kv-staging.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile-workers", type=int, default=4)
    parser.add_argument("--min-rows", type=int, default=512)
    a = parser.parse_args()
    print(json.dumps(build(a.plan, a.output, a.compile_workers, a.min_rows)))


if __name__ == "__main__":
    main()
