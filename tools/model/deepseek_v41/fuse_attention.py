"""Fuse shared DSv4.1 attention, including selected INT8 KV preparation."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path

from tools.build.files import atomic_json


def selected_chain(program, index, describe, contract):
    """Consume the shared chain and its disjoint, same-output PV schedules."""
    expected = [
        "private_rows.kv_int8_gather_attention",
        "private_rows.attention_cube_qk",
        "private_rows.attention_cube_softmax",
        "private_rows.attention_cube_pv",
    ]
    chain = program[index : index + 4]
    specs = [describe(v) for v in chain]
    if len(specs) != 4 or any(s is None for s in specs) or [s["op"] for s in specs] != expected:
        raise ValueError("Require the original gather/QK/softmax/PV sequence")
    first = chain[3]
    pv = [first]
    end = index + 4
    while end < len(program):
        candidate = program[end]
        spec = describe(candidate)
        if spec is None or spec["op"] != expected[3]:
            break
        if candidate["buffers"][1:] != first["buffers"][1:] or any(
            spec["parameters"].get(k) != specs[3]["parameters"].get(k)
            for k in ("rows", "heads", "selected", "dimension", "_owner_token_capacity")
        ):
            raise ValueError("PV schedules must share their physical output and reduction")
        pv.append(candidate)
        end += 1
    capacity = specs[0]["parameters"]["_owner_token_capacity"]
    ranges = contract.get("work_ranges", {})
    target = ranges.get(chain[0]["buffers"][0], dict(min_rows=1, max_rows=capacity))
    intervals = sorted(
        (r["min_rows"], r["max_rows"])
        for step in pv
        for r in [ranges.get(step["buffers"][0], dict(min_rows=1, max_rows=capacity))]
    )
    cursor = target["min_rows"]
    for low, high in intervals:
        if low != cursor or high < low:
            raise ValueError("PV owner schedules must cover the gather domain without overlap")
        cursor = high + 1
    if cursor != target["max_rows"] + 1:
        raise ValueError("PV owner schedules do not cover the gather domain")
    return chain[:3] + pv


def selected_work_key(grid, partition, extent, decoder):
    """Source and bounded-decoder domains coexist within one packet."""
    import hashlib

    return hashlib.sha256(
        json.dumps([grid, partition, extent, decoder], sort_keys=True).encode()
    ).hexdigest()[:16]


def selected_packet_ranges(packets, minimum):
    """Keep the measured small-row schedule within the existing target family."""
    import copy

    if minimum < 1:
        raise ValueError("Selected attention requires a positive minimum row count")
    result = []
    for packet in packets:
        low = packet.get("min_rows", 1)
        high = packet.get("max_rows", packet["rows"])
        if (
            packet.get("role", "target") == "target"
            and not packet.get("batch_decode", False)
            and low < minimum <= high
        ):
            small = copy.deepcopy(packet)
            small["max_rows"] = minimum - 1
            packet = copy.deepcopy(packet)
            packet["min_rows"] = minimum
            result.append(small)
        result.append(packet)
    return result


def build_selected(
    plan, output, workers=4, minimum=512, grid_limit=64, softmax_heads=1, model_dir=None
):
    """Fuse the existing private selected-KV chain without changing its views."""
    import hashlib
    import inspect
    import math
    from kernels.model.rows import row_count
    from tools.build.files import file_hash
    from tools.model.deepseek_v41.batch import compile_task, DTYPES, SIZES
    from tools.model.deepseek_v41.assembly import compact_buffers, publish_package, validate_rows
    from tools.operators.ascend.attention_fused import selected_int8
    from tools.operators.ascend.build import operand
    from tools.operators.ascend.geometry import physical_shapes
    from tools.operators.ascend.private_rows import indirect

    if output.exists() or grid_limit not in (8, 16, 32, 64):
        raise ValueError("Require a new output directory and a supported physical grid")
    model = json.loads(plan.read_text())
    validate_rows(model, plan.parent, model_dir=model_dir)
    base = output / "source"
    publish_package(model, plan.parent, base)
    contract = model["batch"]
    peers = len(model["ranks"])
    descriptions, prototypes, tasks = {}, {}, {}

    def describe(step):
        if step.get("op") != "kernel":
            return None
        path = step["manifest"]
        if path not in descriptions:
            descriptions[path] = json.loads((base / path).read_text())
        return descriptions[path]

    def column_ids(step):
        return contract["pointers"][step["buffers"][1]]

    changed = 0
    expected = [
        "private_rows.kv_int8_gather_attention",
        "private_rows.attention_cube_qk",
        "private_rows.attention_cube_softmax",
        "private_rows.attention_cube_pv",
    ]
    for rank in model["ranks"]:
        rank["packets"] = selected_packet_ranges(rank["packets"], minimum)
        definitions = {v["name"]: v for v in rank["buffers"]}

        def buffer(name, size):
            item = dict(name=name, bytes=size)
            if name not in definitions:
                definitions[name] = item
                rank["buffers"].append(item)
            elif definitions[name]["bytes"] != size:
                raise ValueError("Fused table extent differs")

        for packet in rank["packets"]:
            if packet.get("min_rows", 1) < minimum:
                continue
            old, body, index, count = packet["body"], [], 0, 0
            while index < len(old):
                desc = describe(old[index])
                if desc is None or desc["op"] != expected[0]:
                    body.append(old[index])
                    index += 1
                    continue
                chain = selected_chain(old, index, describe, contract)
                specs = [describe(v) for v in chain[:4]]
                g, q, s, p = [column_ids(v) for v in chain[:4]]
                gp, qp, sp, pp = [v["parameters"] for v in specs]
                heads = qp["heads"]
                if heads not in (8, 16, 32, 64):
                    body.append(old[index])
                    index += 1
                    continue
                if not (q[1] == g[6] == p[1] and q[2] == s[0] and g[7] == s[1] and s[2] == p[0]):
                    raise ValueError("Attention intermediates do not share their original views")
                if len(chain[2]["buffers"]) != 4 or any(
                    len(v["buffers"]) != 3 for v in (chain[0], chain[1], chain[3])
                ):
                    raise ValueError("Unexpected shared attention inputs")
                capacities = {v["parameters"]["_owner_token_capacity"] for v in specs}
                if len(capacities) != 1 or sp["heads"] != heads or pp["heads"] != heads:
                    raise ValueError("Attention owner geometries differ")
                capacity = capacities.pop()
                partition = contract.get("partitioned_tables", {}).get(chain[0]["buffers"][0])
                if partition not in (None, peers):
                    raise ValueError("Unsupported query partitioning")
                for step in chain:
                    role = step["buffers"][0]
                    if contract.get("partitioned_tables", {}).get(role) != partition:
                        raise ValueError("Selected fusion requires common query partitioning")
                arguments = {
                    k: v for k, v in gp.items() if not k.startswith("_") and k != "bulk_store"
                }
                arguments.update(rows=capacity, heads=heads, softmax_heads=softmax_heads)
                identity = json.dumps(arguments, sort_keys=True)
                if identity not in prototypes:
                    function = selected_int8(**dict(arguments, rows=row_count(capacity, True)))
                    tensors = [
                        function.buffer_map[v] for v in function.params if v in function.buffer_map
                    ]
                    owned = [v for v in tensors if v.name != "sink"]
                    columns = {v.name: i for i, v in enumerate(owned)}
                    _, identities, grid = indirect(
                        function, columns, contract["owners"], capacity, table_columns=len(columns)
                    )
                    if identities[3:] != ["sink"]:
                        raise ValueError("Unexpected fused shared input ABI")
                    writes = {
                        "output",
                        "kv_workspace",
                        "mask_workspace",
                        "score_workspace",
                        "probability_workspace",
                        "output_workspace",
                    }
                    metadata = dict(
                        _owner_token_capacity=capacity,
                        _owner_source_sha256=file_hash(Path(inspect.getfile(selected_int8))),
                        _owner_aliases={v.name + "_owner": str(v.dtype) for v in owned},
                        _owner_access=[
                            dict(column=i, write=v.name in writes) for i, v in enumerate(owned)
                        ],
                        _owner_writes=sorted(writes),
                    )
                    task = dict(
                        key=f"@selected_fusion{len(tasks)}",
                        module="tools.operators.ascend.attention_fused",
                        name="selected_int8",
                        arguments=arguments,
                        capacity=capacity,
                        dynamic=dict(rows=1),
                        columns=columns,
                        owners=contract["owners"],
                        table_columns=len(columns),
                        metadata=metadata,
                        grid_limit=grid_limit,
                        specs=[
                            operand("OwnerWork", "i32", [8192, 2]),
                            operand("OwnerPointers", "u64", [contract["owners"], len(columns) + 1]),
                            operand("PacketActive", "i32", [1]),
                            operand("sink", "f32", [heads]),
                        ],
                    )
                    tasks[identity] = task
                    shapes = physical_shapes(function, capacity)
                    sizes = {
                        t.name: (shape, SIZES[DTYPES[str(t.dtype)]])
                        for t, shape in zip(tensors, shapes, strict=True)
                        if t.name in columns
                    }
                    prototypes[identity] = task["key"], grid, sizes
                key, grid, sizes = prototypes[identity]
                roles = dict(
                    query=q[0],
                    swa=g[0],
                    swa_scales=g[1],
                    long_kv=g[2],
                    long_scales=g[3],
                    swa_indices=g[4],
                    long_indices=g[5],
                    output=p[2],
                    kv_workspace=g[6],
                    mask_workspace=g[7],
                    score_workspace=q[2],
                    probability_workspace=s[2],
                    output_workspace=p[3],
                )
                ids = []
                for name, (shape, item_size) in sizes.items():
                    col = contract["columns"][roles[name]]
                    actual = math.prod(shape) * item_size
                    bound = (
                        capacity * col["row_bytes"]
                        if col["row_bytes"] is not None
                        else definitions[col["buffer"]]["bytes"]
                    )
                    if actual > bound or (col["row_bytes"] is not None and actual != bound):
                        raise ValueError("Fused owner view changes physical geometry: " + name)
                    ids.append(roles[name])
                digest = hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]
                pointer = "batch.selected_attention.pointers." + digest
                replay = contract.get("decoder_replay")
                decoder = bool(
                    replay
                    and any(name in replay["tables"] for v in chain for name in v["buffers"][:2])
                )
                work_digest = selected_work_key(
                    grid,
                    partition,
                    contract.get("work_ranges", {}).get(chain[0]["buffers"][0]),
                    decoder,
                )
                work = "batch.selected_attention.work." + work_digest
                contract["pointers"][pointer] = ids
                contract["work"][work] = grid
                if partition is not None:
                    contract.setdefault("partitioned_tables", {}).update(
                        {pointer: partition, work: partition}
                    )
                extent = contract.get("work_ranges", {}).get(chain[0]["buffers"][0])
                if extent:
                    contract.setdefault("work_ranges", {})[work] = dict(extent)
                if decoder:
                    replay["tables"] = sorted(set(replay["tables"]) | {work, pointer})
                buffer(pointer, contract["owners"] * (len(ids) + 1) * 8)
                buffer(work, 8192 * 2 * 4)
                body.append(
                    dict(
                        op="kernel",
                        manifest=key,
                        owner_work=grid,
                        buffers=[work, pointer, contract["active"], chain[2]["buffers"][3]],
                    )
                )
                index += len(chain)
                count += 1
                changed += 1
            if count not in (0, 3, 40):
                raise ValueError("Fused packet must cover the complete target or draft layer stack")
            packet["body"] = body
    if not changed:
        raise ValueError("No selected private attention packet matched")
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        replacements = dict(pool.map(compile_task, [(v, str(base)) for v in tasks.values()]))
    for rank in model["ranks"]:
        for packet in rank["packets"]:
            for step in packet["body"] + packet["head"]:
                if step.get("manifest") in replacements:
                    step["manifest"] = replacements[step["manifest"]]
    live_work = {
        s["buffers"][0]
        for r in model["ranks"]
        for p in r["packets"]
        for s in p["body"] + p["head"]
        if s["op"] == "kernel" and "owner_work" in s
    }
    contract["work"] = {n: v for n, v in contract["work"].items() if n in live_work}
    contract["work_ranges"] = {
        n: v for n, v in contract.get("work_ranges", {}).items() if n in live_work
    }
    compact_buffers(model, owner_tables=True)
    if "decoder_replay" in contract:
        live = set(contract["work"]) | set(contract["pointers"])
        contract["decoder_replay"]["tables"] = sorted(
            set(contract["decoder_replay"]["tables"]) & live
        )
    validate_rows(model, base, model_dir=model_dir)
    assets = publish_package(model, base, output / "package")
    report = dict(
        changed_chains=changed,
        kernels=len(tasks),
        assets=assets,
        model=str(output / "package/model.json"),
        grid_limit=grid_limit,
    )
    atomic_json(output / "selected-fusion.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-int8", action="store_true")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--grid-limit", type=int, choices=(8, 16, 32, 64), default=64)
    parser.add_argument("--softmax-heads", type=int, choices=(1, 4, 8), default=1)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tile-m", type=int, choices=[16, 32, 64], default=32)
    parser.add_argument("--tile-k", type=int, choices=[128, 256], default=128)
    parser.add_argument("--reuse-kv", action="store_true")
    parser.add_argument(
        "--min-rows",
        type=int,
        default=2,
        help="Fuse rows from this measured boundary; retain smaller target rows in the same family",
    )
    parser.add_argument("--compile-workers", type=int, default=3)
    args = parser.parse_args()
    if args.selected_int8:
        print(
            json.dumps(
                build_selected(
                    args.plan,
                    args.output,
                    args.compile_workers,
                    args.min_rows,
                    args.grid_limit,
                    args.softmax_heads,
                    args.model_dir,
                )
            )
        )
        return
    if args.plan.parent.resolve() != args.output.parent.resolve():
        parser.error("Input and output must share the asset directory")
    if args.min_rows < 2:
        parser.error("Fused prefill requires a minimum of two rows")
    from tools.model.deepseek_v41.build import compile_task
    from tools.operators.ascend.build import operand
    from tools.operators.ascend.attention_fused import score_tiles

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
                cached_tiles = score_tiles(
                    c, parameters["ratio"], parameters["compressed"], args.tile_m, tile_k
                )
                for name, dtype, shape, size in (
                    (
                        "scores",
                        "f32",
                        [blocks, args.tile_m, cached_tiles * tile_k],
                        blocks * cached_tiles * args.tile_m * tile_k * 4,
                    ),
                    (
                        "probabilities",
                        "bf16",
                        [blocks, args.tile_m, tile_k],
                        blocks * args.tile_m * tile_k * 2,
                    ),
                    ("output", "f32", [blocks, args.tile_m, 512], blocks * args.tile_m * 512 * 4),
                ):
                    geometry = f"{cached_tiles}." if name == "scores" else ""
                    name = f"attention.fused.{m}.{args.tile_m}.{tile_k}.{int(args.reuse_kv)}.{geometry}{name}"
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
                fused = dict(
                    op="kernel",
                    manifest=tasks[key]["placeholder"],
                    buffers=step["buffers"][:4]
                    + softmax[0]["buffers"][2:4]
                    + [pv["buffers"][4]]
                    + [s[0] for s in scratch],
                )
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
