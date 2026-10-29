"""Compile shared token-row A2 execution with indirect owner-private state.

Input is the immutable complete INT8-KV contract, before SP/replay. Dense and
expert operations consume aggregate rows; only causal/state operations use the
same-source indirect wrapper. No checkpoint payload is transformed online.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import importlib
import inspect
import json
import math
import multiprocessing
from pathlib import Path

from tools.build.files import atomic_json, file_hash
from tools.model.deepseek_v41.converge import (
    canonical,
    publish_package,
    run_transform,
    validate_rows,
)
from tools.model.row_contract import resolve

SIZES = dict(i8=1, i32=4, i64=8, f16=2, bf16=2, f32=4, u64=8)
DTYPES = dict(
    bfloat16="bf16", float16="f16", float32="f32", int8="i8", int32="i32", int64="i64", uint64="u64"
)


def compile_task(item):
    task, base = item
    from kernels.model.rows import row_count
    from tools.operators.ascend.build import compile_kernel
    from tools.operators.ascend.private_rows import indirect

    arguments = dict(task["arguments"])
    m = row_count(task["capacity"], True)
    for name, factor in task["dynamic"].items():
        arguments[name] = m * factor
    function = getattr(importlib.import_module(task["module"]), task["name"])(**arguments)
    parameters = dict(task["arguments"])
    op = task["module"].split(".")[-1] + "." + task["name"]
    capacity = task["capacity"]
    if "columns" in task:
        function, _, _ = indirect(
            function, task["columns"], task["owners"], capacity, table_columns=task["table_columns"]
        )
        parameters.update(task["metadata"])
        op = "private_rows." + op.replace(".", "_")
        capacity = 4096
    path = compile_kernel(
        function, op, task["specs"], parameters, Path(base) / "cache", row_capacity=capacity
    )
    return task["key"], str(path.relative_to(base))


def build(
    plan,
    output,
    owners,
    capacity,
    rows,
    workers,
    *,
    sequence_parallel=False,
    shared_attention=False,
    reduce_scatter=False,
    fused_attention=False,
):
    from kernels.model.rows import row_count
    from tools.operators.ascend.build import operand
    from tools.operators.ascend.geometry import physical_shapes
    from tools.operators.ascend.private_rows import indirect

    base = plan.parent
    model = canonical(json.loads(plan.read_text()))
    if sequence_parallel and rows != model["ranks"][0]["packets"][1]["rows"]:
        raise ValueError("Sequence transforms require the source packet capacity")
    if sequence_parallel or fused_attention:
        current = output.with_name(output.stem + "-reference.json")
        atomic_json(current, model)
        if sequence_parallel:
            target = output.with_name(output.stem + "-sequence.json")
            run_transform(
                "sequence_parallel",
                current,
                target,
                workers,
                ["--min-rows", "512"]
                + (["--reduce-scatter"] if reduce_scatter else [])
                + (["--shared-attention"] if shared_attention else []),
            )
            current = target
        if fused_attention:
            target = output.with_name(output.stem + "-attention.json")
            run_transform(
                "fuse_attention",
                current,
                target,
                workers,
                [
                    "--min-rows",
                    "2",
                    "--tile-m",
                    "64",
                    "--tile-k",
                    "128",
                    "--reuse-kv",
                ],
            )
            current = target
        model = json.loads(current.read_text())
    old_capacity = model["capacity"]
    model["capacity"] = capacity
    model.pop("row_selections", None)
    specs, tasks, columns, work, work_ranges = {}, {}, [], {}, {}
    column_ids = {}
    pointer_tables = {}
    expected_columns = None
    geometry_cache = {}
    indirect_cache = {}
    source_hashes = {}
    for rank in model["ranks"]:
        original = {b["name"]: b for b in rank["buffers"]}
        definitions = {}
        rank_columns = []
        old_packets = rank["packets"]
        packets = []
        private = []

        def define(name, size, source=None):
            if name in definitions:
                if definitions[name]["bytes"] != size:
                    raise ValueError(
                        f"Conflicting physical buffer {name}: {definitions[name]['bytes']} != {size}"
                    )
            else:
                definitions[name] = dict(source or {}, name=name, bytes=size)
            return name

        def immutable(name):
            return any(k in original[name] for k in ("weight", "data", "prepared_weight"))

        def runtime_buffer(name, family, size=None):
            old = original[name]
            if immutable(name):
                return define(name, old["bytes"], old)
            label = "batch." + family + "." + name
            if label in definitions:
                return define(label, definitions[label]["bytes"] if size is None else size)
            size = old["bytes"] if size is None else size
            if "view" in old and name.startswith("sequence."):
                view = deepcopy(old["view"])
                view["parent"] = runtime_buffer(view["parent"], family)
                return define(label, size, dict(view=view))
            return define(label, size)

        def binding(name, shape, dtype, *, family, source_rows, head=False, alignment=1):
            size = math.prod(shape) * SIZES[dtype]
            if alignment > 1:
                packet_cap = owners if family == "decode" else rows
                size = (packet_cap + owners * (alignment - 1)) * (size // packet_cap)
            old = original.get(name)
            if old and immutable(name):
                if old["bytes"] == size:
                    return define(name, size, old)
                if "data" not in old:
                    raise ValueError("Immutable matrix shape changed: " + name)
                # Only per-row constants are resized, with an explicit periodic
                # identity check; matrix payloads remain shared and immutable.
                data = (base / old["data"]["file"]).read_bytes()
                unit = data[: SIZES[dtype]]
                if not unit or data != unit * (len(data) // len(unit)):
                    raise ValueError("Cannot resize a nonuniform constant: " + name)
                target = base / "data" / f"batch-{family}-{name}.bin"
                payload = unit * (size // len(unit))
                if target.exists():
                    if target.read_bytes() != payload:
                        raise ValueError("Existing batch constant differs: " + str(target))
                else:
                    target.write_bytes(payload)
                return define(
                    f"batch.{family}.{name}",
                    size,
                    dict(data=dict(file=str(target.relative_to(base)), sha256=file_hash(target))),
                )
            fixed = name.startswith(("cache.", "kv.int8.work."))
            label = name if fixed else "batch." + family + "." + name
            if fixed:
                parent = old.get("view", {}).get("parent")
                if parent:
                    parent_size = original[parent]["bytes"] * capacity // old_capacity
                    define(parent, parent_size)
                    if parent not in private:
                        private.append(parent)
                    define(label, size, dict(view=dict(parent=parent, offset=0)))
                else:
                    define(label, size)
                if label not in private:
                    private.append(label)
            else:
                if old and "view" in old and name.startswith("sequence."):
                    return runtime_buffer(name, family, size)
                # Row workspaces are independent allocations; dropping obsolete
                # static views makes mixed-owner slices unambiguous.
                define(label, size)
            return label

        def source_spec(step):
            if step["manifest"] not in specs:
                specs[step["manifest"]] = json.loads((base / step["manifest"]).read_text())
            return specs[step["manifest"]]

        def kernel(step, packet, family, *, head=False, custom=None):
            source = source_spec(step) if custom is None else custom
            module_name, name = source["op"].split(".")
            module = importlib.import_module("tools.operators.ascend." + module_name)
            factory = getattr(module, name)
            parameters = {k: v for k, v in source["parameters"].items() if not k.startswith("_")}
            if head and module_name == "topk" and name == "finalize":
                parameters["output_count"] = 1
            old_rows = packet["rows"]
            packet_cap = owners if family == "decode" else rows
            axis = step.get("row_expression")
            source_rows = resolve(axis, old_rows) if axis else step.get("fixed_rows", old_rows)
            cap = owners if head else (resolve(axis, packet_cap) if axis else packet_cap)
            dynamic = {}
            for key in ("rows", "tokens", "expert_rows"):
                if key in parameters:
                    # Output head has its own row axis, independent of aggregate M.
                    divisor = 1 if head else source_rows
                    if parameters[key] % divisor:
                        raise ValueError("Nonintegral row factor")
                    dynamic[key] = parameters[key] // divisor
                    parameters[key] = cap * dynamic[key]
            if module_name == "memory" and name == "gather_quantized_rows":
                dynamic["table_rows"] = 1
                parameters["table_rows"] = cap
            if module_name == "routing_parallel" and name == "prefix":
                parameters["scan_rows"] = cap
            for key in (
                "capacity",
                "compressed_capacity",
                "table_rows",
                "length",
                "logical_capacity",
            ):
                if key not in dynamic and parameters.get(key) in (old_capacity, old_capacity // 2):
                    parameters[key] = parameters[key] * capacity // old_capacity
            if module_name == "topk" and name == "merge":
                prior = "batch." + family + "." + step["buffers"][0]
                parameters["sources"] = definitions[prior]["bytes"] // (cap * 1024 * 4)
            args = dict(parameters)
            m = row_count(cap, True)
            for key, factor in dynamic.items():
                args[key] = m * factor
            indirect_op = module_name.startswith(
                ("attention", "indexer", "compressor", "kv_int8")
            ) or (module_name == "memory" and name == "scatter_rows")
            geometry_key = json.dumps([module_name, name, parameters, dynamic, cap], sort_keys=True)
            if geometry_key not in geometry_cache:
                function = factory(**args)
                shapes = physical_shapes(function, cap)
                tensor_params = [
                    function.buffer_map[p] for p in function.params if p in function.buffer_map
                ]
                local_shapes = [
                    owner_geometry(p, cap) if indirect_op else (None, None, 1)
                    for p in tensor_params
                ]
                geometry_cache[geometry_key] = function, shapes, tensor_params, local_shapes
            function, shapes, tensor_params, local_shapes = geometry_cache[geometry_key]
            operands, bound = [], []
            for port, old_operand, shape, old_name, (_, _, alignment) in zip(
                tensor_params,
                source["operands"],
                shapes,
                step["buffers"],
                local_shapes,
                strict=True,
            ):
                if (
                    head
                    and module_name == "topk"
                    and name == "finalize"
                    and old_operand["access"] == "write"
                ):
                    old_name += ".compact"
                dtype = DTYPES[str(port.dtype)]
                operands.append(operand(port.name, dtype, shape, old_operand["access"]))
                bound.append(
                    binding(
                        old_name,
                        shape,
                        dtype,
                        family=family,
                        source_rows=old_rows,
                        head=head,
                        alignment=alignment if (old_operand["access"] == "scratch") else 1,
                    )
                )
            indirect_op = module_name.startswith(
                ("attention", "indexer", "compressor", "kv_int8")
            ) or (module_name == "memory" and name == "scatter_rows")
            task = dict(
                module=module.__name__,
                name=name,
                arguments=parameters,
                dynamic=dynamic,
                capacity=cap,
                specs=operands,
            )
            result = dict(op="kernel", buffers=bound)
            if axis is not None and not head:
                result["row_expression"] = axis
            if head:
                result["owners"] = True
            if indirect_op:
                indices, accesses, aliases = {}, [], {}
                column_mapping = []
                for port, op, buffer, (first, last, alignment) in zip(
                    tensor_params, operands, bound, local_shapes, strict=True
                ):
                    definition = definitions[buffer]
                    if any(k in definition for k in ("weight", "data", "prepared_weight")):
                        continue
                    # All dynamic owner operands have a leading linear row
                    # dimension, possibly multiplied by heads. Validate the
                    # actual primitive rather than assuming declared layouts.
                    if alignment == 0:
                        stride = None
                        if buffer not in private:
                            raise ValueError("Unexpected fixed mutable owner operand: " + buffer)
                    else:
                        if alignment > 1 and op["access"] != "scratch":
                            raise ValueError("Only scratch operands may require padded owner rows")
                        stride = math.prod(last) * SIZES[op["dtype"]] // cap
                    column = dict(
                        buffer=buffer,
                        row_bytes=stride,
                        row_alignment=max(1, alignment),
                        decode=(family == "decode") if stride is not None else None,
                    )
                    identity = json.dumps(column, sort_keys=True)
                    if identity not in column_ids:
                        column_ids[identity] = len(columns)
                        columns.append(column)
                    index = column_ids[identity]
                    if column not in rank_columns:
                        rank_columns.append(column)
                    local_index = len(indices)
                    indices[port.name] = local_index
                    column_mapping.append(index)
                    accesses.append(
                        dict(
                            column=local_index,
                            write=op["access"] in ("write", "read_write", "scratch"),
                        )
                    )
                    aliases[port.name + "_owner"] = str(port.dtype)
                # The final pointer table width is filled after every rank.
                wrapper_key = (geometry_key, tuple(indices.items()))
                if wrapper_key not in indirect_cache:
                    _, identities, grid = indirect(function, indices, owners, cap)
                    indirect_cache[wrapper_key] = identities, grid
                identities, grid = indirect_cache[wrapper_key]
                import hashlib

                work_name = (
                    "batch.work."
                    + hashlib.sha256(
                        json.dumps(
                            [grid, step["owner_range"]] if "owner_range" in step else grid,
                            sort_keys=True,
                        ).encode()
                    ).hexdigest()[:16]
                )
                work[work_name] = grid
                if "owner_range" in step:
                    work_ranges[work_name] = dict(
                        min_rows=step["owner_range"][0],
                        max_rows=packet_cap
                        if step["owner_range"][1] == old_rows
                        else step["owner_range"][1],
                    )
                define(work_name, 8192 * 2 * 4)
                pointer_name = (
                    "batch.pointers."
                    + hashlib.sha256(json.dumps(column_mapping).encode()).hexdigest()[:16]
                )
                pointer_tables[pointer_name] = column_mapping
                result.update(
                    owner_work=grid,
                    buffers=[work_name, pointer_name, "batch.active"]
                    + [bound[[p.name for p in tensor_params].index(n)] for n in identities[3:]],
                )
                source_file = Path(inspect.getfile(factory))
                if source_file not in source_hashes:
                    source_hashes[source_file] = file_hash(source_file)
                task.update(
                    columns=indices,
                    owners=owners,
                    specs=[
                        operand("OwnerWork", "i32", [8192, 2]),
                        operand("OwnerPointers", "u64", [owners, 1]),
                        operand("PacketActive", "i32", [1]),
                    ]
                    + [operands[[p.name for p in tensor_params].index(n)] for n in identities[3:]],
                    metadata=dict(
                        _owner_token_capacity=cap,
                        _owner_access=accesses,
                        _owner_aliases=aliases,
                        _owner_writes=[
                            p.name
                            for p, a in zip(
                                [p for p in tensor_params if p.name in indices],
                                accesses,
                                strict=True,
                            )
                            if a["write"]
                        ],
                        _owner_source_sha256=source_hashes[source_file],
                    ),
                )
            key = json.dumps(task, sort_keys=True)
            if key not in tasks:
                task["key"] = f"@batch{len(tasks)}"
                tasks[key] = task
            result["manifest"] = tasks[key]["key"]
            return result

        for source in old_packets:
            family = "decode" if source["rows"] == 1 else "prefill"
            prefix_lengths = any(
                source_spec(s)["parameters"].get("length_input", False)
                for s in source["body"]
                if s["op"] == "kernel"
            )
            packet = deepcopy(source)
            packet.update(
                rows=owners if family == "decode" else rows,
                min_rows=1 if source.get("min_rows", 0) < 3 else source["min_rows"],
                batch_decode=family == "decode",
            )
            packet["coordinates"] = {
                k: ("batch." + family + "." + v if isinstance(v, str) else v)
                for k, v in source["coordinates"].items()
            }
            packet["coordinates"]["device_indices"] = True
            # Coordinates are uploaded as a common packet boundary, including
            # roles unused by this family's device attention implementation.
            for role, old in source["coordinates"].items():
                if not isinstance(old, str):
                    continue
                size = original[old]["bytes"] * packet["rows"] // source["rows"]
                if role in ("c1_indices", "c2_indices") and not prefix_lengths:
                    size = size * capacity // old_capacity
                define("batch." + family + "." + old, size)

            for part in ("body", "head"):
                steps = []
                if part == "body" and family == "decode":
                    # Device-generated context indices use each owner's position
                    # rows; arithmetic is independent of aggregate batch length.
                    for ratio in (1, 2):
                        c = source["coordinates"]
                        stub = dict(
                            op="kernel",
                            buffers=[c["slots"], c[f"c{ratio}_indices"], c[f"c{ratio}_lengths"]],
                        )
                        custom = dict(
                            op="model_ops.causal_indices",
                            parameters=dict(
                                rows=1, capacity=old_capacity, ratio=ratio, dense=not prefix_lengths
                            ),
                            operands=[
                                dict(access="read"),
                                dict(access="write"),
                                dict(access="write"),
                            ],
                        )
                        steps.append(kernel(stub, source, family, custom=custom))
                for step in source[part]:
                    if step["op"] == "kernel":
                        if step.get("owner_range", [1])[0] > packet["rows"]:
                            continue
                        op = source_spec(step)["op"]
                        if op == "memory.last_row":
                            continue
                        is_head = part == "head" and op != "model_ops.norm"
                        if part == "head" and is_head and (not steps or "owners" not in steps[-1]):
                            # Gather each selected owner's last norm row before
                            # executing the shared batched vocabulary projection.
                            norm = (
                                "batch."
                                + family
                                + "."
                                + (
                                    "m1.final.norm"
                                    if family == "decode"
                                    else f"m{source['rows']}.final.norm"
                                )
                            )
                            last = (
                                "batch."
                                + family
                                + "."
                                + (
                                    "m1.final.last"
                                    if family == "decode"
                                    else f"m{source['rows']}.final.last"
                                )
                            )
                            define(last, owners * 5120 * 2)
                            steps.append(gather_head(norm, last, owners, packet["rows"], tasks))
                        steps.append(kernel(step, source, family, head=is_head))
                    else:
                        s = deepcopy(step)
                        for key in (
                            "source",
                            "destination",
                            "buffer",
                            "indices",
                            "gather",
                            "restore",
                            "workspace",
                            "values",
                            "scales",
                            "mask",
                        ):
                            if isinstance(s.get(key), str) and s[key] in original:
                                old = s[key]
                                if immutable(old):
                                    define(old, original[old]["bytes"], original[old])
                                else:
                                    size = (
                                        original[old]["bytes"]
                                        if key == "workspace"
                                        else original[old]["bytes"]
                                        * packet["rows"]
                                        // source["rows"]
                                    )
                                    runtime_buffer(old, family, size)
                                s[key] = old if immutable(old) else "batch." + family + "." + old
                        for key in ("buffers", "groups"):
                            names = s.get(key)
                            if names is None:
                                continue
                            entries = [names] if isinstance(names, str) else names
                            mapped = []
                            for old in entries:
                                if immutable(old):
                                    define(old, original[old]["bytes"], original[old])
                                    mapped.append(old)
                                else:
                                    size = original[old]["bytes"]
                                    if old.startswith("m") and old not in (s.get("workspace"),):
                                        # W4 group counts are expert-sized; their
                                        # allocation does not grow with rows.
                                        if not old.endswith(".counts"):
                                            size = size * packet["rows"] // source["rows"]
                                    label = "batch." + family + "." + old
                                    runtime_buffer(old, family, size)
                                    mapped.append(label)
                            s[key] = mapped[0] if isinstance(names, str) else mapped
                        if s["op"] in ("gather", "reduce") and not s.get("fixed_rows"):
                            s["elements"] = s["elements"] * packet["rows"] // source["rows"]
                        if s["op"] == "w4":
                            s["rows"] = s["rows"] * packet["rows"] // source["rows"]
                        if (
                            s["op"] == "gather"
                            and family == "decode"
                            and s["source"].endswith(".engram.local")
                        ):
                            destination = s["destination"]
                            gathered = define(
                                destination + ".gathered", definitions[destination]["bytes"]
                            )
                            s["destination"] = gathered
                            task = dict(
                                module="tools.operators.ascend.memory",
                                name="gather_columns",
                                arguments=dict(rows=owners, width=25600),
                                dynamic=dict(rows=1),
                                capacity=owners,
                                specs=[
                                    operand("gathered", "bf16", [8, owners, 3200]),
                                    operand("out", "bf16", [owners, 25600], "write"),
                                ],
                            )
                            key = json.dumps(task, sort_keys=True)
                            if key not in tasks:
                                task["key"] = f"@batch{len(tasks)}"
                                tasks[key] = task
                            steps.append(s)
                            steps.append(
                                dict(
                                    op="kernel",
                                    manifest=tasks[key]["key"],
                                    buffers=[gathered, destination],
                                )
                            )
                        else:
                            steps.append(s)
                packet[part] = steps
            for key in ("best_value", "best_index", "logits"):
                packet[key] = "batch." + family + "." + source[key]
            packet["best_index"] += ".compact"
            packet["best_value"] += ".compact"
            packet["reset"] = source["reset"]
            packets.append(packet)
        print(f"Rank {rank['rank']} geometry ready", flush=True)
        rank["packets"] = packets
        rank["buffers"] = list(definitions.values())
        if expected_columns is None:
            expected_columns = rank_columns
        elif expected_columns != rank_columns:
            raise ValueError("Ranks disagree on private pointer-column geometry")
    for task in tasks.values():
        if "columns" in task:
            task["specs"][1]["shape"] = [owners, len(task["columns"]) + 1]
            task["table_columns"] = len(task["columns"])
    # Work names used only by another rank are deliberately shared across TP.
    for rank in model["ranks"]:
        rank["buffers"].extend(
            [
                dict(name="batch.active", bytes=4),
                dict(name="batch.endpoints", bytes=owners * 4),
                dict(name="batch.head.errors", bytes=owners * 4),
            ]
        )
        rank["buffers"].extend(
            dict(name=n, bytes=owners * (len(mapping) + 1) * 8)
            for n, mapping in pointer_tables.items()
        )
        known = {b["name"] for b in rank["buffers"]}
        rank["buffers"].extend(dict(name=n, bytes=8192 * 2 * 4) for n in work if n not in known)
    model["batch"] = dict(
        owners=owners,
        columns=columns,
        private=private,
        pointers=pointer_tables,
        active="batch.active",
        endpoints="batch.endpoints",
        work=work,
        work_ranges=work_ranges,
        prefill_quantum=min(512, rows) if sequence_parallel else 1,
    )
    print(
        f"Compile {len(tasks)} shared batch kernels; {len(columns)} pointer columns; {len(private)} private allocations/views",
        flush=True,
    )
    atomic_json(output.with_suffix(".build.json"), dict(model=model, tasks=list(tasks.values())))
    return compile_contract(model, list(tasks.values()), base, output, workers)


def compile_contract(model, tasks, base, output, workers):
    for task in tasks:
        if "metadata" in task:
            module = importlib.import_module(task["module"])
            factory = getattr(module, task["name"])
            if task["metadata"]["_owner_source_sha256"] != file_hash(
                Path(inspect.getfile(factory))
            ):
                raise ValueError("Owner primitive source changed; reconstruct the batch contract")
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        compiled = dict(pool.map(compile_task, ((t, base) for t in tasks)))
    for rank in model["ranks"]:
        for packet in rank["packets"]:
            for step in packet["body"] + packet["head"]:
                if step["op"] == "kernel":
                    step["manifest"] = compiled[step["manifest"]]
    validate_rows(model, base)
    atomic_json(output, model)
    return model


def owner_geometry(port, cap):
    from tilelang import tvm

    first, last = physical_shapes_single(port, 1), physical_shapes_single(port, cap)
    if first == last:
        return first, last, 0
    if first[1:] != last[1:]:
        return first, last, 1
    variables = {}
    extent = math.prod(port.shape)
    tvm.tir.stmt_functor.post_order_visit(
        extent, lambda n: variables.setdefault(n, n) if isinstance(n, tvm.tir.Var) else None
    )
    row = next(iter(variables)) if len(variables) == 1 else None
    analyzer = tvm.arith.Analyzer()
    for alignment in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        if cap % alignment:
            continue
        volume = math.prod(last)
        if (
            volume % cap == 0
            and row is not None
            and analyzer.can_prove(
                extent == ((row + alignment - 1) // alignment) * alignment * (volume // cap)
            )
        ):
            return first, last, alignment
        if volume % cap == 0 and all(
            math.prod(physical_shapes_single(port, q))
            == ((q + alignment - 1) // alignment) * alignment * (volume // cap)
            for q in range(1, cap + 1)
        ):
            return first, last, alignment
    raise ValueError("Unsupported owner row padding: " + port.name)


def physical_shapes_single(port, rows):
    from tilelang import tvm

    variables = {}
    for dim in port.shape:
        tvm.tir.stmt_functor.post_order_visit(
            dim, lambda n: variables.setdefault(n, rows) if isinstance(n, tvm.tir.Var) else None
        )
    return [
        int(tvm.arith.Analyzer().simplify(tvm.tir.stmt_functor.substitute(d, variables)))
        for d in port.shape
    ]


def gather_head(source, target, owners, rows, tasks):
    from tools.operators.ascend.build import operand

    task = dict(
        module="tools.operators.ascend.memory",
        name="gather_rows",
        arguments=dict(rows=owners, table_rows=rows, width=5120, index_dtype="int32"),
        dynamic=dict(rows=1),
        capacity=owners,
        specs=[
            operand("table", "bf16", [rows, 5120]),
            operand("indices", "i32", [owners]),
            operand("out", "bf16", [owners, 5120], "write"),
            operand("errors", "i32", [owners], "write"),
        ],
    )
    key = json.dumps(task, sort_keys=True)
    if key not in tasks:
        task["key"] = f"@batch{len(tasks)}"
        tasks[key] = task
    return dict(
        op="kernel",
        manifest=tasks[key]["key"],
        buffers=[source, "batch.endpoints", target, "batch.head.errors"],
        owners=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--plan", type=Path)
    source.add_argument("--rebuild-contract", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--package-output", type=Path)
    p.add_argument("--owners", type=int, default=128)
    p.add_argument("--capacity", type=int, default=4096)
    p.add_argument("--rows", type=int, default=2048)
    p.add_argument("--compile-workers", type=int, default=4)
    p.add_argument("--sequence-parallel", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--shared-attention", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--reduce-scatter", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--fused-attention", action=argparse.BooleanOptionalAction, default=True)
    a = p.parse_args()
    input_path = a.plan or a.rebuild_contract
    if (
        a.output.parent.resolve() != input_path.parent.resolve()
        or a.output.resolve() == input_path.resolve()
    ):
        p.error("Output must share the source asset directory and preserve its input")
    if (
        not 2 <= a.owners <= 128
        or not 128 <= a.capacity <= 16384
        or a.capacity % 256
        or not a.owners <= a.rows <= min(a.capacity, 4096)
    ):
        p.error("Invalid owner, context or token capacity")
    if a.rebuild_contract:
        contract = json.loads(a.rebuild_contract.read_text())
        model = compile_contract(
            contract["model"], contract["tasks"], input_path.parent, a.output, a.compile_workers
        )
    else:
        if a.reduce_scatter and not a.sequence_parallel:
            p.error("ReduceScatter requires the shared sequence transform")
        model = build(
            a.plan,
            a.output,
            a.owners,
            a.capacity,
            a.rows,
            a.compile_workers,
            sequence_parallel=a.sequence_parallel,
            shared_attention=a.shared_attention,
            reduce_scatter=a.reduce_scatter,
            fused_attention=a.fused_attention,
        )
    if a.package_output:
        publish_package(model, input_path.parent, a.package_output)


if __name__ == "__main__":
    main()
