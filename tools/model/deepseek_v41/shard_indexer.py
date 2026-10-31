"""Partition prefill index scores/TopK by row and gather exact selected indices.

Owner-private plans partition the complete dynamic prefill domain through the
shared row/address-table contract. Flat reference plans retain their original
short-fragment dispatch. Both reuse the existing indexer and TopK arithmetic;
request state and global attention consumers retain their layouts.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import importlib
import hashlib
import inspect
import json
import math
import multiprocessing
from pathlib import Path

from tools.build.files import atomic_json
from tools.model.row_contract import expression


def build_private(plan, output, workers=4):
    """Partition shared query rows while preserving owner-private key bindings."""
    from kernels.model.rows import row_count
    from tools.build.files import file_hash
    from tools.model.deepseek_v41.batch import compile_task, DTYPES, SIZES
    from tools.model.deepseek_v41.assembly import compact_buffers, publish_package, validate_rows
    from tools.model.deepseek_v41.sequence_parallel import external_reads
    from tools.operators.ascend import indexer_cube, memory
    from tools.operators.ascend.build import operand
    from tools.operators.ascend.geometry import physical_shapes
    from tools.operators.ascend.private_rows import indirect

    model = json.loads(plan.read_text())
    if output.exists() or not model.get("batch") or model["batch"].get("partitioned_tables"):
        raise ValueError("Require an unpartitioned owner-private plan and new output directory")
    validate_rows(model, plan.parent)
    base = output / "source"
    publish_package(model, plan.parent, base)
    contract = model["batch"]
    ranks = len(model["ranks"])
    tasks, descriptions, index_templates = {}, {}, {}
    column_ids = {json.dumps(c, sort_keys=True): i for i, c in enumerate(contract["columns"])}
    partitions = contract.setdefault("partitioned_tables", {})

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
            tasks[identity] = dict(task, key=f"@private_index{len(tasks)}")
        return tasks[identity]["key"]

    changed = 0
    for rank in model["ranks"]:
        definitions = {d["name"]: d for d in rank["buffers"]}

        def define(name, size):
            if name not in definitions:
                definitions[name] = dict(name=name, bytes=size)
                rank["buffers"].append(definitions[name])
            elif definitions[name]["bytes"] != size:
                raise ValueError("Inconsistent partition buffer extent: " + name)
            return name

        for packet in rank["packets"]:
            if packet.get("batch_decode"):
                continue
            capacity = packet["rows"]
            if ranks < 2 or capacity % ranks:
                raise ValueError("Query capacity must be divisible by the rank count")
            count = capacity // ranks
            body, steps, index, sources = [], packet["body"], 0, 0
            while index < len(steps):
                old = steps[index]
                descriptor = describe(old["manifest"]) if old["op"] == "kernel" else None
                if (
                    descriptor is None
                    or descriptor["op"] != "private_rows.indexer_cube_causal_logits"
                ):
                    body.append(old)
                    index += 1
                    continue
                if "owner_work" not in old:
                    raise ValueError("Private indexer requires indirect owner work")
                sources += 1
                p = descriptor["parameters"]
                if not p.get("length_input") or p.get("_owner_token_capacity") != capacity:
                    raise ValueError("Partition requires length-driven complete query rows")
                if p["_owner_source_sha256"] != file_hash(
                    Path(inspect.getfile(indexer_cube.causal_logits))
                ):
                    raise ValueError("Private indexer factory differs from its source identity")
                original_mapping = contract["pointers"][old["buffers"][1]]
                columns = [deepcopy(contract["columns"][i]) for i in original_mapping]
                if len(columns) != 8 or any(c["row_alignment"] != 1 for c in columns):
                    raise ValueError("Partition requires the original unpadded indexer operands")
                mapping = {}
                for slot, width, dtype in (
                    (0, 4096, "i8"),
                    (1, 32, "f16"),
                    (4, 32, "bf16"),
                    (5, 1, "i32"),
                ):
                    column = columns[slot]
                    source = column["buffer"]
                    if column["row_bytes"] != width * SIZES[dtype] or column["decode"] is not False:
                        raise ValueError("Unexpected query row geometry")
                    packed = define(source + ".index_partition", count * width * SIZES[dtype])
                    mapping[source] = packed
                    params = dict(
                        rows=capacity,
                        count=count,
                        width=width,
                        first=rank["rank"],
                        stride=ranks,
                        dtype={"i8": "int8", "f16": "float16", "bf16": "bfloat16", "i32": "int32"}[
                            dtype
                        ],
                        dynamic_count=True,
                    )
                    manifest = register(
                        memory,
                        "partition_rows",
                        params,
                        [
                            operand("source", dtype, [capacity, width]),
                            operand("out", dtype, [count, width], "write"),
                        ],
                        capacity,
                    )
                    body.append(dict(op="kernel", manifest=manifest, buffers=[source, packed]))
                for slot in (6, 7):
                    column = columns[slot]
                    if not column["row_bytes"] or column["decode"] is not False:
                        raise ValueError("Indexer workspace must consist of shared query rows")
                    mapping[column["buffer"]] = define(
                        column["buffer"] + ".index_partition", count * column["row_bytes"]
                    )
                for column in columns:
                    if column["row_bytes"] is not None:
                        column.update(buffer=mapping[column["buffer"]], row_partition=ranks)

                geometry = (
                    old["manifest"],
                    count,
                    tuple((c["row_bytes"], definitions[c["buffer"]]["bytes"]) for c in columns),
                )
                if geometry not in index_templates:
                    args = {k: v for k, v in p.items() if not k.startswith("_")}
                    args["rows"] = count
                    args.setdefault("direct_sum", True)
                    function = indexer_cube.causal_logits(**dict(args, rows=row_count(count, True)))
                    tensors = [
                        function.buffer_map[param]
                        for param in function.params
                        if param in function.buffer_map
                    ]
                    aliases = {tensor.name + "_owner": str(tensor.dtype) for tensor in tensors}
                    if aliases != p["_owner_aliases"] or list(aliases) != list(p["_owner_aliases"]):
                        raise ValueError("Indexer owner tensor order changed")
                    for live_rows in (1, count):
                        for tensor, shape, column in zip(
                            tensors, physical_shapes(function, live_rows), columns, strict=True
                        ):
                            expected = (
                                live_rows * column["row_bytes"]
                                if column["row_bytes"] is not None
                                else definitions[column["buffer"]]["bytes"]
                            )
                            if math.prod(shape) * SIZES[DTYPES[str(tensor.dtype)]] != expected:
                                raise ValueError(
                                    "Partition changes indexer physical operand geometry"
                                )
                    local_columns = {tensor.name: i for i, tensor in enumerate(tensors)}
                    _, identities, grid = indirect(
                        function,
                        local_columns,
                        contract["owners"],
                        count,
                        table_columns=len(columns),
                    )
                    if len(identities) != 3:
                        raise ValueError("Private indexer has unexpected external operands")
                    metadata = {k: v for k, v in p.items() if k.startswith("_owner_")}
                    metadata["_owner_token_capacity"] = count
                    manifest = register(
                        indexer_cube,
                        "causal_logits",
                        args,
                        descriptor["operands"],
                        count,
                        columns=local_columns,
                        owners=contract["owners"],
                        table_columns=len(columns),
                        metadata=metadata,
                    )
                    index_templates[geometry] = manifest, grid
                manifest, grid = index_templates[geometry]
                ids = []
                for column in columns:
                    identity = json.dumps(column, sort_keys=True)
                    if identity not in column_ids:
                        column_ids[identity] = len(contract["columns"])
                        contract["columns"].append(column)
                    ids.append(column_ids[identity])
                pointer = (
                    "batch.pointers." + hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]
                )
                work = (
                    "batch.work."
                    + hashlib.sha256(
                        json.dumps([grid, ranks], sort_keys=True).encode()
                    ).hexdigest()[:16]
                )
                contract["pointers"][pointer] = ids
                contract["work"][work] = grid
                partitions.update({pointer: ranks, work: ranks})
                define(pointer, contract["owners"] * (len(columns) + 1) * 8)
                define(work, 8192 * 2 * 4)
                body.append(
                    dict(
                        op="kernel",
                        manifest=manifest,
                        owner_work=grid,
                        buffers=[work, pointer, contract["active"]],
                    )
                )
                index += 1
                while index < len(steps):
                    step = steps[index]
                    if step["op"] != "kernel":
                        raise ValueError("Indexer must be followed by complete TopK")
                    spec = describe(step["manifest"])
                    if not spec["op"].startswith("topk."):
                        raise ValueError("Unexpected consumer between indexer and TopK")
                    if spec["op"] == "topk.local" and step["buffers"][1] not in mapping:
                        raise ValueError("Indexer and TopK must share identical causal lengths")
                    specs = deepcopy(spec["operands"])
                    for source, port in zip(step["buffers"], specs, strict=True):
                        if port["shape"][0] != capacity:
                            raise ValueError("TopK must use shared query-major rows")
                        port["shape"][0] = count
                        if source not in mapping:
                            mapping[source] = define(
                                source + ".index_partition",
                                math.prod(port["shape"]) * SIZES[port["dtype"]],
                            )
                    module_name, name = spec["op"].split(".")
                    module = importlib.import_module("tools.operators.ascend." + module_name)
                    arguments = {
                        k: v for k, v in spec["parameters"].items() if not k.startswith("_")
                    }
                    arguments["rows"] = count
                    path = register(module, name, arguments, specs, count)
                    body.append(
                        dict(
                            op="kernel",
                            manifest=path,
                            buffers=[mapping[b] for b in step["buffers"]],
                            row_expression=expression(f"(m + {ranks - 1}) // {ranks}"),
                        )
                    )
                    index += 1
                    if spec["op"] == "topk.finalize":
                        selected, values = step["buffers"][1:]
                        if any(
                            c["buffer"] == values for c in contract["columns"]
                        ) or external_reads(
                            steps[index:] + packet["head"],
                            {values},
                            lambda step: describe(step["manifest"]),
                        ):
                            raise ValueError(
                                "Global TopK values require an explicit gather contract"
                            )
                        break
                else:
                    raise ValueError("Missing TopK finalization")
                gathered = define("batch.index_partition.gathered", capacity * 512 * 4)
                body.append(
                    dict(
                        op="gather",
                        source=mapping[selected],
                        destination=gathered,
                        elements=count * 512,
                        fp32=False,
                        int32=True,
                        elements_expression=expression(f"((m + {ranks - 1}) // {ranks}) * 512"),
                    )
                )
                path = register(
                    memory,
                    "interleave_rows",
                    dict(
                        rows=capacity,
                        count=count,
                        width=512,
                        ranks=ranks,
                        dtype="int32",
                        dynamic_count=True,
                    ),
                    [
                        operand("source", "i32", [capacity, 512]),
                        operand("out", "i32", [capacity, 512], "write"),
                    ],
                    capacity,
                )
                body.append(dict(op="kernel", manifest=path, buffers=[gathered, selected]))
            if sources != 8:
                raise ValueError("Expected all eight owner-private indexer sources")
            packet["body"] = body
            changed += sources
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
        changed_bindings=changed,
        compiled=len(tasks),
        model=str(output / "package/model.json"),
        resident_bytes=[
            sum(b["bytes"] for b in rank["buffers"] if "view" not in b) for rank in model["ranks"]
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-rows", type=int)
    parser.add_argument("--compile-workers", type=int, default=4)
    args = parser.parse_args()
    if json.loads(args.plan.read_text()).get("batch"):
        if args.min_rows not in (None, 1):
            parser.error("Private query partition uses the complete dynamic row domain")
        print(json.dumps(build_private(args.plan, args.output, args.compile_workers)), flush=True)
        return
    args.min_rows = args.min_rows or 512
    if args.output.parent.resolve() != args.plan.parent.resolve():
        parser.error("Output must share the input plan's asset directory")
    from tools.model.deepseek_v41.build import compile_task
    from tools.operators.ascend import memory
    from tools.operators.ascend.build import operand

    model = json.loads(args.plan.read_text())
    base = args.plan.parent
    compiled = {}
    sizes = {"i8": 1, "i32": 4, "f16": 2, "bf16": 2, "f32": 4}
    names = {"i8": "int8", "i32": "int32", "f16": "float16", "bf16": "bfloat16"}

    def kernel(module, name, parameters, specs, capacity):
        identity = json.dumps([module.__name__, name, parameters, specs, capacity], sort_keys=True)
        if identity not in compiled:
            compiled[identity] = dict(
                module=module.__name__,
                name=name,
                arguments=parameters,
                specs=specs,
                row_capacity=capacity,
                dynamic_arguments={"rows": 1},
                placeholder=f"@index_shard{len(compiled)}",
            )
        return compiled[identity]["placeholder"]

    for rank in model["ranks"]:
        if len(rank["packets"]) != 2:
            raise ValueError("Expected one original decode and prefill packet")
        fallback = rank["packets"][1]
        capacity = fallback["rows"]
        if capacity % 8 or not 2 < args.min_rows <= capacity:
            raise ValueError("Expected TP8-divisible capacity and a nonempty short fallback")
        count = capacity // 8
        local = deepcopy(fallback)
        fallback["max_rows"] = args.min_rows - 1
        local["min_rows"] = args.min_rows
        definitions = {d["name"]: d for d in rank["buffers"]}

        def temp(name, shape, dtype):
            name = "index.shard." + name
            size = math.prod(shape) * sizes[dtype]
            if name not in definitions:
                definition = dict(name=name, bytes=size)
                definitions[name] = definition
                rank["buffers"].append(definition)
            elif definitions[name]["bytes"] != size:
                raise ValueError("Inconsistent local workspace geometry: " + name)
            return name

        def partition(source, width, dtype, repeat_last=False):
            output = temp(source, [count, width], dtype)
            specs = [
                operand("source", dtype, [capacity, width]),
                operand("out", dtype, [count, width], "write"),
            ]
            params = dict(
                rows=capacity,
                count=count,
                width=width,
                first=rank["rank"],
                stride=8,
                dtype=names[dtype],
                repeat_last=repeat_last,
            )
            params["dynamic_count"] = True
            manifest = kernel(memory, "partition_rows", params, specs, capacity)
            return output, dict(op="kernel", manifest=manifest, buffers=[source, output])

        body, steps, index, sources = [], local["body"], 0, 0
        while index < len(steps):
            step = steps[index]
            descriptor = (
                json.loads((base / step["manifest"]).read_text())
                if step["op"] == "kernel"
                else None
            )
            if descriptor is None or descriptor["op"] != "indexer_cube.causal_logits":
                body.append(step)
                index += 1
                continue
            sources += 1
            params = descriptor["parameters"]
            length = params["length"]
            mapping = {}
            for slot, width, dtype in (
                (0, 4096, "i8"),
                (1, 32, "f16"),
                (4, 32, "bf16"),
                (5, length, "i32"),
            ):
                old = step["buffers"][slot]
                packed, pack_step = partition(old, width, dtype, repeat_last=slot == 5)
                mapping[old] = packed
                body.append(pack_step)
            for slot in (6, 7):
                spec = descriptor["operands"][slot]
                shape = list(spec["shape"])
                shape[0] = shape[0] // 8
                mapping[step["buffers"][slot]] = temp(step["buffers"][slot], shape, spec["dtype"])

            def local_step(old, manifest):
                parameters = {
                    k: v for k, v in manifest["parameters"].items() if not k.startswith("_")
                }
                parameters["rows"] = count
                specs = deepcopy(manifest["operands"])
                for source, spec in zip(old["buffers"], specs, strict=True):
                    if source in mapping:
                        spec["shape"][0] //= 8
                    elif spec["shape"][0] == capacity and manifest["op"].startswith("topk."):
                        spec["shape"][0] = count
                        mapping[source] = temp(source, spec["shape"], spec["dtype"])
                module_name, name = manifest["op"].split(".")
                module = importlib.import_module("tools.operators.ascend." + module_name)
                path = kernel(module, name, parameters, specs, count)
                instruction = dict(
                    op="kernel",
                    manifest=path,
                    buffers=[mapping.get(b, b) for b in old["buffers"]],
                    fixed_rows=count,
                )
                instruction.pop("fixed_rows")
                instruction["row_expression"] = expression("(m + 7) // 8")
                body.append(instruction)

            local_step(step, descriptor)
            index += 1
            if steps[index]["op"] != "kernel":
                raise ValueError("Indexer must be followed by TopK")
            lengths = steps[index]["buffers"][1]
            packed_lengths, pack_step = partition(lengths, 1, "i32")
            mapping[lengths] = packed_lengths
            body.append(pack_step)
            while index < len(steps):
                next_step = steps[index]
                manifest = json.loads((base / next_step["manifest"]).read_text())
                if not manifest["op"].startswith("topk."):
                    raise ValueError("Unexpected operator inside index TopK")
                local_step(next_step, manifest)
                index += 1
                if manifest["op"] == "topk.finalize":
                    selected = next_step["buffers"][1]
                    break
            gathered = temp("gathered", [capacity, 512], "i32")
            transport = dict(
                op="gather",
                source=mapping[selected],
                destination=gathered,
                elements=count * 512,
                fp32=False,
                int32=True,
                fixed_rows=True,
            )
            transport.pop("fixed_rows")
            transport["elements_expression"] = expression("((m + 7) // 8) * 512")
            body.append(transport)
            specs = [
                operand("source", "i32", [capacity, 512]),
                operand("out", "i32", [capacity, 512], "write"),
            ]
            params = dict(rows=capacity, count=count, width=512, ranks=8, dtype="int32")
            params["dynamic_count"] = True
            path = kernel(memory, "interleave_rows", params, specs, capacity)
            body.append(dict(op="kernel", manifest=path, buffers=[gathered, selected]))
        if sources != 8:
            raise ValueError("Expected all eight indexer source layers")
        local["body"] = body
        rank["packets"].append(local)
        print(
            dict(
                rank=rank["rank"],
                indexers=sources,
                resident_bytes=sum(d["bytes"] for d in rank["buffers"] if d.get("view") is None),
            ),
            flush=True,
        )
    manifests = {}
    with ProcessPoolExecutor(
        max_workers=args.compile_workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        for placeholder, path in pool.map(
            compile_task, [(task, str(base)) for task in compiled.values()]
        ):
            manifests[placeholder] = path
            print(dict(compiled=len(manifests), total=len(compiled)), flush=True)
    for rank in model["ranks"]:
        for packet in rank["packets"]:
            for step in packet["body"] + packet["head"]:
                if step["op"] == "kernel":
                    step["manifest"] = manifests.get(step["manifest"], step["manifest"])
    atomic_json(args.output, model)


if __name__ == "__main__":
    main()
