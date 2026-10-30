"""Bind owner-local bounded decoder rows to existing native A2 arithmetic."""

from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import hashlib
import importlib
import inspect
import json
import multiprocessing
from pathlib import Path

from tools.build.files import atomic_json, file_hash
from tools.model.deepseek_v41.batch import compile_task
from tools.model.deepseek_v41.converge import compact_buffers, publish_package, validate_rows
from tools.model.row_contract import expression


def build(plan, output, keep=128, workers=4, final_prompt_only=False):
    from tools.operators.ascend.build import operand
    from tools.operators.ascend.private_rows import indirect
    from kernels.model.rows import row_count

    if output.exists() or not 128 <= keep <= 512:
        raise ValueError("Require a new output directory and 128..512 retained rows")
    model = json.loads(plan.read_text())
    contract = model.get("batch")
    peers = len(model["ranks"])
    if not contract or peers != 8 or contract.get("decoder_replay"):
        raise ValueError("Require a complete TP8 owner-row plan")
    validate_rows(model, plan.parent)
    base = output / "source"
    publish_package(model, plan.parent, base)
    descriptions, tasks = {}, {}
    decoder_columns, decoder_tables, maps = {}, set(), {}

    def describe(step):
        path = step["manifest"]
        if path not in descriptions:
            descriptions[path] = json.loads((base / path).read_text())
        return descriptions[path]

    def register(task):
        identity = json.dumps(task, sort_keys=True)
        if identity not in tasks:
            tasks[identity] = dict(task, key=f"@decoder_rows{len(tasks)}")
        return tasks[identity]["key"]

    counts = []
    for rank in model["ranks"]:
        definitions = {b["name"]: b for b in rank["buffers"]}

        def define(name, size, view=None):
            definition = dict(name=name, bytes=size)
            if view is not None:
                definition["view"] = view
            if name not in definitions:
                rank["buffers"].append(definition)
                definitions[name] = definition
            elif definitions[name]["bytes"] != size or definitions[name].get("view") != view:
                raise ValueError("Conflicting decoder buffer: " + name)
            return name

        def clone_buffer(name, replacements):
            if name in replacements:
                return replacements[name]
            old = definitions[name]
            if "view" not in old:
                return name
            parent = clone_buffer(old["view"]["parent"], replacements)
            if parent == old["view"]["parent"]:
                return name
            target = "decoder." + name
            define(target, old["bytes"], dict(old["view"], parent=parent))
            replacements[name] = target
            return target

        def move(source, target, width, dtype, capacity, stride, map_name):
            bound = capacity // stride
            errors = define(f"batch.decoder.errors.{stride}", bound * 4)
            arguments = dict(
                rows=bound,
                table_rows=capacity,
                width=width,
                dtype={"bf16": "bfloat16", "i32": "int32"}[dtype],
                index_dtype="int32",
            )
            specs = [
                operand("table", dtype, [capacity, width]),
                operand("indices", "i32", [bound]),
                operand("out", dtype, [bound, width], "write"),
                operand("errors", "i32", [bound], "write"),
            ]
            manifest = register(
                dict(
                    module="tools.operators.ascend.memory",
                    name="gather_rows",
                    arguments=arguments,
                    specs=specs,
                    capacity=bound,
                    dynamic=dict(rows=1),
                )
            )
            step = dict(op="kernel", manifest=manifest, buffers=[source, map_name, target, errors])
            if stride != 1:
                step["row_expression"] = expression(f"(m + {stride - 1}) // {stride}")
            return step

        def gather(source, target, width, dtype, capacity):
            return dict(
                op="gather",
                source=source,
                destination=target,
                elements=capacity // peers * width,
                fp32=dtype == "float32",
                int32=dtype == "int32",
                elements_expression=expression(f"((m + {peers - 1}) // {peers}) * {width}"),
            )

        for packet in rank["packets"]:
            if packet.get("batch_decode"):
                continue
            capacity = packet["rows"]
            if capacity % peers:
                raise ValueError("Decoder row capacity must cover equal peer partitions")
            boundary = (
                next(i for i, s in enumerate(packet["body"]) if s.get("name") == "layer20") + 1
            )
            body, tail = packet["body"][:boundary], deepcopy(packet["body"][boundary:])
            coordinates = packet["coordinates"]
            replacements = {}
            for key, name in coordinates.items():
                if not isinstance(name, str):
                    continue
                old = definitions[name]
                if "view" in old or any(k in old for k in ("data", "weight", "prepared_weight")):
                    raise ValueError("Decoder coordinates require standalone mutable row buffers")
                target = "decoder." + name
                define(target, old["bytes"])
                replacements[name] = target
            packet["decoder_coordinates"] = {
                k: replacements.get(v, v) if isinstance(v, str) else v
                for k, v in coordinates.items()
            }
            if not coordinates.get("device_indices"):
                raise ValueError("Bounded decoder requires device-built compressed index views")
            sp = any(
                step.get("buffers", [None])[0] == "batch.prefill.sequence.m2048.residual.a"
                for step in tail[:12]
                if step.get("buffers")
            )
            stride = peers if sp else 1
            map_name = define(
                "batch.decoder.map.sequence" if sp else "batch.decoder.map.global",
                capacity // stride * 4,
            )
            maps[map_name] = dict(
                buffer=map_name, source_partition=1, destination_partition=stride, contiguous=sp
            )
            query_map = define("batch.decoder.map.query", capacity // peers * 4) if sp else map_name
            if sp:
                maps[query_map] = dict(
                    buffer=query_map, source_partition=peers, destination_partition=peers
                )
            prepare, select = [], []
            prefix = "batch.prefill.m2048."
            if capacity != 2048:
                raise ValueError("Current packet contract requires a 2048-row shared arena")
            for suffix, width in [("residual.a", 20480), ("collapsed", 5120), ("selected", 512)]:
                target = (
                    ("batch.prefill.sequence.m2048." + suffix)
                    if sp and suffix != "selected"
                    else prefix + suffix
                )
                dtype = "int32" if suffix == "selected" else "bfloat16"
                kind = "i32" if suffix == "selected" else "bf16"
                if sp and suffix == "selected":
                    target = prefix + "selected.index_partition"
                source = prefix + suffix
                size = capacity * width * (4 if dtype == "int32" else 2)
                if definitions[source]["bytes"] != size:
                    raise ValueError("Decoder boundary source geometry differs: " + source)
                if sp:
                    prepare.append(gather(target, source, width, dtype, capacity))
                    select.append(
                        move(
                            source,
                            target,
                            width,
                            kind,
                            capacity,
                            stride,
                            query_map if suffix == "selected" else map_name,
                        )
                    )
                else:
                    scratch = prefix + ("residual.b" if suffix == "residual.a" else "norm.attn")
                    if definitions[scratch]["bytes"] < size:
                        raise ValueError("Decoder compaction scratch is too small")
                    if definitions[scratch]["bytes"] != size:
                        scratch = define(
                            f"batch.decoder.scratch.{suffix}", size, dict(parent=scratch, offset=0)
                        )
                    select.append(move(source, scratch, width, kind, capacity, 1, map_name))
                    identity = define("batch.decoder.identity", capacity * 4)
                    if "data" not in definitions[identity]:
                        path = base / "data/decoder-identity.bin"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(b"".join(i.to_bytes(4, "little") for i in range(capacity)))
                        definitions[identity]["data"] = dict(
                            file=str(path.relative_to(base)), sha256=file_hash(path)
                        )
                    select.append(move(scratch, target, width, kind, capacity, 1, identity))
            if sp:
                count = capacity // peers
                args = dict(
                    rows=capacity,
                    count=count,
                    width=128,
                    first=rank["rank"],
                    stride=peers,
                    dtype="int32",
                    dynamic_count=True,
                )
                specs = [
                    operand("source", "i32", [capacity, 128]),
                    operand("out", "i32", [count, 128], "write"),
                ]
                manifest = register(
                    dict(
                        module="tools.operators.ascend.memory",
                        name="partition_rows",
                        arguments=args,
                        specs=specs,
                        capacity=capacity,
                        dynamic=dict(rows=1),
                    )
                )
                select.append(
                    dict(
                        op="kernel",
                        manifest=manifest,
                        buffers=[
                            replacements[coordinates["swa_indices"]],
                            "batch.query_attention.swa_indices",
                        ],
                    )
                )
            first_attention = True
            for step in tail + packet["head"]:
                if step["op"] == "kernel":
                    desc = describe(step)
                    if "owner_work" in step:
                        old_pointer, old_work = step["buffers"][1], step["buffers"][0]
                        for access in desc["parameters"]["_owner_access"]:
                            role = contract["columns"][
                                contract["pointers"][old_pointer][access["column"]]
                            ]["buffer"]
                            if (
                                access["write"]
                                and role.startswith("cache.")
                                and not role.startswith("cache.swa.")
                            ):
                                raise ValueError(
                                    "Decoder boundary precedes a persistent shared KV/key source: "
                                    + role
                                )
                        ids = []
                        for old_id in contract["pointers"][old_pointer]:
                            col = deepcopy(contract["columns"][old_id])
                            if col["row_bytes"] is not None:
                                col["buffer"] = clone_buffer(col["buffer"], replacements)
                                col["decoder"] = True
                                identity = json.dumps(col, sort_keys=True)
                                if identity not in decoder_columns:
                                    decoder_columns[identity] = len(contract["columns"])
                                    contract["columns"].append(col)
                                ids.append(decoder_columns[identity])
                            else:
                                ids.append(old_id)
                        digest = hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]
                        pointer = "batch.decoder.pointers." + digest
                        work = "decoder." + old_work
                        contract["pointers"][pointer] = ids
                        contract["work"][work] = deepcopy(contract["work"][old_work])
                        for field, name, old in [
                            ("work_ranges", work, old_work),
                            ("partitioned_tables", work, old_work),
                            ("partitioned_tables", pointer, old_pointer),
                        ]:
                            if old in contract.get(field, {}):
                                contract[field][name] = deepcopy(contract[field][old])
                        define(pointer, contract["owners"] * (len(ids) + 1) * 8)
                        define(work, 8192 * 2 * 4)
                        decoder_tables.update([pointer, work])
                        step["buffers"][:2] = [work, pointer]
                        if first_attention and desc["op"] in [
                            "private_rows.kv_int8_gather_attention",
                            "private_rows.attention_fused_selected_int8",
                        ]:
                            p = desc["parameters"]
                            if not p.get("reuse_long"):
                                raise ValueError(
                                    "Expected retained long KV at the decoder boundary"
                                )
                            module, name = (
                                ("kv_int8", "gather_attention")
                                if "kv_int8" in desc["op"]
                                else ("attention_fused", "selected_int8")
                            )
                            factory = getattr(
                                importlib.import_module("tools.operators.ascend." + module), name
                            )
                            params = {
                                k: v for k, v in p.items() if not k.startswith("_") and k != "rows"
                            }
                            params["reuse_long"] = False
                            owner_capacity = p["_owner_token_capacity"]
                            function = factory(**dict(params, rows=row_count(owner_capacity, True)))
                            owned = {
                                v.removesuffix("_owner"): i
                                for i, v in enumerate(p["_owner_aliases"])
                            }
                            _, _, grid = indirect(
                                function, owned, contract["owners"], owner_capacity
                            )
                            work += ".fresh_long"
                            define(work, 8192 * 2 * 4)
                            contract["work"][work] = grid
                            if old_work in contract.get("partitioned_tables", {}):
                                contract["partitioned_tables"][work] = contract[
                                    "partitioned_tables"
                                ][old_work]
                            if old_work in contract.get("work_ranges", {}):
                                contract["work_ranges"][work] = deepcopy(
                                    contract["work_ranges"][old_work]
                                )
                            decoder_tables.add(work)
                            step["buffers"][0] = work
                            step["owner_work"] = grid
                            metadata = {k: v for k, v in p.items() if k.startswith("_owner_")}
                            metadata["_owner_source_sha256"] = file_hash(
                                Path(inspect.getfile(factory))
                            )
                            step["uncompacted_manifest"] = step["manifest"]
                            step["manifest"] = register(
                                dict(
                                    module="tools.operators.ascend." + module,
                                    name=name,
                                    arguments=dict(params, rows=owner_capacity),
                                    capacity=owner_capacity,
                                    dynamic=dict(rows=1),
                                    columns=owned,
                                    owners=contract["owners"],
                                    table_columns=len(owned),
                                    metadata=metadata,
                                    specs=desc["operands"],
                                )
                            )
                            first_attention = False
                    else:
                        step["buffers"] = [clone_buffer(n, replacements) for n in step["buffers"]]
                elif step["op"] in ["reduce", "gather", "exchange"]:
                    step["source"] = clone_buffer(step["source"], replacements)
                    step["destination"] = clone_buffer(step["destination"], replacements)
                elif step["op"] == "snapshot":
                    step["buffer"] = clone_buffer(step["buffer"], replacements)
                elif step["op"] not in ["fork", "switch", "join", "w4"]:
                    raise ValueError("Unsupported decoder operation: " + step["op"])
            if first_attention:
                raise ValueError("Decoder boundary has no selected attention")
            # SWA coordinates change on every final-window entry, including a
            # full 128-row final packet that needs no hidden-row compaction.
            selected_steps = len(select) - int(sp and final_prompt_only)
            marker = dict(op="decoder_rows", keep=keep, prepare=len(prepare), select=selected_steps)
            packet["body"] = body + prepare + [marker] + select + tail
            packet["head"].insert(0, dict(op="decoder_rows", keep=keep, prepare=0, select=0))
            counts.append(
                dict(
                    rank=rank["rank"],
                    family=packet["min_rows"],
                    prepare=len(prepare),
                    select=len(select),
                )
            )
    contract["decoder_replay"] = dict(
        keep=keep,
        final_prompt_only=final_prompt_only,
        tables=sorted(decoder_tables),
        maps=list(maps.values()),
    )
    if final_prompt_only:
        # Consumer SWA is always reconstructed from the final prompt window;
        # prefix snapshots contain only producer state and shared KV sources.
        decoder_state = {
            f"cache.swa.{layer}{suffix}" for layer in range(21, 40) for suffix in ("", ".scale")
        }
        contract["state"] = [s for s in contract["state"] if s["buffer"] not in decoder_state]
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
    live_tables = set(contract["pointers"]) | set(contract["work"])
    contract["decoder_replay"]["tables"] = sorted(
        set(contract["decoder_replay"]["tables"]) & live_tables
    )
    validate_rows(model, base)
    assets = publish_package(model, base, output / "package")
    report = dict(
        model=str(output / "package/model.json"),
        keep=keep,
        final_prompt_only=final_prompt_only,
        kernels=len(tasks),
        assets=assets,
        families=counts,
        additional_owner_bytes=0,
    )
    atomic_json(output / "decoder-replay.json", report)
    return report
