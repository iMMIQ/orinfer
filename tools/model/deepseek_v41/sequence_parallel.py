"""Reuse native row kernels for contiguous TP8 prefill sequence partitions.

Large packets keep residuals, mHC, normalization and routing private to each
rank. Existing tensor-parallel attention and expert projections consume gathered
inputs. The default retains AllReduce arithmetic; --reduce-scatter enables the
native collective candidate, whose floating reduction order needs validation.
"""

import argparse
import importlib
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
import math
import multiprocessing
from pathlib import Path

from tools.build.files import atomic_json
from tools.model.row_contract import expression

SIZES = {"i8": 1, "i32": 4, "bf16": 2, "f16": 2, "f32": 4}
NAMES = {"bf16": "bfloat16", "f32": "float32"}


def external_reads(program, candidates, descriptor):
    """Find consumers of each local output before its next global overwrite."""
    live, required = set(candidates), set()
    for step in program:
        if not live:
            break
        if step["op"] == "kernel":
            ports = list(zip(step["buffers"], descriptor(step)["operands"], strict=True))
            required.update(
                name
                for name, port in ports
                if name in live and port["access"] in ("read", "read_write")
            )
            live.difference_update(name for name, port in ports if port["access"] != "read")
        elif step["op"] in ("gather", "reduce", "exchange"):
            if step["source"] in live:
                required.add(step["source"])
            live.discard(step["destination"])
        elif step["op"] == "snapshot" and step["buffer"] in live:
            required.add(step["buffer"])
        elif step["op"] == "w4":
            required.update(live.intersection(step["buffers"][:5]))
            live.discard(step["buffers"][5])
    return required


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile-workers", type=int, default=4)
    parser.add_argument("--min-rows", type=int, default=512)
    parser.add_argument("--reduce-scatter", action="store_true")
    parser.add_argument("--shared-attention", action="store_true")
    args = parser.parse_args()
    if args.output.parent.resolve() != args.plan.parent.resolve():
        parser.error("Input and output must share their asset directory")
    from tools.model.deepseek_v41.build import compile_task
    from tools.operators.ascend.build import operand
    from tools.operators.ascend.geometry import physical_shapes
    from kernels.model.rows import row_count

    base = args.plan.parent
    model = json.loads(args.plan.read_text())
    tasks, descriptors = {}, {}
    geometries = {}

    def descriptor(step):
        path = step["manifest"]
        if path not in descriptors:
            descriptors[path] = json.loads((base / path).read_text())
        return descriptors[path]

    def kernel(op, parameters, specs, capacity):
        key = json.dumps([op, parameters, specs, capacity], sort_keys=True)
        if key not in tasks:
            module, name = op.split(".")
            tasks[key] = dict(
                module="tools.operators.ascend." + module,
                name=name,
                arguments=parameters,
                specs=specs,
                row_capacity=capacity,
                dynamic_arguments={"rows": 1},
                placeholder=f"@sequence{len(tasks)}",
            )
            descriptors[tasks[key]["placeholder"]] = dict(
                op=op, parameters=parameters, operands=specs
            )
        return tasks[key]["placeholder"]

    for rank in model["ranks"]:
        definitions = {d["name"]: d for d in rank["buffers"]}
        packets = [
            p
            for p in rank["packets"]
            if p.get("min_rows", 0) > 0 and p.get("max_rows", p["rows"]) == p["rows"]
        ]
        if len(packets) != 1:
            raise ValueError("Expected one selected large-row packet")
        packet = packets[0]
        capacity = packet["rows"]
        if not packet["min_rows"] <= args.min_rows <= capacity:
            raise ValueError("Selected minimum must lie inside the original large packet")
        if args.min_rows > packet["min_rows"]:
            fallback = deepcopy(packet)
            fallback["max_rows"] = args.min_rows - 1
            rank["packets"].insert(rank["packets"].index(packet), fallback)
            packet["min_rows"] = args.min_rows
        if capacity % 8:
            raise ValueError("Sequence capacity must be divisible by eight")
        count = capacity // 8
        mapping, ready = {}, set()

        def temp(name, size, view=None):
            name = "sequence." + name
            definition = dict(name=name, bytes=size)
            if view is not None:
                definition["view"] = view
            if name not in definitions:
                definitions[name] = definition
                rank["buffers"].append(definition)
            elif definitions[name] != definition:
                raise ValueError("Conflicting sequence buffer: " + name)
            return name

        def view(source, width, dtype):
            layout = dict(parent=source, offset=rank["rank"] * count * width * SIZES[dtype])
            layout.update(
                row_capacity=capacity,
                offset_expression=expression(
                    f"((m + 7) // 8) * {rank['rank'] * width * SIZES[dtype]}"
                ),
            )
            return temp(
                "view." + source,
                count * width * SIZES[dtype],
                layout,
            )

        def zero_tail(source, width, dtype):
            path = kernel(
                "memory.zero_row_tail",
                dict(
                    rows=capacity,
                    capacity=capacity,
                    width=width,
                    dtype=NAMES[dtype],
                    partitions=8,
                ),
                [
                    operand("positions", "i32", [capacity]),
                    operand("out", dtype, [capacity, width], "read_write"),
                ],
                capacity,
            )
            return dict(
                op="kernel", manifest=path, buffers=[packet["coordinates"]["slots"], source]
            )

        def gather(source, dtype, width, trace=False):
            if source not in mapping:
                raise ValueError("Missing private gather source: " + source)
            transport = dict(
                op="gather",
                source=mapping[source],
                destination=source,
                elements=count * width,
                fp32=dtype == "f32",
                fixed_rows=True,
            )
            if dtype == "i32":
                transport["int32"] = True
            elif dtype == "i8":
                # The same A8 bytes travel as an INT32 view; no conversion.
                if width % 4:
                    raise ValueError("Packed A8 transport requires four-byte rows")
                transport["source"] = temp(
                    "transport." + mapping[source],
                    definitions[mapping[source]]["bytes"],
                    dict(parent=mapping[source], offset=0),
                )
                transport["destination"] = temp(
                    "transport." + source,
                    definitions[source]["bytes"],
                    dict(parent=source, offset=0),
                )
                transport["elements"] //= 4
                transport["int32"] = True
            transport.pop("fixed_rows")
            transport["elements_expression"] = expression(
                f"((m + 7) // 8) * {transport['elements'] // count}"
            )
            if trace:
                transport["trace_only"] = True
            else:
                ready.add(source)
            return transport

        def local(step, manifest):
            parameters = {k: v for k, v in manifest["parameters"].items() if not k.startswith("_")}
            if parameters["rows"] != capacity:
                raise ValueError("Local kernel has an unexpected row capacity")
            parameters["rows"] = count
            key = json.dumps([manifest["op"], parameters], sort_keys=True)
            if key not in geometries:
                module, name = manifest["op"].split(".")
                factory = getattr(importlib.import_module("tools.operators.ascend." + module), name)
                geometries[key] = physical_shapes(
                    factory(**dict(parameters, rows=row_count(count, True))), count
                )
            specs = deepcopy(manifest["operands"])
            for source, spec, shape in zip(step["buffers"], specs, geometries[key], strict=True):
                if any(key in definitions[source] for key in ("weight", "data", "prepared_weight")):
                    continue
                if spec["access"] == "read" and source not in mapping:
                    raise ValueError("Local input has no preceding producer: " + source)
                spec["shape"] = shape
                if spec["access"] != "read":
                    size = math.prod(spec["shape"]) * SIZES[spec["dtype"]]
                    mapping[source] = temp(source, size)
                    ready.discard(source)
            path = kernel(manifest["op"], parameters, specs, count)
            result = dict(
                op="kernel",
                manifest=path,
                buffers=[mapping.get(b, b) for b in step["buffers"]],
                fixed_rows=count,
            )
            result.pop("fixed_rows")
            result["row_expression"] = expression("(m + 7) // 8")
            return result

        norm_gathers = set()
        norm_source, norm_producer = None, None
        for index, step in enumerate(packet["body"]):
            if step["op"] != "kernel":
                continue
            manifest = descriptor(step)
            if manifest["op"] == "model_ops.norm" and step["buffers"][2].endswith(".norm.attn"):
                norm_source, norm_producer = step["buffers"][2], index
            elif norm_source is not None and manifest["op"] != "vector.quantize":
                if any(
                    source == norm_source and spec["access"] in ("read", "read_write")
                    for source, spec in zip(step["buffers"], manifest["operands"], strict=True)
                ):
                    norm_gathers.add(norm_producer)

        body, reductions, local_kernels = [], 0, 0
        premix = packet["coordinates"]["premix"]
        body.append(zero_tail(premix, 8, "f32"))
        mapping[premix] = view(premix, 8, "f32")
        for index, step in enumerate(packet["body"]):
            op = step["op"]
            if op == "embed":
                body.append(step)
                source = step["destination"]
                body.append(zero_tail(source, 5120, "bf16"))
                mapping[source] = view(source, 5120, "bf16")
                ready.add(source)
                continue
            if op == "reduce":
                reductions += 1
                if step["elements"] != capacity * 5120:
                    raise ValueError("Unexpected sequence reduction geometry")
                dtype = "f32" if step["fp32"] else "bf16"
                destination = step["destination"]
                if args.reduce_scatter:
                    body.append(zero_tail(step["source"], 5120, dtype))
                    mapping[destination] = temp(destination, count * 5120 * SIZES[dtype])
                    collective = dict(
                        step,
                        destination=mapping[destination],
                        elements=count * 5120,
                        scatter=True,
                        fixed_rows=True,
                    )
                    collective.pop("fixed_rows")
                    collective["elements_expression"] = expression("((m + 7) // 8) * 5120")
                    body.append(collective)
                    ready.discard(destination)
                else:
                    body.append(step)
                    mapping[destination] = view(destination, 5120, dtype)
                    ready.add(destination)
                continue
            if op == "snapshot":
                body.append(gather(step["buffer"], "bf16", 20480, trace=True))
                body.append(step)
                continue
            if op != "kernel":
                body.append(step)
                continue
            manifest = descriptor(step)
            kind, buffers = manifest["op"], step["buffers"]
            if kind == "engram.hadamard32":
                body.append(gather(buffers[0], "bf16", 20480))
            is_local = (
                kind.startswith("vector.mhc_")
                or kind == "hc_prefill.reduce"
                or (kind == "floating.partials" and manifest["parameters"].get("variance"))
                or (kind.startswith("floating.") and any("router." in b for b in buffers))
                or kind == "routing.route"
                or (kind == "model_ops.norm" and buffers[0].endswith(".collapsed"))
                or (kind == "vector.quantize" and buffers[0] in mapping)
                or (kind == "model_ops.convert" and buffers[0].endswith(".expert.reduced"))
            )
            if is_local:
                body.append(local(step, manifest))
                local_kernels += 1
                if kind == "model_ops.norm" and index in norm_gathers:
                    body.append(gather(buffers[2], "bf16", 5120))
                elif kind == "routing.route":
                    body.append(gather(buffers[2], "i32", 8))
                    body.append(gather(buffers[3], "f32", 8))
                elif kind == "vector.quantize":
                    body.append(gather(buffers[1], "i8", 5120))
                    body.append(gather(buffers[2], "f32", 1))
                continue
            for source, spec in zip(buffers, manifest["operands"], strict=True):
                if (
                    spec["access"] in ("read", "read_write")
                    and source in mapping
                    and source not in ready
                ):
                    raise ValueError("Global consumer reads unavailable private rows: " + source)
                if spec["access"] != "read" and source in mapping:
                    ready.add(source)
            body.append(step)
            if kind == "engram.gate":
                mapping[buffers[-1]] = view(buffers[-1], 20480, "bf16")
                ready.add(buffers[-1])
        if reductions != (80):
            raise ValueError("Expected all 80 body reductions")
        body.append(gather(packet["head"][0]["buffers"][0], "bf16", 5120))
        if args.shared_attention:
            shared_body, index, shared_groups = [], 0, 0
            # Each KV/Qa projection and both norm weights refer to the same
            # checkpoint ranges on all ranks; Qb remains tensor-parallel.
            root_definitions = {d["name"]: d for d in model["ranks"][0]["buffers"]}
            while index < len(body):
                step = body[index]
                if (
                    step["op"] != "kernel"
                    or descriptor(step)["op"] != "dense.projection"
                    or not step["buffers"][4].endswith(".kv")
                ):
                    shared_body.append(step)
                    index += 1
                    continue
                if index < 3 or len(shared_body) < 3:
                    raise ValueError("Shared attention is missing its quantization/gather prefix")
                group = body[index - 1 : index + 10]
                expected = [
                    "fork",
                    "dense.projection",
                    "model_ops.norm",
                    "model_ops.partial_rope",
                    "memory.scatter_rows",
                    "switch",
                    "dense.projection",
                    "model_ops.norm_quant",
                    "dense.projection",
                    "model_ops.partial_rope",
                    "join",
                ]
                actual = [
                    descriptor(op)["op"] if op["op"] == "kernel" else op["op"] for op in group
                ]
                if len(actual) > 4 and actual[4] == "kv_int8.scatter":
                    expected[4] = "kv_int8.scatter"
                if actual != expected:
                    raise ValueError("Unsupported KV/Qa branch: " + str(actual))
                kv, norm, rope, scatter = group[1:5]
                qa, norm_quant, qb, query_rope = group[6:10]
                outputs = {
                    name
                    for original in (kv, norm, rope, qa, norm_quant)
                    for name, port in zip(
                        original["buffers"], descriptor(original)["operands"], strict=True
                    )
                    if port["access"] in ("write", "read_write")
                }
                gathered = {rope["buffers"][-1], *norm_quant["buffers"][2:4]}
                missing = external_reads(
                    body[index + 10 :] + packet["head"], outputs - gathered, descriptor
                )
                if missing:
                    raise ValueError(
                        "Shared attention needs additional output gathers: " + str(missing)
                    )
                for original in [kv, norm, qa, norm_quant]:
                    for binding in original["buffers"]:
                        if any(
                            key in definitions[binding] for key in ("weight", "prepared_weight")
                        ):
                            if definitions[binding] != root_definitions.get(binding):
                                raise ValueError(
                                    "Shared projection uses rank-specific weights: " + binding
                                )
                if (
                    not qa["buffers"][3].startswith("w.")
                    or not qa["buffers"][4].endswith(".q.a")
                    or not qb["buffers"][0].endswith(".q.norm.quant")
                ):
                    raise ValueError("Unexpected shared query projection geometry")
                before = shared_body[-3:]
                if (
                    [op["op"] for op in before] != ["gather", "gather", "fork"]
                    or not before[0].get("int32")
                    or not (before[0].get("fixed_rows") or "elements_expression" in before[0])
                    or before[0]["elements"] != count * 1280
                    or not before[1]["destination"].endswith(".scale")
                ):
                    raise ValueError("Shared attention must replace its original wide A8 gathers")
                quant = definitions[before[0]["destination"]].get("view", {}).get("parent")
                private_quant = definitions[before[0]["source"]].get("view", {}).get("parent")
                if (
                    quant != kv["buffers"][0]
                    or before[1]["destination"] != kv["buffers"][2]
                    or private_quant != mapping[kv["buffers"][0]]
                    or before[1]["source"] != mapping[kv["buffers"][2]]
                ):
                    raise ValueError("Shared KV consumes a different quantization group")
                del shared_body[-3:]
                shared_body.append(dict(op="fork"))
                for original in [kv, norm, rope]:
                    manifest = descriptor(original)
                    if original is rope:
                        for binding, spec in zip(
                            original["buffers"][1:3], manifest["operands"][1:3], strict=True
                        ):
                            width = math.prod(spec["shape"][1:])
                            mapping[binding] = view(binding, width, spec["dtype"])
                    shared_body.append(local(original, manifest))
                shared_body.append(dict(op="switch"))
                for original in [qa, norm_quant]:
                    shared_body.append(local(original, descriptor(original)))
                shared_body.append(dict(op="join"))
                shared_body.append(gather(rope["buffers"][-1], "bf16", 512))
                shared_body.append(scatter)
                shared_body.append(gather(norm_quant["buffers"][2], "i8", 1280))
                shared_body.append(gather(norm_quant["buffers"][3], "f32", 1))
                shared_body.extend([qb, query_rope])
                index += 10
                shared_groups += 1
                local_kernels += 5
            if shared_groups != reductions // 2:
                raise ValueError("Shared KV/Qa branches differ from the selected layer count")
            body = shared_body
        packet["body"] = body
        print(
            dict(
                rank=rank["rank"],
                local_kernels=local_kernels,
                reductions=reductions,
                tasks=len(tasks),
                reduce_scatter=args.reduce_scatter,
            ),
            flush=True,
        )
    manifests = {}
    with ProcessPoolExecutor(
        max_workers=args.compile_workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        for placeholder, path in pool.map(compile_task, [(t, str(base)) for t in tasks.values()]):
            manifests[placeholder] = path
            print(dict(compiled=len(manifests), total=len(tasks)), flush=True)
    for rank in model["ranks"]:
        for packet in rank["packets"]:
            for step in packet["body"] + packet["head"]:
                if step["op"] == "kernel":
                    step["manifest"] = manifests.get(step["manifest"], step["manifest"])
    atomic_json(args.output, model)


if __name__ == "__main__":
    main()
