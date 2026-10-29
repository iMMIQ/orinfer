"""Build a native DSv4.1 INT8/group64/FP16-scale attention-cache plan.

Prefill decodes active ranges into two shared BF16 workspaces, reusing each
compressed source across its consumers. Decode fuses selected-row loads and
conversion. --reference-bf16 stores identical quantized values in BF16 only as
an offline integration reference; the original model plan remains immutable.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path

from tools.build.files import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile-workers", type=int, default=4)
    parser.add_argument("--reference-bf16", action="store_true")
    args = parser.parse_args()
    if args.output.parent.resolve() != args.plan.parent.resolve():
        parser.error("Input and output must share the asset directory")
    from tools.model.deepseek_v41.build import compile_task
    from tools.operators.ascend.build import operand

    model = json.loads(args.plan.read_text())
    base, tasks, descriptors = args.plan.parent, {}, {}

    def descriptor(step):
        path = step["manifest"]
        if path not in descriptors:
            descriptors[path] = json.loads((base / path).read_text())
        return descriptors[path]

    def kernel(name, parameters, specs, capacity):
        key = json.dumps([name, parameters, specs, capacity], sort_keys=True)
        if key not in tasks:
            tasks[key] = dict(
                module="tools.operators.ascend.kv_int8",
                name=name,
                arguments=parameters,
                specs=specs,
                row_capacity=capacity,
                dynamic_arguments={"rows": 1} if capacity is not None else {},
                placeholder=f"@kv_int8{len(tasks)}",
            )
        return tasks[key]["placeholder"]

    def cache(name):
        return name.startswith(("cache.swa.", "cache.long."))

    for rank in model["ranks"]:
        definitions = {d["name"]: d for d in rank["buffers"]}
        tables = {}
        for packet in rank["packets"]:
            for step in packet["body"]:
                if step["op"] != "kernel":
                    continue
                manifest = descriptor(step)
                if manifest["op"] == "memory.scatter_rows" and cache(step["buffers"][2]):
                    parameters = manifest["parameters"]
                    if parameters["width"] != 512 or parameters["dtype"] != "bfloat16":
                        raise ValueError("Expected the original BF16 attention cache")
                    if parameters["index_dtype"] != "int32":
                        raise ValueError("Expected native INT32 cache positions")
                    tables[step["buffers"][2]] = parameters["table_rows"]
        if len(tables) != 44:
            raise ValueError("Expected 40 SWA and four shared compressed cache banks")

        def temp(name, size, view=None):
            definition = dict(name=name, bytes=size)
            if view is not None:
                definition["view"] = view
            if name not in definitions:
                definitions[name] = definition
                rank["buffers"].append(definition)
            elif definitions[name] != definition:
                raise ValueError("Conflicting INT8 cache buffer: " + name)
            return name

        original_bytes = sum(definitions[name]["bytes"] for name in tables)
        for name, capacity in tables.items():
            if definitions[name] != dict(name=name, bytes=capacity * 512 * 2):
                raise ValueError("Cache allocation differs from its physical BF16 shape")
            if not args.reference_bf16:
                definitions[name]["bytes"] = capacity * 512
            temp(name + ".scale", capacity * 8 * 2)
        if not args.reference_bf16:
            swa_work = temp("kv.int8.work.swa", model["capacity"] * 512 * 2)
            long_work = temp("kv.int8.work.long", model["capacity"] * 512 * 2)
        writes, prefills, gathers = 0, 0, 0
        for packet in rank["packets"]:
            body, loaded_long = [], None
            for old in packet["body"]:
                if old["op"] != "kernel":
                    body.append(old)
                    continue
                manifest = descriptor(old)
                op, buffers = manifest["op"], old["buffers"]
                parameters = manifest["parameters"]
                row_capacity = parameters.get("_row_capacity")
                rows = parameters.get("rows")
                if op == "memory.scatter_rows" and cache(buffers[2]):
                    name = buffers[2]
                    capacity = tables[name]
                    dtype = "bf16" if args.reference_bf16 else "i8"
                    path = kernel(
                        "scatter",
                        dict(rows=rows, table_rows=capacity, reference_bf16=args.reference_bf16),
                        [
                            operand("source", "bf16", [rows, 512]),
                            operand("indices", "i32", [rows]),
                            operand("cache", dtype, [capacity, 512], "read_write"),
                            operand("scales", "f16", [capacity, 8], "read_write"),
                            operand("errors", "i32", [rows], "write"),
                        ],
                        row_capacity,
                    )
                    body.append(
                        dict(
                            old, manifest=path, buffers=buffers[:3] + [name + ".scale", buffers[3]]
                        )
                    )
                    if name.startswith("cache.long."):
                        loaded_long = None
                    writes += 1
                    continue
                if args.reference_bf16:
                    body.append(old)
                    continue
                if op == "model_ops.gather_attention":
                    capacity, long_capacity = tables[buffers[0]], tables[buffers[1]]
                    selected = 640 if parameters["compressed"] else 128
                    params = {k: v for k, v in parameters.items() if not k.startswith("_")}
                    path = kernel(
                        "gather_attention",
                        params,
                        [
                            operand("swa", "i8", [capacity, 512]),
                            operand("swa_scales", "f16", [capacity, 8]),
                            operand("long_kv", "i8", [long_capacity, 512]),
                            operand("long_scales", "f16", [long_capacity, 8]),
                            operand("swa_indices", "i32", [rows, 128]),
                            operand("long_indices", "i32", [rows, 512]),
                            operand("out", "bf16", [rows, selected, 512], "write"),
                            operand("mask", "i32", [rows, selected], "write"),
                        ],
                        row_capacity,
                    )
                    body.append(
                        dict(
                            old,
                            manifest=path,
                            buffers=[
                                buffers[0],
                                buffers[0] + ".scale",
                                buffers[1],
                                buffers[1] + ".scale",
                            ]
                            + buffers[2:],
                        )
                    )
                    gathers += 1
                    continue
                if op in ("attention_prefill.qk", "attention_prefill.pv"):
                    if not row_capacity or old.get("fixed_rows") is not None:
                        raise ValueError("Expected a dynamic whole-packet attention consumer")
                    short, long = buffers[1:3]
                    long_capacity = tables[long]
                    long_view = temp(
                        f"kv.int8.work.long.{long_capacity}",
                        long_capacity * 512 * 2,
                        dict(parent=long_work, offset=0),
                    )
                    if op == "attention_prefill.qk":
                        for name, output, ratio, windowed in (
                            (short, swa_work, 1, parameters.get("windowed", True)),
                            (long, long_view, parameters["ratio"], False),
                        ):
                            if name == long and (
                                not parameters["compressed"] or loaded_long == long
                            ):
                                continue
                            capacity = tables[name]
                            path = kernel(
                                "decode_prefix",
                                dict(rows=rows, capacity=capacity, ratio=ratio, windowed=windowed),
                                [
                                    operand("cache", "i8", [capacity, 512]),
                                    operand("scales", "f16", [capacity, 8]),
                                    operand("positions", "i32", [rows]),
                                    operand("out", "bf16", [capacity, 512], "read_write"),
                                ],
                                row_capacity,
                            )
                            body.append(
                                dict(
                                    op="kernel",
                                    manifest=path,
                                    buffers=[name, name + ".scale", buffers[3], output],
                                )
                            )
                            if name == long:
                                loaded_long = long
                            prefills += 1
                    body.append(dict(old, buffers=[buffers[0], swa_work, long_view] + buffers[3:]))
                    continue
                if any(cache(name) for name in buffers):
                    raise ValueError("Unconverted attention-cache consumer: " + op)
                body.append(old)
            packet["body"] = body
        print(
            dict(
                rank=rank["rank"],
                writes=writes,
                prefix_decodes=prefills,
                selected_gathers=gathers,
                original_cache_bytes=original_bytes,
                cache_bytes=sum(
                    definitions[name]["bytes"] + definitions[name + ".scale"]["bytes"]
                    for name in tables
                ),
                tasks=len(tasks),
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
