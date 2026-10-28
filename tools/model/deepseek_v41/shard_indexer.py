"""Partition prefill index scores/TopK by row and gather exact selected indices.

The original packet remains the fallback for short fragments. New kernels reuse
the existing indexer and TopK arithmetic with fixed local row capacity; weight
banks, request state and the global attention consumers retain their layouts.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import importlib
import json
import math
import multiprocessing
from pathlib import Path

from tools.build.files import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-rows", type=int, default=512)
    parser.add_argument("--compile-workers", type=int, default=4)
    args = parser.parse_args()
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
                body.append(
                    dict(
                        op="kernel",
                        manifest=path,
                        buffers=[mapping.get(b, b) for b in old["buffers"]],
                        fixed_rows=count,
                    )
                )

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
            body.append(
                dict(
                    op="gather",
                    source=mapping[selected],
                    destination=gathered,
                    elements=count * 512,
                    fp32=False,
                    int32=True,
                    fixed_rows=True,
                )
            )
            specs = [
                operand("source", "i32", [capacity, 512]),
                operand("out", "i32", [capacity, 512], "write"),
            ]
            params = dict(rows=capacity, count=count, width=512, ranks=8, dtype="int32")
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
