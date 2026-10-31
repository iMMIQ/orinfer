"""Build bounded long-context A2 rows with the shared decode attention primitives.

SWA stores a ring large enough for one packet plus its causal window. The
indexer keeps exact full candidate coverage and a bounded Cube scratch tile;
TopK uses the existing recursive four-way merge. Both execution families read
the same INT8 cache and use the existing QK/normalized-softmax/PV factories.
"""

import argparse
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import shutil

from tools.build.files import atomic_json
from tools.model.deepseek_v41.assembly import (
    canonical,
    compact_buffers,
    publish_package,
    validate_rows,
)
from tools.model.deepseek_v41.service import state_layout


def prepare(model, base, capacity, storage, length_input=False, gather_slots=1):
    from tools.operators.ascend.geometry import physical_shapes

    old_capacity = model["capacity"]
    descriptors = {}
    metadata_cache = {}
    (base / "native-meta").mkdir(exist_ok=True)

    def spec(step):
        name = step["manifest"]
        if name not in descriptors:
            descriptors[name] = json.loads((base / name).read_text())
        return descriptors[name]

    def metadata(op, parameters, accesses):
        signature = json.dumps([op, parameters, accesses], sort_keys=True)
        if signature in metadata_cache:
            return metadata_cache[signature]
        module, name = op.split(".")
        factory = getattr(importlib.import_module("tools.operators.ascend." + module), name)
        function = factory(**parameters)
        tensors = [function.buffer_map[p] for p in function.params if p in function.buffer_map]
        shapes = physical_shapes(function, parameters["rows"])
        dtypes = dict(int8="i8", int32="i32", float16="f16", bfloat16="bf16", float32="f32")
        operands = [
            dict(name=t.name, dtype=dtypes[str(t.dtype)], shape=shape, access=access)
            for t, shape, access in zip(tensors, shapes, accesses, strict=True)
        ]
        descriptor = dict(op=op, parameters=parameters, operands=operands)
        key = hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()
        name = "native-meta/" + key + ".json"
        atomic_json(base / name, descriptor)
        descriptors[name] = descriptor
        metadata_cache[signature] = name
        return name

    for rank in model["ranks"]:
        definitions = {b["name"]: b for b in rank["buffers"]}
        decode = rank["packets"][0]
        prefill = rank["packets"][1]
        source_prefix = decode["coordinates"]["slots"].removesuffix("slots")
        target_prefix = prefill["coordinates"]["slots"].removesuffix("slots")
        templates = []
        for index, step in enumerate(decode["body"]):
            if step["op"] == "kernel" and spec(step)["op"] == "kv_int8.gather_attention":
                chain = deepcopy(decode["body"][index : index + 4])
                if [spec(s)["op"] for s in chain] != [
                    "kv_int8.gather_attention",
                    "attention_cube.qk",
                    "attention_cube.softmax",
                    "attention_cube.pv",
                ]:
                    raise ValueError("Expected the shared selected-attention chain")
                templates.append(chain)
        if len(templates) != 40:
            raise ValueError("Expected forty complete decode attention templates")

        def define(step):
            descriptor = spec(step)
            for name, port in zip(step["buffers"], descriptor["operands"], strict=True):
                if name not in definitions:
                    size = 1
                    for d in port["shape"]:
                        size *= d
                    size *= dict(i8=1, i32=4, f16=2, bf16=2, f32=4)[port["dtype"]]
                    definitions[name] = dict(name=name, bytes=size)
                    rank["buffers"].append(definitions[name])

        layer = 0
        body = []
        index = 0
        while index < len(prefill["body"]):
            step = prefill["body"][index]
            kind = spec(step)["op"] if step["op"] == "kernel" else None
            if kind in ("kv_int8.decode_prefix", "attention_prefill.selected_mask"):
                index += 1
                continue
            if kind == "attention_prefill.qk":
                for template in templates[layer]:
                    new = deepcopy(template)
                    descriptor = deepcopy(spec(template))
                    params = {
                        k: v for k, v in descriptor["parameters"].items() if not k.startswith("_")
                    }
                    params["rows"] = prefill["rows"]
                    new["buffers"] = [
                        name.replace(source_prefix, target_prefix, 1)
                        if name.startswith(source_prefix)
                        else name
                        for name in new["buffers"]
                    ]
                    new["manifest"] = metadata(
                        descriptor["op"], params, [p["access"] for p in descriptor["operands"]]
                    )
                    define(new)
                    body.append(new)
                while spec(prefill["body"][index])["op"] != "attention_prefill.pv":
                    index += 1
                    if prefill["body"][index]["op"] != "kernel":
                        raise ValueError("Packed attention chain has an unexpected boundary")
                layer += 1
                index += 1
                continue
            body.append(deepcopy(step))
            index += 1
        if layer != 40:
            raise ValueError("Expected forty prefill attention replacements")
        prefill["body"] = body

        for packet in rank["packets"]:
            causal_lengths = {}
            if length_input:
                coordinates = packet["coordinates"]
                for ratio in (1, 2):
                    name = coordinates[f"c{ratio}_indices"]
                    causal_lengths[name] = coordinates[f"c{ratio}_lengths"]
                    definitions[name]["bytes"] = packet["rows"] * 4
                for item in packet["body"]:
                    if item["op"] == "kernel" and spec(item)["op"] == "model_ops.causal_indices":
                        causal_lengths[item["buffers"][1]] = item["buffers"][2]
            body = []
            index = 0
            old = packet["body"]
            while index < len(old):
                step = deepcopy(old[index])
                if step["op"] != "kernel":
                    body.append(step)
                    index += 1
                    continue
                descriptor = deepcopy(spec(step))
                kind = descriptor["op"]
                params = {
                    k: v for k, v in descriptor["parameters"].items() if not k.startswith("_")
                }
                accesses = [p["access"] for p in descriptor["operands"]]
                if kind == "kv_int8.scatter" and step["buffers"][2].startswith("cache.swa."):
                    params.update(table_rows=storage, logical_capacity=old_capacity)
                    step["manifest"] = metadata(kind, params, accesses)
                    for name, width in [(step["buffers"][2], 512), (step["buffers"][3], 16)]:
                        definitions[name]["bytes"] = storage * width
                elif kind == "kv_int8.gather_attention":
                    params["swa_storage_rows"] = storage
                    if packet is prefill:
                        params["tile_slots"] = gather_slots
                    step["manifest"] = metadata(kind, params, accesses)
                elif kind == "model_ops.causal_indices" and length_input:
                    params["dense"] = False
                    step["manifest"] = metadata(kind, params, accesses)
                    definitions[step["buffers"][1]]["bytes"] = packet["rows"] * 4
                elif kind.startswith("indexer"):
                    # The causal index view is contiguous and contains only -1
                    # or the key's own slot; the Cube factory preserves its law.
                    # Keep the source's measured Cube tile, mask and rounding
                    # choices when changing capacity and scratch geometry.
                    # Quantized decode uses the same Cube factory with one
                    # query row and vector masking.
                    length = params["length"]
                    if kind != "indexer_cube.causal_logits":
                        params = dict(tile_s=128, vector_mask=True)
                    params.update(rows=packet["rows"], length=length, workspace_tiles=64)
                    if length_input:
                        step["buffers"][5] = causal_lengths[step["buffers"][5]]
                        params["length_input"] = True
                    workspace = (
                        packet["coordinates"]["slots"].removesuffix("slots") + "index.bounded"
                    )
                    if kind != "indexer_cube.causal_logits":
                        step["buffers"].append(workspace)
                        accesses.append("scratch")
                    else:
                        step["buffers"][-1] = workspace
                    step["manifest"] = metadata("indexer_cube.causal_logits", params, accesses)
                    define(step)
                elif kind == "topk.local":
                    length = params["length"]
                    if length not in (old_capacity, old_capacity // 2):
                        raise ValueError("Unexpected non-context TopK in the model body")
                    params["length"] = length * capacity // old_capacity
                    step["manifest"] = metadata(kind, params, accesses)
                    body.append(step)
                    source = step["buffers"][-1]
                    selected = params.get("selected", 512)
                    block = 4096 if selected <= 512 else 2048
                    sources = (params["length"] + block - 1) // block
                    stage = 0
                    index += 1
                    while spec(old[index])["op"] == "topk.merge":
                        index += 1
                    final = deepcopy(old[index])
                    if spec(final)["op"] != "topk.finalize":
                        raise ValueError("Context TopK must finalize its recursive merge")
                    while sources > 1:
                        output = source + ".merge" + str(stage)
                        merge = dict(
                            op="kernel",
                            manifest=metadata(
                                "topk.merge",
                                dict(rows=packet["rows"], sources=sources, selected=selected),
                                ["read", "write"],
                            ),
                            buffers=[source, output],
                        )
                        define(merge)
                        body.append(merge)
                        source = output
                        sources = (sources + 3) // 4
                        stage += 1
                    final["buffers"][0] = source
                    body.append(final)
                    index += 1
                    continue
                body.append(step)
                index += 1
            packet["body"] = body
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capacity", type=int, default=262144)
    parser.add_argument("--swa-storage-rows", type=int, default=4096)
    parser.add_argument("--owners", type=int, default=32)
    parser.add_argument("--compile-workers", type=int, default=4)
    parser.add_argument("--resume-preparation", action="store_true")
    parser.add_argument("--length-input", action="store_true")
    parser.add_argument("--gather-slots", type=int, choices=(1, 4, 8, 16), default=1)
    parser.add_argument("--reuse-cache", type=Path, action="append", default=[])
    args = parser.parse_args()
    if args.output.exists() and not args.resume_preparation:
        parser.error("Preserve existing build directories")
    source = canonical(json.loads(args.plan.read_text()))
    rows = source["ranks"][0]["packets"][1]["rows"]
    if (
        args.swa_storage_rows < rows + 127
        or args.swa_storage_rows % 256
        or not source["capacity"] <= args.capacity <= 262144
        or args.capacity % 256
    ):
        parser.error("Long context needs an aligned SWA ring covering one packet and its window")
    base = args.output / "source"
    if args.resume_preparation:
        source = canonical(json.loads((base / "model.json").read_text()))
    else:
        publish_package(source, args.plan.parent, base)
    cache_root = base / "cache"
    cache_root.mkdir(exist_ok=True)
    for cache in args.reuse_cache:
        if not cache.is_dir():
            parser.error(f"Compiled cache directory does not exist: {cache}")
        for entry in cache.iterdir():
            if (entry / "kernel.json").is_file() and (entry / "kernel.so").is_file():
                target = cache_root / entry.name
                if not target.exists():
                    shutil.copytree(
                        entry, target, copy_function=lambda a, b: Path(b).hardlink_to(a)
                    )
    source = prepare(
        source, base, args.capacity, args.swa_storage_rows, args.length_input, args.gather_slots
    )
    atomic_json(base / "model-selected-context-source.json", source)
    from tools.model.deepseek_v41 import batch
    from tools.model.deepseek_v41.build import rope_tables

    model = batch.build(
        base / "model-selected-context-source.json",
        base / "model-batch.json",
        args.owners,
        args.capacity,
        rows,
        args.compile_workers,
        sequence_parallel=True,
        shared_attention=True,
        reduce_scatter=True,
        fused_attention=False,
    )
    model["batch"]["state"] = state_layout(model, args.swa_storage_rows)
    rope_root = base / ("rope-" + str(args.capacity))
    (rope_root / "data").mkdir(parents=True, exist_ok=True)
    model["rope"] = rope_tables(rope_root, args.capacity)
    for asset in model["rope"].values():
        asset["file"] = str((rope_root / asset["file"]).relative_to(base))
    model = compact_buffers(model)
    validate_rows(model, base)
    atomic_json(base / "model-long.json", model)
    count = publish_package(model, base, args.output / "package")
    print(
        json.dumps(
            dict(
                package=str(args.output / "package/model.json"),
                assets=count,
                capacity=args.capacity,
                owners=args.owners,
            )
        )
    )


if __name__ == "__main__":
    main()
