"""Gather routed INT8 activations, scales and route metadata in one message."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
import multiprocessing
from pathlib import Path

from orinfer_build.common.files import atomic_json
from orinfer_build.models.deepseek_v41.assembly import (
    compact_buffers,
    publish_package,
    validate_rows,
)
from orinfer_build.models.deepseek_v41.batch import compile_task
from orinfer_build.contracts.row_contract import expression, resolve


def zero_view(name, definitions):
    seen = set()
    while "view" in definitions[name]:
        if name in seen:
            raise ValueError("Cyclic transport view")
        seen.add(name)
        view = definitions[name]["view"]
        if view.get("offset", 0) or view.get("offset_expression"):
            raise ValueError("Routed transport requires a whole-buffer alias")
        name = view["parent"]
    return name


def routed_chain(steps, index, describe, definitions, capacity, ranks):
    chain = steps[index : index + 5]
    if [s["op"] for s in chain] != ["gather", "gather", "kernel", "gather", "gather"]:
        return None
    first, weights, quant, activation, scale = chain
    if any(s.get("trace_only") for s in (first, weights, activation, scale)):
        return None
    spec = describe(quant)
    if spec["op"] != "vector.quantize" or len(quant["buffers"]) != 3:
        return None
    if (
        spec["parameters"].get("fp16_query_scale", False)
        or spec["parameters"].get("tile_rows", 1) != 1
    ):
        return None
    width = spec["parameters"]["width"]
    if width % 32 or first.get("fp32") or not first.get("int32") or not weights.get("fp32"):
        return None
    count = (capacity + ranks - 1) // ranks
    routes = first["elements"] // count
    if routes < 8 or routes % 8 or not activation.get("int32") or not scale.get("fp32"):
        return None
    x, s = quant["buffers"][1:]
    if zero_view(activation["source"], definitions) != x or scale["source"] != s:
        return None
    if {first["destination"], weights["destination"]}.intersection(quant["buffers"]):
        raise ValueError("Local quantization depends on gathered route metadata")
    quant_roots = [zero_view(n, definitions) for n in quant["buffers"]]
    metadata_roots = {zero_view(s["source"], definitions) for s in (first, weights)}
    gathered_roots = {zero_view(s["destination"], definitions) for s in (first, weights)}
    if metadata_roots.intersection(quant_roots[1:]) or gathered_roots.intersection(quant_roots):
        raise ValueError("Local quantization aliases route metadata")
    for gather, columns in (
        (first, routes),
        (weights, routes),
        (activation, width // 4),
        (scale, 1),
    ):
        if gather["elements"] != count * columns or "elements_expression" not in gather:
            raise ValueError("Routed collective extent differs")
        if any(
            resolve(gather["elements_expression"], m) != ((m + ranks - 1) // ranks) * columns
            for m in range(1, capacity + 1)
        ):
            raise ValueError("Routed collective dynamic extent differs")
    if "row_expression" not in quant or any(
        resolve(quant["row_expression"], m) != (m + ranks - 1) // ranks
        for m in range(1, capacity + 1)
    ):
        raise ValueError("Routed quantization must retain the local row axis")
    return dict(
        normalized=quant["buffers"][0],
        width=width,
        routes=routes,
        ranks=ranks,
        source=[x, s, first["source"], weights["source"]],
        destination=[
            zero_view(activation["destination"], definitions),
            scale["destination"],
            first["destination"],
            weights["destination"],
        ],
    )


def alias_root(name, definitions):
    seen = set()
    while "view" in definitions[name]:
        if name in seen:
            raise ValueError("Cyclic transport alias")
        seen.add(name)
        name = definitions[name]["view"]["parent"]
    return name


def accesses(step, describe, batch):
    op = step["op"]
    if op == "kernel":
        variants = [step]
        if step.get("uncompacted_manifest"):
            variants.append(dict(step, manifest=step["uncompacted_manifest"]))
        ports = []
        for variant in variants:
            spec = describe(variant)
            ports.extend(
                (name, p["access"] in ("read", "read_write"), p["access"] != "read")
                for name, p in zip(step["buffers"], spec["operands"], strict=True)
            )
            for access in spec["parameters"].get("_owner_access", []):
                column = batch["pointers"][step["buffers"][1]][access["column"]]
                # The hazard ABI records mutability; writable state can also
                # consume its previous value. Analyze both launch variants.
                ports.append((batch["columns"][column]["buffer"], True, access["write"]))
        return ports
    if op in ("gather", "reduce", "exchange"):
        return [(step["source"], True, False), (step["destination"], False, True)]
    if op in ("w4", "w4_a4"):
        return [(name, index != 5, index == 5) for index, name in enumerate(step["buffers"])]
    if op == "snapshot":
        return [(step["buffer"], True, False)]
    return []


def activation_readers(steps, start, activation, local_outputs, definitions, describe, batch):
    """Find every old A8 reader before its next overwrite, including aliases."""
    active = {activation, *local_outputs}
    readers = []
    for index in range(start, len(steps)):
        step = steps[index]
        ports = accesses(step, describe, batch)
        for name, read, _ in ports:
            root = alias_root(name, definitions)
            if not read or root not in active:
                continue
            if root != activation:
                raise ValueError("A local quantized output remains live after routed gather")
            if (
                step["op"] != "kernel"
                or name != step["buffers"][0]
                or describe(step)["op"] not in ("dense.projection", "memory.gather_quantized_rows")
            ):
                raise ValueError("Unsupported reader of gathered routed activation")
            if zero_view(name, definitions) != activation:
                raise ValueError("A8 reader must use the complete logical table")
            if index not in readers:
                readers.append(index)
        for name, _, write in ports:
            root = alias_root(name, definitions)
            if write and root in active:
                if (
                    zero_view(name, definitions) != root
                    or definitions[name]["bytes"] != definitions[root]["bytes"]
                ):
                    raise ValueError(
                        "Partial overwrite of a routed activation requires an explicit layout"
                    )
                active.remove(root)
        if not active:
            break
    if not readers:
        raise ValueError("Routed activation has no supported readers")
    return readers


def strided_task(spec, stride):
    args = {k: v for k, v in spec["parameters"].items() if not k.startswith("_")}
    capacity = spec["parameters"]["_row_capacity"]
    module, name = spec["op"].split(".")
    if spec["op"] == "dense.projection" and args["integer"]:
        args["activation_stride"] = stride
        dynamic = {"rows": 1}
    elif spec["op"] == "memory.gather_quantized_rows":
        args["table_stride"] = stride
        if args["rows"] % capacity or args["table_rows"] != capacity:
            raise ValueError("Expert gather must preserve its original row multipliers")
        dynamic = dict(rows=args["rows"] // capacity, table_rows=1)
    else:
        raise ValueError("Unsupported strided A8 consumer")
    specs = deepcopy(spec["operands"])
    if specs[0]["dtype"] != "i8" or specs[0]["shape"][0] != capacity:
        raise ValueError("Strided A8 input must use the original global row capacity")
    specs[0]["shape"][-1] = stride
    return dict(
        module="kernels.ascend." + module,
        name=name,
        arguments=args,
        specs=specs,
        capacity=capacity,
        dynamic=dynamic,
        grid_limit=spec["parameters"].get("_grid_limit", 64),
    )


def build(plan, output, workers=2, model_dir=None):
    if output.exists():
        raise ValueError("Require a new output directory")
    model = json.loads(plan.read_text())
    ranks = len(model["ranks"])
    base = output / "source"
    publish_package(model, plan.parent, base)
    tasks, descriptors, counts, strided = {}, {}, [], {}
    original_selections = deepcopy(model.get("row_selections", {}))
    selections = model.setdefault("row_selections", {})

    def describe(step):
        path = step["manifest"]
        if path not in descriptors:
            descriptors[path] = json.loads((base / path).read_text())
        return descriptors[path]

    def register(source):
        identity = json.dumps(source, sort_keys=True)
        if identity not in tasks:
            tasks[identity] = dict(source, key=f"@routed_transport{len(tasks)}")
        return tasks[identity]["key"]

    def transport(name, args, specs):
        return register(
            dict(
                module="kernels.ascend.transport",
                name=name,
                arguments=args,
                specs=specs,
                capacity=args["rows"],
                dynamic={"rows": 1},
            )
        )

    def stride_manifest(old, stride):
        key = (old, stride)
        if key not in strided:
            spec = describe(dict(manifest=old))
            replacement = register(strided_task(spec, stride))
            strided[key] = replacement
            variants = []
            for variant in original_selections.get(old, []):
                variant = deepcopy(variant)
                variant["kernels"] = [stride_manifest(k, stride) for k in variant["kernels"]]
                variants.append(variant)
            if variants:
                if replacement in selections and selections[replacement] != variants:
                    raise ValueError("Equivalent strided slots have different row selections")
                selections[replacement] = variants
        return strided[key]

    for rank in model["ranks"]:
        definitions = {b["name"]: b for b in rank["buffers"]}
        for packet in rank["packets"]:
            if packet.get("batch_decode") or packet.get("role", "target") != "target":
                continue
            capacity = packet["rows"]
            old, chains, changes = packet["body"], {}, {}
            for index in range(len(old)):
                chain = routed_chain(old, index, describe, definitions, capacity, ranks)
                if chain is None:
                    continue
                width, routes = chain["width"], chain["routes"]
                local = (capacity + ranks - 1) // ranks
                pitch = ((width // 4 + 2 * routes + 8) // 8) * 8
                prefix = f"collective.routed.m{capacity}.w{width}.r{routes}"
                send, receive, activation = (
                    prefix + ".send",
                    prefix + ".receive",
                    prefix + ".activation",
                )
                for definition in (
                    dict(name=send, bytes=local * pitch * 4),
                    dict(name=receive, bytes=ranks * local * pitch * 4),
                    dict(
                        name=activation,
                        bytes=capacity * pitch * 4,
                        view=dict(parent=receive, offset=0),
                    ),
                ):
                    name = definition["name"]
                    if name not in definitions:
                        definitions[name] = definition
                        rank["buffers"].append(definition)
                    elif definitions[name] != definition:
                        raise ValueError("Routed transport workspace differs")
                root = chain["destination"][0]
                indices = activation_readers(
                    old + packet["head"],
                    index + 5,
                    root,
                    chain["source"][:2],
                    definitions,
                    describe,
                    model["batch"],
                )
                for reader in indices:
                    if reader >= len(old):
                        raise ValueError(
                            "Routed FFN activation unexpectedly reaches the output head"
                        )
                    step = deepcopy(old[reader])
                    step["buffers"][0] = activation
                    for field in ("manifest", "uncompacted_manifest"):
                        if step.get(field):
                            step[field] = stride_manifest(step[field], pitch * 4)
                    if reader in changes and changes[reader] != step:
                        raise ValueError("Routed activation consumer scopes overlap")
                    changes[reader] = step
                args = dict(rows=capacity, width=width, routes=routes, ranks=ranks)
                specs = [
                    dict(name=n, dtype=d, shape=shape, access="read")
                    for n, d, shape in (
                        ("x", "bf16", [local, width]),
                        ("indices", "i32", [local, routes]),
                        ("weights", "f32", [local, routes]),
                    )
                ]
                specs.append(dict(name="out", dtype="i32", shape=[local, pitch], access="write"))
                pack = transport("quantize_routed", args, specs)
                metadata = [
                    dict(name=n, dtype=d, shape=shape, access="write")
                    for n, d, shape in (
                        ("scales", "f32", [capacity]),
                        ("indices", "i32", [capacity, routes]),
                        ("weights", "f32", [capacity, routes]),
                    )
                ]
                unpack = transport(
                    "unpack_routed_metadata",
                    args,
                    [dict(name="source", dtype="i32", shape=[ranks * local, pitch], access="read")]
                    + metadata,
                )
                chains[index] = [
                    dict(
                        op="kernel",
                        manifest=pack,
                        buffers=[chain["normalized"], *chain["source"][2:], send],
                    ),
                    dict(
                        op="gather",
                        source=send,
                        destination=receive,
                        elements=local * pitch,
                        int32=True,
                        fp32=False,
                        elements_expression=expression(f"((m + {ranks - 1}) // {ranks}) * {pitch}"),
                    ),
                    dict(
                        op="kernel", manifest=unpack, buffers=[receive, *chain["destination"][1:]]
                    ),
                ]
            body, index = [], 0
            while index < len(old):
                if index in chains:
                    body.extend(chains[index])
                    index += 5
                else:
                    body.append(changes.get(index, old[index]))
                    index += 1
            packet["body"] = body
            if chains:
                counts.append(
                    dict(rank=rank["rank"], minimum=packet.get("min_rows", 1), groups=len(chains))
                )
    if not tasks:
        raise ValueError("No routed sequence collectives found")
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        replacements = dict(pool.map(compile_task, [(t, str(base)) for t in tasks.values()]))
    for rank in model["ranks"]:
        for packet in rank["packets"]:
            for step in packet["body"]:
                for field in ("manifest", "uncompacted_manifest"):
                    if step.get(field) in replacements:
                        step[field] = replacements[step[field]]
    referenced = {
        s.get(k)
        for r in model["ranks"]
        for p in r["packets"]
        for section in ("body", "head")
        for s in p.get(section, [])
        for k in ("manifest", "uncompacted_manifest")
    }
    model["row_selections"] = {
        replacements.get(slot, slot): [
            dict(v, kernels=[replacements.get(k, k) for k in v["kernels"]]) for v in variants
        ]
        for slot, variants in selections.items()
        if slot in referenced or replacements.get(slot, slot) in referenced
    }
    # A new physical stride still belongs to the same operator factory. Keep
    # all of that factory's published geometries on one compiler/source identity.
    from orinfer_build.models.deepseek_v41.publication import manifests

    identities = {}
    for name in manifests(model):
        spec = json.loads((base / name).read_text())
        identities.setdefault(spec["op"], set()).add(spec["source_sha256"])
    mixed = {op for op, values in identities.items() if len(values) > 1}
    if mixed:
        if mixed - {"dense.projection", "memory.gather_quantized_rows"}:
            raise ValueError("Input contains unrelated mixed operator sources")
        from orinfer_build.models.deepseek_v41.build import rebuild_kernels

        convergence = base / "model-strided.json"
        atomic_json(convergence, model)
        rebuild_kernels(convergence, workers, operators=mixed)
        model = json.loads(convergence.with_name("model-strided-selected.json").read_text())
    compact_buffers(model)
    validate_rows(model, base, model_dir=model_dir)
    assets = publish_package(model, base, output / "package")
    report = dict(
        model=str(output / "package/model.json"),
        kernels=len(tasks),
        assets=assets,
        families=counts,
        activation_unpack=False,
        quantization_fused=True,
    )
    atomic_json(output / "routed-collectives.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--compile-workers", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(build(args.plan, args.output, args.compile_workers, args.model_dir)))


if __name__ == "__main__":
    main()
