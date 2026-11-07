"""Attach the shared native vision program and VL routing to an existing A2 plan."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
import multiprocessing
from pathlib import Path
import re

from orinfer_build.common.files import atomic_json, file_hash
from orinfer_build.contracts.row_contract import expression


def coordinate_steps(packet):
    """Follow the same source/decoder boundary as the native packet executor."""
    coordinates = packet["coordinates"]
    for step in packet["body"] + packet["head"]:
        if step["op"] == "decoder_rows":
            coordinates = packet["decoder_coordinates"]
        elif step["op"] == "kernel":
            yield step, coordinates


def routing_mask(definitions, coordinates, packet_rows, spec, step, rank, peers):
    """Bind a full row mask or its existing contiguous sequence partition."""
    capacity = spec["operands"][0]["shape"][0]
    name = coordinates["base_cos"].removesuffix("base_cos") + "image_rows"
    coordinates["image_rows"] = name
    definitions[name] = dict(name=name, bytes=packet_rows * 4)
    if step.get("row_expression") is None and capacity == packet_rows:
        return name
    count = (packet_rows + peers - 1) // peers
    expected = expression(f"(m + {peers - 1}) // {peers}")
    if capacity != count or step.get("row_expression") != expected:
        raise ValueError("Vision router has an unsupported row domain")
    local = name + ".partition"
    definitions[local] = dict(
        name=local,
        bytes=count * 4,
        view=dict(
            parent=name,
            offset=rank * count * 4,
            row_capacity=packet_rows,
            offset_expression=expression(f"((m + {peers - 1}) // {peers}) * {rank * 4}"),
        ),
    )
    return local


def rebind_masks(model, base):
    specs = {}
    for rank in model["ranks"]:
        definitions = {b["name"]: b for b in rank["buffers"]}
        for packet in rank["packets"]:
            if packet.get("role", "target") != "target":
                continue
            for step, coordinates in coordinate_steps(packet):
                path = step["manifest"]
                if path not in specs:
                    specs[path] = json.loads((base / path).read_text())
                spec = specs[path]
                if spec["op"] != "routing.route":
                    continue
                if not spec["parameters"].get("vision_bias"):
                    raise ValueError("Existing target router has no vision bias ABI")
                step["buffers"][-1] = routing_mask(
                    definitions,
                    coordinates,
                    packet["rows"],
                    spec,
                    step,
                    rank["rank"],
                    len(model["ranks"]),
                )
        rank["buffers"] = list(definitions.values())


def hardlinks(source, destination):
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if not target.samefile(path):
                raise ValueError("Conflicting immutable asset: " + str(target))
        else:
            target.hardlink_to(path)


def attach(plan, vision, output, workers):
    plan, vision, output = plan.resolve(), vision.resolve(), output.resolve()
    if output.exists():
        raise ValueError("Preserve existing vision plans: " + str(output))
    model = json.loads(plan.read_text())
    program = json.loads(vision.read_text())
    if model.get("vision"):
        if model["vision"]["spec"] != program["spec"]:
            raise ValueError("Replacing vision requires the same frontend and VL routing contract")
        # Reuse language arithmetic and bind masks through its declared row
        # domains. A local row expression also appears before decoder_rows.
        rebind_masks(model, plan.parent)
        prefix = "vision-" + file_hash(vision)[:16]
        output.mkdir(parents=True)
        hardlinks(plan.parent, output)
        hardlinks(vision.parent, output / prefix)
        for operation in program["body"]:
            operation["manifest"] = prefix + "/" + operation["manifest"]
        model["vision"] = program
        atomic_json(output / "model.json", model)
        atomic_json(output / "attach-result.json", dict(compiled_templates=0, replaced_vision=True))
        return output / "model.json"

    from orinfer_build.models.deepseek_v41.build import compile_task, kernel_rebuild_task

    output.mkdir(parents=True)
    # Baseline operators are immutable and reused without recompiling text kernels.
    hardlinks(plan.parent, output)
    hardlinks(vision.parent, output / "vision")
    for operation in program["body"]:
        operation["manifest"] = "vision/" + operation["manifest"]
    model["vision"] = program
    tasks = {}
    pending = []
    specs = {}
    for rank in model["ranks"]:
        definitions = {item["name"]: item for item in rank["buffers"]}
        for packet in rank["packets"]:
            for step, coordinates in coordinate_steps(packet):
                path = step["manifest"]
                if path not in specs:
                    specs[path] = json.loads((plan.parent / path).read_text())
                spec = specs[path]
                if (
                    spec["op"] == "floating.partials"
                    or spec["op"] == "routing.route"
                    and packet.get("role", "target") == "draft"
                ):
                    tasks.setdefault(path, kernel_rebuild_task(spec, path))
                    pending.append((step, "manifest"))
                    continue
                if spec["op"] != "routing.route":
                    continue
                match = re.fullmatch(r"w\.layers\.(\d+)\.ffn\.gate\.bias\..*", step["buffers"][1])
                if match is None:
                    raise ValueError("Unexpected target routing bias")
                capacity, experts = spec["operands"][0]["shape"]
                if experts != 384 or spec["parameters"].get("selected", 6) != 6:
                    raise ValueError("Unexpected target router geometry")
                mask = routing_mask(
                    definitions,
                    coordinates,
                    packet["rows"],
                    spec,
                    step,
                    rank["rank"],
                    len(model["ranks"]),
                )
                bias = f"vision.layers.{match[1]}.ffn.gate.bias_vl"
                definitions[bias] = dict(
                    name=bias,
                    bytes=experts * 4,
                    prepared_tensor=dict(
                        name=f"layers.{match[1]}.ffn.gate.bias_vl", dtype="F32", shape=[experts]
                    ),
                )
                ports = deepcopy(spec["operands"]) + [
                    dict(name="image_bias", dtype="f32", shape=[experts], access="read"),
                    dict(name="image_rows", dtype="i32", shape=[capacity], access="read"),
                ]
                if path not in tasks:
                    tasks[path] = dict(
                        module="kernels.ascend.routing",
                        name="route",
                        arguments=dict(
                            rows=capacity, experts=experts, selected=6, vision_bias=True
                        ),
                        specs=ports,
                        row_capacity=capacity,
                        dynamic_arguments=dict(rows=1),
                        grid_limit=spec["parameters"].get("_grid_limit", 64),
                        placeholder=path,
                    )
                step["buffers"].extend([bias, mask])
                pending.append((step, "manifest"))
        rank["buffers"] = list(definitions.values())
    for variants in model.get("row_selections", {}).values():
        for variant in variants:
            for i, path in enumerate(variant["kernels"]):
                spec = specs.setdefault(path, json.loads((plan.parent / path).read_text()))
                if spec["op"] == "floating.partials":
                    tasks.setdefault(path, kernel_rebuild_task(spec, path))
                    pending.append((variant["kernels"], i))
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        replacements = dict(
            pool.map(compile_task, [(task, str(output)) for task in tasks.values()])
        )
    for binding, key in pending:
        binding[key] = replacements[binding[key]]
    atomic_json(output / "model.json", model)
    atomic_json(
        output / "attach-result.json",
        dict(
            compiled_templates=len(tasks),
            vision_operators=len(set(op["manifest"] for op in program["body"])),
            layers=model["layers"],
        ),
    )
    return output / "model.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--vision", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile-workers", type=int, default=3)
    args = parser.parse_args()
    print(attach(args.plan, args.vision, args.output, args.compile_workers))


if __name__ == "__main__":
    main()
