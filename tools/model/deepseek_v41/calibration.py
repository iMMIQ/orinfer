"""Prepare read-only native activation traces from an installed A2 execution plan."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
import tempfile

from tools.build.files import atomic_json, file_hash
from tools.build.shared_json import decode
from tools.model.deepseek_v41.vision_attach import hardlinks
from tools.model.publication import source_path


def instrument(model, read_spec):
    """Retain numerical instructions; serialize branches for host snapshots."""
    model = deepcopy(model)
    specs, points = {}, []
    for rank in model["ranks"]:
        definitions = {d["name"]: d for d in rank["buffers"]}
        for packet_index, packet in enumerate(rank["packets"]):
            if packet.get("role", "target") != "target":
                continue
            capacity = packet["rows"]
            layer = None
            steps = []

            def observe(label, buffer, *, fixed=False, projection=None):
                definition = definitions[buffer]
                # Fractional sequence partitions cannot use the legacy
                # bytes/capacity row slicing. Retain the bounded allocation;
                # consumers select valid rows from the paired frame contract.
                whole = fixed or definition["bytes"] % capacity != 0
                name = f"P{packet_index}.L{layer:02}.{label}"
                steps.append(
                    dict(op="snapshot", name=name, buffer=buffer, fixed_rows=whole, diagnostic=True)
                )
                point = dict(
                    rank=rank["rank"],
                    packet=packet_index,
                    capacity=capacity,
                    name=name,
                    buffer=buffer,
                    definition=definition,
                    fixed_rows=whole,
                )
                if projection is not None:
                    point["projection"] = projection
                points.append(point)

            for step in packet["body"]:
                if step.get("diagnostic"):
                    raise ValueError("Use the frozen source plan, without existing diagnostics")
                if step["op"] in ("fork", "switch", "join"):
                    continue
                steps.append(step)
                if step["op"] == "kernel":
                    path = step["manifest"]
                    if path not in specs:
                        specs[path] = read_spec(path)
                    spec, args = specs[path], step["buffers"]
                    for name in args:
                        match = re.search(r"(?:parts\.)?w\.layers\.(\d+)\.", name)
                        if match:
                            layer = int(match[1])
                            break
                    if layer is None or any("dspark" in name for name in args):
                        continue
                    kind = spec["op"]
                    if kind == "model_ops.norm":
                        for role in ("attn", "ffn"):
                            if f".{role}_norm." in args[1]:
                                observe(f"{role}.norm.bf16", args[2])
                                observe(f"{role}.norm.fp32", args[3])
                    elif kind == "dense.projection" and ".engram." not in args[1]:
                        label = re.sub(r"^w\.layers\.\d+\.", "", args[1]).split(".weight.")[0]
                        if label.startswith("w.shared."):
                            label = "shared." + label.split(".")[-1]
                        label = re.sub(r"[^a-zA-Z0-9_.-]", "_", label)
                        observe(label + ".input", args[0])
                        observe(label + ".token_scale", args[2])
                        observe(
                            label + ".output",
                            args[4],
                            projection=dict(
                                step=step,
                                manifest=path,
                                weight=definitions[args[1]],
                                channel_scale=definitions[args[3]],
                            ),
                        )
                    elif kind == "routing.route":
                        observe("routing.indices", args[2])
                        observe("routing.weights", args[3])
                    elif kind == "memory.gather_quantized_rows":
                        observe("expert.gather", args[2])
                        observe("expert.input.i8", args[3])
                        observe("expert.input.scale", args[4])
                elif step["op"] == "w4":
                    layer = int(re.search(r"\.L(\d+)\.", step["name"])[1])
                    label = "up" if step["output"] == 4608 else "down"
                    observe("expert." + label + ".input", step["buffers"][0])
                    observe("expert." + label + ".token_scale", step["buffers"][4])
                    observe(
                        "expert." + label + ".output",
                        step["buffers"][5],
                        projection=dict(
                            step=step,
                            weight=definitions[step["buffers"][1]],
                            channel_scale=definitions[step["buffers"][2]],
                        ),
                    )
                    if step.get("groups"):
                        observe("routing.counts", step["groups"], fixed=True)
            packet["body"] = steps
    return model, dict(points=points, manifests=specs)


def prepare(plan, output):
    plan, output = Path(plan).resolve(strict=True), Path(output).resolve()
    if output.exists() or output.is_relative_to(plan.parent):
        raise ValueError("Use a new directory outside the source execution package")
    model = decode(json.loads(plan.read_text()))
    if len(model.get("ranks", [])) != 8 or not model.get("batch"):
        raise ValueError("Expected a complete DSv4.1 TP8 batch plan")
    result, metadata = instrument(
        model, lambda name: json.loads(source_path(plan.parent, name).read_text())
    )
    metadata.update(
        source_plan_sha256=file_hash(plan),
        weights_changed=False,
        kernels_changed=False,
        scope="Offline activation tracing, not a performance execution plan",
        required_environment={
            "ORINFER_A2_BATCH_TRACE": "PATH",
            "ORINFER_A2_BATCH_TRACE_ALL_RANKS": "1",
            "HCCL_DETERMINISTIC": "strict",
        },
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".activation-calibration-", dir=output.parent) as temp:
        stage = Path(temp) / "package"
        stage.mkdir()
        hardlinks(plan.parent, stage)
        atomic_json(stage / "calibration.json", result, compact=True)
        atomic_json(stage / "calibration-points.json", metadata, compact=True)
        stage.rename(output)
    return output / "calibration.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.plan, args.output))


if __name__ == "__main__":
    main()
