"""Prepare read-only native activation traces from an installed A2 execution plan."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
import tempfile

from orinfer_build.common.files import atomic_json, file_hash
from orinfer_build.common.shared_json import decode
from orinfer_build.models.deepseek_v41.vision_attach import hardlinks
from orinfer_build.packaging.publication import source_path


def instrument(model, read_spec, *, layers=None, labels=None, expert_rows=None, roles=("target",)):
    """Retain numerical instructions; serialize branches for host snapshots."""
    model = deepcopy(model)
    layers = None if layers is None else set(layers)
    labels = None if labels is None else re.compile(labels)
    roles = set(roles)
    if not roles or not roles <= {"target", "draft"}:
        raise ValueError("Select target and/or draft trace roles")
    if expert_rows is not None and expert_rows < 1:
        raise ValueError("Expert snapshot row limit must be positive")
    specs, points = {}, []
    for rank in model["ranks"]:
        definitions = {d["name"]: d for d in rank["buffers"]}
        for packet_index, packet in enumerate(rank["packets"]):
            role = packet.get("role", "target")
            if role not in roles:
                continue
            capacity = packet["rows"]
            selected = None
            routing = None
            for instruction in packet["body"]:
                if instruction["op"] == "kernel":
                    path = instruction["manifest"]
                    if path not in specs:
                        specs[path] = read_spec(path)
                    description = specs[path]
                    if description["op"] == "routing.route":
                        selected = description["parameters"]["selected"]
                        routing = {k: description["parameters"][k] for k in ("selected", "experts")}
                        break
            layer = None
            steps = []
            head_observed = set()

            def observe(label, buffer, *, fixed=False, projection=None):
                if layers is not None and layer not in layers:
                    return
                if labels is not None and not labels.fullmatch(label):
                    return
                definition = definitions[buffer]
                captured_rows = None
                if expert_rows is not None and label.startswith("expert.up."):
                    # Legacy unfused plans infer the same row multiplier from
                    # their declared projection, rather than a model constant.
                    multiplier = selected or step.get("rows", 0) // capacity
                    if not multiplier:
                        raise ValueError("Expert trace requires declared routed row geometry")
                    routed_capacity = capacity * multiplier
                    if definition["bytes"] % routed_capacity:
                        raise ValueError("Expert snapshot requires a whole routed row stride")
                    captured_rows = min(expert_rows, routed_capacity)
                    name = f"calibration.P{packet_index}.L{layer}.{label}"
                    view = dict(
                        name=name,
                        bytes=definition["bytes"] // routed_capacity * captured_rows,
                        view=dict(parent=buffer, offset=0),
                    )
                    rank["buffers"].append(view)
                    buffer = name
                    definition = view
                    fixed = True
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
                    role=role,
                    draft_block=packet.get("coordinates", {}).get("draft_block"),
                    name=name,
                    buffer=buffer,
                    definition=definition,
                    fixed_rows=whole,
                )
                if projection is not None:
                    point["projection"] = projection
                if routing is not None:
                    point["routing"] = routing
                if captured_rows is not None:
                    point["captured_rows"] = captured_rows
                points.append(point)

            for step in packet["body"]:
                if step.get("diagnostic"):
                    raise ValueError("Use the frozen source plan, without existing diagnostics")
                if step["op"] in ("fork", "switch", "join"):
                    continue
                if step["op"] == "snapshot" and expert_rows is not None:
                    # Keep layer coverage markers without exporting every full
                    # HC hidden allocation in an otherwise bounded trace.
                    parent = step["buffer"]
                    marker = f"calibration.marker.{len(rank['buffers'])}"
                    rank["buffers"].append(
                        dict(name=marker, bytes=2, view=dict(parent=parent, offset=0))
                    )
                    step.update(buffer=marker, fixed_rows=True)
                steps.append(step)
                if step["op"] == "kernel":
                    path = step["manifest"]
                    if path not in specs:
                        specs[path] = read_spec(path)
                    spec, args = specs[path], step["buffers"]
                    for name in args:
                        match = re.search(r"(?:parts\.)?w\.(layers|mtp)\.(\d+)\.", name)
                        if match:
                            layer = int(match[2]) + (40 if match[1] == "mtp" else 0)
                            break
                    if layer is None or (
                        role == "target" and any("dspark" in name for name in args)
                    ):
                        continue
                    kind = spec["op"]
                    if kind == "model_ops.norm":
                        for norm_role in ("attn", "ffn"):
                            if f".{norm_role}_norm." in args[1]:
                                observe(f"{norm_role}.norm.bf16", args[2])
                                observe(f"{norm_role}.norm.fp32", args[3])
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
                    elif kind == "expert.gather_split":
                        observe("expert.table.i8", args[0])
                        observe("expert.table.scale", args[1])
                        observe("routing.counts", args[3], fixed=True)
                    elif (
                        role == "draft"
                        and kind == "topk.add_bias_argmax"
                        and "base" not in head_observed
                    ):
                        observe("draft.base", args[0], fixed=True)
                        head_observed.add("base")
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
            if role == "draft":
                # Markov endpoints use a physical per-owner stride. Keep the
                # small allocation intact; readers select actual owner slots.
                for label, key in (
                    ("draft.confidence", "confidence"),
                    ("draft.tokens", "best_index"),
                ):
                    if key in packet:
                        observe(label, packet[key], fixed=True)
                packet["body"] = steps
    return model, dict(points=points, manifests=specs)


def collect_routed(trace, metadata, samples, output, *, role="draft"):
    """Collect declared routed tables with conversation-disjoint fold labels."""
    import numpy as np

    metadata = json.loads(Path(metadata).read_text())
    samples = json.loads(Path(samples).read_text())["samples"]
    fold_names = {"calibration": 0, "fit": 0, "tune": 1, "validation": 1, "test": 2}
    owners, conversations = {}, {}
    for index, sample in enumerate(samples):
        owner = sample.get("owner", index)
        fold = fold_names[sample["split"]]
        conversation = sample["conversation"]
        if owner in owners or conversations.setdefault(conversation, fold) != fold:
            raise ValueError("Owner identity or conversation folds overlap")
        owners[owner] = fold
    points = {}
    labels = ("expert.table.i8", "expert.table.scale", "routing.indices", "routing.weights")
    for point in metadata["points"]:
        if point["rank"] != 0 or point.get("role") != role:
            continue
        label = next((x for x in labels if point["name"].endswith("." + x)), None)
        if label:
            layer = int(point["name"].split(".L", 1)[1].split(".", 1)[0])
            points.setdefault((point["packet"], layer), {})[label] = point
    collected = {}
    for frame in sorted(Path(trace).iterdir()):
        if not frame.is_dir():
            continue
        segments = [tuple(map(int, value.split("-"))) for value in frame.name.split("_")]
        row_owners = np.repeat([s[0] for s in segments], [s[2] for s in segments])
        for (_, layer), bindings in points.items():
            if set(bindings) != set(labels):
                raise ValueError("Routed calibration points are incomplete")
            table = bindings["expert.table.i8"]
            first = frame / ("p0-" + table["name"] + ".bin")
            if not first.exists():
                continue
            geometry = table["routing"]
            selected, experts = geometry["selected"], geometry["experts"]
            width = table["definition"]["bytes"] // table["capacity"]

            def read(label, dtype):
                return np.fromfile(frame / ("p0-" + bindings[label]["name"] + ".bin"), dtype=dtype)

            inputs = read("expert.table.i8", "i1").reshape(-1, width)
            scale = read("expert.table.scale", "<f4")
            indices = read("routing.indices", "<i4").reshape(len(row_owners), -1)[:, :selected]
            weights = read("routing.weights", "<f4").reshape(len(row_owners), -1)[:, :selected]
            if (
                len(inputs) != len(row_owners)
                or len(scale) != len(inputs)
                or np.any((indices < 0) | (indices >= experts))
            ):
                raise ValueError("Native trace differs from its declared routed geometry")
            values = dict(
                inputs=inputs,
                scale=scale,
                indices=indices,
                weights=weights,
                owner=row_owners,
                fold=np.array([owners[int(owner)] for owner in row_owners]),
                importance=np.concatenate([1 / (np.arange(s[2]) + 1) for s in segments]),
            )
            for key, value in values.items():
                collected.setdefault(layer, {}).setdefault(key, []).append(value)
    if not collected:
        raise ValueError("No matching native routed frames")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for layer, values in collected.items():
        np.savez(output / f"layer{layer}.npz", **{k: np.concatenate(v) for k, v in values.items()})
    return output


def prepare(plan, output, **options):
    plan, output = Path(plan).resolve(strict=True), Path(output).resolve()
    if output.exists() or output.is_relative_to(plan.parent):
        raise ValueError("Use a new directory outside the source execution package")
    model = decode(json.loads(plan.read_text()))
    if len(model.get("ranks", [])) != 8 or not model.get("batch"):
        raise ValueError("Expected a complete DSv4.1 TP8 batch plan")
    result, metadata = instrument(
        model, lambda name: json.loads(source_path(plan.parent, name).read_text()), **options
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
    parser.add_argument("--layers", type=int, nargs="+")
    parser.add_argument("--labels", help="Full-match expression selecting observed tensor labels")
    parser.add_argument("--roles", nargs="+", choices=["target", "draft"], default=["target"])
    parser.add_argument(
        "--expert-rows", type=int, help="Bound captured up-projection rows per rank"
    )
    args = parser.parse_args()
    print(
        prepare(
            args.plan,
            args.output,
            layers=args.layers,
            labels=args.labels,
            expert_rows=args.expert_rows,
            roles=args.roles,
        )
    )


if __name__ == "__main__":
    main()
