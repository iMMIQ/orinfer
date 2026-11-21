"""Export, attach, and publish BF16 vision banks and native VL programs for A2."""

from orinfer_build.common.paths import ROOT

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import re
import shutil
import struct
import tempfile

from orinfer_build.common.files import atomic_json, file_hash
from orinfer_build.contracts.row_contract import expression
from orinfer_build.models.deepseek_v41.publication import prepare_execution
from orinfer_build.weights.safetensors_source import local_header


def certify_values(path, config):
    """Bound V from RMSNorm and checkpoint weights, before FP16 PV compilation.

    Q/K, weights and residuals keep BF16. The V boundary first rounds to BF16;
    a conservative Cauchy bound includes both BF16 rounding steps. The check is
    offline and reads only norm1 and QKV tensors from the semantic vision shard.
    """
    import numpy as np

    start, header = local_header(path)
    shapes = tensor_shapes(config)
    hidden = config["vision_config"]["hidden_size"]
    bounds = []
    with Path(path).open("rb") as file:

        def read(name):
            item = header[name]
            if item["dtype"] != "BF16" or item["shape"] != shapes[name]:
                raise ValueError("Vision value range tensor differs: " + name)
            begin, end = item["data_offsets"]
            file.seek(start + begin)
            data = file.read(end - begin)
            if len(data) != end - begin:
                raise ValueError("Truncated value range tensor: " + name)
            bits = np.frombuffer(data, dtype="<u2").astype(np.uint32) << 16
            values = bits.view(np.float32).reshape(item["shape"]).astype(np.float64)
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite vision value range tensor: " + name)
            return values

        for layer in range(config["vision_config"]["num_hidden_layers"]):
            prefix = f"vision.blocks.{layer}"
            gamma = read(prefix + ".norm1.weight")
            weight = read(prefix + ".attn.wqkv.weight")[2 * hidden :]
            bias = read(prefix + ".attn.wqkv.bias")[2 * hidden :]
            bound = math.sqrt(hidden) * np.linalg.norm(weight * gamma, axis=1) + abs(bias)
            bounds.append(float(np.max(bound) * 1.02))
    if not bounds or not math.isfinite(max(bounds)) or max(bounds) >= 65504:
        raise ValueError("Vision V exceeds the certified FP16 compute range")
    return dict(
        dtype="f16",
        maximum=max(bounds),
        per_layer=bounds,
        limit=65504,
        bf16_rounding_margin=1.02,
        small_value_absolute_error_bound=2**-14,
    )


def tensor_shapes(config):
    vision = config["vision_config"]
    hidden = vision["hidden_size"]
    intermediate = vision["intermediate_size"]
    patch = vision["patch_size"]
    merge = vision["downsample_ratio"]
    output = config["text_config"]["hidden_size"]
    if min(hidden, intermediate, patch, merge, output, vision["num_hidden_layers"]) <= 0:
        raise ValueError("Invalid vision dimensions")
    shapes = {
        "vision.patch_embed.proj.weight": [hidden, 3 * patch * patch],
        "vision.patch_embed.proj.bias": [hidden],
        "vision.norm.weight": [hidden],
        "aligner.w1.weight": [output, hidden * merge * merge],
        "aligner.w1.bias": [output],
        "aligner.w2.weight": [output, output],
        "aligner.w2.bias": [output],
        "image_start": [output],
        "image_end": [output],
        "image_newline": [output],
    }
    for layer in range(vision["num_hidden_layers"]):
        prefix = f"vision.blocks.{layer}"
        shapes.update(
            {
                f"{prefix}.norm1.weight": [hidden],
                f"{prefix}.norm2.weight": [hidden],
                f"{prefix}.attn.wqkv.weight": [3 * hidden, hidden],
                f"{prefix}.attn.wqkv.bias": [3 * hidden],
                f"{prefix}.attn.wo.weight": [hidden, hidden],
                f"{prefix}.attn.wo.bias": [hidden],
                f"{prefix}.mlp.w1.weight": [2 * intermediate, hidden],
                f"{prefix}.mlp.w2.weight": [hidden, intermediate],
            }
        )
    return shapes


def export(source, output):
    source, output = Path(source).resolve(), Path(output)
    config = json.loads((source / "config.json").read_text())
    if config.get("quantization_config", {}).get("quant_method") != "ascend":
        raise ValueError("Use the matching Ascend checkpoint: image outputs use its target basis")
    if output.exists():
        raise ValueError("Preserve existing vision exports: " + str(output))
    index = json.loads((source / "quant_model_weights.safetensors.index.json").read_text())[
        "weight_map"
    ]
    shapes = tensor_shapes(config)
    names = sorted(set(shapes) | {name for name in index if name.endswith(".gate.bias_vl")})
    headers, sources, tensors = {}, {}, {}
    offset = 0
    try:
        for name in names:
            path = (source / index[name]).resolve()
            if not path.is_relative_to(source):
                raise ValueError("Vision source leaves checkpoint directory")
            if path not in headers:
                headers[path] = local_header(path)
                sources[path] = path.open("rb")
            start, header = headers[path]
            item = header[name]
            shape = item["shape"]
            if name in shapes and (item["dtype"] != "BF16" or shape != shapes[name]):
                raise ValueError("Vision tensor shape/dtype differs: " + name)
            if name not in shapes and (item["dtype"] != "F32" or len(shape) != 1 or shape[0] < 1):
                raise ValueError("VL routing bias requires an FP32 vector")
            physical = list(shape)
            if name == "vision.patch_embed.proj.weight":
                physical[1] = (shape[1] + 127) // 128 * 128
            elements = 1
            for dim in physical:
                elements *= dim
            size = elements * (2 if item["dtype"] == "BF16" else 4)
            tensors[name] = dict(
                dtype=item["dtype"], shape=physical, data_offsets=[offset, offset + size]
            )
            offset += size
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=output.name + ".", dir=output.parent))
        try:
            encoded = json.dumps(tensors, separators=(",", ":")).encode()
            encoded += b" " * (-len(encoded) % 8)
            digest = hashlib.sha256()
            size = 0
            with (temporary / "vision.safetensors").open("xb") as destination:

                def write(data):
                    nonlocal size
                    destination.write(data)
                    digest.update(data)
                    size += len(data)

                write(struct.pack("<Q", len(encoded)))
                write(encoded)
                for name in names:
                    path = (source / index[name]).resolve()
                    start, header = headers[path]
                    item = header[name]
                    file = sources[path]
                    begin, end = item["data_offsets"]
                    file.seek(start + begin)
                    if tensors[name]["shape"] != item["shape"]:
                        rows, width = item["shape"]
                        padded = tensors[name]["shape"][1]
                        zero = bytes((padded - width) * 2)
                        for _ in range(rows):
                            data = file.read(width * 2)
                            if len(data) != width * 2:
                                raise ValueError("Truncated patch projection")
                            write(data)
                            write(zero)
                    else:
                        remaining = end - begin
                        while remaining:
                            data = file.read(min(remaining, 8 << 20))
                            if not data:
                                raise ValueError("Truncated vision tensor: " + name)
                            write(data)
                            remaining -= len(data)
                if size != 8 + len(encoded) + offset:
                    raise ValueError("Vision payload differs from its declared geometry")
            manifest = dict(
                schema_version=1,
                target_basis="ascend_preprocessed",
                precision="bf16",
                vision=config["vision_config"],
                hidden=config["text_config"]["hidden_size"],
                logical_shapes=shapes,
                patch_reduction=tensors["vision.patch_embed.proj.weight"]["shape"][1],
                file="vision.safetensors",
                bytes=size,
                sha256=digest.hexdigest(),
                tensors=len(tensors),
                value_compute=certify_values(temporary / "vision.safetensors", config),
            )
            atomic_json(temporary / "vision.json", manifest)
            atomic_json(temporary / "config.json", config)
            atomic_json(
                temporary / "model.safetensors.index.json",
                dict(weight_map=dict.fromkeys(tensors, "vision.safetensors")),
            )
            temporary.rename(output)
            return manifest
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    finally:
        for file in sources.values():
            file.close()


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
    hardlinks(plan.parent, output)
    hardlinks(vision.parent, output / "vision")
    for operation in program["body"]:
        operation["manifest"] = "vision/" + operation["manifest"]
    model["vision"] = program
    tasks, pending, specs = {}, [], {}
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


def publish(plan, source_model, vision_weights, output_model, execution_cache):
    plan, source_model, vision_weights = (
        plan.resolve(),
        source_model.resolve(),
        vision_weights.resolve(),
    )
    output_model, execution_cache = output_model.resolve(), execution_cache.resolve()
    if (
        output_model.exists()
        or output_model.is_relative_to(execution_cache)
        or execution_cache.is_relative_to(output_model)
    ):
        raise ValueError("Use a new model directory and a separate execution cache")
    model = json.loads(plan.read_text())
    if not model.get("vision"):
        raise ValueError("The plan must contain the validated native vision program")
    index = json.loads((source_model / model["index"]["file"]).read_text())
    added = json.loads((vision_weights / "model.safetensors.index.json").read_text())["weight_map"]
    if index["weight_map"].keys() & added.keys():
        raise ValueError("Vision tensors conflict with the language checkpoint")
    if set(added.values()) != {"vision.safetensors"}:
        raise ValueError("Require one semantic vision shard")
    index["weight_map"].update(added)
    release = deepcopy(json.loads((source_model / "release-manifest.json").read_text()))
    record = json.loads((vision_weights / "vision.json").read_text())
    if record["target_basis"] != "ascend_preprocessed":
        raise ValueError("Vision and language target bases differ")
    certificate = record.get("value_compute", {})
    if (
        certificate.get("dtype") != "f16"
        or not 0 <= certificate.get("maximum", float("inf")) < 65504
    ):
        raise ValueError("Export vision assets with a certified FP16 V range")
    start, header = local_header(vision_weights / "vision.safetensors")
    hashes = {}
    with (vision_weights / "vision.safetensors").open("rb") as file:
        for name in sorted(added):
            begin, end = header[name]["data_offsets"]
            file.seek(start + begin)
            remaining, digest = end - begin, hashlib.sha256()
            while remaining:
                chunk = file.read(min(remaining, 8 << 20))
                if not chunk:
                    raise ValueError("Truncated vision release shard")
                digest.update(chunk)
                remaining -= len(chunk)
            hashes[name] = digest.hexdigest()
    release.setdefault("logical_shapes", {}).update(record["logical_shapes"])
    if "value_compute" in record:
        release["vision_value_compute"] = record["value_compute"]
    release["files"]["vision.safetensors"] = dict(
        bytes=record["bytes"], sha256=record["sha256"], tensors=hashes
    )
    execution_cache.mkdir(parents=True, exist_ok=True)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".vision-model-", dir=output_model.parent) as temporary:
        stage = Path(temporary) / "model"
        stage.mkdir()
        for name in set(index["weight_map"].values()):
            source = vision_weights / name if name == "vision.safetensors" else source_model / name
            target = stage / name
            if source.resolve().is_relative_to(
                vision_weights if name == "vision.safetensors" else source_model
            ) is False or not target.resolve().is_relative_to(stage):
                raise ValueError("Weight shard leaves its model directory")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.hardlink_to(source)
        for name in [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "generation_config.json",
            "chat_template.jinja",
            "LICENSE",
            "SOURCE-README.md",
        ]:
            if (source_model / name).exists():
                shutil.copyfile(source_model / name, stage / name)
        atomic_json(stage / "model.safetensors.index.json", index)
        model["index"]["sha256"] = file_hash(stage / "model.safetensors.index.json")
        with tempfile.TemporaryDirectory(
            prefix=".vision-execution-", dir=execution_cache
        ) as package_temp:
            package_stage = Path(package_temp) / "package"
            package_stage.mkdir()
            prepare_execution(model, plan.parent, package_stage)
            shutil.copytree(plan.parent / "licenses", package_stage / "licenses")
            shutil.copyfile(
                ROOT / "THIRD_PARTY_NOTICES.md",
                package_stage / "licenses/THIRD_PARTY_NOTICES.md",
            )
            digest = file_hash(package_stage / "package.json")
            package = execution_cache / digest
            if package.exists():
                raise ValueError("Execution package already exists")
            package_stage.rename(package)
        descriptor = json.loads((source_model / "cache/model.json").read_text())
        descriptor["execution_package"] = digest
        (stage / "cache").mkdir()
        atomic_json(stage / "cache/model.json", descriptor)
        atomic_json(stage / "release-manifest.json", release)
        stage.rename(output_model)
    return dict(
        model=str(output_model),
        execution_package=str(package),
        execution_digest=digest,
        vision_tensors=len(added),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p_export = sub.add_parser("export", help="Export vision.safetensors from checkpoint")
    p_export.add_argument("--source", type=Path, required=True)
    p_export.add_argument("--output", type=Path, required=True)
    p_attach = sub.add_parser("attach", help="Attach a compiled vision program to an A2 plan")
    p_attach.add_argument("--plan", type=Path, required=True)
    p_attach.add_argument("--vision", type=Path, required=True)
    p_attach.add_argument("--output", type=Path, required=True)
    p_attach.add_argument("--compile-workers", type=int, default=3)
    p_pub = sub.add_parser("publish", help="Publish vision release shards and execution package")
    for name in ["plan", "source-model", "vision-weights", "output-model", "execution-cache"]:
        p_pub.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.command == "export":
        print(json.dumps(export(args.source, args.output), indent=2))
    elif args.command == "attach":
        print(attach(args.plan, args.vision, args.output, args.compile_workers))
    else:
        print(
            json.dumps(
                publish(
                    args.plan,
                    args.source_model,
                    args.vision_weights,
                    args.output_model,
                    args.execution_cache,
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
