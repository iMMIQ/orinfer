"""Export BF16 vision banks in the Ascend target basis without language weights."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import tempfile

from tools.build.files import atomic_json
from tools.model.safetensors_source import local_header


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
