"""Publish byte-preserving Flash Next shards with stable component/layer names.

CPU tables retain their already compact row containers. Execution packages stay
in the external package cache; this tool never rebuilds kernels or quantizes.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import shutil

from orinfer_build.packaging.compact import compact_weights
from orinfer_build.packaging.publication import clone_cpu_assets, staged_directory
from orinfer_build.common.files import write_json


def shard_name(name):
    layer = re.match(r"W_blk\.(\d+)\.", name)
    if layer and "_exps.weight_" in name:
        first = int(layer[1]) // 3 * 3
        return f"experts-l{first:03d}-l{first + 2:03d}.safetensors"
    if name.startswith("HC13_"):
        return "hc-up-bf16-packed.safetensors"
    if name.startswith("Q4_output_") or name.startswith("W_output"):
        return "output.safetensors"
    if name.startswith("Q4_"):
        return "projections-q4.safetensors"
    if name.startswith("DW_"):
        return "mtp.safetensors"
    if name.startswith("Vision_"):
        return "vision.safetensors"
    if layer:
        first = int(layer[1]) // 8 * 8
        return f"dense-l{first:03d}-l{first + 7:03d}.safetensors"
    if name.startswith("W_e8p_") or name == "PleConvHalf":
        return "output.safetensors"
    raise ValueError(f"Unrecognized Flash Next weight role: {name}")


def partitions(tensors, limit):
    grouped = defaultdict(list)
    for tensor in tensors:
        grouped[shard_name(tensor["name"])].append(tensor)
    for filename, part in sorted(grouped.items()):
        part.sort(key=lambda t: t["name"])
        size = sum(s[2] for t in part for s in t["segments"])
        if size > limit:
            raise ValueError(f"{filename}: {size} payload bytes exceed shard limit {limit}")
        yield filename, part


def repack(source, output, shard_bytes=2 * 1024**3):
    source = source.resolve(strict=True)
    if output.resolve().is_relative_to(source) or shard_bytes <= 0:
        raise ValueError("Invalid destination or shard size")
    descriptor = json.loads((source / "cache/model.json").read_text())
    if descriptor.get("schema_version") != 1 or descriptor.get("architecture") != "flash_next":
        raise ValueError("Expected a schema 1 Flash Next model")
    payloads = (source / "cache/weights").glob("*.safetensors")
    output.parent.mkdir(parents=True, exist_ok=True)
    required = sum(p.stat().st_size for p in payloads) + 2 * 1024**3
    if shutil.disk_usage(output.parent).free < required:
        raise ValueError(f"Repacking requires {required} free bytes")
    with staged_directory(output) as target:
        target.mkdir()
        for path in source.iterdir():
            if path.is_file() and path.name not in ("weight-publication.json", "MODEL_README.md"):
                shutil.copyfile(path, target / path.name)
        clone_cpu_assets(source, target)
        shards = compact_weights(source, target, descriptor, shard_bytes, partitions)
        write_json(target / "cache/model.json", descriptor)
    return dict(
        model=str(output), gpu_shards=shards, execution_package=descriptor["execution_package"]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-mib", type=int, default=2048)
    args = parser.parse_args()
    print(json.dumps(repack(args.model, args.output, args.shard_mib * 1024**2), indent=2))


if __name__ == "__main__":
    main()
