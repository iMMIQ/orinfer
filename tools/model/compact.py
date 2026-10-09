"""Losslessly compact prepared models and publish execution packages separately."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.model.publication import (  # noqa: E402
    AssetReader,
    file_hash,
    load_model,
    source_path,
    staged_directory,
    write_json,
)
from tools.model.safetensors_source import read_header  # noqa: E402


def header(path):
    with path.open("rb") as stream:

        def read(offset, size):
            stream.seek(offset)
            return stream.read(size)

        return read_header(read, file_size=path.stat().st_size)


def write_shard(path, tensors, metadata=None):
    """Stream byte ranges into a standard container without materializing tensors.

    Each entry contains name, dtype, shape, and (file, offset, bytes) segments.
    Returns absolute tensor offsets and full-file SHA256 for CPU asset identities.
    """
    spec = {}
    if metadata:
        spec["__metadata__"] = metadata
    cursor = 0
    for item in tensors:
        size = sum(s[2] for s in item["segments"])
        if item["name"] in spec or size == 0:
            raise ValueError("Duplicate or empty tensor")
        spec[item["name"]] = dict(
            dtype=item["dtype"], shape=item["shape"], data_offsets=[cursor, cursor + size]
        )
        cursor += size
    raw = json.dumps(spec, separators=(",", ":"), ensure_ascii=False).encode()
    raw += b" " * (-len(raw) % 8)
    prefix = struct.pack("<Q", len(raw)) + raw
    digest = hashlib.sha256(prefix)
    temporary = path.with_suffix(".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("xb") as out:
            out.write(prefix)
            for item in tensors:
                payload = hashlib.sha256()
                for source, offset, size in item["segments"]:
                    with source.open("rb") as stream:
                        stream.seek(offset)
                        while size:
                            block = stream.read(min(size, 8 * 1024**2))
                            if not block:
                                raise EOFError(f"Truncated tensor source: {source}")
                            out.write(block)
                            digest.update(block)
                            payload.update(block)
                            size -= len(block)
                if item.get("sha256") and payload.hexdigest() != item["sha256"]:
                    raise ValueError(f"Tensor payload identity changed: {item['name']}")
        # Standard parser validates shapes, dtype sizes, contiguous offsets and file extent.
        header(temporary)
        temporary.rename(path)
    finally:
        temporary.unlink(missing_ok=True)
    return (
        {n: len(prefix) + v["data_offsets"][0] for n, v in spec.items() if n != "__metadata__"},
        digest.hexdigest(),
    )


def groups(items, limit, size, compatible=lambda a, b: True):
    current, total = [], 0
    for item in items:
        count = size(item)
        if current and (total + count > limit or not compatible(current[-1], item)):
            yield current
            current, total = [], 0
        current.append(item)
        total += count
    if current:
        yield current


def compact_weights(source, target, descriptor, limit):
    origin = source / "cache/weights"
    index = json.loads((origin / "model.safetensors.index.json").read_text())
    buffers = {b["data"]["tensor"]: b for b in descriptor["metadata"]["buffers"] if b.get("data")}
    if set(index["weight_map"]) != set(buffers):
        raise ValueError("Weight index differs from model tensor contract")
    tensors, headers = [], {}
    for name, filename in index["weight_map"].items():
        path = source_path(origin.resolve(), filename)
        if path not in headers:
            headers[path] = header(path)
        begin, spec = headers[path]
        info = spec[name]
        layout = spec.get("__metadata__", {}).get(f"orin.layout.{name}")
        if layout != buffers[name]["layout"]:
            raise ValueError("Weight physical layout mismatch")
        first, end = info["data_offsets"]
        tensors.append(
            dict(
                name=name,
                dtype=info["dtype"],
                shape=info["shape"],
                layout=layout,
                segments=[(path, begin + first, end - first)],
                sha256=buffers[name]["data"]["sha256"],
            )
        )
    partitions = list(groups(tensors, limit, lambda t: t["segments"][0][2]))
    mapping = {}
    for i, partition in enumerate(partitions):
        name = f"model-{i + 1:05d}-of-{len(partitions):05d}.safetensors"
        write_shard(
            target / "cache/weights" / name,
            partition,
            {f"orin.layout.{t['name']}": t["layout"] for t in partition},
        )
        mapping.update((t["name"], name) for t in partition)
        print(f"GPU shard {i + 1}/{len(partitions)} verified", flush=True)
    index["weight_map"] = mapping
    write_json(target / "cache/weights/model.safetensors.index.json", index)
    return len(partitions)


def compact_cpu(source, target, descriptor, limit):
    identity = descriptor["metadata"].get("input_assets")
    if identity is None:
        return 0
    path = source_path(source.resolve(), identity["file"])
    if file_hash(path) != identity["sha256"]:
        raise ValueError("CPU input metadata identity mismatch")
    assets = json.loads(path.read_text())
    count = 0
    for kind, width in (("embedding", 2560), ("ple", 160)):
        parts = assets[kind]
        outputs = []
        first = 0
        for part in parts:
            if part["first"] != first or part["rows"] <= 0:
                raise ValueError("Discontinuous CPU row parts")
            first += part["rows"]

        def compatible(a, b):
            return a.get("book") == b.get("book") and ("scale_offset" in a) == ("scale_offset" in b)

        def size(p):
            return p["rows"] * ((2 + width // 4) if "book" in p else width + 2)

        partitions = list(groups(parts, limit, size, compatible))
        for i, partition in enumerate(partitions):
            encoded = "book" in partition[0]
            rows = sum(p["rows"] for p in partition)
            codes, scales = [], []
            for part in partition:
                source_file = source_path(source.resolve(), part["file"])
                if file_hash(source_file) != part["sha256"]:
                    raise ValueError("CPU row shard identity mismatch")
                start, spec = header(source_file)

                def contains(offset, dtype, shape):
                    return any(
                        start + t["data_offsets"][0] == offset
                        and t["dtype"] == dtype
                        and t["shape"] == shape
                        for n, t in spec.items()
                        if n != "__metadata__"
                    )

                row_width = 2 + width // 4 if encoded else width
                dtype = "U8" if encoded else "I8"
                if not contains(part["offset"], dtype, [part["rows"], row_width]):
                    raise ValueError("CPU row container layout mismatch")
                codes.append((source_file, part["offset"], part["rows"] * row_width))
                if not encoded:
                    if not contains(part["scale_offset"], "F16", [part["rows"]]):
                        raise ValueError("CPU row scale layout mismatch")
                    scales.append((source_file, part["scale_offset"], part["rows"] * 2))
            tensors = [
                dict(
                    name="rows",
                    dtype="U8" if encoded else "I8",
                    shape=[rows, row_width],
                    segments=codes,
                )
            ]
            if not encoded:
                tensors.append(dict(name="scales", dtype="F16", shape=[rows], segments=scales))
            filename = f"cache/cpu/{kind}-{i + 1:05d}-of-{len(partitions):05d}.safetensors"
            offsets, digest = write_shard(target / filename, tensors)
            part = dict(
                file=filename,
                sha256=digest,
                first=partition[0]["first"],
                rows=rows,
                offset=offsets["rows"],
            )
            if encoded:
                part["book"] = partition[0]["book"]
            else:
                part["scale_offset"] = offsets["scales"]
            outputs.append(part)
            count += 1
            print(f"CPU {kind} shard {i + 1}/{len(partitions)} verified", flush=True)
        assets[kind] = outputs
    output = target / identity["file"]
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, assets)
    identity["sha256"] = file_hash(output)
    return count


def compact_package(origin, package, cache):
    """Deduplicate byte-identical kernel assets into one indexed safetensors bundle."""
    package = copy.deepcopy(package)
    reader = AssetReader(origin)
    assets, verified = {}, {}
    with staged_directory(cache / ".compact-assets") as staging:
        staging.mkdir()
        tensors = []
        for kernel in package["kernels"]:
            for field in ("module", "source", "host_abi"):
                asset = kernel[field]
                key = (asset["file"], asset.get("tensor"), asset["sha256"])
                if key not in verified:
                    raw = reader.read(asset)
                    digest = asset["sha256"]
                    if digest not in assets:
                        asset_path = staging / digest
                        asset_path.write_bytes(raw)
                        tensors.append(
                            dict(
                                name=digest,
                                dtype="U8",
                                shape=[len(raw)],
                                segments=[(asset_path, 0, len(raw))],
                                sha256=digest,
                            )
                        )
                        assets[digest] = True
                    verified[key] = digest
                kernel[field] = dict(
                    file="assets.safetensors", tensor=verified[key], sha256=verified[key]
                )
        write_shard(staging / "assets.safetensors", tensors)
        for name in ("lib", "LICENSE", "COPYING.LESSER", "COPYING", "THIRD_PARTY_NOTICES.md"):
            path = origin / name
            if path.is_dir():
                shutil.copytree(path, staging / name)
            elif path.is_file():
                shutil.copyfile(path, staging / name)
        raw = (json.dumps(package, separators=(",", ":"), ensure_ascii=False) + "\n").encode()
        digest = hashlib.sha256(raw).hexdigest()
        (staging / "package.json").write_bytes(raw)
        for path in staging.iterdir():
            if path.name in assets:
                path.unlink()
    built = cache / ".compact-assets"
    if (cache / digest).exists():
        if (cache / digest / "package.json").read_bytes() != raw:
            raise ValueError("Conflicting execution package")
        shutil.rmtree(built)
    else:
        built.rename(cache / digest)
    return digest


def compact(source, output, package_cache, shard_bytes=2 * 1024**3):
    source = source.resolve(strict=True)
    if output.resolve().is_relative_to(source) or shard_bytes <= 0:
        raise ValueError("Invalid destination or shard size")
    descriptor, origin, package = load_model(source)
    payloads = set((source / "cache/weights").glob("*.safetensors"))
    identity = descriptor["metadata"].get("input_assets")
    if identity:
        assets = json.loads(source_path(source, identity["file"]).read_text())
        payloads.update(
            source_path(source, p["file"]) for kind in ("embedding", "ple") for p in assets[kind]
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    required = sum(p.stat().st_size for p in payloads) + 2 * 1024**3
    if shutil.disk_usage(output.parent).free < required:
        raise ValueError(f"Compaction requires {required} free bytes, including staging headroom")
    with staged_directory(output) as target:
        target.mkdir()
        for path in source.iterdir():
            if path.is_file() and path.name not in ("weight-publication.json", "MODEL_README.md"):
                shutil.copyfile(path, target / path.name)
        gpu = compact_weights(source, target, descriptor, shard_bytes)
        cpu = compact_cpu(source, target, descriptor, shard_bytes)
        digest = compact_package(origin, package, package_cache)
        descriptor["execution_package"] = digest
        write_json(target / "cache/model.json", descriptor)
        source_info = target / "source.json"
        if source_info.exists():
            info = json.loads(source_info.read_text())
            info["execution_package"] = digest
            write_json(source_info, info)
    return dict(model=str(output), execution_package=digest, gpu_shards=gpu, cpu_shards=cpu)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--package-cache", required=True, type=Path)
    parser.add_argument("--shard-mib", type=int, default=2048)
    args = parser.parse_args()
    print(
        json.dumps(
            compact(args.model, args.output, args.package_cache, args.shard_mib * 1024**2), indent=2
        )
    )


if __name__ == "__main__":
    main()
