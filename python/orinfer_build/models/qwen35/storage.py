"""Bounded reads, shard adoption and lossless K-major E8P publication."""

import hashlib
import json
import math
from pathlib import Path
import tempfile
import numpy as np
from safetensors.numpy import save_file
from orinfer_build.packaging.compact import write_shard
from orinfer_build.common.files import file_hash, write_json
from orinfer_build.packaging.publication import link_or_copy, source_path
from orinfer_build.weights.safetensors_source import local_header

DTYPES = {
    "BF16": "bf16",
    "F16": "f16",
    "F32": "f32",
    "I8": "i8",
    "U8": "u8",
    "I32": "i32",
    "U32": "u32",
}


class Weights:
    def __init__(self, directory, destination, *, packed_k=0):
        self.source, self.root = Path(directory).resolve(strict=True), destination
        self.index = json.loads((self.source / "model.safetensors.index.json").read_text())
        self.headers, self.buffers, self.scopes, self.mapping, self.extra = {}, {}, {}, {}, {}
        self.paths = {}
        if type(packed_k) is not int or packed_k not in (0, 64, 128):
            raise ValueError("Invalid E8P packed K layout")
        self.packed_k = packed_k
        self.directory = self.root / "cache/weights"
        self.directory.mkdir(parents=True, exist_ok=True)

    def repack(self, path, file, begin, header):
        """Publish one physical W4 copy; all other tensor bytes pass through."""
        metadata = dict(header.get("__metadata__", {}))
        packed = [name for name in header if name.endswith(".packed")]
        missing_layout = any(
            name != "__metadata__" and f"orin.layout.{name}" not in metadata for name in header
        )
        if (not packed or not self.packed_k) and not missing_layout:
            return path, begin, header
        tensors = []
        with tempfile.TemporaryDirectory(prefix=".repack-", dir=self.directory) as tmp:
            for name, info in header.items():
                if name == "__metadata__":
                    continue
                metadata.setdefault(f"orin.layout.{name}", "native_contiguous")
                first, last = info["data_offsets"]
                shape = info["shape"]
                segments = [(path, begin + first, last - first)]
                if name in packed and self.packed_k:
                    if (
                        info["dtype"] != "U32"
                        or len(shape) not in (2, 3)
                        or min(shape) <= 0
                        or shape[-1] % (self.packed_k // 8)
                        or metadata.get(f"orin.layout.{name}") != "e8p-q4-row-u32"
                    ):
                        raise ValueError(f"Invalid source E8P weight layout: {name}")
                    source = np.memmap(
                        path, mode="r", offset=begin + first, dtype="<u4", shape=shape
                    )
                    n, vectors = shape[-2:]
                    codes = self.packed_k // 8
                    data = np.ascontiguousarray(
                        source.reshape(n, vectors // codes, codes).transpose(1, 0, 2)
                        if len(shape) == 2
                        else source.reshape(shape[0], n, vectors // codes, codes).transpose(
                            0, 2, 1, 3
                        )
                    )
                    payload = Path(tmp) / f"{len(tensors)}.bin"
                    data.tofile(payload)
                    shape = list(data.shape)
                    segments = [(payload, 0, last - first)]
                    metadata[f"orin.layout.{name}"] = f"e8p-q4-k{self.packed_k}-u32"
                tensors.append(dict(name=name, dtype=info["dtype"], shape=shape, segments=segments))
            destination = self.directory / file
            write_shard(destination, tensors, metadata)
        offset, rewritten = local_header(destination)
        return destination, offset, rewritten

    def info(self, name):
        file = self.index["weight_map"][name]
        if file not in self.headers:
            path = source_path(self.source, file)
            expected = self.index["metadata"]["orinfer.sha256"][file]
            if file_hash(path) != expected:
                raise ValueError(f"Quantized shard identity changed: {file}")
            begin, header = local_header(path)
            path, begin, header = self.repack(path, file, begin, header)
            self.headers[file] = begin, header
            self.paths[file] = path
        begin, header = self.headers[file]
        return self.paths[file], begin, header[name]

    def raw(self, name):
        path, begin, info = self.info(name)
        first, last = info["data_offsets"]
        with path.open("rb") as stream:
            stream.seek(begin + first)
            raw = stream.read(last - first)
        if len(raw) != last - first:
            raise EOFError(name)
        return raw, info

    def array(self, name):
        if name in self.extra:
            return self.extra[name]
        raw, info = self.raw(name)
        if info["dtype"] == "BF16":
            return (
                (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16)
                .view(np.float32)
                .reshape(info["shape"])
            )
        dtype = {"F16": "<f2", "F32": "<f4", "I8": "i1", "U8": "u1", "I32": "<i4", "U32": "<u4"}[
            info["dtype"]
        ]
        return np.frombuffer(raw, dtype=dtype).reshape(info["shape"])

    def immutable(self, name, *, cast=None, shape=None):
        target = name if cast is None else f"Runtime.{name}.{cast}"
        if target in self.buffers:
            return target
        path, begin, info = self.info(name)
        dtype = info["dtype"]
        dimensions = info["shape"] if shape is None else list(shape)
        if math.prod(dimensions) != math.prod(info["shape"]):
            raise ValueError("Weight view changes element count")
        if cast is not None:
            original = self.array(name)
            data = np.ascontiguousarray(original, dtype=cast)
            if not np.isfinite(data).all():
                raise ValueError(f"Nonfinite runtime cast: {name}")
            if (
                info["dtype"] == "BF16"
                and cast == "float16"
                and np.max(np.abs(data.astype(np.float32) - original), initial=0) > 2**-25
            ):
                raise ValueError(f"BF16 weight exceeds lossless FP16 range: {name}")
            self.extra[target] = data.reshape(dimensions)
            raw = data.tobytes()
            dtype = {"float16": "F16", "float32": "F32"}[cast]
            self.mapping[target] = "runtime.safetensors"
            payload = hashlib.sha256(raw).hexdigest()
            layout = "native_contiguous"
        else:
            file = self.index["weight_map"][name]
            if path != self.directory / file:
                link_or_copy(path, self.directory / file)
            self.mapping[target] = file
            digest = hashlib.sha256()
            first, last = info["data_offsets"]
            with path.open("rb") as stream:
                stream.seek(begin + first)
                remaining = last - first
                while remaining:
                    block = stream.read(min(remaining, 8 * 1024**2))
                    if not block:
                        raise EOFError(name)
                    remaining -= len(block)
                    digest.update(block)
            payload = digest.hexdigest()
            layout = (
                self.headers[file][1]
                .get("__metadata__", {})
                .get(f"orin.layout.{name}", "native_contiguous")
            )
        self.buffers[target] = dict(
            name=target,
            dtype=DTYPES[dtype],
            shape=dimensions,
            layout=layout,
            alignment=256,
            access="read",
            data=dict(tensor=target, sha256=payload),
        )
        self.scopes[target] = "weights"
        return target

    def constant(self, name, data):
        data = np.ascontiguousarray(data)
        dtype = {
            np.dtype("float16"): "f16",
            np.dtype("float32"): "f32",
            np.dtype("int32"): "i32",
            np.dtype("uint32"): "u32",
            np.dtype("int8"): "i8",
        }[data.dtype]
        self.extra[name] = data
        self.buffers[name] = dict(
            name=name,
            dtype=dtype,
            shape=list(data.shape),
            layout="native_contiguous",
            alignment=256,
            access="read",
            data=dict(tensor=name, sha256=hashlib.sha256(data.tobytes()).hexdigest()),
        )
        self.scopes[name] = "weights"
        self.mapping[name] = "runtime.safetensors"
        return name

    def combined_projection(self, name, sources):
        """Losslessly join output rows while keeping the original integer codec."""
        suffixes = [
            ".int8" if p + ".int8" in self.index["weight_map"] else ".packed" for p in sources
        ]
        if len(set(suffixes)) != 1:
            raise ValueError("Combined projections require the same integer representation")
        suffix = suffixes[0]
        values = [self.array(p + suffix) for p in sources]
        if suffix == ".packed" and (self.packed_k != 128 or any(v.ndim != 3 for v in values)):
            raise ValueError("Combined compressed projections require K128 E8P banks")
        target = self.constant(
            name + suffix, np.concatenate(values, axis=0 if suffix == ".int8" else 1)
        )
        if suffix == ".packed":
            self.buffers[target]["layout"] = "e8p-q4-k128-u32"
        self.constant(name + ".scale", np.concatenate([self.array(p + ".scale") for p in sources]))
        return name

    def fixed(self, name, dtype, shape, scope="workspace"):
        if name in self.buffers:
            raise ValueError(f"Duplicate storage: {name}")
        self.buffers[name] = dict(
            name=name,
            dtype=dtype,
            shape=list(shape),
            layout="native_contiguous",
            alignment=256,
            access="read_write",
        )
        self.scopes[name] = scope
        return name

    def combined(self, name, inputs):
        data = np.concatenate([self.array(n) for n in inputs]).astype(np.float16)
        if not np.isfinite(data).all():
            raise ValueError("Nonfinite projection weight")
        return self.constant(name, data)

    def finish(self):
        save_file(
            self.extra,
            str(self.directory / "runtime.safetensors"),
            metadata={f"orin.layout.{n}": self.buffers[n]["layout"] for n in self.extra},
        )
        # Assembled banks replace individually stored tensors. Prune unbound
        # source data from each adopted shard without changing bound bytes.
        for filename in set(self.mapping.values()) - {"runtime.safetensors"}:
            path = self.directory / filename
            begin, specs = local_header(path)
            names = {n for n, f in self.mapping.items() if f == filename}
            if names == set(specs) - {"__metadata__"}:
                continue
            entries = []
            for name in sorted(names):
                info = specs[name]
                first, last = info["data_offsets"]
                entries.append(
                    dict(
                        name=name,
                        dtype=info["dtype"],
                        shape=info["shape"],
                        segments=[(path, begin + first, last - first)],
                    )
                )
            write_shard(path, entries, specs.get("__metadata__"))
        total = sum(
            math.prod(b["shape"])
            * {"f16": 2, "f32": 4, "bf16": 2, "i8": 1, "u8": 1, "i32": 4, "u32": 4}[b["dtype"]]
            for b in self.buffers.values()
            if b["access"] == "read"
        )
        write_json(
            self.directory / "model.safetensors.index.json",
            dict(metadata=dict(total_size=total), weight_map=self.mapping),
        )
        return total
