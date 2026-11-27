"""Export byte-preserving native A2 safetensors in layer and hash-head shards."""

from orinfer_build.common.paths import ROOT


import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct

import numpy as np

from orinfer_build.common.files import atomic_json, file_hash
from orinfer_build.common.shared_json import decode
from orinfer_build.models.deepseek_v41.weight_layout import transposed_nz_file_chunks
from orinfer_build.weights.safetensors_source import local_header, SIZES

CHUNK = 8 << 20
DTYPES = {
    "bf16": "BF16",
    "f16": "F16",
    "f32": "F32",
    "f64": "F64",
    "i8": "I8",
    "u8": "U8",
    "u16": "U16",
    "i16": "I16",
    "i32": "I32",
    "u32": "U32",
    "i64": "I64",
    "u64": "U64",
}
NUMPY = {
    "I8": "i1",
    "U8": "u1",
    "BF16": "<u2",
    "F16": "<f2",
    "F32": "<f4",
    "U64": "<u8",
    "I64": "<i8",
    "U32": "<u4",
    "I32": "<i4",
}


def physical_contract(rank, package, specs=None):
    """Read immutable tensor geometry from the same public operator bindings."""
    specs = {} if specs is None else specs
    buffers = {b["name"]: b for b in rank["buffers"]}
    physical = {}
    for packet in rank["packets"]:
        for step in packet["body"] + packet["head"]:
            if step["op"] in ("w4", "w8"):
                e, k, n = step["experts"], step["input"], step["output"]
                int8 = step["op"] == "w8"
                if int8 and step.get("weight_layout", "nd") != "nd":
                    raise ValueError("INT8 grouped weights require ND storage")
                packed = (
                    [e, k // 64, n // 16, 16, 32]
                    if step.get("weight_layout") == "transposed_nz"
                    else [e, k, n if int8 else n // 2]
                )
                for name, dtype, shape in zip(
                    step["buffers"][1:4],
                    ["I8", "BF16" if int8 else "U64", "F32"],
                    [packed, [e, n] if int8 else [e, 1, n], [e, n]],
                    strict=True,
                ):
                    physical[name] = dtype, shape
            elif step["op"] == "kernel":
                for path in [step["manifest"], step.get("uncompacted_manifest")]:
                    if path is None:
                        continue
                    if path not in specs:
                        specs[path] = json.loads((package / path).read_text())
                    for name, port in zip(step["buffers"], specs[path]["operands"], strict=True):
                        shape, dtype = port["shape"], DTYPES[port["dtype"]]
                        if (
                            all(type(d) is int and d > 0 for d in shape)
                            and math.prod(shape) * SIZES[dtype] == buffers[name]["bytes"]
                        ):
                            physical.setdefault(name, (dtype, shape))
    return physical


def record_chunks(weight, scale, rows, width, block_rows=16384):
    """Two sequential input ranges produce unchanged value/scale bytes in row order."""
    if rows < 1 or width < 1 or width % 32 or block_rows < 1:
        raise ValueError("Host records require positive group-32 INT8 rows")
    scale_bytes = width // 32 * 4
    with Path(weight[0]).open("rb", buffering=8 << 20) as w:
        with Path(scale[0]).open("rb", buffering=8 << 20) as s:
            w.seek(weight[1])
            s.seek(scale[1])
            for first in range(0, rows, block_rows):
                count = min(block_rows, rows - first)
                values = w.read(count * width)
                scales = s.read(count * scale_bytes)
                if len(values) != count * width or len(scales) != count * scale_bytes:
                    raise EOFError("Truncated host INT8 value/scale range")
                out = np.empty((count, width + scale_bytes), dtype=np.uint8)
                out[:, :width] = np.frombuffer(values, dtype=np.uint8).reshape(count, width)
                out[:, width:] = np.frombuffer(scales, dtype=np.uint8).reshape(count, scale_bytes)
                yield memoryview(out).cast("B")


def _plan_w4_nz_transforms(model, key_for_buffer):
    """Rewrite ND W4 steps in place to transposed-NZ and return {key: [E, K, N]}."""
    transforms = {}
    for rank in model["ranks"]:
        definitions = {b["name"]: b for b in rank["buffers"]}
        workspace_sizes = {}
        for packet in rank["packets"]:
            for step in packet["body"] + packet["head"]:
                if step["op"] != "w4":
                    continue
                if step.get("weight_layout", "nd") != "nd":
                    raise ValueError("Prepacking requires the original ND representation")
                if step.get("expected_expert_rows") != 16:
                    raise ValueError("NZ capacity/workspace selection was measured with hint16")
                e, k, n = step["experts"], step["input"], step["output"]
                if e != 48 or (k, n) not in [(5120, 4608), (2304, 5120)]:
                    raise ValueError("Unmeasured W4 NZ geometry")
                target = definitions[step["buffers"][1]]
                key = key_for_buffer(target)
                geometry = [e, k, n]
                if key in transforms and transforms[key] != geometry:
                    raise ValueError("Conflicting W4 physical shapes")
                transforms[key] = geometry
                if "prepared_tensor" in target:
                    target["prepared_tensor"]["shape"] = [e, k // 64, n // 16, 16, 32]
                step["weight_layout"] = "transposed_nz"
                workspace = step["workspace"]
                required = 32 << 20 if step["rows"] <= 192 else 320 << 20
                workspace_sizes[workspace] = max(workspace_sizes.get(workspace, 0), required)
        for name, size in workspace_sizes.items():
            definitions[name]["bytes"] = size
    return transforms


class Export:
    def __init__(
        self,
        source,
        output,
        attribution=None,
        execution_output=None,
        source_execution=None,
        *,
        w4_layout="nd",
        pack_host=False,
    ):
        if w4_layout not in ("nd", "transposed_nz"):
            raise ValueError(f"Unsupported W4 layout: {w4_layout}")
        self.w4_layout = w4_layout
        self.pack_host = pack_host
        self.source, self.output = Path(source).resolve(), Path(output).resolve()
        self.execution_output = Path(execution_output).resolve() if execution_output else None
        if self.execution_output and (
            self.execution_output.is_relative_to(self.output)
            or self.output.is_relative_to(self.execution_output)
        ):
            raise ValueError("Weights and execution output directories must be separate")
        self.attribution = Path(attribution).resolve() if attribution else self.source
        descriptor = json.loads((self.source / "cache/model.json").read_text())
        source_cache = (
            Path(source_execution) if source_execution else self.source / "cache/packages"
        )
        self.package = source_cache / descriptor["execution_package"]
        self.model = deepcopy(decode(json.loads((self.package / "model.json").read_text())))
        self.old_index = json.loads((self.source / self.model["index"]["file"]).read_text())
        self.headers, self.entries, self.groups = {}, {}, defaultdict(list)
        self.index = dict(
            metadata=dict(format="orinfer-a2-physical-v1"), weight_map={}, row_shards={}
        )
        self.manifest = dict(
            schema_version=1,
            source_package=descriptor["execution_package"],
            layers_per_gpu_shard=4,
            host_sharding="layer/hash-head/half-head",
            files={},
            logical_shapes={},
        )

    def tensor(self, name):
        file = self.source / self.old_index["weight_map"][name]
        if file not in self.headers:
            self.headers[file] = local_header(file)
        start, header = self.headers[file]
        spec = header[name]
        return dict(
            file=str(file),
            offset=start + spec["data_offsets"][0],
            dtype=spec["dtype"],
            shape=spec["shape"],
        )

    def plan(self):
        specs = {}
        nz_banks = {}
        if self.w4_layout == "transposed_nz":
            nz_banks = _plan_w4_nz_transforms(self.model, lambda b: b["prepared_weight"])
            self.index["metadata"]["w4_layout"] = "transposed_nz"
            self.manifest["w4_layout"] = "transposed_nz"
        for rank in self.model["ranks"]:
            physical = physical_contract(rank, self.package, specs)
            for b in rank["buffers"]:
                if "prepared_weight" in b:
                    key = "bank:" + b["prepared_weight"]
                    if b["name"] not in physical:
                        raise ValueError(f"Missing physical contract: {b['name']}")
                    dtype, shape = physical[b["name"]]
                    if b["prepared_weight"] in nz_banks:
                        source = dict(
                            nz_source=str(self.package / b["prepared_weight"]),
                            offset=0,
                            geometry=nz_banks[b["prepared_weight"]],
                        )
                    else:
                        source = dict(file=str(self.package / b["prepared_weight"]), offset=0)
                elif "weight" in b:
                    w = b["weight"]
                    key = "transform:" + json.dumps(w, sort_keys=True)
                    dtype = (
                        "F32" if w.get("float32") or w["kind"] != "matrix" else DTYPES[w["dtype"]]
                    )
                    shape = (
                        [w["rows"], w["columns"]]
                        if w["kind"] == "matrix"
                        else [b["bytes"] // SIZES[dtype]]
                    )
                    source = dict(transform=w)
                elif "data" in b:
                    key = "data:" + b["data"]["file"]
                    if b["name"] not in physical:
                        raise ValueError(f"Missing constant contract: {b['name']}")
                    dtype, shape = physical[b["name"]]
                    source = dict(file=str(self.package / b["data"]["file"]), offset=0)
                else:
                    continue
                if math.prod(shape) * SIZES[dtype] != b["bytes"]:
                    raise ValueError("Physical tensor byte extent differs")
                if key not in self.entries:
                    self.entries[key] = dict(
                        source=source, dtype=dtype, shape=shape, bytes=b["bytes"], refs=[]
                    )
                entry = self.entries[key]
                if entry["dtype"] != dtype or entry["bytes"] != b["bytes"]:
                    raise ValueError("Conflicting shared weight interpretation")
                entry["refs"].append((rank["rank"], b))
        for i, entry in enumerate(self.entries.values()):
            ranks = sorted({r for r, _ in entry["refs"]})
            name = entry["refs"][0][1]["name"]
            match = re.search(r"(?:layers\.|L)(\d+)", name)
            layer = int(match[1]) if match else 39
            if len(ranks) > 1:
                group = "weights/shared/model-common.safetensors"
            else:
                first = layer // 4 * 4
                group = (
                    f"weights/gpu/rank-{ranks[0]:02}/layers-{first:02}-{first + 3:02}.safetensors"
                )
            tensor = f"native.{i:05}.{name}"
            self.add(group, tensor, entry)
            for _, b in entry["refs"]:
                for field in ["prepared_weight", "weight", "data"]:
                    b.pop(field, None)
                b["prepared_tensor"] = dict(name=tensor, dtype=entry["dtype"], shape=entry["shape"])
        embed = self.tensor("embed.weight")
        self.add(
            "weights/shared/embedding.safetensors",
            "embed.weight",
            dict(
                source=embed,
                dtype=embed["dtype"],
                shape=embed["shape"],
                bytes=math.prod(embed["shape"]) * SIZES[embed["dtype"]],
            ),
        )
        layout = json.loads((self.package / self.model["engram"]["file"]).read_text())
        if self.pack_host:
            self.index["metadata"]["host_row_layout"] = "i8-f32-g32-v1"
            self.index["row_records"] = {}
            self.manifest["host_row_layout"] = "i8-f32-g32-v1"
        for layer in layout["layers"]:
            prefix = f"layers.{layer['layer']}.engram.embed"
            weight = f"{prefix}.weight"
            scale = f"{prefix}.scale"
            sources = {name: self.tensor(name) for name in [weight, scale]}
            self.manifest["logical_shapes"].update(
                {name: source["shape"] for name, source in sources.items()}
            )
            first = 0
            if self.pack_host:
                w_src, s_src = sources[weight], sources[scale]
                width = w_src["shape"][1]
                if (
                    w_src["dtype"] != "I8"
                    or s_src["dtype"] != "F32"
                    or s_src["shape"] != [w_src["shape"][0], width // 32]
                    or width % 32
                ):
                    raise ValueError("Host value/scale dtype or geometry differs")
                scale_width = width // 32
                parts = []
                part_index = 0
                for head, rows in enumerate(layer["primes"]):
                    half = (rows + 1) // 2
                    for part, begin in enumerate(range(0, rows, half)):
                        count = min(half, rows - begin)
                        group = f"weights/host/layer-{layer['layer']:02}/head-{head:02}-part-{part:02}.safetensors"
                        name = f"{prefix}.records.part{part_index:04}"
                        part_index += 1
                        parts.append(name)
                        row_bytes = width + scale_width * 4
                        self.add(
                            group,
                            name,
                            dict(
                                dtype="U8",
                                shape=[count, row_bytes],
                                bytes=count * row_bytes,
                                source=dict(
                                    row_record=dict(
                                        weight=[
                                            w_src["file"],
                                            w_src["offset"] + (first + begin) * width,
                                        ],
                                        scale=[
                                            s_src["file"],
                                            s_src["offset"] + (first + begin) * scale_width * 4,
                                        ],
                                        rows=count,
                                        width=width,
                                    )
                                ),
                            ),
                        )
                    first += rows
                self.index["row_records"][prefix] = dict(format="i8-f32-g32-v1", parts=parts)
            else:
                for name in sources:
                    self.index["row_shards"][name] = []
                for head, rows in enumerate(layer["primes"]):
                    half = (rows + 1) // 2
                    for part, begin in enumerate(range(0, rows, half)):
                        count = min(half, rows - begin)
                        group = f"weights/host/layer-{layer['layer']:02}/head-{head:02}-part-{part:02}.safetensors"
                        for logical, source in sources.items():
                            dtype, width = source["dtype"], source["shape"][1]
                            name = f"{logical}.head_{head:02}.part_{part:02}"
                            self.index["row_shards"][logical].append(name)
                            self.add(
                                group,
                                name,
                                dict(
                                    dtype=dtype,
                                    shape=[count, width],
                                    bytes=count * width * SIZES[dtype],
                                    source=dict(
                                        file=source["file"],
                                        offset=source["offset"]
                                        + (first + begin) * width * SIZES[dtype],
                                    ),
                                ),
                            )
                    first += rows
            if first != layer["rows"]:
                raise ValueError("Hash-head partitions do not cover the original table")
        for group, entries in self.groups.items():
            size = sum(e["bytes"] for _, e in entries)
            if size > 4 << 30:
                raise ValueError(f"Shard too large: {group} {size}")
        return self

    def add(self, group, name, entry):
        if name in self.index["weight_map"]:
            raise ValueError("Duplicate exported tensor")
        self.index["weight_map"][name] = group
        self.groups[group].append((name, entry))

    def array(self, name):
        source = self.tensor(name)
        return np.memmap(
            source["file"],
            mode="r",
            dtype=NUMPY[source["dtype"]],
            offset=source["offset"],
            shape=tuple(source["shape"]),
        )

    def chunks(self, entry):
        source = entry["source"]
        if "nz_source" in source:
            yield from transposed_nz_file_chunks(
                source["nz_source"], source["offset"], *source["geometry"]
            )
            return
        if "row_record" in source:
            yield from record_chunks(**source["row_record"])
            return
        if "file" in source:
            with open(source["file"], "rb") as f:
                f.seek(source["offset"])
                remaining = entry["bytes"]
                while remaining:
                    data = f.read(min(CHUNK, remaining))
                    if not data:
                        raise EOFError("Truncated source tensor")
                    yield data
                    remaining -= len(data)
            return
        w = source["transform"]
        if w["kind"] == "matrix":
            original = self.array(w["name"])
            original = original.reshape(original.shape[0], -1)
            dtype = entry["dtype"]
            step = max(1, CHUNK // (w["columns"] * SIZES[dtype]))
            for first in range(0, w["rows"], step):
                a = original[
                    w["first_row"] + first : w["first_row"] + min(w["rows"], first + step),
                    w["first_column"] : w["first_column"] + w["columns"],
                ]
                if w.get("float32") and w["dtype"] == "bf16":
                    a = a.astype("<u4") << 16
                yield memoryview(np.ascontiguousarray(a)).cast("B")
        elif w["kind"] == "pad":
            source = self.tensor(w["name"])
            n = math.prod(source["shape"]) * SIZES[source["dtype"]]
            if n > entry["bytes"] or source["dtype"] != "F32":
                raise ValueError("Invalid padded weight")
            yield from self.chunks(dict(source=source, bytes=n))
            for first in range(n, entry["bytes"], CHUNK):
                yield bytes(min(CHUNK, entry["bytes"] - first))
        elif w["kind"] == "channel_product":
            a, b = self.array(w["left"]), self.array(w["right"])
            if a.dtype != np.dtype("<u2") or b.shape != a.shape:
                raise ValueError("Invalid BF16 channel product")
            fa = (a.astype("<u4") << 16).view("<f4")
            fb = (b.astype("<u4") << 16).view("<f4")
            yield memoryview(np.ascontiguousarray(fa * fb)).cast("B")
        else:
            raise ValueError("Unknown offline transform")

    def write_shard(self, group):
        path = self.output / group
        path.parent.mkdir(parents=True, exist_ok=True)
        header = {"__metadata__": {"format": "orinfer-a2-physical-v1"}}
        offset = 0
        for name, entry in self.groups[group]:
            header[name] = dict(
                dtype=entry["dtype"],
                shape=entry["shape"],
                data_offsets=[offset, offset + entry["bytes"]],
            )
            offset += entry["bytes"]
        raw = json.dumps(header, separators=(",", ":")).encode()
        raw += b" " * ((-len(raw)) % 8)
        digest = hashlib.sha256()
        tensor_hashes = {}
        with path.with_suffix(".partial").open("wb") as out:
            prefix = struct.pack("<Q", len(raw)) + raw
            out.write(prefix)
            digest.update(prefix)
            for name, entry in self.groups[group]:
                count = 0
                tensor_hash = hashlib.sha256()
                for chunk in self.chunks(entry):
                    out.write(chunk)
                    digest.update(chunk)
                    tensor_hash.update(chunk)
                    count += len(chunk)
                if count != entry["bytes"]:
                    raise ValueError("Exported tensor extent differs")
                tensor_hashes[name] = tensor_hash.hexdigest()
            out.flush()
            os.fsync(out.fileno())
        path.with_suffix(".partial").replace(path)
        local_header(path)
        return group, dict(
            bytes=path.stat().st_size, sha256=digest.hexdigest(), tensors=tensor_hashes
        )

    def export(self, workers):
        if self.execution_output is None:
            raise ValueError("Export requires a separate execution output directory")
        self.output.mkdir(parents=True, exist_ok=False)
        atomic_json(
            self.output / "export-state.json", dict(phase="writing", files=len(self.groups))
        )
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for group, record in pool.map(self.write_shard, self.groups):
                self.manifest["files"][group] = record
                print(
                    json.dumps(
                        dict(
                            file=group,
                            bytes=record["bytes"],
                            completed=len(self.manifest["files"]),
                            total=len(self.groups),
                        )
                    ),
                    flush=True,
                )
                atomic_json(
                    self.output / "export-state.json",
                    dict(
                        phase="writing",
                        completed=len(self.manifest["files"]),
                        files=len(self.groups),
                    ),
                )
        self.finish()

    def finish(self):
        for name in [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "generation_config.json",
            "chat_template.jinja",
        ]:
            shutil.copyfile(self.source / name, self.output / name)
        for name in ["LICENSE", "README.md"]:
            path = self.source / name
            if name == "README.md" and (self.source / "SOURCE-README.md").exists():
                path = self.source / "SOURCE-README.md"
            if not path.exists():
                path = self.attribution / name
            if path.exists():
                shutil.copyfile(
                    path, self.output / ("SOURCE-" + name if name == "README.md" else name)
                )
        index = self.output / "model.safetensors.index.json"
        atomic_json(index, self.index)
        self.model["index"] = dict(file=index.name, sha256=file_hash(index))
        target = self.execution_output / ".building"
        target.mkdir(parents=True)
        from orinfer_build.models.deepseek_v41.publication import prepare_execution

        prepare_execution(self.model, self.package, target)
        digest = file_hash(target / "package.json")
        target.rename(target.with_name(digest))
        desc = json.loads((self.source / "cache/model.json").read_text())
        desc["execution_package"] = digest
        (self.output / "cache").mkdir(exist_ok=True)
        atomic_json(self.output / "cache/model.json", desc)
        atomic_json(self.output / "release-manifest.json", self.manifest)
        atomic_json(
            self.output / "export-state.json",
            dict(
                phase="exported_pending_validation",
                files=len(self.groups),
                bytes=sum(v["bytes"] for v in self.manifest["files"].values()),
            ),
        )
        repo = ROOT
        licenses = self.execution_output / digest / "licenses"
        licenses.mkdir(exist_ok=True)
        for source, name in [
            ("LICENSE", "orinfer-LICENSE"),
            ("COPYING", "orinfer-COPYING"),
            ("kernels/ascend/LICENSE.tilelang", "TileLang-LICENSE"),
            ("python/orinfer_build/models/deepseek_v41/LICENSE", "DeepSeek-adapter-LICENSE"),
        ]:
            shutil.copyfile(repo / source, licenses / name)
        for root in [self.output, self.execution_output / digest]:
            for path in root.rglob("*.json"):
                path.chmod(0o644)


class TensorRelease(Export):
    """Replace offline tensor payloads, sharing semantic and experimental exports."""

    def __init__(self, *args, overlay=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.overlay = overlay
        old = json.loads((self.source / "release-manifest.json").read_text())
        self.old_records = old["files"]
        package = self.manifest["source_package"]
        self.manifest.update(deepcopy(old))
        self.manifest.update(
            source_package=package, files=deepcopy(self.old_records) if overlay else {}
        )
        self.index = deepcopy(self.old_index)
        self.transforms, self.payloads, self.payload_dtypes, self.override_groups = {}, {}, {}, {}
        self.rewrite_groups = set()

    def replace_tensor(self, definition, values, *, suffix, dtype=None, group):
        original = definition["prepared_tensor"]["name"]
        replacement = original + suffix
        if original in self.transforms:
            if self.transforms[original] != replacement:
                raise ValueError("Inconsistent replacement of a shared weight")
            previous = self.payloads[replacement]
            if (
                previous.shape != values.shape
                or previous.dtype != values.dtype
                or (previous is not values and not np.array_equal(previous, values))
            ):
                raise ValueError("Shared weight replacement values differ")
        else:
            if replacement in self.index["weight_map"]:
                raise ValueError("Replacement tensor name already exists")
            self.transforms[original] = replacement
            self.payloads[replacement] = np.ascontiguousarray(values)
            self.payload_dtypes[replacement] = dtype or definition["prepared_tensor"]["dtype"]
            self.override_groups[replacement] = group
        definition.update(
            bytes=values.nbytes,
            prepared_tensor=dict(
                name=replacement, dtype=self.payload_dtypes[replacement], shape=list(values.shape)
            ),
        )

    def prepare_overrides(self):
        if self.overlay:
            for name, values in self.payloads.items():
                group = self.override_groups[name]
                self.index["weight_map"][name] = group
                self.groups[group].append(
                    (
                        name,
                        dict(
                            dtype=self.payload_dtypes[name],
                            shape=list(values.shape),
                            bytes=values.nbytes,
                            source=dict(replacement=name),
                        ),
                    )
                )
        else:
            for group, record in self.old_records.items():
                path = self.source / group
                start, header = local_header(path)
                for name in sorted(record["tensors"], key=lambda n: header[n]["data_offsets"][0]):
                    spec = header[name]
                    entry = dict(
                        dtype=spec["dtype"],
                        shape=spec["shape"],
                        bytes=spec["data_offsets"][1] - spec["data_offsets"][0],
                        source=dict(file=str(path), offset=start + spec["data_offsets"][0]),
                    )
                    if name in self.transforms:
                        replacement = self.transforms[name]
                        values = self.payloads[replacement]
                        self.index["weight_map"][replacement] = self.index["weight_map"].pop(name)
                        entry.update(
                            dtype=self.payload_dtypes[replacement],
                            shape=list(values.shape),
                            bytes=values.nbytes,
                            source=dict(replacement=replacement),
                        )
                        name = replacement
                        self.rewrite_groups.add(group)
                    self.groups[group].append((name, entry))
            for name, values in self.payloads.items():
                if name in self.index["weight_map"]:
                    continue
                group = self.override_groups[name]
                self.index["weight_map"][name] = group
                self.groups[group].append(
                    (
                        name,
                        dict(
                            dtype=self.payload_dtypes[name],
                            shape=list(values.shape),
                            bytes=values.nbytes,
                            source=dict(replacement=name),
                        ),
                    )
                )
                self.rewrite_groups.add(group)

    def add_tensor(self, name, values, *, dtype, group):
        if name in self.index["weight_map"] or name in self.payloads:
            raise ValueError("Additional tensor name already exists")
        self.payloads[name] = np.ascontiguousarray(values)
        self.payload_dtypes[name] = dtype
        self.override_groups[name] = group

    def chunks(self, entry):
        name = entry["source"].get("replacement")
        if name is None:
            yield from super().chunks(entry)
        else:
            raw = memoryview(self.payloads[name]).cast("B")
            for first in range(0, len(raw), CHUNK):
                yield raw[first : first + CHUNK]

    def write_shard(self, group):
        if self.overlay or group in self.rewrite_groups:
            return super().write_shard(group)
        target = self.output / group
        target.parent.mkdir(parents=True, exist_ok=True)
        target.hardlink_to(self.source / group)
        return group, deepcopy(self.old_records[group])

    def export(self, workers):
        if not self.overlay:
            return super().export(workers)
        self.output.mkdir(parents=True, exist_ok=False)
        for group in self.old_records:
            target = self.output / group
            target.parent.mkdir(parents=True, exist_ok=True)
            target.hardlink_to(self.source / group)
        for group in self.groups:
            name, record = self.write_shard(group)
            self.manifest["files"][name] = record
        return self.finish()


def verify_release(output, workers=4, execution_output=None):
    output = Path(output)
    manifest = json.loads((output / "release-manifest.json").read_text())
    index = json.loads((output / "model.safetensors.index.json").read_text())

    def check(item):
        name, record = item
        from orinfer_build.weights.safetensors_source import relative_name

        relative_name(name)
        path = output / name
        if path.stat().st_size != record["bytes"] or file_hash(path) != record["sha256"]:
            raise ValueError(f"Release checksum differs: {name}")
        _, header = local_header(path)
        tensors = {n: t for n, t in header.items() if n != "__metadata__"}
        if tensors.keys() != record["tensors"].keys():
            raise ValueError("Release tensor inventory differs")
        return name, tensors

    with ThreadPoolExecutor(max_workers=workers) as pool:
        headers = dict(pool.map(check, manifest["files"].items()))
    observed = {name: file for file, tensors in headers.items() for name in tensors}
    if observed != index["weight_map"]:
        raise ValueError("Release index does not match its containers")
    actual_files = {str(p.relative_to(output)) for p in (output / "weights").rglob("*.safetensors")}
    if actual_files != headers.keys():
        raise ValueError("Unexpected or missing release shards")
    if (output / "cache/packages").exists() or any(output.rglob("*.so")):
        raise ValueError("Weight release must not bundle execution plans or operators")
    checked = None
    if execution_output is not None:
        descriptor = json.loads((output / "cache/model.json").read_text())
        package = Path(execution_output) / descriptor["execution_package"]
        plan = json.loads((package / "model.json").read_text())
        if file_hash(output / plan["index"]["file"]) != plan["index"]["sha256"]:
            raise ValueError("Release plan index identity differs")
        checked = 0
        for rank in plan["ranks"]:
            for binding in rank["buffers"]:
                if any(k in binding for k in ["weight", "prepared_weight", "data"]):
                    raise ValueError("Release retains an online weight transformation")
                if "prepared_tensor" not in binding:
                    continue
                tensor = binding["prepared_tensor"]
                info = headers[index["weight_map"][tensor["name"]]][tensor["name"]]
                if (
                    info["dtype"] != tensor["dtype"]
                    or info["shape"] != tensor["shape"]
                    or info["data_offsets"][1] - info["data_offsets"][0] != binding["bytes"]
                ):
                    raise ValueError("Release binding dtype/shape/extent differs")
                checked += 1
    logical_shapes = manifest["logical_shapes"]
    for logical, parts in index["row_shards"].items():
        if not parts or len(set(parts)) != len(parts):
            raise ValueError("Invalid host-row shard list")
        specs = [headers[index["weight_map"][n]][n] for n in parts]
        shape = logical_shapes[logical]
        expected_dtype = "I8" if logical.endswith(".weight") else "F32"
        if sum(s["shape"][0] for s in specs) != shape[0] or any(
            s["shape"][1:] != shape[1:] or s["dtype"] != expected_dtype for s in specs
        ):
            raise ValueError("Host-row shards do not cover the logical table")
    for prefix, records in index.get("row_records", {}).items():
        parts = records["parts"]
        if records["format"] != "i8-f32-g32-v1" or not parts or len(set(parts)) != len(parts):
            raise ValueError("Invalid host-row records")
        values = logical_shapes[prefix + ".weight"]
        scales = logical_shapes[prefix + ".scale"]
        if values[1] % 32 or scales != [values[0], values[1] // 32]:
            raise ValueError("Host record logical value/scale shapes differ")
        if any(prefix + suffix in index["row_shards"] for suffix in (".weight", ".scale")):
            raise ValueError("Ambiguous split and packed host records")
        specs = [headers[index["weight_map"][n]][n] for n in parts]
        if sum(s["shape"][0] for s in specs) != values[0] or any(
            s["dtype"] != "U8" or s["shape"][1:] != [values[1] + scales[1] * 4] for s in specs
        ):
            raise ValueError("Host records do not cover their logical table")
    proof = dict(
        phase="verified_pending_load",
        files=len(headers),
        bytes=sum(r["bytes"] for r in manifest["files"].values()),
        prepared_bindings=checked,
        full_file_checksums=True,
        container_dtype_shape_and_row_coverage=True,
    )
    return proof


class _IncrementalRelease(Export):
    def write_shard(self, group):
        if group in self.rewrite_groups:
            return super().write_shard(group)
        target = self.output / group
        target.parent.mkdir(parents=True, exist_ok=True)
        target.hardlink_to(self.source / group)
        return group, deepcopy(self.old_records[group])


class PrepackRelease(_IncrementalRelease):
    """Prepare transposed-NZ W4 shards from an existing ND release, hardlinking unchanged shards."""

    def plan(self):
        self.index = deepcopy(self.old_index)
        self.index["metadata"]["w4_layout"] = "transposed_nz"
        old_manifest = json.loads((self.source / "release-manifest.json").read_text())
        self.old_records = old_manifest["files"]
        self.manifest["logical_shapes"] = old_manifest["logical_shapes"]
        self.manifest["w4_layout"] = "transposed_nz"
        self.rewrite_groups = set()
        transforms = _plan_w4_nz_transforms(self.model, lambda b: b["prepared_tensor"]["name"])
        for group, record in self.old_records.items():
            path = self.source / group
            if path not in self.headers:
                self.headers[path] = local_header(path)
            start, header = self.headers[path]
            names = sorted(record["tensors"], key=lambda n: header[n]["data_offsets"][0])
            for name in names:
                spec = header[name]
                offset = start + spec["data_offsets"][0]
                source = dict(file=str(path), offset=offset)
                shape = spec["shape"]
                if name in transforms:
                    e, k, n = transforms[name]
                    if spec["dtype"] != "I8" or shape != [e, k, n // 2]:
                        raise ValueError("Original packed W4 tensor contract differs")
                    source = dict(nz_source=str(path), offset=offset, geometry=[e, k, n])
                    shape = [e, k // 64, n // 16, 16, 32]
                    self.rewrite_groups.add(group)
                self.groups[group].append(
                    (
                        name,
                        dict(
                            dtype=spec["dtype"],
                            shape=shape,
                            bytes=spec["data_offsets"][1] - spec["data_offsets"][0],
                            source=source,
                        ),
                    )
                )
        if len(transforms) != 640 or len(self.rewrite_groups) != 80:
            raise ValueError("Expected complete 40-layer TP8 up/down W4 banks in 80 GPU shards")
        return self


class HostRecordRelease(_IncrementalRelease):
    """Interleave split host INT8/FP32 shards into packed row records, hardlinking GPU shards."""

    def plan(self):
        self.index = deepcopy(self.old_index)
        self.index["metadata"]["host_row_layout"] = "i8-f32-g32-v1"
        if self.index.get("row_records"):
            raise ValueError("Host records are already packed")
        self.index["row_records"] = {}
        old = json.loads((self.source / "release-manifest.json").read_text())
        self.manifest.update({k: v for k, v in old.items() if k != "files"})
        self.manifest["source_package"] = json.loads(
            (self.source / "cache/model.json").read_text()
        )["execution_package"]
        self.manifest["host_row_layout"] = "i8-f32-g32-v1"
        self.old_records = old["files"]
        self.rewrite_groups = set()
        replacements = {}
        removed = set()
        for logical, values in list(self.index["row_shards"].items()):
            if not logical.endswith(".weight"):
                continue
            prefix = logical.removesuffix(".weight")
            scales = self.index["row_shards"].get(prefix + ".scale")
            if not scales or len(values) != len(scales):
                raise ValueError("Host value/scale shard inventory differs")
            parts = []
            for i, (value, scale) in enumerate(zip(values, scales, strict=True)):
                group = self.index["weight_map"][value]
                if group != self.index["weight_map"][scale]:
                    raise ValueError("Host value/scale pair must share one semantic shard")
                path = self.source / group
                if path not in self.headers:
                    self.headers[path] = local_header(path)
                start, header = self.headers[path]
                w, s = header[value], header[scale]
                rows, width = w["shape"]
                if (
                    w["dtype"] != "I8"
                    or s["dtype"] != "F32"
                    or s["shape"] != [rows, width // 32]
                    or width % 32
                ):
                    raise ValueError("Host value/scale dtype or geometry differs")
                name = prefix + f".records.part{i:04}"
                entry = dict(
                    dtype="U8",
                    shape=[rows, width + width // 32 * 4],
                    bytes=rows * (width + width // 32 * 4),
                    source=dict(
                        row_record=dict(
                            weight=[str(path), start + w["data_offsets"][0]],
                            scale=[str(path), start + s["data_offsets"][0]],
                            rows=rows,
                            width=width,
                        )
                    ),
                )
                replacements[value] = (name, entry)
                removed.add(scale)
                self.index["weight_map"].pop(value)
                self.index["weight_map"].pop(scale)
                self.index["weight_map"][name] = group
                self.rewrite_groups.add(group)
                parts.append(name)
            self.index["row_records"][prefix] = dict(format="i8-f32-g32-v1", parts=parts)
            self.index["row_shards"].pop(logical)
            self.index["row_shards"].pop(prefix + ".scale")
        if not replacements:
            raise ValueError("Missing split host row tables")
        for group, record in self.old_records.items():
            path = self.source / group
            if path not in self.headers:
                self.headers[path] = local_header(path)
            start, header = self.headers[path]
            for name in sorted(record["tensors"], key=lambda n: header[n]["data_offsets"][0]):
                if name in removed:
                    continue
                if name in replacements:
                    self.groups[group].append(replacements[name])
                else:
                    spec = header[name]
                    self.groups[group].append(
                        (
                            name,
                            dict(
                                dtype=spec["dtype"],
                                shape=spec["shape"],
                                bytes=spec["data_offsets"][1] - spec["data_offsets"][0],
                                source=dict(file=str(path), offset=start + spec["data_offsets"][0]),
                            ),
                        )
                    )
        return self


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("export", "prepack", "pack-host"), default="export")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--source-execution", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--execution-output", type=Path, help="Separate execution-package cache directory"
    )
    parser.add_argument(
        "--attribution", type=Path, help="Original checkpoint license and documentation"
    )
    parser.add_argument("--w4-layout", choices=("nd", "transposed_nz"), default="nd")
    parser.add_argument("--pack-host", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        print(
            json.dumps(verify_release(args.output, args.workers, args.execution_output)), flush=True
        )
        return
    if args.source is None:
        parser.error("Export requires --source")
    if args.mode in ("prepack", "pack-host"):
        if args.source_execution is None or args.execution_output is None:
            parser.error(f"Mode {args.mode} requires --source-execution and --execution-output")
        cls = PrepackRelease if args.mode == "prepack" else HostRecordRelease
        job = cls(
            args.source,
            args.output,
            execution_output=args.execution_output,
            source_execution=args.source_execution,
        ).plan()
        print(
            json.dumps(
                dict(
                    shards=len(job.groups),
                    rewritten_shards=len(job.rewrite_groups),
                    reused_shards=len(job.groups) - len(job.rewrite_groups),
                )
            ),
            flush=True,
        )
        if not args.plan_only:
            job.export(args.workers)
        return
    exporter = Export(
        args.source,
        args.output,
        args.attribution,
        args.execution_output,
        args.source_execution,
        w4_layout=args.w4_layout,
        pack_host=args.pack_host,
    ).plan()
    print(
        json.dumps(
            dict(
                files=len(exporter.groups),
                bytes=sum(e["bytes"] for xs in exporter.groups.values() for _, e in xs),
                shards={k: sum(e["bytes"] for _, e in v) for k, v in exporter.groups.items()},
            )
        ),
        flush=True,
    )
    if not args.plan_only:
        exporter.export(args.workers)


if __name__ == "__main__":
    main()
