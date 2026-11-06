"""Export selected routed up banks as INT8, preserving the published W3 basis."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
import multiprocessing
from pathlib import Path
import re

import numpy as np

from tools.model.deepseek_v41.export_release import Export
from tools.model.safetensors_source import local_header


def bf16_scale(values):
    """Round positive row scales before choosing the signed INT8 codes."""
    values = np.asarray(values, dtype="<f4")
    if not np.isfinite(values).all() or np.any(values <= 0) or np.any(values > 1e30):
        raise ValueError("Expert scales must be finite and positive within BF16 range")
    bits = values.view("<u4")
    encoded = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype("<u2")
    decoded = (encoded.astype("<u4") << 16).view("<f4")
    if np.any(decoded <= 0):
        raise ValueError("Expert scales underflow the BF16 representation")
    return encoded, decoded


def block_rotation(integer_source):
    """Require the checkpoint's complete repeated orthogonal block-32 basis."""
    file = Path(integer_source) / "optional/quarot.safetensors"
    start, header = local_header(file)
    tensor = header["global_rotation"]
    if tensor["dtype"] != "F32" or tensor["shape"] != [5120, 5120]:
        raise ValueError("Expected DSv4.1 global rotation")
    full = np.memmap(
        file, dtype="<f4", mode="r", offset=start + tensor["data_offsets"][0], shape=(5120, 5120)
    )
    rotation = np.array(full[:32, :32])
    # The checkpoint stores a rounded basis (its Gram diagonal is about
    # 1.000214). Preserve those coefficients rather than renormalizing it.
    if not np.allclose(rotation.T @ rotation, np.eye(32), rtol=0, atol=1e-3):
        raise ValueError("Global rotation block must be approximately orthogonal")
    for first in range(0, 5120, 32):
        rows = full[first : first + 32]
        if (
            not np.array_equal(rows[:, first : first + 32], rotation)
            or np.count_nonzero(rows[:, :first])
            or np.count_nonzero(rows[:, first + 32 :])
        ):
            raise ValueError("Global rotation requires repeated, isolated block-32 sections")
    return rotation


def prepare_bank(arguments):
    """Stream one expert at a time; retain the W3 codes from its AWQ basis."""
    from tools.model.deepseek_v41.official import Checkpoint
    from tools.model.deepseek_v41.prepare import Checkpoint as IntegerCheckpoint

    official, integer, output, layer, rank, rotation = arguments
    directory = Path(output) / f"layer{layer}" / f"rank{rank}"
    directory.mkdir(parents=True, exist_ok=False)
    names = [
        f"layers.{layer}.ffn.experts.{e}.w3.{suffix}"
        for e in range(rank * 48, (rank + 1) * 48)
        for suffix in ("weight", "weight_scale")
    ]
    reference = Checkpoint(official)
    baseline = IntegerCheckpoint(integer, "quant_model_weights.safetensors.index.json", names)
    try:
        norm = reference.read(f"layers.{layer}.ffn_norm.weight")
        weights = np.memmap(directory / "up-i8.bin", mode="w+", dtype="i1", shape=(48, 5120, 4608))
        scales = np.empty((48, 4608), dtype="<u2")
        for local in range(48):
            prefix = f"layers.{layer}.ffn.experts.{rank * 48 + local}"
            matrix = reference.matrix(prefix + ".w1.weight")
            if matrix.shape != (2304, 5120) or norm.shape != (5120,):
                raise ValueError("Official expert geometry differs")
            canonical = ((matrix * norm).reshape(2304, 160, 32) @ rotation).reshape(2304, 5120)
            encoded, scale = bf16_scale(np.maximum(abs(canonical).max(1) / 127, 1e-30))
            weights[local, :, :2304] = (
                np.rint(canonical / scale[:, None]).clip(-127, 127).astype("i1").T
            )
            scales[local, :2304] = encoded
            packed = baseline.read(prefix + ".w3.weight", "I8", [1152, 5120])
            low, high = (packed & 15).astype("i1"), (packed >> 4).astype("i1")
            codes = np.empty((2304, 5120), dtype="i1")
            codes[0::2] = np.where(low >= 8, low - 16, low)
            codes[1::2] = np.where(high >= 8, high - 16, high)
            weights[local, :, 2304:] = codes.T
            scale = baseline.read(prefix + ".w3.weight_scale", "F32", [2304, 1])[:, 0]
            scales[local, 2304:] = bf16_scale(scale)[0]
        weights.flush()
        scales.tofile(directory / "scale-bf16.bin")
        metadata = dict(
            schema=1,
            layer=layer,
            rank=rank,
            geometry=[48, 5120, 4608],
            official=str(official),
            integer_source=str(integer),
            w1_basis="official-norm-fold-block32",
            w3_codes="published-int4",
            scale="bf16-rne",
        )
        (directory / "preparation.json").write_text(json.dumps(metadata, indent=2))
        return metadata
    finally:
        baseline.close()
        reference.close()


class ExpertRelease(Export):
    """Consume bounded offline banks; retain routing, activation and state programs."""

    def __init__(self, *args, prepared, layers, **kwargs):
        super().__init__(*args, **kwargs)
        self.prepared = Path(prepared)
        self.layers = sorted(set(layers))
        if not self.layers or any(
            type(layer) is not int or not 0 <= layer < 40 for layer in layers
        ):
            raise ValueError("Select target expert layers within 0..39")

    def plan(self):
        self.index = deepcopy(self.old_index)
        old = json.loads((self.source / "release-manifest.json").read_text())
        self.old_records = old["files"]
        source_package = self.manifest["source_package"]
        self.manifest.update(deepcopy(old))
        self.manifest.update(source_package=source_package, files={})
        self.transforms, self.rewrite_groups, calls = {}, set(), []
        for ri, rank in enumerate(self.model["ranks"]):
            # A4 replay and A8 packets share physical workspaces. An INT8 bank
            # consumes the existing A8 gather/GMM/SwiGLU subgraph in both.
            ordinary = {}
            for packet in rank["packets"]:
                steps = packet["body"]
                for i, step in enumerate(steps):
                    if step["op"] == "w4" and step.get("name", "").endswith(".up"):
                        ordinary[(step["name"], packet["rows"])] = deepcopy(steps[i - 1 : i + 2])
            for packet in rank["packets"]:
                steps = packet["body"]
                for i, step in enumerate(steps):
                    match = re.search(r"\.L(\d+)\.up$", step.get("name", ""))
                    if step["op"] != "w4_a4" or not match or int(match[1]) not in self.layers:
                        continue
                    trio = ordinary.get((step["name"], packet["rows"]))
                    if trio is None or i == 0 or i + 1 == len(steps):
                        raise ValueError("Promoted A4 bank requires its matching A8 up subgraph")
                    gather, finish = steps[i - 1], steps[i + 1]
                    specifications = [
                        json.loads((self.package / s["manifest"]).read_text())
                        for s in (gather, finish)
                    ]
                    if (
                        [s["op"] for s in specifications]
                        != ["expert.gather_split", "expert.finish"]
                        or gather["buffers"][:3] != trio[0]["buffers"][:3]
                        or gather["buffers"][3] != trio[1]["groups"]
                        or finish["buffers"][-2:] != trio[2]["buffers"][1:3]
                        or step["rows"] != 2 * trio[1]["rows"]
                    ):
                        raise ValueError("A4 and A8 expert subgraphs have different row contracts")
                    steps[i - 1 : i + 2] = trio
            definitions = {b["name"]: b for b in rank["buffers"]}
            originals = {
                name: b["prepared_tensor"]["name"]
                for name, b in definitions.items()
                if "prepared_tensor" in b
            }
            found = set()
            for pi, packet in enumerate(rank["packets"]):
                for program in ("body", "head"):
                    for si, step in enumerate(packet[program]):
                        match = re.search(r"\.L(\d+)\.up$", step.get("name", ""))
                        if step["op"] != "w4" or not match:
                            continue
                        layer = int(match[1])
                        if layer not in self.layers:
                            continue
                        e, k, n = step["experts"], step["input"], step["output"]
                        if (len(self.model["ranks"]), e, k, n) != (8, 48, 5120, 4608):
                            raise ValueError(
                                "Prepared expert conversion requires DSv4.1 TP8 up banks"
                            )
                        files = self.prepared / f"layer{layer}" / f"rank{ri}"
                        for slot, suffix, dtype, shape, size, filename in [
                            (1, ".w1_int8_up_codes", "I8", [e, k, n], e * k * n, "up-i8.bin"),
                            (2, ".bf16_scale", "BF16", [e, n], e * n * 2, "scale-bf16.bin"),
                        ]:
                            bank = definitions[step["buffers"][slot]]
                            original = originals[bank["name"]]
                            if original not in self.transforms:
                                original_shape = (
                                    (
                                        [e, k // 64, n // 16, 16, 32]
                                        if step.get("weight_layout") == "transposed_nz"
                                        else [e, k, n // 2]
                                    )
                                    if slot == 1
                                    else [e, 1, n]
                                )
                                original_dtype = "I8" if slot == 1 else "U64"
                                source_tensor = self.tensor(original)
                                if (source_tensor["shape"], source_tensor["dtype"]) != (
                                    original_shape,
                                    original_dtype,
                                ):
                                    raise ValueError(
                                        "Original expert physical bank contract differs"
                                    )
                                file = files / filename
                                metadata = json.loads((files / "preparation.json").read_text())
                                expected = dict(
                                    schema=1,
                                    layer=layer,
                                    rank=ri,
                                    geometry=[e, k, n],
                                    w1_basis="official-norm-fold-block32",
                                    w3_codes="published-int4",
                                    scale="bf16-rne",
                                )
                                if any(
                                    metadata.get(key) != value for key, value in expected.items()
                                ):
                                    raise ValueError("Prepared expert precision contract differs")
                                if file.stat().st_size != size:
                                    raise ValueError("Prepared INT8 expert bank extent differs")
                                self.transforms[original] = dict(
                                    name=original + suffix,
                                    dtype=dtype,
                                    shape=shape,
                                    bytes=size,
                                    source=dict(file=str(file), offset=0),
                                )
                            replacement = self.transforms[original]
                            bank.update(
                                bytes=size,
                                prepared_tensor={
                                    key: replacement[key] for key in ("name", "dtype", "shape")
                                },
                            )
                        step.update(op="w8", weight_layout="nd")
                        found.add(layer)
                        calls.append((ri, pi, program, si))
            if found != set(self.layers):
                raise ValueError("Every rank must expose the selected expert layers")
        # A changed physical bank must have no unconverted consumer or view.
        for ri, rank in enumerate(self.model["ranks"]):
            changed = {
                b["name"]
                for b in rank["buffers"]
                if b.get("prepared_tensor", {}).get("name")
                in {t["name"] for t in self.transforms.values()}
            }
            if any(b.get("view", {}).get("parent") in changed for b in rank["buffers"]):
                raise ValueError("Promoted expert banks cannot have another buffer view")
            allowed = set(calls)
            for pi, packet in enumerate(rank["packets"]):
                for program in ("body", "head"):
                    for si, step in enumerate(packet[program]):
                        if (
                            changed.intersection(step.get("buffers", []))
                            and (ri, pi, program, si) not in allowed
                        ):
                            raise ValueError("Promoted expert bank has another consumer")
        for group, record in self.old_records.items():
            path = self.source / group
            start, header = local_header(path)
            for name in sorted(record["tensors"], key=lambda key: header[key]["data_offsets"][0]):
                tensor = header[name]
                entry = dict(
                    dtype=tensor["dtype"],
                    shape=tensor["shape"],
                    bytes=tensor["data_offsets"][1] - tensor["data_offsets"][0],
                    source=dict(file=str(path), offset=start + tensor["data_offsets"][0]),
                )
                if name in self.transforms:
                    replacement = deepcopy(self.transforms[name])
                    new_name = replacement.pop("name")
                    self.index["weight_map"][new_name] = self.index["weight_map"].pop(name)
                    self.manifest.get("logical_shapes", {}).pop(name, None)
                    self.manifest.setdefault("logical_shapes", {})[new_name] = replacement["shape"]
                    name, entry = new_name, replacement
                    self.rewrite_groups.add(group)
                self.groups[group].append((name, entry))
        if any(sum(e["bytes"] for _, e in entries) >= 4 << 30 for entries in self.groups.values()):
            raise ValueError("Expert promotion exceeds the four-GiB semantic shard budget")
        self.manifest["expert_precision"] = dict(
            layers=self.layers,
            w1="int8-official-norm-fold-block32-rotation",
            w3="published-int4-codes-expanded-to-int8",
            scale="bf16",
            calls=len(calls),
        )
        return self

    def write_shard(self, group):
        if group in self.rewrite_groups:
            return super().write_shard(group)
        target = self.output / group
        target.parent.mkdir(parents=True, exist_ok=True)
        target.hardlink_to(self.source / group)
        return group, deepcopy(self.old_records[group])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-execution", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execution-output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--official", type=Path)
    parser.add_argument("--integer-source", type=Path)
    args = parser.parse_args()
    args.layers = sorted(set(args.layers))
    if not args.layers or any(not 0 <= layer < 40 for layer in args.layers):
        parser.error("Select target expert layers within 0..39")
    if args.prepare:
        if args.official is None or args.integer_source is None:
            parser.error("--prepare requires --official and --integer-source")
        rotation = block_rotation(args.integer_source)
        jobs = [
            (args.official, args.integer_source, args.prepared, layer, rank, rotation)
            for layer in args.layers
            for rank in range(8)
        ]
        with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            for result in pool.map(prepare_bank, jobs):
                print(json.dumps(result), flush=True)
    job = ExpertRelease(
        args.source,
        args.output,
        source_execution=args.source_execution,
        execution_output=args.execution_output,
        prepared=args.prepared,
        layers=args.layers,
    ).plan()
    print(json.dumps(dict(banks=len(job.transforms), shards=len(job.rewrite_groups))), flush=True)
    job.export(args.workers)


if __name__ == "__main__":
    main()
