"""Promote selected original BF16 matrices while preserving other E8P values."""

import copy
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from orinfer_build.common.files import file_hash, write_json
from orinfer_build.packaging.compact import write_shard
from orinfer_build.packaging.publication import staged_directory, link_or_copy
from orinfer_build.weights.safetensors_source import Source, local_header
from orinfer_build.quantization.flash_next_aux import int8_rows
from orinfer_build.quantization.vq import rotate


def upgrade(args, *, configuration, signature, layers, layer_prefix):
    """Retain existing E8P values and re-read only protected matrices from BF16."""
    original, source = args.source_dir.resolve(strict=True), args.upgrade_from.resolve(strict=True)
    config = json.loads((source / "config.json").read_text())
    original_config = json.loads((original / "config.json").read_text())
    configuration(config, quantized=True)
    configuration(original_config)
    origin = json.loads((original / "source.json").read_text())
    if signature(config) != signature(original_config) or (
        config["quantization_config"].get("source"),
        config["quantization_config"].get("source_revision"),
    ) != (origin["repo"], origin["revision"]):
        raise ValueError("BF16 source differs from the encoded model identity")
    state = json.loads((source / "progress.json").read_text())
    if not state.get("complete"):
        raise ValueError("Source E8P conversion is incomplete")
    index = json.loads((source / "model.safetensors.index.json").read_text())
    for filename in set(index["weight_map"].values()):
        if file_hash(source / filename) != index["metadata"]["orinfer.sha256"][filename]:
            raise ValueError("Encoded source shard checksum changed")
    bf16 = Source(original / "model.safetensors.index.json", directory=original)
    selected = [
        name
        for name in bf16.weight_map
        if any(name.startswith(f"{layer_prefix}{layer}.") for layer in layers)
        and name.endswith(
            (
                "q_proj.weight",
                "k_proj.weight",
                "v_proj.weight",
                "gate_proj.weight",
                "up_proj.weight",
                "down_proj.weight",
            )
        )
    ]
    if len(selected) != len(layers) * 6:
        raise ValueError("Protected matrices are incomplete")
    from safetensors import safe_open

    text = original_config.get("text_config", original_config)
    widths = [text["hidden_size"], text["intermediate_size"]]
    with safe_open(
        str(source / index["weight_map"][f"orinfer.rotation.k{widths[0]}.signs"]), framework="numpy"
    ) as common:
        signs = {k: common.get_tensor(f"orinfer.rotation.k{k}.signs") for k in widths}
    contract = json.dumps(
        dict(
            base_conversion=state["contract"],
            base_index=file_hash(source / "model.safetensors.index.json"),
            source=origin,
            precision="rotated-w8",
            layers=layers,
            implementation=file_hash(Path(__file__)),
        ),
        sort_keys=True,
    )
    with staged_directory(args.output) as out:
        out.mkdir()
        payloads = out / ".payloads"
        payloads.mkdir()
        replacement, statistics = {}, {}
        for number, name in enumerate(sorted(selected)):
            _, _, info = bf16.tensor(name)
            if info["dtype"] != "BF16" or len(info["shape"]) != 2:
                raise ValueError("Protected weights must be original BF16 matrices")
            n, k = info["shape"]
            codes, scales = payloads / f"{number}.codes", payloads / f"{number}.scale"
            digest, sse, norm = hashlib.sha256(), 0.0, 0.0
            with codes.open("wb") as pc, scales.open("wb") as ps:
                for first in range(0, n, args.rows):
                    count = min(args.rows, n - first)
                    raw = bf16.rows(name, first, count)
                    digest.update(raw)
                    values = (
                        (np.frombuffer(raw, "<u2").astype(np.uint32) << 16)
                        .view(np.float32)
                        .reshape(count, k)
                    )
                    reference = rotate(values, signs[k])
                    code, scale = int8_rows(reference)
                    pc.write(code.tobytes())
                    ps.write(scale.tobytes())
                    error = code.astype(np.float32) * scale.astype(np.float32)[:, None] - reference
                    sse += float(np.square(error).sum(dtype=np.float64))
                    norm += float(np.square(reference).sum(dtype=np.float64))
            replacement[name + ".int8"] = dict(
                name=name + ".int8",
                dtype="I8",
                shape=[n, k],
                segments=[(codes, 0, codes.stat().st_size)],
            )
            replacement[name + ".scale"] = dict(
                name=name + ".scale",
                dtype="F16",
                shape=[n],
                segments=[(scales, 0, scales.stat().st_size)],
            )
            statistics[name] = dict(
                name=name,
                shape=[n, k],
                kind="w8-rotated",
                rows=n,
                sse=sse,
                norm=norm,
                source_sha256=digest.hexdigest(),
            )
            print("PROMOTED", name, "relative_mse", sse / max(norm, 1e-30), flush=True)
        mapping, hashes, shards = {}, {}, copy.deepcopy(state["shards"])
        for filename in sorted(set(index["weight_map"].values())):
            path = source / filename
            begin, header = local_header(path)
            affected = [
                name for name in selected if index["weight_map"].get(name + ".packed") == filename
            ]
            if not affected:
                link_or_copy(path, out / filename)
                names = [n for n in header if n != "__metadata__"]
            else:
                entries = []
                metadata = dict(header.get("__metadata__", {}))
                removed = {name + suffix for name in affected for suffix in (".packed", ".scale")}
                for name, info in header.items():
                    if name == "__metadata__" or name in removed:
                        continue
                    first, last = info["data_offsets"]
                    entries.append(
                        dict(
                            name=name,
                            dtype=info["dtype"],
                            shape=info["shape"],
                            segments=[(path, begin + first, last - first)],
                        )
                    )
                for name in affected:
                    metadata.pop("orin.layout." + name + ".packed", None)
                    for suffix in (".int8", ".scale"):
                        entry = replacement[name + suffix]
                        entries.append(entry)
                        metadata["orin.layout." + entry["name"]] = "native_contiguous"
                metadata["contract"] = contract
                write_shard(out / filename, entries, metadata)
                names = [e["name"] for e in entries]
            mapping.update({n: filename for n in names})
            hashes[filename] = file_hash(out / filename)
            if filename in shards:
                shards[filename].update(sha256=hashes[filename], names=names)
        for path in payloads.iterdir():
            path.unlink()
        payloads.rmdir()
        size = sum((out / f).stat().st_size - local_header(out / f)[0] for f in hashes)
        for path in source.iterdir():
            if (
                path.is_file()
                and path.suffix != ".safetensors"
                and path.name
                not in ("config.json", "progress.json", "model.safetensors.index.json")
            ):
                shutil.copyfile(path, out / path.name)
        config["quantization_config"]["rotated_w8_layers"] = list(layers)
        state.update(contract=contract, complete=True, weight_bytes=size, shards=shards)
        state["quantized"].update(statistics)
        write_json(out / "config.json", config)
        write_json(out / "progress.json", state)
        write_json(
            out / "model.safetensors.index.json",
            dict(metadata={"total_size": size, "orinfer.sha256": hashes}, weight_map=mapping),
        )
        print("COMPLETE", size, flush=True)
