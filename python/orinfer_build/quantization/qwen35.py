"""Resume bounded original-BF16 Qwen 27B conversion to our table Q4A8.

Uses the Flash Next scalar Q4 codec, not the two-bit E8P expert encoding.
Original tensors, W8 exceptions, rotation signs and the full table are counted.
This publishes weights; execution-package construction is a separate step.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import time

from orinfer_build.common.gpu import gpu_lock

import numpy as np
from safetensors.numpy import save_file

from orinfer_build.packaging.compact import write_shard
from orinfer_build.weights.safetensors_source import Source, SIZES, local_header
from orinfer_build.common.files import atomic_json, file_hash as digest
from orinfer_build.quantization.flash_next import bf16, equivalent
from orinfer_build.quantization.flash_next_aux import int8_rows
from orinfer_build.quantization.row_q4 import MAGNITUDES


def policy(name, info):
    shape = info["shape"]
    if (
        info["dtype"] != "BF16"
        or len(shape) != 2
        or shape[1] % 128
        or not name.endswith(".weight")
        or "norm" in name
        or name.startswith("model.visual.")
        or min(shape) < 128
    ):
        return "original"
    if name.startswith("mtp.") or name.endswith(
        (
            "embed_tokens.weight",
            "linear_attn.out_proj.weight",
            "self_attn.o_proj.weight",
        )
    ):
        return "int8-row"
    return "q4-row"


def group(name):
    layer = re.match(r"model\.language_model\.layers\.(\d+)\.", name)
    if layer:
        return f"model-language-layer-{int(layer[1]):03d}"
    if name.startswith("model.visual."):
        return "model-vision"
    if name.startswith("mtp."):
        return "model-mtp"
    return "model-shared-" + name.split(".")[-2]


def tasks(source, chunk_bytes):
    queue = []
    for name in sorted(source.weight_map):
        _, _, info = source.tensor(name)
        kind = policy(name, info)
        shape = info["shape"]
        rows = shape[0] if shape else 1
        step = rows if kind == "original" else max(1, chunk_bytes // (2 * shape[1]))
        for first in range(0, rows, step):
            queue.append(
                dict(
                    name=name,
                    kind=kind,
                    shape=shape,
                    dtype=info["dtype"],
                    first=first,
                    rows=min(step, rows - first),
                )
            )
    return queue


def save_original(path, name, info, raw, contract):
    spec = dict(__metadata__=dict(contract=contract))
    spec[name] = dict(dtype=info["dtype"], shape=info["shape"], data_offsets=[0, len(raw)])
    header = json.dumps(spec, separators=(",", ":")).encode()
    header += b" " * (-len(header) % 8)
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(header)))
        f.write(header)
        f.write(raw)


def publish(output, queue, records, signs, contract):
    """Merge chunks by component/layer without retaining duplicate tensor bytes."""
    by_group = {}
    for task in queue:
        by_group.setdefault(group(task["name"]), {}).setdefault(task["name"], []).append(task)
    mapping, sizes, hashes = {}, 0, {}
    for family, tensors in by_group.items():
        entries, layouts = [], {}
        for name, chunks in tensors.items():
            kind, shape = chunks[0]["kind"], chunks[0]["shape"]
            fields = (
                [name]
                if kind == "original"
                else [name + ".packed" if kind == "q4-row" else name + ".int8", name + ".scale"]
            )
            for field in fields:
                segments = []
                for task in chunks:
                    path = output / "chunks" / records[task_key(task)]["file"]
                    begin, h = local_header(path)
                    first, last = h[field]["data_offsets"]
                    segments.append((path, begin + first, last - first))
                dtype = h[field]["dtype"]
                physical = (
                    shape
                    if kind == "original"
                    else [shape[0]]
                    if field.endswith(".scale")
                    else [shape[0], shape[1] // (2 if kind == "q4-row" else 1)]
                )
                entries.append(dict(name=field, dtype=dtype, shape=physical, segments=segments))
                layouts[field] = (
                    "q4-row-packed-little-nibbles"
                    if kind == "q4-row" and field.endswith(".packed")
                    else "native_contiguous"
                )
        filename = family + ".safetensors"
        target = output / filename
        metadata = dict(contract=contract, **{f"orin.layout.{n}": v for n, v in layouts.items()})
        temporary = target.with_suffix(".merge")
        if temporary.exists():
            temporary.unlink()
        write_shard(temporary, entries, metadata)
        if target.exists() and not equivalent(target, temporary):
            raise ValueError("Previously merged weight differs")
        temporary.replace(target)
        hashes[filename] = digest(target)
        mapping.update({e["name"]: filename for e in entries})
        sizes += sum(sum(s[2] for s in e["segments"]) for e in entries)
    common = {f"orinfer.rotation.k{k}.signs": v for k, v in signs.items()}
    common["orinfer.q4.table"] = np.asarray((*MAGNITUDES, *(-x for x in MAGNITUDES)), np.int8)
    save_file(common, str(output / "quantization.safetensors"), metadata=dict(contract=contract))
    mapping.update({n: "quantization.safetensors" for n in common})
    sizes += sum(v.nbytes for v in common.values())
    hashes["quantization.safetensors"] = digest(output / "quantization.safetensors")
    atomic_json(
        output / "model.safetensors.index.json",
        dict(metadata={"total_size": sizes, "orinfer.sha256": hashes}, weight_map=mapping),
    )
    return sizes


def task_key(task):
    return hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--source-dir", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpu-lock", type=Path, default=Path("artifacts/gpu-experiment.lock"))
    p.add_argument("--chunk-mib", type=int, default=32)
    p.add_argument("--max-chunks", type=int)
    p.add_argument("--inventory-only", action="store_true")
    a = p.parse_args()
    if not 1 <= a.chunk_mib <= 64 or (a.max_chunks is not None and a.max_chunks < 1):
        p.error("Invalid chunk size/count")
    a.output.mkdir(parents=True, exist_ok=True)
    origin = json.loads((a.source / "source.json").read_text())
    config = json.loads((a.source / "config.json").read_text())
    if config.get("quantization_config") is not None:
        raise ValueError("Conversion requires the original floating checkpoint")
    text = config.get("text_config", config)
    if config.get("model_type") != "qwen3_5" or any(
        text.get(k) != v
        for k, v in dict(
            hidden_size=5120, intermediate_size=17408, num_hidden_layers=64, vocab_size=248320
        ).items()
    ):
        raise ValueError("Expected original Qwen3.8-27B architecture")
    source = Source(
        a.source / "model.safetensors.index.json",
        directory=a.source_dir,
        repo=origin["repo"],
        revision=origin["revision"],
        cache=a.output / "source-headers",
    )
    implementation = hashlib.sha256()
    for module in ("qwen35.py", "row_q4.py", "row_q4_gpu.py", "vq.py", "flash_next_aux.py"):
        implementation.update(Path(__file__).with_name(module).read_bytes())
    contract = json.dumps(
        dict(
            format="orinfer.qwen35.q4a8.v1",
            source=origin,
            source_index_sha256=digest(a.source / "model.safetensors.index.json"),
            seed=20261002,
            rotation="signed-block128",
            table=list(MAGNITUDES),
            policy="dense-q4-embedding-output-mtp-w8-v1",
            chunk_mib=a.chunk_mib,
            implementation=implementation.hexdigest(),
        ),
        sort_keys=True,
    )
    progress = a.output / "progress.json"
    state = (
        json.loads(progress.read_text())
        if progress.exists()
        else dict(contract=contract, complete=False, chunks={})
    )
    if not state["chunks"] and not state["complete"]:
        state["contract"] = contract
    if state["contract"] != contract:
        raise ValueError("Quantization source/configuration changed")
    if state["complete"]:
        index = json.loads((a.output / "model.safetensors.index.json").read_text())
        for file, expected in index["metadata"]["orinfer.sha256"].items():
            if digest(a.output / file) != expected:
                raise ValueError("Final quantized shard changed")
        shutil.rmtree(a.output / "chunks", ignore_errors=True)
        print("COMPLETE: verified existing output", flush=True)
        return
    queue = tasks(source, a.chunk_mib * 1024**2)
    widths = sorted({t["shape"][1] for t in queue if t["kind"] == "q4-row"})
    signs = {
        k: np.random.default_rng(20261002 + k).choice(np.asarray([-1, 1], np.int8), k)
        for k in widths
    }
    total_parameters = sum(math.prod(source.tensor(n)[2]["shape"]) for n in source.weight_map)
    state.update(
        expected_chunks=len(queue),
        parameters=total_parameters,
        policy_counts={
            k: sum(t["rows"] * math.prod(t["shape"][1:]) for t in queue if t["kind"] == k)
            for k in ("q4-row", "int8-row", "original")
        },
    )
    estimated = sum(
        math.prod(t["shape"]) * (0.5 if t["kind"] == "q4-row" else 1) + 2 * t["shape"][0]
        if t["kind"] != "original"
        else math.prod(t["shape"]) * SIZES[t["dtype"]]
        for t in {t["name"]: t for t in queue}.values()
    )
    state["estimated_weight_bytes"] = math.ceil(estimated)
    atomic_json(progress, state)
    print("INVENTORY", state["policy_counts"], "chunks", len(queue), flush=True)
    if a.inventory_only:
        return
    remaining = estimated * (1 - len(state["chunks"]) / len(queue))
    if shutil.disk_usage(a.output).free < remaining + estimated + 2 * 1024**3:
        raise ValueError("Insufficient space for conversion and final merged weights")
    chunks = a.output / "chunks"
    chunks.mkdir(exist_ok=True)
    completed = 0
    for task in queue:
        key = task_key(task)
        path = chunks / (key + ".safetensors")
        if key in state["chunks"]:
            if digest(path) != state["chunks"][key]["sha256"]:
                raise ValueError("Completed quantized chunk changed")
            continue
        started = time.monotonic()
        name, kind = task["name"], task["kind"]
        filename, begin, info = source.tensor(name)
        raw = (
            source.read(
                filename,
                begin + info["data_offsets"][0],
                info["data_offsets"][1] - info["data_offsets"][0],
            )
            if kind == "original"
            else source.rows(name, task["first"], task["rows"])
        )
        temporary = path.with_suffix(".tmp")
        if kind == "original":
            save_original(temporary, name, info, raw, contract)
        else:
            floating = bf16(raw, (task["rows"], task["shape"][1]))
            if kind == "q4-row":
                with gpu_lock(a.gpu_lock):
                    from orinfer_build.quantization.row_q4_gpu import fit

                    encoded = fit(floating, signs[task["shape"][1]])
                data = {name + ".packed": encoded.packed, name + ".scale": encoded.scales}
            else:
                integer, scale = int8_rows(floating)
                data = {name + ".int8": integer, name + ".scale": scale}
            save_file(data, str(temporary), metadata=dict(contract=contract))
            del floating, data
        source_hash = hashlib.sha256(raw).hexdigest()
        del raw
        local_header(temporary)
        with temporary.open("rb") as f:
            os.fsync(f.fileno())
        if path.exists() and not equivalent(path, temporary):
            raise ValueError("Unrecorded quantized chunk differs")
        temporary.replace(path)
        state["chunks"][key] = dict(
            file=path.name,
            sha256=digest(path),
            seconds=time.monotonic() - started,
            source_sha256=source_hash,
        )
        atomic_json(progress, state)
        completed += 1
        print("CONVERTED", len(state["chunks"]), "/", len(queue), name, task["first"], flush=True)
        if a.max_chunks is not None and completed >= a.max_chunks:
            return
    for file in a.source.iterdir():
        if (
            file.name not in ("source.json", "config.json", "model.safetensors.index.json")
            and file.is_file()
        ):
            shutil.copyfile(file, a.output / file.name)
    config["quantization_config"] = dict(
        quant_method="orinfer_q4_int8",
        version=1,
        rotation="signed-block128",
        table_tensor="orinfer.q4.table",
        scale="fp16-output-row",
        compute_dtype="int8_quality",
        source=origin["repo"],
        source_revision=origin["revision"],
        weight_overrides=state["policy_counts"],
    )
    atomic_json(a.output / "config.json", config)
    state["weight_bytes"] = publish(a.output, queue, state["chunks"], signs, contract)
    state.update(
        complete=True,
        effective_weight_bits=8 * state["weight_bytes"] / total_parameters,
        execution_package_ready=False,
    )
    atomic_json(progress, state)
    shutil.rmtree(chunks)
    print(
        "COMPLETE", state["weight_bytes"], state["effective_weight_bits"], "bits/weight", flush=True
    )


if __name__ == "__main__":
    main()
