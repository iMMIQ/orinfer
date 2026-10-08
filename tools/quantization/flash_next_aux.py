"""Streaming PLE E8P rows and mixed-precision non-expert text weights."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import struct
import time

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

from tools.quantization.embedding_vq import pack
from tools.quantization.flash_next import atomic_json, bf16, digest, equivalent


def policy(name, info):
    if "ngram_embedding." in name:
        return "e8p-embedding"
    if ".experts." in name:
        return "experts"
    if name.startswith("mtp.fc_"):
        return "original"
    if (
        len(info["shape"]) == 2
        and name.endswith(".weight")
        and not any(
            s in name for s in ("hyper_connection", "norm", ".mlp.gate.", ".shared_expert_gate.")
        )
    ):
        return "int8-row"
    return "original"


def int8_rows(raw):
    if raw.ndim != 2 or not np.isfinite(raw).all():
        raise ValueError("Invalid dense floating matrix")
    scale = np.maximum(np.abs(raw).max(1) / 127, 2**-24).astype(np.float16)
    if not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Unrepresentable row scale")
    return np.rint(raw / scale.astype(np.float32)[:, None]).clip(-127, 127).astype(np.int8), scale


def tasks(source, chunk_bytes=32 * 1024**2, *, component="text"):
    if component not in ("text", "mtp"):
        raise ValueError("Invalid Flash component")
    result = []
    for name in sorted(source.weight_map):
        selected = (
            name.startswith("mtp.")
            if component == "mtp"
            else name.startswith("model.language_model.") or name == "lm_head.weight"
        )
        if not selected or ".experts." in name:
            continue
        filename, begin, info = source.tensor(name)
        kind = policy(name, info)
        shape = info["shape"]
        if kind != "original" and info["dtype"] != "BF16":
            raise ValueError("Quantization requires original BF16")
        if kind == "e8p-embedding" and (len(shape) != 2 or shape[1] != 160):
            raise ValueError("Unexpected PLE table geometry")
        if kind == "original":
            ranges = [(0, shape[0] if shape else 1)]
        else:
            rows = max(1, chunk_bytes // (shape[1] * 2))
            ranges = [(start, min(rows, shape[0] - start)) for start in range(0, shape[0], rows)]
        for start, count in ranges:
            identity = hashlib.sha256(f"{name}:{start}".encode()).hexdigest()[:24]
            result.append(
                {
                    "tensor": name,
                    "kind": kind,
                    "source_shape": shape,
                    "dtype": info["dtype"],
                    "first_row": start,
                    "rows": count,
                    "filename": f"aux-{identity}.safetensors",
                }
            )
    return result


def prefetch(source, queue, *, workers=2):
    """Read an ordered, bounded window while the consumer fits/saves weights.

    tasks() has warmed the immutable headers before workers share Source.
    No more than workers source blocks are live, including the yielded block.
    Only selected missing tasks are submitted; resume and partial runs do not
    fetch already completed or unrequested weights.
    """
    if type(workers) is not int or not 1 <= workers <= 2:
        raise ValueError("Auxiliary read workers must be in 1..2")

    def read(task):
        started = time.perf_counter()
        filename, begin, info = source.tensor(task["tensor"])
        if task["kind"] == "original":
            size = info["data_offsets"][1] - info["data_offsets"][0]
            if size > 64 * 1024**2:
                raise ValueError("Unexpected large critical tensor")
            raw = source.read(filename, begin + info["data_offsets"][0], size)
        else:
            raw = source.rows(task["tensor"], task["first_row"], task["rows"])
        return raw, time.perf_counter() - started

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {i: pool.submit(read, task) for i, task in enumerate(queue[:workers])}
        try:
            for i, task in enumerate(queue):
                yield task, pending.pop(i).result()
                next_index = i + workers
                if next_index < len(queue):
                    pending[next_index] = pool.submit(read, queue[next_index])
        finally:
            for future in pending.values():
                future.cancel()


def convert(
    source,
    output,
    encoder,
    *,
    source_contract,
    seed,
    max_chunks=None,
    kind_filter=None,
    component="text",
):
    if max_chunks is not None and (type(max_chunks) is not int or max_chunks < 0):
        raise ValueError("Auxiliary chunk limit must be nonnegative")
    contract = json.dumps(
        {
            "format": "orinfer.flash_next.aux.v1",
            "source_contract": source_contract,
            "seed": seed,
            "basis": "integer-e8p-spread29-v1",
            "embedding_rotation": "paley20-walsh8",
            "dense": "int8-row-rne-f16-scale",
            "critical": "original",
        },
        sort_keys=True,
    )
    state_path = output / "aux-progress.json"
    state = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else {"contract": contract, "shards": {}, "complete": False}
    )
    if state["contract"] != contract:
        raise ValueError("Auxiliary conversion contract changed")
    queue = tasks(source, component=component)
    state["expected_shards"] = len(queue)
    pending = []
    signs = np.random.default_rng(seed).choice(np.array([-1, 1], np.int8), 160)
    for task in queue:
        if kind_filter is not None and task["kind"] != kind_filter:
            continue
        path = output / task["filename"]
        if task["filename"] in state["shards"]:
            if digest(path) != state["shards"][task["filename"]]["sha256"]:
                raise ValueError("Corrupt auxiliary shard")
            with safe_open(path, framework="np") as f:
                if (f.metadata() or {}).get("contract") != contract:
                    raise ValueError("Auxiliary shard contract mismatch")
            continue
        pending.append(task)
    if max_chunks is not None:
        pending = pending[:max_chunks]
    for task, (raw, read_seconds) in prefetch(source, pending):
        path = output / task["filename"]
        started = time.perf_counter()
        name, kind = task["tensor"], task["kind"]
        _, _, info = source.tensor(name)
        if kind == "original":
            # Critical tensors are small: HC, router, normalization, convolution
            # and scalar metadata. Their original bytes/dtype remain exact.
            count = info["data_offsets"][1] - info["data_offsets"][0]
            data = None
        else:
            floating = bf16(raw, (task["rows"], task["source_shape"][1]))
            if kind == "int8-row":
                q, scale = int8_rows(floating)
                data = {"weight": q, "scale": scale}
            else:
                codes, scale = encoder.fit_arrays(floating, signs, rotation="full")
                data = {"rows": pack(codes, scale), "table": encoder.table, "signs": signs}
            del floating
        source_hash = hashlib.sha256(raw).hexdigest()
        meta = {
            "contract": contract,
            "source_tensor": name,
            "kind": kind,
            "source_range_sha256": source_hash,
            "source_shape": json.dumps(info["shape"]),
            "first_row": str(task["first_row"]),
            "rows": str(task["rows"]),
        }
        # A crash after rename but before recording is recovered by pinned,
        # deterministic recomputation and byte-for-byte artifact comparison.
        temporary = path.with_suffix(".tmp")
        if kind == "original":
            prefix = json.dumps(
                {
                    "__metadata__": meta,
                    name: {
                        "dtype": info["dtype"],
                        "shape": info["shape"],
                        "data_offsets": [0, count],
                    },
                },
                sort_keys=True,
            ).encode()
            prefix += b" " * (-len(prefix) % 8)
            with temporary.open("wb") as f:
                f.write(struct.pack("<Q", len(prefix)))
                f.write(prefix)
                f.write(raw)
        else:
            save_file(data, str(temporary), metadata=meta)
        del raw
        with temporary.open("rb") as f:
            os.fsync(f.fileno())
        if path.exists() and not equivalent(path, temporary):
            raise ValueError("Unrecorded auxiliary shard differs")
        temporary.replace(path)
        with safe_open(path, framework="np") as f:
            if f.metadata() != meta:
                raise ValueError("Auxiliary metadata did not round-trip")
            if kind == "original":
                view = f.get_slice(name)
                if view.get_shape() != info["shape"] or view.get_dtype() != info["dtype"]:
                    raise ValueError("Original tensor dtype/shape changed")
            else:
                for key, value in data.items():
                    if not np.array_equal(f.get_tensor(key), value):
                        raise ValueError("Quantized payload did not round-trip")
        state["shards"][task["filename"]] = dict(
            task,
            sha256=digest(path),
            bytes=path.stat().st_size,
            source_range_sha256=source_hash,
            source_read_seconds=read_seconds,
            seconds=read_seconds + time.perf_counter() - started,
        )
        state["complete"] = len(state["shards"]) == len(queue)
        atomic_json(state_path, state)
        print(
            json.dumps(
                {
                    "stage": "aux",
                    "done": len(state["shards"]),
                    "total": len(queue),
                    "tensor": name,
                    "first_row": task["first_row"],
                    "seconds": time.perf_counter() - started,
                }
            ),
            flush=True,
        )
    return state
