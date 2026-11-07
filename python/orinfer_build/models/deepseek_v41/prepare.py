"""Prepare immutable A2 W4 banks on disk; online execution reads physical bytes."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import time

import numpy as np

from orinfer_build.common.files import atomic_json
from orinfer_build.weights.local import LocalSource


class Transpose:
    def __init__(self, cache):
        source = Path(__file__).with_name("transpose.c")
        stat = source.stat()
        key = f"{platform.machine()}-{stat.st_size}-{stat.st_mtime_ns}"
        cache = Path(cache)
        cache.mkdir(parents=True, exist_ok=True)
        library = cache / f"transpose-{key}.so"
        if not library.exists():
            with tempfile.TemporaryDirectory(dir=cache) as temporary:
                stage = Path(temporary) / "transpose.so"
                subprocess.run(
                    ["cc", "-O3", "-std=c11", "-shared", "-fPIC", str(source), "-o", str(stage)],
                    check=True,
                )
                os.replace(stage, library)
        self.library = ctypes.CDLL(str(library.resolve()))
        self.call = self.library.orinfer_transpose_u8
        self.call.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_size_t] * 3
        self.call.restype = None

    def __call__(self, source, destination):
        if (
            source.dtype != np.uint8
            or destination.dtype != np.uint8
            or source.ndim != 2
            or destination.shape != source.shape[::-1]
            or not source.flags.c_contiguous
            or not destination.flags.writeable
            or destination.strides[1] != 1
            or destination.strides[0] < source.shape[0]
            or np.shares_memory(source, destination)
        ):
            raise ValueError("Invalid packed-byte transpose buffers")
        self.call(
            source.ctypes.data,
            destination.ctypes.data,
            *source.shape,
            destination.strides[0],
        )


class Checkpoint(LocalSource):
    def __init__(self, root, index, names):
        super().__init__(root, index)
        try:
            for name in sorted(set(names)):
                self.tensor(name)
        except BaseException:
            self.close()
            raise

    def read(self, name, kind, shape=None):
        tensor = self.metadata(name)
        if tensor["dtype"] != kind or (shape is not None and tensor["shape"] != list(shape)):
            raise ValueError(f"{name}: unexpected checkpoint dtype/shape")
        dtype = {"I8": np.uint8, "F32": np.dtype("<f4")}[kind]
        return np.frombuffer(self.read_rows(name), dtype=dtype).reshape(tensor["shape"])


def tensor_names(weight):
    if weight.get("kind") == "shared":
        suffix = "weight_scale" if weight["plane"] == "scale" else "weight"
        parts = ("w2",) if weight["plane"] == "down" else ("w1", "w3")
        return [f"{weight['prefix']}.{part}.{suffix}" for part in parts]
    names = []
    suffix = {"packed": "weight", "scale": "weight_scale", "assist": "scale_bias"}[weight["plane"]]
    for expert in weight["experts"]:
        for part in weight["parts"]:
            names.extend([f"{expert}.{part}.{suffix}", f"{expert}.{part}.weight_offset"])
    return names


def convert(weight, checkpoint, transpose):
    experts, parts = weight["experts"], weight["parts"]
    k, n, plane = weight["input"], weight["output"], weight["plane"]
    pn = n // len(parts)
    shape, dtype = {
        "packed": ((len(experts), k, n // 2), np.uint8),
        "scale": ((len(experts), 1, n), np.dtype("<u8")),
        "assist": ((len(experts), n), np.dtype("<f4")),
    }[plane]
    bank = np.empty(shape, dtype=dtype)
    for e, expert in enumerate(experts):
        for p, part in enumerate(parts):
            prefix = expert + "." + part
            offset = checkpoint.read(prefix + ".weight_offset", "F32")
            if not np.all(offset == 0):
                raise ValueError(f"{prefix}: asymmetric W4 is unsupported")
            if plane == "packed":
                data = checkpoint.read(prefix + ".weight", "I8", (pn // 2, k))
                destination = bank[e, :, p * (pn // 2) : (p + 1) * (pn // 2)]
                transpose(data, destination)
            elif plane == "scale":
                data = checkpoint.read(prefix + ".weight_scale", "F32", (pn, 1))
                bank[e, 0, p * pn : (p + 1) * pn] = data.view("<u4").reshape(-1)
            else:
                data = checkpoint.read(prefix + ".scale_bias", "F32")
                if data.ndim != 2 or data.shape[0] != pn:
                    raise ValueError(f"{prefix}: assist shape differs")
                total = np.zeros(pn, dtype="<f4")
                for column in range(data.shape[1]):
                    total += data[:, column]
                bank[e, p * pn : (p + 1) * pn] = total
    return bank


def convert_shared(weight, checkpoint):
    rank, prefix, plane = weight["rank"], weight["prefix"], weight["plane"]
    if not 0 <= rank < 8:
        raise ValueError("Shared-expert rank must be within TP8")
    if plane == "down":
        data = checkpoint.read(prefix + ".w2.weight", "I8")
        if data.ndim != 2 or data.shape[1] % 8:
            raise ValueError("Shared down projection cannot be partitioned over TP8")
        local = data.shape[1] // 8
        return data[:, rank * local : (rank + 1) * local].copy()
    if plane not in ("up", "scale"):
        raise ValueError("Unknown shared-expert bank plane")
    kind, suffix = ("F32", "weight_scale") if plane == "scale" else ("I8", "weight")
    gate = checkpoint.read(prefix + ".w1." + suffix, kind)
    up = checkpoint.read(prefix + ".w3." + suffix, kind, gate.shape)
    if gate.ndim != 2 or gate.shape[0] % 8 or (plane == "scale" and gate.shape[1] != 1):
        raise ValueError("Shared gate/up projection cannot be partitioned over TP8")
    local = gate.shape[0] // 8
    bank = np.concatenate(
        (gate[rank * local : (rank + 1) * local], up[rank * local : (rank + 1) * local])
    )
    return bank.reshape(-1) if plane == "scale" else bank


def prepare(plan, model_dir, output, workers):
    start = time.perf_counter()
    base = plan.resolve().parent
    if output.resolve().parent != base:
        raise ValueError("Prepared plan must share the original plan's asset directory")
    model = json.loads(plan.read_text())
    storage = base / "weights"
    manifests = storage / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    transpose = Transpose(storage / "compiler")
    jobs = [
        (rank["rank"], definition)
        for rank in model["ranks"]
        for definition in rank["buffers"]
        if definition.get("weight", {}).get("kind") == "w4"
    ]
    # Keep W4 job order and converter identity stable so existing physical
    # banks resume without retransposing or rewriting their payloads.
    jobs += [
        (rank["rank"], definition)
        for rank in model["ranks"]
        for definition in rank["buffers"]
        if definition.get("weight", {}).get("kind") == "shared"
    ]
    names = {name for _, definition in jobs for name in tensor_names(definition["weight"])}
    checkpoint = Checkpoint(model_dir, model["index"]["file"], names)
    converter = {
        "layout_version": 1,
        "source": Path(__file__).with_name("transpose.c").read_text(),
        "transform": inspect.getsource(convert),
    }

    def one(indexed_job):
        index, (rank, definition) = indexed_job
        weight = definition["weight"]
        source_files = sorted({checkpoint.index[n] for n in tensor_names(weight)})
        shared = weight["kind"] == "shared"
        identity = {
            "converter": {"layout_version": 1, "transform": inspect.getsource(convert_shared)}
            if shared
            else converter,
            "weight": weight,
            "model_index": model["index"],
            "source_files": {f: checkpoint.signatures[f] for f in source_files},
        }
        shared_key = hashlib.sha256(json.dumps(weight, sort_keys=True).encode()).hexdigest()
        manifest = manifests / (
            f"r{rank}-shared-{shared_key}.json" if shared else f"r{rank}-b{index:04}.json"
        )
        target = None
        if manifest.exists():
            record = json.loads(manifest.read_text())
            if record["identity"] == identity and record["bytes"] == definition["bytes"]:
                candidate = (base / record["file"]).resolve()
                if not candidate.is_relative_to(base):
                    raise ValueError("Prepared weight path leaves asset directory")
                if candidate.is_file() and candidate.stat().st_size == definition["bytes"]:
                    target = candidate
        reused = target is not None
        if not reused:
            bank = (
                convert_shared(weight, checkpoint)
                if shared
                else convert(weight, checkpoint, transpose)
            )
            data = memoryview(bank).cast("B")
            if len(data) != definition["bytes"]:
                raise ValueError("Prepared weight extent differs from execution plan")
            target = storage / f"r{rank}-b{index:04}-{weight['plane']}-{time.time_ns()}.bin"
            with tempfile.NamedTemporaryFile(dir=storage, prefix=".stage-", delete=False) as stage:
                temporary = Path(stage.name)
                try:
                    stage.write(data)
                    stage.flush()
                    os.fsync(stage.fileno())
                    os.replace(temporary, target)
                finally:
                    temporary.unlink(missing_ok=True)
        file = str(target.relative_to(base))
        atomic_json(manifest, {"identity": identity, "file": file, "bytes": definition["bytes"]})
        definition["prepared_weight"] = file
        del definition["weight"]
        return {
            "rank": rank,
            "plane": weight["plane"],
            "bytes": definition["bytes"],
            "reused": reused,
        }

    total_bytes, completed, converted_bytes, converted_banks = 0, 0, 0, 0
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for result in pool.map(one, enumerate(jobs)):
                completed += 1
                total_bytes += result["bytes"]
                if not result["reused"]:
                    converted_bytes += result["bytes"]
                    converted_banks += 1
                print(
                    json.dumps({"completed": completed, "total": len(jobs), **result}), flush=True
                )
    finally:
        checkpoint.close()
    atomic_json(output, model)
    summary = {
        "plan": str(output),
        "banks": completed,
        "logical_bytes": total_bytes,
        "converted_bytes": converted_bytes,
        "converted_banks": converted_banks,
        "reused_bytes": total_bytes - converted_bytes,
        "reused_banks": completed - converted_banks,
        "seconds": time.perf_counter() - start,
        "transpose": "aarch64 NEON 8x8" if platform.machine() == "aarch64" else "portable scalar",
        "payload_scan": False,
        "batch_numpy_compare": False,
        "resume": "source size/mtime/inode, converter identity, output size and completion metadata",
    }
    atomic_json(storage / "summary.json", summary)
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("workers must be within 1..16")
    prepare(args.plan, args.model_dir, args.output, args.workers)


if __name__ == "__main__":
    main()
