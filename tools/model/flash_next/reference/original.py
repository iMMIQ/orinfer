"""Bounded original-precision reads for a layerwise Flash Next teacher.

Only pinned source ranges are read. No community quantizer or student codec
participates. This reader alone is not a teacher forward or quality result.
"""

import hashlib
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from tools.model.safetensors_source import SIZES


DTYPES = {
    "F16": "<f2",
    "F32": "<f4",
    "F64": "<f8",
    "I8": "i1",
    "U8": "u1",
    "I16": "<i2",
    "U16": "<u2",
    "I32": "<i4",
    "U32": "<u4",
    "I64": "<i8",
    "U64": "<u8",
    "BOOL": "?",
}


class Original:
    def __init__(self, source, config, *, max_read_bytes=64 * 1024**2):
        if type(max_read_bytes) is not int or max_read_bytes < 1:
            raise ValueError("Original read budget must be positive")
        self.source, self.config, self.max_read_bytes = source, config, max_read_bytes
        self.parts = {
            name: None
            for name in source.weight_map
            if name.startswith("model.language_model.") or name == "lm_head.weight"
        }
        self.reads = {}

    def info(self, name):
        if name not in self.parts:
            raise ValueError("Expected an original text tensor")
        return self.source.tensor(name)[2]

    def shape(self, name):
        return tuple(self.info(name)["shape"])

    def record(self, name, start, count, raw):
        key = f"{name}:{start}:{count}"
        digest = hashlib.sha256(raw).hexdigest()
        if key in self.reads and self.reads[key]["sha256"] != digest:
            raise ValueError("Original source range changed between reads")
        self.reads[key] = {
            "tensor": name,
            "first": start,
            "count": count,
            "bytes": len(raw),
            "sha256": digest,
        }

    @staticmethod
    def decode(raw, dtype, shape):
        if dtype == "BF16":
            value = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
        elif dtype in DTYPES:
            value = np.frombuffer(raw, dtype=DTYPES[dtype]).copy()
        else:
            raise ValueError("Unsupported original tensor dtype")
        value = value.reshape(shape)
        if value.dtype.kind == "f" and not np.isfinite(value).all():
            raise ValueError("Nonfinite original reference weights")
        return value

    def rows(self, name, start, count):
        info = self.info(name)
        shape = info["shape"]
        if (
            not shape
            or type(start) is not int
            or type(count) is not int
            or start < 0
            or count < 1
            or start + count > shape[0]
        ):
            raise ValueError("Invalid original row range")
        size = count * math.prod(shape[1:]) * SIZES[info["dtype"]]
        if size > self.max_read_bytes:
            raise ValueError("Split original tensor into bounded rows or experts")
        raw = self.source.rows(name, start, count)
        self.record(name, start, count, raw)
        return self.decode(raw, info["dtype"], (count, *shape[1:]))

    def tensor(self, name):
        shape = self.shape(name)
        if shape:
            return self.rows(name, 0, shape[0])
        filename, begin, info = self.source.tensor(name)
        raw = self.source.read(filename, begin + info["data_offsets"][0], SIZES[info["dtype"]])
        self.record(name, 0, 1, raw)
        return self.decode(raw, info["dtype"], ())

    def batch_rows(self, requests, *, workers=4, max_total_bytes=16 * 1024**2):
        """Parallel small original row reads, bounded and validated up front."""
        if type(workers) is not int or not 1 <= workers <= 8:
            raise ValueError("Original row workers must be in 1..8")
        requests = list(dict.fromkeys(requests))
        total = 0
        for name, start, count in requests:
            info = self.info(name)
            shape = info["shape"]
            if (
                not shape
                or type(start) is not int
                or type(count) is not int
                or start < 0
                or count < 1
                or start + count > shape[0]
            ):
                raise ValueError("Invalid original batch row range")
            size = count * math.prod(shape[1:]) * SIZES[info["dtype"]]
            if size > self.max_read_bytes:
                raise ValueError("Unbounded original row read")
            total += size
        if total > max_total_bytes:
            raise ValueError("Original row batch exceeds memory budget")
        # info() warms source headers before workers can share a source file.
        result = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = {pool.submit(self.rows, *request): request for request in requests}
            try:
                for future in as_completed(pending):
                    result[pending[future]] = future.result()
            except Exception:
                for future in pending:
                    future.cancel()
                raise
        return result
