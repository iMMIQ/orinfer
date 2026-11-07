"""Bounded reads of the official FP4-expert / FP8-backbone checkpoint."""

import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile

import numpy as np

from orinfer_build.weights.local import LocalSource


def e4m3_table():
    bits = np.arange(256, dtype=np.uint16)
    exponent, fraction = (bits >> 3) & 15, bits & 7
    value = np.where(
        exponent == 0,
        fraction * 2.0**-9,
        (1 + fraction / 8) * np.exp2(exponent.astype(np.int16) - 7),
    )
    value = np.where(bits & 128, -value, value).astype(np.float32)
    value[[127, 255]] = np.nan
    return value


FP8_VALUES = e4m3_table()
FP4_VALUES = np.array(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32
)


def decode_scale(bits):
    if np.any(bits == 255):
        raise ValueError("Official E8M0 scale contains NaN")
    return np.exp2(bits.astype(np.int16) - 127).astype(np.float32)


class Decoder:
    """Bounded offline CPU decoder, with exact FP32 scale multiplication."""

    def __init__(self, cache):
        source = Path(__file__).with_name("official_decode.c")
        compiler = subprocess.check_output(["cc", "--version"])
        key = hashlib.sha256(
            source.read_bytes() + compiler + platform.machine().encode()
        ).hexdigest()
        cache = Path(cache)
        cache.mkdir(parents=True, exist_ok=True)
        library = cache / f"official-{key}.so"
        if not library.exists():
            with tempfile.TemporaryDirectory(dir=cache) as temporary:
                staged = Path(temporary) / "decoder.so"
                subprocess.run(
                    [
                        "cc",
                        "-O3",
                        "-std=c11",
                        "-fopenmp",
                        "-shared",
                        "-fPIC",
                        str(source),
                        "-o",
                        str(staged),
                    ],
                    check=True,
                )
                os.replace(staged, library)
        self.library = ctypes.CDLL(str(library.resolve()))
        self.call = self.library.orinfer_official_fp4
        self.call.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_size_t] * 2
        self.call.restype = None

    def fp4(self, packed, scale):
        if (
            packed.ndim != 2
            or packed.dtype != np.uint8
            or not packed.flags.c_contiguous
            or packed.shape[1] % 16
            or scale.dtype != np.float32
            or not scale.flags.c_contiguous
            or scale.shape != (packed.shape[0], packed.shape[1] // 16)
        ):
            raise ValueError("Official FP4 decoder geometry or dtype differs")
        out = np.empty((packed.shape[0], packed.shape[1] * 2), dtype=np.float32)
        self.call(packed.ctypes.data, scale.ctypes.data, out.ctypes.data, *out.shape)
        return out


class Checkpoint(LocalSource):
    def __init__(self, directory, decoder=None):
        super().__init__(directory)
        self.decoder = decoder
        config = json.loads((self.directory / "config.json").read_text())
        quant = config["quantization_config"]
        if (
            quant.get("expert_dtype") != "fp4"
            or quant.get("scale_fmt") != "ue8m0"
            or quant.get("weight_block_size") != [32, 32]
        ):
            raise ValueError("Expected the official FP4/FP8 mixed checkpoint with E8M0 scales")

    def read(self, name, first_row=0, rows=None, max_bytes=4 << 30):
        meta = self.metadata(name)
        shape = meta["shape"]
        count = shape[0] - first_row if rows is None else rows
        dtype = meta["dtype"]
        if dtype not in {"BF16", "F32", "F8_E4M3", "F8_E8M0", "I8"}:
            raise ValueError(f"Unsupported official tensor dtype: {dtype}")
        data = self.read_rows(name, first_row, count, max_bytes=max_bytes)
        storage = {"BF16": "<u2", "F32": "<f4"}.get(dtype, "u1")
        value = np.frombuffer(data, dtype=storage).reshape(count, *shape[1:])
        if dtype == "BF16":
            return (value.astype(np.uint32) << 16).view(np.float32)
        if dtype == "F8_E4M3":
            return FP8_VALUES[value]
        if dtype == "F8_E8M0":
            return decode_scale(value)
        return value

    def matrix(self, name, first_row=0, rows=None):
        meta = self.metadata(name)
        value = self.read(name, first_row, rows)
        if meta["dtype"] == "F8_E4M3":
            count = value.shape[0]
            scale_name = name.removesuffix("weight") + "scale"
            scale_shape = self.metadata(scale_name)["shape"]
            row_block = 1 if scale_shape[0] == meta["shape"][0] else 32
            if scale_shape != [
                (meta["shape"][0] + row_block - 1) // row_block,
                (meta["shape"][1] + 31) // 32,
            ]:
                raise ValueError("Official FP8 scale geometry differs")
            lo, hi = first_row // row_block, (first_row + count + row_block - 1) // row_block
            scale = self.read(scale_name, lo, hi - lo)
            scale = np.repeat(np.repeat(scale, row_block, axis=0), 32, axis=1)
            return (
                value
                * scale[first_row % row_block : first_row % row_block + count, : value.shape[1]]
            )
        if meta["dtype"] == "I8":
            scale = self.read(name.removesuffix("weight") + "scale", first_row, rows)
            if self.decoder is not None:
                return self.decoder.fp4(value, scale)
            codes = np.empty((*value.shape[:-1], value.shape[-1] * 2), dtype=np.uint8)
            codes[..., 0::2], codes[..., 1::2] = value & 15, value >> 4
            return FP4_VALUES[codes] * np.repeat(scale, 32, axis=-1)
        return value

    def rows(self, name, indices):
        ids = np.asarray(indices, dtype=np.int64)
        unique, inverse = np.unique(ids, return_inverse=True)
        values = []
        for index in unique:
            values.append(self.matrix(name, int(index), 1)[0])
        return np.stack(values)[inverse].reshape(*ids.shape, -1)
