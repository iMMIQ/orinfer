"""Two integer E8P vectors per eight weights, with one FP16 output-row scale.

Each U32 contains the coarse U16 in its low half and the residual U16 in its
high half. Both stages use our existing E8P basis. Integer coefficients are
global codec parameters, never per-group scales; the sum is exact in INT8.
"""

from dataclasses import dataclass
import math

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

from orinfer_build.quantization.e8p import basis, nearest
from orinfer_build.quantization.vq import e8p_decode, rotate


def validate_coefficients(coefficients):
    if (
        not isinstance(coefficients, tuple)
        or len(coefficients) != 2
        or any(type(v) is not int for v in coefficients)
        or not coefficients[0] > coefficients[1] > 0
        or math.gcd(*coefficients) != 1
        or 11 * sum(coefficients) > 127
    ):
        raise ValueError("Expected coprime positive coarse/residual coefficients within INT8")
    return coefficients


@dataclass(frozen=True)
class Weights:
    packed: np.ndarray
    scales: np.ndarray
    signs: np.ndarray
    coefficients: tuple[int, int] = (4, 1)

    def validate(self):
        validate_coefficients(self.coefficients)
        if self.packed.dtype != np.uint32 or self.packed.ndim != 2:
            raise ValueError("Expected U32 [outputs, inputs/8] E8P-Q4 codes")
        n, k = self.packed.shape[0], self.packed.shape[1] * 8
        if min(n, k) <= 0 or k % 128:
            raise ValueError("Expected nonempty block128 E8P-Q4 geometry")
        if (
            self.scales.dtype != np.float16
            or self.scales.shape != (n,)
            or not np.isfinite(self.scales).all()
            or (self.scales <= 0).any()
        ):
            raise ValueError("Expected one positive finite FP16 scale per output row")
        if (
            self.signs.dtype != np.int8
            or self.signs.shape != (k,)
            or not np.isin(self.signs, [-1, 1]).all()
        ):
            raise ValueError("Expected one fixed +/-1 sign per input channel")
        return n, k

    def integers(self):
        n, k = self.validate()
        table = basis()
        first = e8p_decode(self.packed.astype(np.uint16), table).astype(np.int16)
        second = e8p_decode((self.packed >> 16).astype(np.uint16), table).astype(np.int16)
        a, b = self.coefficients
        return (a * first + b * second).reshape(n, k).astype(np.int8)

    def dequantize(self):
        return self.integers().astype(np.float32) * self.scales.astype(np.float32)[:, None]

    @property
    def bits_per_weight(self):
        n, k = self.validate()
        # Conservatively count a full shared INT8 basis and codec coefficients.
        return (
            8 * (self.packed.nbytes + self.scales.nbytes + self.signs.nbytes + 2048 + 8) / (n * k)
        )

    def save(self, path):
        self.validate()
        save_file(
            dict(packed=self.packed, scales=self.scales, signs=self.signs, basis=basis()),
            str(path),
            metadata=dict(
                codec="orinfer-e8p-q4-v1",
                coefficients=",".join(map(str, self.coefficients)),
                layout="u32-low-coarse-high-residual",
                rotation="signed-block128",
            ),
        )

    @classmethod
    def load(cls, path):
        with safe_open(str(path), framework="numpy") as source:
            metadata = source.metadata() or {}
            if (
                metadata.get("codec") != "orinfer-e8p-q4-v1"
                or metadata.get("layout") != "u32-low-coarse-high-residual"
                or metadata.get("rotation") != "signed-block128"
                or set(source.keys()) != {"packed", "scales", "signs", "basis"}
                or not np.array_equal(source.get_tensor("basis"), basis())
            ):
                raise ValueError("E8P-Q4 codec/basis contract mismatch")
            result = cls(
                source.get_tensor("packed"),
                source.get_tensor("scales"),
                source.get_tensor("signs"),
                tuple(int(v) for v in metadata.get("coefficients", "").split(",")),
            )
        result.validate()
        return result


def fit(weights, signs, *, coefficients=(4, 1), iterations=2):
    """Bounded CPU oracle; fit the actual stored FP16 scale and joint INT8 sum."""
    a, b = validate_coefficients(coefficients)
    raw = np.asarray(weights, dtype=np.float32)
    signs = np.asarray(signs)
    if (
        raw.ndim != 2
        or min(raw.shape) <= 0
        or raw.shape[1] % 128
        or not np.isfinite(raw).all()
        or signs.dtype != np.int8
        or signs.shape != (raw.shape[1],)
        or not np.isin(signs, [-1, 1]).all()
        or type(iterations) is not int
        or not 0 <= iterations <= 8
    ):
        raise ValueError("Invalid E8P-Q4 fitting inputs")
    rotated = rotate(raw, signs)
    table = basis()
    n, k = raw.shape
    packed = np.empty((n, k // 8), np.uint32)
    scales = np.empty(n, np.float16)

    def quantize(x):
        v = x.reshape(-1, 8)
        ids = np.concatenate([nearest(v[i : i + 256], table) for i in range(0, len(v), 256)])
        return ids.reshape(-1, k // 8), e8p_decode(ids, table).reshape(-1, k).astype(np.float32)

    for row in range(n):
        w = rotated[row : row + 1]
        initial = np.sqrt(np.mean(w * w)) / np.float32(4 * a)
        best = np.inf
        for multiplier in (0.9, 1.0, 1.1):
            scale = np.float16(max(initial * multiplier, 2**-24))
            if not np.isfinite(scale):
                raise ValueError("E8P-Q4 scale cannot be represented as FP16")
            first, q0 = quantize(w / (np.float32(scale) * a))
            second, q1 = quantize((w / np.float32(scale) - a * q0) / b)
            for _ in range(iterations):
                q = a * q0 + b * q1
                scale = np.float16(
                    max(float(np.sum(w * q) / max(float(np.sum(q * q)), 1e-30)), 2**-24)
                )
                if not np.isfinite(scale):
                    raise ValueError("E8P-Q4 scale cannot be represented as FP16")
                first, q0 = quantize((w / np.float32(scale) - b * q1) / a)
                second, q1 = quantize((w / np.float32(scale) - a * q0) / b)
            loss = np.sum((w - np.float32(scale) * (a * q0 + b * q1)) ** 2)
            if loss < best:
                best = loss
                packed[row] = first[0].astype(np.uint32) | (second[0].astype(np.uint32) << 16)
                scales[row] = scale
    result = Weights(packed, scales, signs.copy(), coefficients)
    result.validate()
    return result
