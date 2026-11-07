"""Weight-only signed block128 scalar Q4 for direct INT8 accumulation.

The reconstructed weights live in the rotated input basis. This codec is an
offline numerical building block; a model adapter must pair its signs with the
activation transform and validate complete-model quality before selecting it.
"""

from dataclasses import dataclass

import numpy as np

from orinfer_build.quantization.vq import rotate


MAGNITUDES = (3, 10, 19, 31, 46, 67, 93, 127)


@dataclass(frozen=True)
class Weights:
    packed: np.ndarray
    scales: np.ndarray
    signs: np.ndarray

    def validate(self):
        if self.packed.dtype != np.uint8 or self.packed.ndim != 2:
            raise ValueError("Expected a two-dimensional packed Q4 byte matrix")
        n, k = self.packed.shape[0], self.packed.shape[1] * 2
        if min(n, k) <= 0 or k % 128:
            raise ValueError("Q4 needs nonempty block128 geometry")
        if (
            self.scales.dtype != np.float16
            or self.scales.shape != (n,)
            or not np.isfinite(self.scales).all()
            or (self.scales <= 0).any()
        ):
            raise ValueError("Q4 needs one positive finite FP16 scale per output row")
        if (
            self.signs.dtype != np.int8
            or self.signs.shape != (k,)
            or not np.isin(self.signs, [-1, 1]).all()
        ):
            raise ValueError("Q4 needs one fixed signed rotation per input channel")
        return n, k

    def integers(self):
        self.validate()
        codes = np.stack((self.packed & 15, self.packed >> 4), axis=-1).reshape(
            self.packed.shape[0], -1
        )
        magnitudes = np.asarray(MAGNITUDES, np.int8)[codes & 7]
        return np.where(codes < 8, magnitudes, -magnitudes).astype(np.int8)

    def dequantize(self):
        return self.integers().astype(np.float32) * self.scales.astype(np.float32)[:, None]

    @property
    def bits_per_weight(self):
        n, k = self.validate()
        # Count the shared 16 signed values conservatively per matrix even
        # when a specialized cubin embeds them as instruction immediates.
        return 8 * (self.packed.nbytes + self.scales.nbytes + self.signs.nbytes + 16) / (n * k)


def fit(weights, signs):
    weights = np.asarray(weights, dtype=np.float32)
    signs = np.asarray(signs)
    if weights.ndim != 2 or min(weights.shape) <= 0 or not np.isfinite(weights).all():
        raise ValueError("Q4 fitting needs a nonempty finite matrix")
    if signs.dtype != np.int8:
        raise ValueError("Rotation signs must use INT8")
    rotated = rotate(weights, signs)
    n, k = rotated.shape
    packed = np.empty((n, k // 2), np.uint8)
    scales = np.empty(n, np.float16)
    levels = np.asarray(MAGNITUDES, np.float32)
    edges = (levels[:-1] + levels[1:]) / 2
    for first in range(0, n, 128):
        raw = rotated[first : first + 128]
        magnitude = np.abs(raw)
        initial = np.sqrt(np.mean(raw * raw, axis=1)) * np.float32(2.7 / 127)
        best = np.full(len(raw), np.inf, np.float32)
        codes = np.empty(raw.shape, np.uint8)
        scale_out = np.empty(len(raw), np.float16)
        for multiplier in (0.9, 1.0, 1.15):
            scale = np.maximum(initial * multiplier, 2**-24)
            for _ in range(4):
                indices = np.searchsorted(edges, magnitude / scale[:, None])
                decoded = levels[indices]
                scale = np.maximum(
                    np.sum(magnitude * decoded, axis=1)
                    / np.maximum(np.sum(decoded * decoded, axis=1), 1e-30),
                    2**-24,
                )
            scale = scale.astype(np.float16)
            if not np.isfinite(scale).all():
                raise ValueError("Q4 row scale cannot be represented as FP16")
            indices = np.searchsorted(edges, magnitude / scale.astype(np.float32)[:, None]).astype(
                np.uint8
            )
            loss = np.sum(
                (magnitude - levels[indices] * scale.astype(np.float32)[:, None]) ** 2, axis=1
            )
            better = loss < best
            best[better] = loss[better]
            codes[better] = indices[better] | np.where(raw[better] < 0, 8, 0).astype(np.uint8)
            scale_out[better] = scale[better]
        packed[first : first + len(raw)] = codes[:, ::2] | (codes[:, 1::2] << 4)
        scales[first : first + len(raw)] = scale_out
    result = Weights(packed, scales, signs.copy())
    result.validate()
    return result
