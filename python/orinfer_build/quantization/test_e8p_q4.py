from pathlib import Path
import tempfile
import unittest

import numpy as np

from orinfer_build.quantization.e8p import basis
from orinfer_build.quantization.e8p_q4 import Weights, fit, validate_coefficients
from orinfer_build.quantization.vq import e8p_decode, rotate


class E8PQ4Tests(unittest.TestCase):
    def test_every_stage_code_and_integer_range(self):
        first = np.arange(65536, dtype=np.uint32).reshape(4096, 16)
        second = (first * 32771 + 97) & 65535
        for coefficients in [(3, 1), (4, 1), (7, 2)]:
            w = Weights(
                first | (second << 16),
                np.ones(4096, np.float16),
                np.ones(128, np.int8),
                coefficients,
            )
            a, b = coefficients
            expected = (
                a * e8p_decode(first.astype(np.uint16), basis()).astype(np.int16)
                + b * e8p_decode(second.astype(np.uint16), basis()).astype(np.int16)
            ).reshape(4096, 128)
            np.testing.assert_array_equal(w.integers(), expected)
            self.assertLessEqual(np.abs(expected).max(), 127)
            self.assertLess(w.bits_per_weight, 4.2)

    def test_fit_zero_rows_determinism_and_serialization(self):
        rng = np.random.default_rng(20261002)
        raw = rng.normal(0, 0.02, (3, 128)).astype(np.float32)
        raw[0] = 0
        signs = rng.choice(np.array([-1, 1], np.int8), 128)
        before = raw.copy()
        for coefficients in [(3, 1), (4, 1), (7, 2)]:
            a = fit(raw, signs, coefficients=coefficients, iterations=1)
            b = fit(raw, signs, coefficients=coefficients, iterations=1)
            np.testing.assert_array_equal(a.packed, b.packed)
            np.testing.assert_array_equal(a.scales, b.scales)
            target = rotate(raw, signs)
            self.assertLess(np.linalg.norm(a.dequantize() - target) / np.linalg.norm(target), 0.12)
            self.assertLess(np.abs(a.dequantize()[0]).max(), 1e-6)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "weights.safetensors"
                a.save(path)
                restored = Weights.load(path)
                self.assertEqual(restored.coefficients, a.coefficients)
                np.testing.assert_array_equal(restored.packed, a.packed)
                np.testing.assert_array_equal(restored.dequantize(), a.dequantize())
        np.testing.assert_array_equal(raw, before)

    def test_reject_invalid_codec_contracts(self):
        for coefficients in [(7, 7), (14, 4), (12, 1), (7.0, 2), (7, 0), [7, 2]]:
            with self.assertRaises(ValueError):
                validate_coefficients(coefficients)
        for raw in [np.ones((1, 127)), np.full((1, 128), np.nan), np.empty((0, 128))]:
            with self.assertRaises(ValueError):
                fit(raw, np.ones(raw.shape[1], np.int8))
        with self.assertRaises(ValueError):
            fit(np.ones((1, 128)), np.zeros(128, np.int8))
        with self.assertRaises(ValueError):
            Weights(
                np.zeros((1, 16), np.uint16), np.ones(1, np.float16), np.ones(128, np.int8)
            ).validate()


if __name__ == "__main__":
    unittest.main()
