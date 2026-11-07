import unittest

import numpy as np

from orinfer_build.quantization.row_q4 import MAGNITUDES, Weights, fit
from orinfer_build.quantization.vq import rotate


class RowQ4Tests(unittest.TestCase):
    def test_every_signed_nibble_and_storage_budget(self):
        codes = np.tile(np.arange(16, dtype=np.uint8), (1, 8))
        packed = codes[:, ::2] | (codes[:, 1::2] << 4)
        w = Weights(packed, np.array([0.03125], np.float16), np.ones(128, np.int8))
        expected = np.tile(np.r_[MAGNITUDES, -np.asarray(MAGNITUDES)], (1, 8))
        np.testing.assert_array_equal(w.integers(), expected)
        np.testing.assert_array_equal(w.dequantize(), expected * 0.03125)
        self.assertEqual(w.bits_per_weight, (64 + 2 + 128 + 16) * 8 / 128)

    def test_weight_only_fit_is_deterministic_with_tail_rows_and_zero_rows(self):
        rng = np.random.default_rng(20261002)
        x = rng.normal(0, 0.02, (131, 256)).astype(np.float32)
        x[-1] = 0
        signs = rng.choice(np.array([-1, 1], np.int8), 256)
        original = x.copy()
        a, b = fit(x, signs), fit(x, signs)
        np.testing.assert_array_equal(a.packed, b.packed)
        np.testing.assert_array_equal(a.scales, b.scales)
        np.testing.assert_array_equal(x, original)
        target = rotate(x, signs)
        self.assertLess(np.linalg.norm(a.dequantize() - target) / np.linalg.norm(target), 0.13)
        self.assertLess(np.abs(a.dequantize()[-1]).max(), 1e-6)
        self.assertLess(a.bits_per_weight, 4.2)

    def test_invalid_geometry_values_and_signs_are_rejected(self):
        for x in [np.ones((2, 127)), np.full((2, 128), np.nan), np.empty((0, 128))]:
            with self.assertRaises(ValueError):
                fit(x, np.ones(x.shape[1], np.int8))
        with self.assertRaises(ValueError):
            fit(np.ones((2, 128)), np.zeros(128, np.int8))
        with self.assertRaises(ValueError):
            Weights(
                np.zeros((2, 64), np.uint8), np.ones(1, np.float16), np.ones(128, np.int8)
            ).validate()


if __name__ == "__main__":
    unittest.main()
