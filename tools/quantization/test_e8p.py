import unittest

import numpy as np

from tools.quantization.e8p import basis, nearest
from tools.quantization.vq import e8p_decode


class E8PEncoderTests(unittest.TestCase):
    def test_basis_geometry_unique_grid_and_exact_nearest(self):
        table = basis()
        grid = e8p_decode(np.arange(65536, dtype=np.uint16), table).astype(np.float32)
        self.assertEqual(len(np.unique(grid, axis=0)), 65536)
        rng = np.random.default_rng(20261002)
        values = np.concatenate(
            (rng.normal(size=(17, 8)) * 4, grid[[0, 1, 255, 32768, 65535]], np.zeros((1, 8)))
        ).astype(np.float32)
        codes = nearest(values, table)
        actual = ((values - grid[codes]) ** 2).sum(1)
        expected = ((values[:, None, :] - grid[None, :, :]) ** 2).sum(2).min(1)
        np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-5)

    def test_reject_nonfinite_and_wrong_shape(self):
        for x in (np.zeros((2, 7)), np.full((1, 8), np.nan)):
            with self.assertRaises(ValueError):
                nearest(x, basis())


if __name__ == "__main__":
    unittest.main()
