import unittest
import numpy as np
from tools.model.deepseek_v41.floating import bf16_parts


class FloatingPartsTests(unittest.TestCase):
    def test_three_parts_reconstruct_finite_model_operands_without_mutation(self):
        rng = np.random.default_rng(20261002)
        source = (
            rng.normal(size=(17, 129)) * np.exp2(rng.integers(-40, 40, size=(17, 129)))
        ).astype("f4")
        before = source.copy()
        parts = bf16_parts(source)
        restored = (parts.astype("u4") << 16).view("f4").sum(0)
        np.testing.assert_array_equal(restored, source)
        np.testing.assert_array_equal(before, source)

    def test_high_part_uses_nearest_even_at_bf16_midpoints(self):
        source = np.array([1.00390625, 1.01171875, -1.00390625], dtype="f4")
        high = (bf16_parts(source)[0].astype("u4") << 16).view("f4")
        np.testing.assert_array_equal(high, [1.0, 1.015625, -1.0])

    def test_invalid_or_excessive_operands_are_rejected(self):
        for value in [np.nan, np.inf, -np.inf, np.finfo("f4").max, 2.0**-140]:
            with self.assertRaises(ValueError):
                bf16_parts(np.array([value], dtype="f4"))


if __name__ == "__main__":
    unittest.main()
