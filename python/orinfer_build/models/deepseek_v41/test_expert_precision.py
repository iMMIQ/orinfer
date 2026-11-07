"""Offline precision contracts that are independent of NPU execution."""

import unittest

import numpy as np

from orinfer_build.models.deepseek_v41.expert_precision import bf16_scale
from orinfer_build.models.deepseek_v41.export_release import physical_contract
from orinfer_build.models.deepseek_v41.floating import bf16_parts


class ExpertPrecisionTests(unittest.TestCase):
    def test_scale_rounding_matches_existing_bf16_boundary(self):
        values = np.array([1e-30, 0.03137, 0.5, 1.00390625, 1.01171875], dtype="f4")
        encoded, decoded = bf16_scale(values)
        np.testing.assert_array_equal(encoded, bf16_parts(values)[0])
        np.testing.assert_array_equal(decoded.view("u4"), encoded.astype("u4") << 16)
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                bf16_scale([bad])

    def test_integer_banks_have_distinct_packing_and_scale_contracts(self):
        def contract(op, layout="nd"):
            step = dict(
                op=op,
                experts=48,
                input=5120,
                output=4608,
                buffers=["x", "w", "s", "bias", "ts", "y"],
                weight_layout=layout,
            )
            return physical_contract(dict(buffers=[], packets=[dict(body=[step], head=[])]), None)

        w4, w8 = contract("w4"), contract("w8")
        self.assertEqual(w4["w"], ("I8", [48, 5120, 2304]))
        self.assertEqual(w8["w"], ("I8", [48, 5120, 4608]))
        self.assertEqual(w8["s"], ("BF16", [48, 4608]))
        with self.assertRaisesRegex(ValueError, "ND storage"):
            contract("w8", "transposed_nz")


if __name__ == "__main__":
    unittest.main()
