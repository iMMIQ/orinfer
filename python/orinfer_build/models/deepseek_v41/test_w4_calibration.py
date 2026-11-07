import unittest

import numpy as np

from orinfer_build.models.deepseek_v41.w4_calibration import unpack_rows
from orinfer_build.models.deepseek_v41.w4_scale_release import (
    reconstruct_gate_bank,
    reconstruct_scales,
)
from orinfer_build.models.deepseek_v41.weight_layout import restore_nd, transposed_nz_blocks


class W4ReconstructionTests(unittest.TestCase):
    def test_gate_rounding_preserves_fused_w3_and_physical_layout(self):
        rng = np.random.default_rng(20261002)
        nd = rng.integers(0, 256, (2, 64, 16), dtype="u1")
        original = np.stack(list(transposed_nz_blocks(nd))).reshape(2, 1, 2, 16, 32)
        replacement = np.ascontiguousarray(nd[0, :, :8].T)
        replacement[0, 0] ^= 1
        bank, changed = reconstruct_gate_bank(original.view("i1"), {0: replacement})
        restored = restore_nd(bank)
        np.testing.assert_array_equal(restored[0, :, :8], replacement.T)
        np.testing.assert_array_equal(restored[:, :, 8:], nd[:, :, 8:])
        np.testing.assert_array_equal(restored[1], nd[1])
        self.assertEqual(changed.sum(), 1)
        self.assertTrue(changed[0, 0])

    def test_code_changes_refresh_assist_even_when_scale_is_unchanged(self):
        scales, assist, fitted, sums = self.fixture()
        fitted[:] = 1
        changed = np.array([[True, False], [False, False]])
        _, bias, count = reconstruct_scales(scales, assist, fitted, sums, changed_codes=changed)
        self.assertEqual(count, 1)
        self.assertEqual(bias[0, 0], 8 * sums[0, 0])
        self.assertEqual(bias[0, 1], assist[0, 1])

    def test_signed_adjacent_output_nibbles(self):
        codes = unpack_rows(np.array([[0x87, 0xF0]], dtype="u1"))
        np.testing.assert_array_equal(codes, [[7, 0], [-8, -1]])

    def fixture(self):
        scales = np.ones((2, 1, 4), dtype="<f4").view("<u4").astype("<u8")
        assist = np.full((2, 4), 123.456, dtype="<f4")
        fitted = np.ones((2, 2), dtype="<f4")
        fitted[0, 0] = 1.04
        return scales, assist, fitted, np.array([[3, -7], [9, 0]], dtype="i4")

    def test_unsigned_a8_compensation_and_untouched_w3_are_preserved(self):
        original, assist, fitted, sums = self.fixture()
        scales, bias, selected = reconstruct_scales(original, assist, fitted, sums, strength=0.5)
        self.assertEqual(selected, 1)
        values = scales.astype("<u4").view("<f4")
        self.assertAlmostEqual(float(values[0, 0, 0]), 1.02, places=6)
        # The signed low nibble carries a -8 offset; assist cancels that offset.
        self.assertAlmostEqual(float(bias[0, 0]), float(8 * sums[0, 0] * values[0, 0, 0]))
        np.testing.assert_array_equal(scales[:, :, 2:], original[:, :, 2:])
        np.testing.assert_array_equal(bias[:, 2:], assist[:, 2:])
        np.testing.assert_array_equal(bias[1], assist[1])
        self.assertEqual(bias[0, 1], assist[0, 1])
        self.assertEqual(assist[0, 0], np.float32(123.456))

    def test_up_section_preserves_gate_scale_and_assist(self):
        original, assist, fitted, sums = self.fixture()
        scales, bias, count = reconstruct_scales(original, assist, fitted, sums, section=1)
        self.assertEqual(count, 1)
        np.testing.assert_array_equal(scales[:, :, :2], original[:, :, :2])
        np.testing.assert_array_equal(bias[:, :2], assist[:, :2])
        value = scales[:, 0, 2:].astype("<u4").view("<f4")
        self.assertAlmostEqual(float(bias[0, 2]), float(8 * sums[0, 0] * value[0, 0]))

    def test_rejects_incompatible_scale_encoding_and_unbounded_adjustments(self):
        args = list(self.fixture())
        args[0][0, 0, 0] |= np.uint64(1) << np.uint64(32)
        with self.assertRaises(ValueError):
            reconstruct_scales(*args)
        args = list(self.fixture())
        args[2][0, 0] = 1.2
        with self.assertRaises(ValueError):
            reconstruct_scales(*args)
        args = list(self.fixture())
        with self.assertRaises(ValueError):
            reconstruct_scales(*args, strength=0)
