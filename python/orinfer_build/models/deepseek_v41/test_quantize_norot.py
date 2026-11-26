"""Unit tests for unrotated native-grid DSv4.1 quantization primitives."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from orinfer_build.models.deepseek_v41.quantize_norot import (
    SSZQuantizer,
    pack_w4_nz_bank,
    quantize_exact_w8,
    quantize_per_channel_i8_f32,
    to_bf16_u16,
)
from orinfer_build.models.deepseek_v41.weight_layout import restore_nd


class NoRotQuantizationTests(unittest.TestCase):
    def test_ssz_w4_beats_minmax_and_handles_zero_rows(self):
        rng = np.random.default_rng(20261002)
        w = rng.standard_normal((16, 256)).astype(np.float32)
        w[0, 0] = 9.5  # Outlier where clipping dramatically reduces MSE
        w[1, :] = 0.0

        with tempfile.TemporaryDirectory() as tmp:
            ssz = SSZQuantizer(Path(tmp))
            q, s, nmse = ssz(w)

        self.assertEqual(q.dtype, np.int8)
        self.assertEqual(s.dtype, np.float32)
        self.assertTrue(np.all((q >= -8) & (q <= 7)))
        self.assertTrue(np.all(q[1] == 0))
        self.assertGreater(s[1], 0.0)

        # Compare against naive symmetric minmax [-7, 7]
        mm_s = np.maximum(np.abs(w).max(axis=1) / 7.0, 1e-30).astype(np.float32)
        mm_q = np.rint(w / mm_s[:, None]).clip(-7, 7).astype(np.int8)
        mm_err = float(np.square(w - mm_q.astype(np.float32) * mm_s[:, None]).sum())
        ssz_err = float(np.square(w - q.astype(np.float32) * s[:, None]).sum())
        self.assertLess(ssz_err, mm_err * 0.75)
        self.assertAlmostEqual(nmse, ssz_err / float(np.square(w).sum()), places=5)

    def test_exact_w8_recovers_mxfp4_grid_losslessly(self):
        rng = np.random.default_rng(20261002)
        fp4_vals = np.array(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
            dtype=np.float32,
        )
        # 8 blocks of 32 elements with UE8M0 scales spanning 2^-6 .. 2^-3 (ratio 8x -> max step 96 <= 127)
        exponents = rng.integers(-6, -2, size=(12, 8))
        scales = np.exp2(exponents).astype(np.float32)
        codes = rng.choice(fp4_vals, size=(12, 256))
        w = codes * np.repeat(scales, 32, axis=1)

        q, enc_bf16, nmse = quantize_exact_w8(w, scales)
        dec_scale = (enc_bf16.astype(np.uint32) << 16).view(np.float32)
        recon = q.astype(np.float32) * dec_scale[:, None]
        self.assertEqual(nmse, 0.0)
        np.testing.assert_array_equal(recon, w)

        # Wide-range row (2^-10 .. 2^0) falls back gracefully to bounded MSE
        wide_scales = np.full((2, 8), 2.0**-10, dtype=np.float32)
        wide_scales[:, 0] = 1.0
        w_wide = codes[:2] * np.repeat(wide_scales, 32, axis=1)
        q2, enc2, nmse2 = quantize_exact_w8(w_wide, wide_scales)
        self.assertEqual(q2.dtype, np.int8)
        self.assertLess(nmse2, 1e-4)
        self.assertTrue(np.all(enc2 > 0))

    def test_pack_w4_nz_bank_roundtrip_and_assist_contract(self):
        rng = np.random.default_rng(20261002)
        q_list = [rng.integers(-8, 8, size=(32, 128), dtype=np.int8) for _ in range(2)]
        s_list = [rng.uniform(0.01, 0.2, size=(32,)).astype(np.float32) for _ in range(2)]
        nz, scale_u64, assist = pack_w4_nz_bank(q_list, s_list)

        self.assertEqual(nz.shape, (2, 128 // 64, 32 // 16, 16, 32))
        self.assertEqual(nz.dtype, np.int8)
        self.assertEqual(scale_u64.shape, (2, 1, 32))
        self.assertEqual(assist.shape, (2, 32))

        restored = restore_nd(nz.view(np.uint8))
        for e in range(2):
            np.testing.assert_array_equal(
                scale_u64[e, 0].astype(np.uint32).view(np.float32), s_list[e]
            )
            expected_assist = (np.float32(8.0) * s_list[e]) * q_list[e].sum(
                axis=1, dtype=np.int32
            ).astype(np.float32)
            np.testing.assert_allclose(assist[e], expected_assist, rtol=0, atol=0)
            lo = (restored[e] & np.uint8(15)).astype(np.int8)
            hi = ((restored[e] >> np.uint8(4)) & np.uint8(15)).astype(np.int8)
            lo = np.where(lo >= 8, lo - 16, lo)
            hi = np.where(hi >= 8, hi - 16, hi)
            unpacked = np.empty((128, 32), dtype=np.int8)
            unpacked[:, 0::2] = lo
            unpacked[:, 1::2] = hi
            np.testing.assert_array_equal(unpacked.T, q_list[e])

    def test_bf16_rne_and_per_channel_i8(self):
        # Exact tie-to-even check: 1.0 + 2^-8 (halfway to next even=0 -> rounds down to 1.0),
        # 1.0 + 2^-7 + 2^-8 (halfway to next even -> rounds up to 1.0 + 2^-6)
        vals = np.array([1.0 + 2.0**-8, 1.0 + 2.0**-7 + 2.0**-8, -3.5], dtype=np.float32)
        dec = (to_bf16_u16(vals).astype(np.uint32) << 16).view(np.float32)
        np.testing.assert_array_equal(dec, np.array([1.0, 1.0 + 2.0**-6, -3.5], dtype=np.float32))

        w = np.array([[0.0, 127.0, -64.0], [-2.0, 1.0, 0.5]], dtype=np.float32)
        q, s = quantize_per_channel_i8_f32(w)
        self.assertEqual(q.dtype, np.int8)
        self.assertEqual(s.shape, (2, 1))
        np.testing.assert_allclose(q.astype(np.float32) * s, w, atol=1e-2)


if __name__ == "__main__":
    unittest.main()
