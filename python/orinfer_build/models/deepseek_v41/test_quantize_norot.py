"""Unit tests for unrotated native-grid DSv4.1 quantization primitives."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from orinfer_build.common.native_c import load_shared_c
from orinfer_build.models.deepseek_v41.quantize_norot import (
    GPTQQuantizer,
    NoRotRelease,
    SSZQuantizer,
    pack_w4_nz_bank,
    quantize_exact_w8,
    quantize_per_channel_i8_f32,
    save_expert_bank,
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

    def test_gptq_w4_beats_ssz_on_correlated_hessian(self):
        rng = np.random.default_rng(20261002)
        # Correlated low-rank + diagonal activation distribution where GPTQ error compensation excels
        basis = rng.standard_normal((512, 24)).astype(np.float32)
        proj = rng.standard_normal((24, 192)).astype(np.float32)
        x = basis @ proj + 0.15 * rng.standard_normal((512, 192)).astype(np.float32)
        x[:, :8] *= 4.0  # Salient input channels
        h = (x.T @ x).astype(np.float64)
        w = rng.standard_normal((32, 192)).astype(np.float32)
        w[0, :] = 0.0  # Zero row safety

        with tempfile.TemporaryDirectory() as tmp:
            ssz = SSZQuantizer(Path(tmp))
            gptq = GPTQQuantizer(Path(tmp), n_threads=4)
            q_ssz, s_ssz, _ = ssz(w)
            q_gptq, s_gptq, nmse_gptq = gptq.quantize_cpu(w, h, damp_frac=0.02, block_size=64)
            q_cd, s_cd, nmse_cd = gptq.quantize_cpu(w, h, damp_frac=0.02, block_size=64, cd_iters=1)

        self.assertEqual(q_gptq.dtype, np.int8)
        self.assertTrue(np.all((q_gptq >= -8) & (q_gptq <= 7)))
        self.assertTrue(np.all(q_gptq[0] == 0))
        self.assertGreater(s_gptq[0], 0.0)

        y_ref = x @ w.T
        err_ssz = float(np.square(x @ (q_ssz.astype(np.float32) * s_ssz[:, None]).T - y_ref).sum())
        err_gptq = float(
            np.square(x @ (q_gptq.astype(np.float32) * s_gptq[:, None]).T - y_ref).sum()
        )
        err_cd = float(np.square(x @ (q_cd.astype(np.float32) * s_cd[:, None]).T - y_ref).sum())
        self.assertLess(err_gptq, err_ssz * 0.55)
        self.assertLessEqual(err_cd, err_gptq * 1.02)
        self.assertLessEqual(nmse_cd, nmse_gptq + 1e-6)

    def test_smoothquant_and_clamp_safe_swiglu_equivalence(self):
        rng = np.random.default_rng(20261002)
        x = rng.standard_normal((64, 128)).astype(np.float32)
        w1 = rng.standard_normal((96, 128)).astype(np.float32) * 0.2
        w3 = rng.standard_normal((96, 128)).astype(np.float32) * 0.2
        w2 = rng.standard_normal((128, 96)).astype(np.float32) * 0.2

        def swiglu(g, u):
            g_c = np.minimum(g, 10.0)
            u_c = np.clip(u, -10.0, 10.0)
            return (g_c / (1.0 + np.exp(-g_c))) * u_c

        # Reference unscaled SwiGLU MLP output
        u_raw = x @ w3.T
        self.assertLess(float(np.abs(u_raw).max()), 8.5)
        y_ref = swiglu(x @ w1.T, u_raw) @ w2.T

        # Apply BOTH input SmoothQuant s_in AND clamp-safe w3/w2 SwiGLU SmoothQuant s_sw >= 1.0
        s_in = rng.uniform(0.4, 2.5, size=(128,)).astype(np.float32)
        s_sw = rng.uniform(1.0, 3.5, size=(96,)).astype(np.float32)
        x_s = x / s_in[None, :]
        w1_s = w1 * s_in[None, :]
        w3_s = (w3 * s_in[None, :]) / s_sw[:, None]
        w2_s = w2 * s_sw[None, :]

        y_sq = swiglu(x_s @ w1_s.T, x_s @ w3_s.T) @ w2_s.T
        np.testing.assert_allclose(y_sq, y_ref, rtol=1e-5, atol=1e-5)

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

    def test_gptq_c11_guards_degenerate_diag_invalid_perm_and_zero_threads(self):
        rng = np.random.default_rng(20261002)
        n, k = 8, 16
        w = rng.standard_normal((n, k)).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            gptq = GPTQQuantizer(Path(tmp), n_threads=0)
            self.assertEqual(gptq.n_threads, 1)

            # Degenerate zero diagonal in hinv_b must stay finite (no NaN/Inf propagation)
            wb = np.ascontiguousarray(w.copy(), dtype=np.float32)
            hb = np.zeros((k, k), dtype=np.float32)
            scales = np.full(n, 0.25, dtype=np.float32)
            qb = np.empty((n, k), dtype=np.int8)
            eb = np.empty((n, k), dtype=np.float32)
            gptq.diag_block(
                wb.ctypes.data,
                hb.ctypes.data,
                scales.ctypes.data,
                n,
                k,
                0,
                qb.ctypes.data,
                eb.ctypes.data,
            )
            self.assertTrue(np.isfinite(eb).all())
            self.assertTrue(np.all((qb >= -8) & (qb <= 7)))

            # Out-of-range perm entries in cd_pass must be skipped safely
            h_mat = np.eye(k, dtype=np.float32)
            h_diag = np.ones(k, dtype=np.float32)
            bad_perm = np.array([-1, k, k + 99] + list(range(k - 3)), dtype=np.int32)
            q_io = np.zeros((n, k), dtype=np.int8)
            r_io = np.ones((n, k), dtype=np.float32)
            gptq.cd_pass(
                h_mat.ctypes.data,
                h_diag.ctypes.data,
                bad_perm.ctypes.data,
                scales.ctypes.data,
                n,
                k,
                0,
                q_io.ctypes.data,
                r_io.ctypes.data,
            )
            self.assertTrue(np.isfinite(r_io).all())

            # Shared native C loader caches compiled .so on disk by content hash
            src = Path(__file__).with_name("ssz_w4.c")
            load_shared_c(src, Path(tmp), prefix="ssz_test", openmp=True, math_lib=True)
            so_files = list(Path(tmp).glob("ssz_test-*.so"))
            self.assertEqual(len(so_files), 1)
            load_shared_c(src, Path(tmp), prefix="ssz_test", openmp=True, math_lib=True)
            self.assertEqual(len(list(Path(tmp).glob("ssz_test-*.so"))), 1)

    def test_save_expert_bank_and_smoothquant_loader(self):
        rng = np.random.default_rng(20261002)
        up_qs = [rng.integers(-8, 8, size=(32, 128), dtype=np.int8) for _ in range(2)]
        up_ss = [rng.uniform(0.01, 0.2, size=(32,)).astype(np.float32) for _ in range(2)]
        dn_qs = [rng.integers(-127, 128, size=(128, 32), dtype=np.int8) for _ in range(2)]
        dn_ss = [rng.integers(1, 1000, size=(32,), dtype=np.uint16) for _ in range(2)]

        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "bank"
            save_expert_bank(dest, up_qs, up_ss, dn_qs, dn_ss, up_w8=False, dn_w8=True)
            self.assertTrue((dest / "up_nz.npy").exists())
            self.assertTrue((dest / "up_scale.npy").exists())
            self.assertTrue((dest / "up_assist.npy").exists())
            self.assertTrue((dest / "down_w8.npy").exists())
            self.assertTrue((dest / "down_scale_bf16.npy").exists())

            rel = NoRotRelease.__new__(NoRotRelease)
            rel.smoothquant_dir = None
            rel.bf16_qkv_layers = []
            rel.w8_layers = []
            s_q, s_attn, s_shw, s_ffn = rel._load_layer_smoothquant(0)
            self.assertEqual(
                (s_q.shape, s_attn.shape, s_shw.shape, s_ffn.shape),
                ((1280,), (5120,), (2304,), (5120,)),
            )
            self.assertTrue(np.all(s_q == 1.0))

            sq_dir = Path(tmp) / "sq"
            sq_dir.mkdir()
            np.savez(
                sq_dir / "L03.npz",
                s_q=np.full(1280, 1.5, dtype=np.float32),
                s_attn=np.full(5120, 2.0, dtype=np.float32),
                s_shw=np.full(2304, 1.25, dtype=np.float32),
                s_ffn=np.full(5120, 0.75, dtype=np.float32),
            )
            rel.smoothquant_dir = sq_dir
            s_q, s_attn, s_shw, s_ffn = rel._load_layer_smoothquant(3)
            self.assertAlmostEqual(float(s_q[0]), 1.5)
            self.assertAlmostEqual(float(s_attn[0]), 2.0)
            self.assertAlmostEqual(float(s_shw[0]), 1.25)
            self.assertAlmostEqual(float(s_ffn[0]), 0.75)


if __name__ == "__main__":
    unittest.main()
