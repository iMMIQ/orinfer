"""Causality and original QSA boundary behavior without native kernels."""

import unittest
import torch
from tools.model.flash_next.reference.qsa import index, select, quantize_kv, dequantize_kv


class QsaTests(unittest.TestCase):
    def test_complete_groups_tail_and_sparse_boundary(self):
        torch.manual_seed(20261002)
        qk = torch.randn(2053, 5, 128).half()
        q, k = index(qk, torch.zeros(128), torch.zeros(128))
        positions = [0, 1, 3, 2047, 2048, 2051, 2052]
        chosen = select(q[positions], k, positions)
        for row, pos in zip(chosen, positions):
            valid = row[row >= 0]
            self.assertEqual(len(valid), min(512, (pos + 1) // 4) * 4 + (pos + 1) % 4)
            self.assertEqual(len(valid.unique()), len(valid))
            self.assertTrue(bool((valid <= pos).all()))
            if pos < 2051:
                self.assertEqual(valid.sort().values.tolist(), list(range(pos + 1)))
        # A partial group must not influence earlier compressed keys or choices.
        changed = qk.clone()
        changed[2052, 4].fill_(100)
        _, other = index(changed, torch.zeros(128), torch.zeros(128))
        self.assertTrue(torch.equal(k, other))

    def test_int8_zero_tiny_and_outlier_groups(self):
        x = torch.zeros(3, 2, 256, dtype=torch.float16)
        x[1].fill_(2**-20)
        x[2, :, 0] = 4
        x[2, :, 64] = -16
        codes, scale = quantize_kv(x)
        self.assertEqual(codes.dtype, torch.int8)
        self.assertTrue(bool((scale > 0).all()))
        self.assertTrue(bool((codes[0] == 0).all()))
        reconstructed = dequantize_kv(codes, scale)
        self.assertTrue(bool(torch.isfinite(reconstructed).all()))
        self.assertTrue(bool((reconstructed[2, :, 64] < 0).all()))
        self.assertLess(float((x.float() - reconstructed).abs().max()), 0.07)
        self.assertEqual(scale.shape, (3, 2, 4))

    def test_relu_is_per_head_before_sum_and_stable_ties(self):
        q = torch.zeros(1, 4, 128)
        q[0, 0, 0] = 1
        q[0, 1, 0] = -1
        k = torch.zeros(514, 128)
        k[:, 0] = torch.arange(514)
        actual = select(q, k, [2055])[0]
        blocks = actual[::4][:512] // 4
        self.assertEqual(set(blocks.tolist()), set(range(2, 514)))
        ties = select(torch.zeros_like(q), k, [2055])[0]
        self.assertEqual(ties[:2048].tolist(), list(range(2048)))


if __name__ == "__main__":
    unittest.main()
