import tempfile
from pathlib import Path
import unittest

import numpy as np

from tools.quantization.vq import Weights, e8p_decode, e8p_sign_table, rotate, save, load


class IntegerVQTests(unittest.TestCase):
    def test_short_sign_table_matches_all_codes(self):
        rng = np.random.default_rng(20261002)
        table = (rng.integers(-63, 64, (256, 8)) * 2).astype(np.int8)
        codes = np.arange(65536, dtype=np.uint16)
        book = e8p_sign_table(table)
        self.assertEqual(book.nbytes, 8192)
        parity = np.array([(int(c) & 255).bit_count() % 2 for c in codes], np.uint16)
        adjusted = (codes & 255) ^ parity
        expected = e8p_decode(codes, table)
        for half in (0, 1):
            positive = book[(codes >> 8) + parity * 256, half * 2]
            negative = book[(codes >> 8) + parity * 256, half * 2 + 1]
            bits = [0, 4, 1, 5] if half == 0 else [2, 6, 3, 7]
            mask = sum(
                ((adjusted >> bit) & 1).astype(np.uint32) * np.uint32(255 << (8 * lane))
                for lane, bit in enumerate(bits)
            )
            decoded = (
                np.ascontiguousarray(positive ^ ((positive ^ negative) & mask))
                .view(np.int8)
                .reshape(-1, 4)
            )
            np.testing.assert_array_equal(decoded, expected[:, half * 4 : half * 4 + 4])
        with self.assertRaises(ValueError):
            e8p_sign_table(np.ones((256, 8), np.int8))

    def test_rotation_preserves_dot_and_is_invertible_with_fixed_signs(self):
        rng = np.random.default_rng(20261002)
        x = rng.normal(size=(5, 256)).astype(np.float32)
        w = rng.normal(size=(7, 256)).astype(np.float32)
        s = rng.choice(np.array([-1, 1], np.int8), 256)
        np.testing.assert_allclose(rotate(x, s) @ rotate(w, s).T, x @ w.T, atol=2e-5, rtol=2e-5)
        inverse = rotate(rotate(x, s), np.ones(256, np.int8)) * s
        np.testing.assert_allclose(inverse, x, atol=1e-6, rtol=1e-5)

    def test_e8p_integer_decode_against_scalar_signed_parity_oracle(self):
        rng = np.random.default_rng(20261002)
        table = (rng.integers(-5, 6, (256, 8)) * 2).astype(np.int8)
        codes = np.array([0, 1, 127, 128, 255, 256, 32768, 65535], np.uint16)
        expected = []
        for c in codes:
            c = int(c)
            parity = (c & 255).bit_count() % 2
            mask = (c & 255) ^ parity
            expected.append(
                [
                    int(table[c >> 8, i]) * (-1 if mask & (1 << bit) else 1) + (1 - 2 * parity)
                    for i, bit in enumerate([0, 4, 1, 5, 2, 6, 3, 7])
                ]
            )
        np.testing.assert_array_equal(e8p_decode(codes, table), expected)

    def test_fixture_roundtrip_group_major_layout_and_reject_overwrite(self):
        rng = np.random.default_rng(20261002)
        for kind, d, dtype in [("vq4", 4, np.uint8), ("e8p", 8, np.uint16)]:
            p = rng.integers(0, 256 if d == 4 else 65536, (3, 256 // d), dtype=dtype)
            table = (rng.integers(-5, 6, (256, d)) * 2).astype(np.int8)
            w = Weights(kind, p, table, np.ones(3, np.float16), np.ones(256, np.int8))
            a, b, s = w.gpu_layout()
            self.assertEqual(a.shape, (2, 3, 128 // d))
            self.assertEqual(b.shape, (256, d // 4))
            np.testing.assert_array_equal(a.transpose(1, 0, 2).reshape(p.shape), p)
            np.testing.assert_array_equal(b.view(np.int8).reshape(table.shape), table)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "w.safetensors"
                save(path, w)
                np.testing.assert_array_equal(load(path).integer_weights(), w.integer_weights())
                with self.assertRaises(FileExistsError):
                    save(path, w)

    def test_patch_signed_value_position_and_disabled_slot(self):
        patches = np.array([[32768 | (128 << 7) | 127, 0]], np.uint16)
        w = Weights(
            "vq4",
            np.zeros((1, 64), np.uint8),
            np.ones((256, 4), np.int8),
            np.ones(1, np.float16),
            np.empty(0, np.int8),
            patches,
        )
        decoded = w.integer_weights()
        self.assertEqual(int(decoded[0, 127]), -128)
        self.assertEqual(int(decoded[0, 128]), 1)
        self.assertEqual(w.nbytes, w.indices.nbytes + w.table.nbytes + w.scales.nbytes + 4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "w.safetensors"
            save(path, w)
            np.testing.assert_array_equal(load(path).integer_weights(), decoded)


if __name__ == "__main__":
    unittest.main()
