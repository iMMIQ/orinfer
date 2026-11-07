import tempfile
from pathlib import Path
import unittest

import numpy as np

from orinfer_build.quantization.vq import (
    Weights,
    e8p_decode,
    e8p_nibble_table,
    e8p_repack_k64,
    e8p_sign_table,
    e8p_selector_table,
    e8p_selector_supports_parity_xor,
    rotate,
    save,
    load,
)


class IntegerVQTests(unittest.TestCase):
    def test_selector_table_matches_all_codes_and_rejects_other_bases(self):
        rng = np.random.default_rng(20261002)
        table = rng.choice(np.array([-10, -6, -2, 0, 2, 6, 10], np.int8), (256, 8))
        book = e8p_selector_table(table)
        self.assertEqual(book.shape, (256, 1))
        self.assertEqual(book.nbytes, 1024)
        codes = np.arange(65536, dtype=np.uint16)
        parity = np.array([(int(c) & 255).bit_count() % 2 for c in codes], np.uint32)
        bases = book[codes >> 8, 0]
        decoded = np.empty((65536, 8), np.int8)
        even = np.array([3, 7, 11, 1, -1, -5, -9, 1], np.int8)
        odd = np.array([1, 5, 9, -1, -3, -7, -11, -1], np.int8)
        for lane, sign_bit in enumerate([0, 4, 1, 5, 2, 6, 3, 7]):
            selector = ((bases >> (4 * lane)) & 7) ^ (((codes ^ parity) >> sign_bit & 1) * 4)
            decoded[:, lane] = np.where(parity, odd[selector], even[selector])
        np.testing.assert_array_equal(decoded, e8p_decode(codes, table))
        expanded = e8p_sign_table(table)[None]
        recovered = np.ascontiguousarray(expanded[0, :256, [0, 2]].T).view(np.int8)
        np.testing.assert_array_equal(recovered.reshape(256, 8).astype(np.int16) - 1, table)
        for value in (4, 14, -14):
            invalid = table.copy()
            invalid[0, 0] = value
            with self.assertRaises(ValueError):
                e8p_selector_table(invalid)
        with self.assertRaises(ValueError):
            e8p_selector_table(table.astype(np.int16))

    def test_xor_parity_matches_all_codes_with_nonzero_basis(self):
        rng = np.random.default_rng(20261002)
        table = rng.choice(np.array([-10, -6, -2, 2, 6, 10], np.int8), (256, 8))
        book = e8p_selector_table(table)
        self.assertTrue(e8p_selector_supports_parity_xor(book))
        codes = np.arange(65536, dtype=np.uint16)
        parity = np.array([(int(c) & 255).bit_count() % 2 for c in codes], np.uint32)
        decoded = np.empty((65536, 8), np.uint8)
        palette = np.array([3, 7, 11, 1, 255, 251, 247, 1], np.uint8)
        for lane, sign_bit in enumerate([0, 4, 1, 5, 2, 6, 3, 7]):
            selector = ((book[codes >> 8, 0] >> (4 * lane)) & 7) ^ (
                ((codes ^ parity) >> sign_bit & 1) * 4
            )
            decoded[:, lane] = palette[selector] ^ (parity * 2).astype(np.uint8)
        np.testing.assert_array_equal(decoded.view(np.int8), e8p_decode(codes, table))
        table[0, 7] = 0
        self.assertFalse(e8p_selector_supports_parity_xor(e8p_selector_table(table)))
        for invalid in (book.astype(np.int32), book.reshape(-1), book | np.uint32(8)):
            with self.assertRaises(ValueError):
                e8p_selector_supports_parity_xor(invalid)

    def test_exact_nibble_decoder_matches_all_codes_and_repacking_is_lossless(self):
        rng = np.random.default_rng(20261002)
        table = rng.choice(np.array([-10, -6, -2, 2, 6, 10], np.int8), (256, 8))
        nibble = e8p_nibble_table(e8p_selector_table(table))[:, 0]
        codes = np.arange(65536, dtype=np.uint16)
        parity = np.array([(int(c) & 255).bit_count() & 1 for c in codes], np.uint32)
        signs = codes.astype(np.uint32) ^ parity
        selectors = (signs & 0x33) | ((signs & 0xCC) << 6)
        magic = np.uint32(0xEEE00E00) ^ (parity * np.uint32(0x11111111))
        mask = np.zeros(65536, np.uint32)
        for byte in range(4):
            mask |= ((magic >> (8 * ((selectors >> (4 * byte)) & 15))) & 255) << (8 * byte)
        words = nibble[codes >> 8] ^ mask
        y = ((words[:, None] >> np.arange(0, 32, 4, dtype=np.uint32)) & 15).astype(np.int16)
        y = np.where(y >= 8, y - 16, y)
        perm = [0, 2, 1, 3, 4, 6, 5, 7]
        np.testing.assert_array_equal(2 * y + 1, e8p_decode(codes, table)[:, perm])
        table[0, 0] = 0
        with self.assertRaisesRegex(ValueError, "nonzero"):
            e8p_nibble_table(e8p_selector_table(table))
        source = rng.integers(0, 65536, (2, 3, 17, 16), dtype=np.uint16)
        packed = e8p_repack_k64(source)
        restored = packed.reshape(2, 3, 2, 17, 8).transpose(0, 1, 3, 2, 4).reshape(source.shape)
        np.testing.assert_array_equal(source, restored)
        self.assertEqual(source.nbytes, packed.nbytes)
        self.assertTrue(packed.flags.c_contiguous)

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
        kind, d, dtype = "e8p", 8, np.uint16
        p = rng.integers(0, 65536, (3, 256 // d), dtype=dtype)
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
            "e8p",
            np.zeros((1, 32), np.uint16),
            np.zeros((256, 8), np.int8),
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
