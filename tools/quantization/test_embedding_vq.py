import unittest

import numpy as np

from tools.quantization.e8p import basis
from tools.quantization.embedding_vq import pack, decode, EmbeddingDecoder
from tools.quantization.rotation import transform
from tools.quantization.vq import e8p_decode


class EmbeddingVQTests(unittest.TestCase):
    def test_native160_row_packets_and_inverse(self):
        rng = np.random.default_rng(20261002)
        codes = rng.integers(0, 65536, (7, 20), dtype=np.uint16)
        scales = np.linspace(0.001, 0.007, 7).astype(np.float16)
        signs = rng.choice(np.array([-1, 1], np.int8), 160)
        packets = pack(codes, scales)
        self.assertEqual(packets.shape, (7, 42))
        restored = decode(packets, basis(), signs)
        rotated = e8p_decode(codes, basis()).reshape(7, 160) * scales.astype(np.float32)[:, None]
        np.testing.assert_allclose(
            transform(restored, signs, mode="full"), rotated, atol=1e-7, rtol=1e-5
        )

    def test_reject_nonfinite_scale(self):
        with self.assertRaises(ValueError):
            pack(np.zeros((1, 20), np.uint16), np.array([np.nan], np.float16))

    def test_cached_decoder_matches_reference_and_observes_metadata_changes(self):
        rng = np.random.default_rng(20261002)
        decoder = EmbeddingDecoder()
        for width in (160, 640):
            signs = rng.choice(np.array([-1, 1], np.int8), width)
            table = basis().copy()
            codes = rng.integers(0, 65536, (3, width // 8), dtype=np.uint16)
            codes[0, 0] = 0
            packets = pack(codes, np.array([0.001, 0.02, 0.5], np.float16))
            for mutation in ("initial", "signs", "table"):
                if mutation == "signs":
                    signs[0] *= -1
                if mutation == "table":
                    table[0, 0] = 2 if table[0, 0] != 2 else -2
                np.testing.assert_array_equal(
                    decoder.decode(packets, table, signs), decode(packets, table, signs)
                )
        self.assertEqual(decoder._book.nbytes, 512 * 1024)
        signs[0] = 0
        with self.assertRaises(ValueError):
            decoder.decode(packets, table, signs)
        signs[0] = 1
        packets[:, :2] = np.array([0], np.float16).view(np.uint8)
        with self.assertRaises(ValueError):
            decoder.decode(packets, table, signs)

    def test_sign_validation_cache_is_bounded(self):
        decoder = EmbeddingDecoder()
        packets = pack(np.zeros((1, 20), np.uint16), np.ones(1, np.float16))
        for i in range(140):
            signs = np.ones(160, np.int8)
            for bit in range(8):
                signs[bit] = -1 if i & (1 << bit) else 1
            decoder.decode(packets, basis(), signs)
        self.assertEqual(len(decoder._signs), 128)


if __name__ == "__main__":
    unittest.main()
