"""Check lossless HC payloads, including escaped exponents and raw special bits."""

import unittest

import numpy as np

from orinfer_build.models.flash_next.hc_bf16 import pack, unpack


class HcBf16CacheTests(unittest.TestCase):
    def test_all_bf16_bit_patterns_round_trip(self):
        raw = np.arange(65536, dtype=np.uint16).reshape(-1, 64)
        encoded = pack(raw)
        self.assertGreater(encoded["patch_count"], 0)
        np.testing.assert_array_equal(unpack(encoded), raw)

    def test_three_tiny_outliers_and_signed_mantissas(self):
        raw = np.arange(64, dtype=np.uint16) + np.uint16(120 << 7)
        raw[::2] |= np.uint16(0x8000)
        raw[:3] = np.array([0, 0x8000, 1], dtype=np.uint16)
        encoded = pack(raw.reshape(1, 64))
        self.assertEqual(encoded["patch_count"], 3)
        np.testing.assert_array_equal(unpack(encoded).reshape(-1), raw)

    def test_no_escape_sentinel_and_weight_bytes(self):
        raw = np.full((2, 64), 120 << 7, dtype=np.uint16)
        encoded = pack(raw)
        self.assertEqual(encoded["patch_count"], 0)
        self.assertEqual(encoded["patches"].nbytes, 8)
        self.assertEqual(encoded["low"].nbytes + encoded["high"].nbytes, 13 * raw.size // 8)
        np.testing.assert_array_equal(unpack(encoded), raw)


if __name__ == "__main__":
    unittest.main()
