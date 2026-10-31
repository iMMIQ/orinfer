"""Independent nibble-address checks for lossless W4 storage transforms."""

import unittest
import tempfile
from pathlib import Path

import numpy as np

from tools.model.deepseek_v41.pack_host_release import record_chunks
from tools.model.deepseek_v41.weight_layout import (
    restore_nd,
    transposed_nz_blocks,
    transposed_nz_file_chunks,
)


class WeightLayoutTests(unittest.TestCase):
    def test_nz_addresses_preserve_signed_int4_codes_and_both_nibbles(self):
        e, k, n = 2, 256, 64
        original = np.random.default_rng(20261002).integers(0, 256, (e, k, n // 2), dtype=np.uint8)
        nz = np.stack(list(transposed_nz_blocks(original))).reshape(e, k // 64, n // 16, 16, 32)
        flat = nz.reshape(-1)
        for expert in range(e):
            for row in [0, 1, 15, 16, 63, 64, 127, 255]:
                for column in [0, 1, 15, 16, 31, 32, 63]:
                    source = int(original[expert, row, column // 2]) >> ((column % 2) * 4) & 15
                    address = (
                        ((expert * (k // 64) + row // 64) * (n // 16) + column // 16) * 16
                        + column % 16
                    ) * 32 + row % 64 // 2
                    actual = int(flat[address]) >> ((row % 2) * 4) & 15
                    self.assertEqual((actual ^ 8) - 8, (source ^ 8) - 8)
        self.assertTrue(np.array_equal(restore_nd(nz), original))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bank.bin"
            path.write_bytes(b"prefix!" + original.tobytes() + b"tail")
            streamed = b"".join(transposed_nz_file_chunks(path, 7, e, k, n))
            self.assertEqual(streamed, nz.tobytes())

    def test_host_records_preserve_scales_bits_and_chunk_boundaries(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "host.bin"
            values = bytes(range(256)) + bytes(range(64))
            scales = bytes.fromhex("000000800000803f0100c07f00000040") * 5
            path.write_bytes(b"prefix!" + values + b"gap" + scales)
            actual = b"".join(record_chunks((path, 7), (path, 7 + len(values) + 3), 5, 64, 2))
            expected = b"".join(
                values[i * 64 : (i + 1) * 64] + scales[i * 8 : (i + 1) * 8] for i in range(5)
            )
            self.assertEqual(actual, expected)
            with self.assertRaises(EOFError):
                b"".join(record_chunks((path, path.stat().st_size - 1), (path, 0), 5, 64))

    def test_storage_alignment_is_explicit(self):
        for shape in [(1, 32, 32), (1, 64, 9), (0, 64, 32)]:
            with self.assertRaises(ValueError):
                list(transposed_nz_blocks(np.zeros(shape, dtype=np.uint8)))
            with self.assertRaises(ValueError):
                list(transposed_nz_file_chunks("unused", 0, shape[0], shape[1], shape[2] * 2))


if __name__ == "__main__":
    unittest.main()
