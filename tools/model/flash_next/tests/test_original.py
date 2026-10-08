import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from tools.model.flash_next.reference.original import Original
from tools.model.safetensors_source import Source


class OriginalTests(unittest.TestCase):
    def test_original_bf16_bits_and_expert_bound_without_full_bank_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            name = "model.language_model.layers.0.mlp.experts.gate_up_proj"
            bits = np.array(
                [0x3F80, 0xBF00, 0x0001, 0x8000, 0x4000, 0xC000, 0x3E80, 0x0000], dtype="<u2"
            )
            data = bits.tobytes()
            header = json.dumps(
                {name: {"dtype": "BF16", "shape": [2, 2, 2], "data_offsets": [0, len(data)]}}
            ).encode()
            shard = root / "weights.safetensors"
            shard.write_bytes(struct.pack("<Q", len(header)) + header + data)
            index = root / "index.json"
            index.write_text(json.dumps({"weight_map": {name: shard.name}}))
            reader = Original(Source(index, directory=root), {}, max_read_bytes=8)
            for expert in (0, 1):
                actual = reader.rows(name, expert, 1)
                expected = (
                    (bits[expert * 4 : expert * 4 + 4].astype(np.uint32) << 16)
                    .view(np.float32)
                    .reshape(1, 2, 2)
                )
                np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
            batched = reader.batch_rows([(name, 1, 1), (name, 0, 1), (name, 1, 1)], workers=2)
            self.assertEqual(len(batched), 2)
            np.testing.assert_array_equal(batched[(name, 1, 1)], reader.rows(name, 1, 1))
            with self.assertRaises(ValueError):
                reader.batch_rows([(name, 0, 1)], max_total_bytes=1)
            self.assertEqual(sum(row["bytes"] for row in reader.reads.values()), 16)
            with self.assertRaises(ValueError):
                reader.tensor(name)
            with self.assertRaises(ValueError):
                reader.rows(name, True, 1)
            # Changes to even a previously finite tiny subnormal are caught.
            with shard.open("r+b") as f:
                f.seek(8 + len(header) + 4)
                f.write(b"\x02\x00")
            with self.assertRaises(ValueError):
                reader.rows(name, 0, 1)

    def test_scalar_metadata_and_nonfinite_are_not_reinterpreted(self):
        np.testing.assert_array_equal(
            Original.decode(struct.pack("<q", 23703573157769), "I64", ()), 23703573157769
        )
        for bits in (0x7F80, 0x7FC0):
            with self.assertRaises(ValueError):
                Original.decode(struct.pack("<H", bits), "BF16", (1,))


if __name__ == "__main__":
    unittest.main()
