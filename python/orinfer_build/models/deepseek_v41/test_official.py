import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from orinfer_build.models.deepseek_v41.official import (
    Checkpoint,
    Decoder,
    FP4_VALUES,
    FP8_VALUES,
    decode_scale,
)


class OfficialCheckpointTests(unittest.TestCase):
    def test_native_fp4_decoding_preserves_every_code_and_scale_boundary(self):
        with tempfile.TemporaryDirectory() as temporary:
            decoder = Decoder(temporary)
            packed = np.tile(np.arange(256, dtype=np.uint8), (3, 1))
            scale = np.tile(decode_scale(np.array([0, 126, 127, 128], dtype=np.uint8)), (3, 4))
            codes = np.empty((3, 512), dtype=np.uint8)
            codes[:, ::2], codes[:, 1::2] = packed & 15, packed >> 4
            expected = FP4_VALUES[codes] * np.repeat(scale, 32, axis=1)
            actual = decoder.fp4(packed, scale)
            np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
            with self.assertRaises(ValueError):
                decoder.fp4(packed[:, ::2], scale)

    def test_float_codes_and_power_of_two_scales(self):
        np.testing.assert_array_equal(FP4_VALUES[:8], [0, 0.5, 1, 1.5, 2, 3, 4, 6])
        self.assertTrue(np.signbit(FP4_VALUES[8]))
        np.testing.assert_array_equal(FP8_VALUES[[0, 1, 56, 126, 184]], [0, 2**-9, 1, 448, -1])
        np.testing.assert_array_equal(
            decode_scale(np.array([126, 127, 128], dtype=np.uint8)), [0.5, 1, 2]
        )
        with self.assertRaises(ValueError):
            decode_scale(np.array([255], dtype=np.uint8))

    def test_official_mixed_matrices_partial_rows_and_gather(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            tensors = {
                "dense.weight": ("F8_E4M3", np.full((64, 32), 56, dtype=np.uint8)),
                "dense.scale": ("F8_E8M0", np.array([[127], [128]], dtype=np.uint8)),
                "expert.weight": ("I8", np.full((2, 16), 0x92, dtype=np.uint8)),
                "expert.scale": ("F8_E8M0", np.array([[127], [128]], dtype=np.uint8)),
                "embed.weight": ("F8_E4M3", np.full((3, 64), 56, dtype=np.uint8)),
                "embed.scale": (
                    "F8_E8M0",
                    np.array([[127, 128], [128, 129], [126, 127]], dtype=np.uint8),
                ),
            }
            header, payload = {}, bytearray()
            for name, (dtype, tensor) in tensors.items():
                begin = len(payload)
                payload.extend(tensor.tobytes())
                header[name] = {
                    "dtype": dtype,
                    "shape": list(tensor.shape),
                    "data_offsets": [begin, len(payload)],
                }
            encoded = json.dumps(header).encode()
            (directory / "weights.safetensors").write_bytes(
                struct.pack("<Q", len(encoded)) + encoded + payload
            )
            (directory / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": dict.fromkeys(tensors, "weights.safetensors")})
            )
            (directory / "config.json").write_text(
                json.dumps(
                    {
                        "quantization_config": {
                            "expert_dtype": "fp4",
                            "scale_fmt": "ue8m0",
                            "weight_block_size": [32, 32],
                        }
                    }
                )
            )
            checkpoint = Checkpoint(directory)
            try:
                actual = checkpoint.matrix("dense.weight", 31, 2)
                np.testing.assert_array_equal(actual[:, 0], [1, 2])
                expert = checkpoint.matrix("expert.weight")
                np.testing.assert_array_equal(expert[:, :2], [[1, -0.5], [2, -1]])
                np.testing.assert_array_equal(
                    checkpoint.rows("dense.weight", [33, 0, 33])[:, 0], [2, 1, 2]
                )
                np.testing.assert_array_equal(
                    checkpoint.rows("embed.weight", [2, 0, 2])[:, [0, 32]],
                    [[0.5, 1], [1, 2], [0.5, 1]],
                )
                with self.assertRaises(ValueError):
                    checkpoint.read("dense.weight", 64, 1)
                with self.assertRaises(ValueError):
                    checkpoint.read("dense.weight", max_bytes=1)
            finally:
                checkpoint.close()


if __name__ == "__main__":
    unittest.main()
