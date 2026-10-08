import io
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np
from safetensors import safe_open

from tools.model.gguf import convert, read_header


def string(value):
    value = value.encode()
    return struct.pack("<Q", len(value)) + value


def fixture(*, q2_kind=42, q2_offset=0, dense_offset=64, truncate=0, duplicate=False):
    header = b"GGUF" + struct.pack("<IQQ", 3, 2, 2)
    header += string("general.alignment") + struct.pack("<II", 4, 32)
    header += string("general.architecture") + struct.pack("<I", 8) + string("qwen4_exp")
    header += string("experts") + struct.pack("<IQQIQ", 2, 64, 2, q2_kind, q2_offset)
    header += string("experts" if duplicate else "norm") + struct.pack(
        "<IQIQ", 1, 3, 1, dense_offset
    )
    header += b"\0" * (-len(header) % 32)
    # Two real Q2_0 groups, including a negative FP16 scale.
    q2 = struct.pack("<e", -0.5) + bytes([0xE4] * 16)
    q2 += struct.pack("<e", 0.25) + bytes([0x1B] * 16)
    payload = q2 + b"\0" * 28 + np.array([1, -2, 0.5], dtype="<f2").tobytes()
    data = header + payload
    return data[:-truncate] if truncate else data, len(header), payload


class GGUFConversionTests(unittest.TestCase):
    def test_conversion_preserves_packed_weights_dense_dtype_and_alignment(self):
        data, start, payload = fixture()
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "source.gguf", Path(directory) / "model.safetensors"
            source.write_bytes(data)
            parsed = convert(source, output)
            self.assertEqual(parsed.data_start, start)
            with safe_open(output, framework="np") as reader:
                self.assertEqual(reader.get_slice("experts").get_shape(), [2, 18])
                self.assertEqual(reader.get_slice("norm").get_dtype(), "F16")
                self.assertEqual(reader.get_tensor("experts").tobytes(), payload[:36])
                np.testing.assert_array_equal(reader.get_tensor("norm"), [1, -2, 0.5])
                self.assertEqual(reader.metadata()["ggml.type.experts"], "42")
                self.assertEqual(reader.metadata()["ggml.dimensions.experts"], "[64, 2]")
            raw = output.read_bytes()
            prefix = 8 + struct.unpack("<Q", raw[:8])[0]
            self.assertEqual(raw[prefix:], payload)
            self.assertEqual(source.read_bytes(), data)

    def test_partial_header_fails_and_does_not_guess_data_extent(self):
        data, start, _ = fixture()
        with self.assertRaises(EOFError):
            read_header(io.BytesIO(data[: start - 8]), file_size=len(data))

    def test_rejects_unknown_type_duplicate_overlap_alignment_and_truncation(self):
        for kwargs in [
            dict(q2_kind=255),
            dict(duplicate=True),
            dict(dense_offset=32),
            dict(q2_offset=1),
            dict(truncate=1),
        ]:
            with self.subTest(kwargs=kwargs):
                data, _, _ = fixture(**kwargs)
                with self.assertRaises(ValueError):
                    read_header(io.BytesIO(data), file_size=len(data))

    def test_existing_output_is_never_overwritten_or_removed(self):
        data, _, _ = fixture()
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "source.gguf", Path(directory) / "model.safetensors"
            source.write_bytes(data)
            output.write_bytes(b"keep")
            with self.assertRaises(ValueError):
                convert(source, output)
            self.assertEqual(output.read_bytes(), b"keep")


if __name__ == "__main__":
    unittest.main()
