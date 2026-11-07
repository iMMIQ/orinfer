"""Persistent pread lifetime, bounded rows and shared format rejection."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from safetensors.numpy import save_file

from orinfer_build.weights.local import LocalSource


class LocalSourceTests(unittest.TestCase):
    def fixture(self, root):
        values = np.arange(24, dtype="<f4").reshape(6, 4)
        save_file({"x": values, "scalar": np.array(7, dtype="<i4")}, root / "weights.safetensors")
        (root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": dict.fromkeys(["x", "scalar"], "weights.safetensors")})
        )
        return values

    def test_reads_only_requested_rows_reuses_fd_and_preserves_scalar(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = self.fixture(root)
            source = LocalSource(root)
            try:
                with patch("orinfer_build.weights.local.os.pread", wraps=os.pread) as read:
                    source.metadata("x")
                    self.assertEqual(read.call_count, 2)
                    fd = source.files["weights.safetensors"]
                    read.reset_mock()
                    self.assertEqual(
                        source.read_rows("x", 2, 2, max_bytes=32), values[2:4].tobytes()
                    )
                    self.assertEqual(read.call_count, 1)
                    self.assertEqual(read.call_args.args[:2], (fd, 32))
                    self.assertEqual(source.read_rows("scalar"), np.array(7, dtype="<i4").tobytes())
                for first, count, budget in [(-1, 1, 32), (5, 2, 32), (0, 2, 31)]:
                    with self.assertRaises(ValueError):
                        source.read_rows("x", first, count, max_bytes=budget)
            finally:
                source.close()
            self.assertFalse(source.files)

    def test_corrupt_header_closes_fd_and_retry_uses_repaired_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = self.fixture(root)
            path = root / "weights.safetensors"
            original = path.read_bytes()
            path.write_bytes(original[:-1])
            source = LocalSource(root)
            with patch("orinfer_build.weights.local.os.close", wraps=os.close) as close:
                with self.assertRaisesRegex(ValueError, "payload size mismatch"):
                    source.metadata("x")
                close.assert_called_once()
            self.assertFalse(source.files)
            path.write_bytes(original)
            try:
                self.assertEqual(source.read_rows("x"), values.tobytes())
            finally:
                source.close()

    def test_resolved_shard_cannot_leave_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            (model / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"x": "outside.safetensors"}})
            )
            (root / "outside").write_bytes(b"private")
            (model / "outside.safetensors").symlink_to(root / "outside")
            source = LocalSource(model)
            with self.assertRaisesRegex(ValueError, "leaves model directory"):
                source.metadata("x")
            self.assertFalse(source.files)
