import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.reference import checkpoint_sha256, reference_path


class ReferenceInputsTests(unittest.TestCase):
    def test_checkpoint_identity_follows_the_supplied_file(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "first", Path(directory) / "second"
            first.write_bytes(b"checkpoint A")
            second.write_bytes(b"checkpoint B")
            self.assertEqual(
                checkpoint_sha256(first), hashlib.sha256(first.read_bytes()).hexdigest()
            )
            self.assertEqual(
                checkpoint_sha256(second), hashlib.sha256(second.read_bytes()).hexdigest()
            )
            self.assertNotEqual(checkpoint_sha256(first), checkpoint_sha256(second))

    def test_external_path_with_spaces_is_not_reinterpreted(self):
        with patch.dict("os.environ", ORINFER_REFERENCE_CHECKPOINT="/external/model with spaces"):
            self.assertEqual(
                reference_path("ORINFER_REFERENCE_CHECKPOINT", "checkpoint"),
                Path("/external/model with spaces"),
            )
