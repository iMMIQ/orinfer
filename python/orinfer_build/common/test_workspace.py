from orinfer_build.common.paths import gpu_lock_path, work_path
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.dev.workspace import initialize


class WorkspaceTests(unittest.TestCase):
    def test_two_checkouts_share_lock_and_cache_but_have_private_builds(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            base = Path(directory)
            a, b, output = base / "one", base / "two", base / "work"
            a.mkdir()
            b.mkdir()
            initialize(a, output, "one")
            initialize(b, output, "two")
            self.assertTrue((a / "target").resolve() != (b / "target").resolve())
            self.assertEqual(gpu_lock_path(a).stat().st_ino, gpu_lock_path(b).stat().st_ino)
            self.assertEqual(
                work_path("cache", "kernels", root=a), work_path("cache", "kernels", root=b)
            )
            initialize(a, output, "one")

    def test_existing_build_data_is_never_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "checkout"
            (root / "target").mkdir(parents=True)
            (root / "target/keep").write_text("data")
            with self.assertRaisesRegex(ValueError, "Preserve existing"):
                initialize(root, base / "work", "one")
            self.assertEqual((root / "target/keep").read_text(), "data")
            self.assertFalse((root / "artifacts/layout.json").exists())
