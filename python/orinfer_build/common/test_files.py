"""Offline metadata failures preserve committed files and release GPU locks."""

import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from orinfer_build.common.files import atomic_json, file_hash, source_hash
from orinfer_build.common.gpu import gpu_lock


class BuildFilesTests(unittest.TestCase):
    def test_failed_metadata_publication_preserves_previous_file(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "state.json"
            atomic_json(path, {"name": "音频"})
            original = path.read_bytes()
            with self.assertRaises(TypeError):
                atomic_json(path, {"bad": object()})
            self.assertEqual(path.read_bytes(), original)
            with patch("orinfer_build.common.files.os.fsync", side_effect=OSError("disk failure")):
                with self.assertRaisesRegex(OSError, "disk failure"):
                    atomic_json(path, {"version": 2})
            self.assertEqual(path.read_bytes(), original)
            with patch.object(Path, "replace", side_effect=OSError("rename failure")):
                with self.assertRaisesRegex(OSError, "rename failure"):
                    atomic_json(path, {"version": 3})
            self.assertEqual(list(root.iterdir()), [path])
            self.assertEqual(file_hash(str(path)), hashlib.sha256(original).hexdigest())

    def test_source_identity_tracks_names_contents_and_stable_order(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            a, b = root / "a.py", root / "b.py"
            a.write_text("a")
            b.write_text("b")
            before = source_hash(root, [a, b])
            self.assertEqual(before, source_hash(root, [b, a, b]))
            a.write_text("changed")
            self.assertNotEqual(before, source_hash(root, [a, b]))
            a.write_text("a")
            c = a.rename(root / "c.py")
            self.assertNotEqual(before, source_hash(root, [c, b]))

    def test_gpu_lock_covers_exception_and_releases_for_other_processes(self):
        fake = SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: False))
        with tempfile.TemporaryDirectory() as folder, patch.dict(sys.modules, torch=fake):
            path = Path(folder) / "gpu.lock"

            def available():
                return (
                    subprocess.run(
                        [
                            sys.executable,
                            "-c",
                            "import fcntl,sys; f=open(sys.argv[1],'a'); "
                            "fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)",
                            str(path),
                        ],
                        capture_output=True,
                    ).returncode
                    == 0
                )

            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                with gpu_lock(path):
                    self.assertFalse(available())
                    raise RuntimeError("cancelled")
            self.assertTrue(available())


class SharedPlanTests(unittest.TestCase):
    def test_shared_plan_protocol_preserves_scalar_types_and_repeated_subtrees(self):
        import json
        from orinfer_build.common.shared_json import decode, encode

        value = {"a": ["op", 1, 1.0], "b": ["op", 1, 1.0]}
        # This protocol vector is also consumed by the Rust plan decoder test.
        expected = {
            "encoding": "shared-json-v1",
            "nodes": ["op", 1, 1.0, [0, 1, 2], {"a": 3, "b": 3}],
            "root": 4,
        }
        self.assertEqual(json.dumps(encode(value)), json.dumps(expected))
        self.assertEqual(json.dumps(decode(expected)), json.dumps(value))
        with self.assertRaisesRegex(ValueError, "precede"):
            decode({"encoding": "shared-json-v1", "nodes": [[0]], "root": 0})

    def test_rearchiving_an_installed_package_omits_derived_payloads(self):
        import json
        import tarfile
        from orinfer_build.packaging.package import archive as model_archive
        from tools.release.package import archive as release_archive

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            package = root / ("a" * 64)
            package.mkdir()
            (package / "generated.bin").write_bytes(b"large derived output")
            (package / "patch.xz").write_bytes(b"packed corrections")
            (package / "package.json").write_text(
                json.dumps({"generated_assets": [{"output": {"file": "generated.bin"}}]})
            )
            for label, encode in [
                ("model", lambda path: model_archive(package, path)),
                ("release", lambda path: release_archive(package, path, 0)),
            ]:
                path = root / (label + ".tar.xz")
                encode(path)
                with tarfile.open(path) as archive:
                    names = archive.getnames()
                    self.assertNotIn(package.name + "/generated.bin", names)
                    self.assertIn(package.name + "/patch.xz", names)
                self.assertEqual((package / "generated.bin").read_bytes(), b"large derived output")
