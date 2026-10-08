"""Execution-library replacements preserve source ownership and publish atomically."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.model.attach_execution import attach


class ExecutionReplacementTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        weights = self.source / "cache/weights"
        weights.mkdir(parents=True)
        (weights / "one.safetensors").write_bytes(b"unchanged weights")
        self.library = self.root / "model.so"
        header = bytearray(20)
        header[:6] = b"\x7fELF\x02\x01"
        header[16:20] = bytes((3, 0, 183, 0))
        self.library.write_bytes(header)
        identity = patch(
            "tools.model.attach_execution.execution_identity",
            return_value=dict(
                abi_version=1,
                package="test-model",
                version="1",
                library=dict(file="lib/model.so", sha256=hashlib.sha256(header).hexdigest()),
            ),
        )
        identity.start()
        self.addCleanup(identity.stop)
        raw = json.dumps(
            dict(
                schema_version=1,
                runtime_abi=1,
                kernels=[],
                execution=dict(
                    abi_version=1,
                    package="test-model",
                    version="1",
                    library=dict(file="lib/model.so", sha256=hashlib.sha256(header).hexdigest()),
                ),
            )
        ).encode()
        digest = hashlib.sha256(raw).hexdigest()
        self.package = self.source / "cache/packages" / digest
        (self.package / "lib").mkdir(parents=True)
        (self.package / "lib/model.so").write_bytes(header)
        (self.package / "package.json").write_bytes(raw)
        (self.source / "cache/model.json").write_text(
            json.dumps(dict(schema_version=1, execution_package=digest))
        )
        (self.source / "config.json").write_text("{}")

    @patch("tools.model.publication.subprocess.run")
    def test_library_replacement_does_not_modify_source(self, check):
        before = (self.package / "package.json").read_bytes()
        output = self.root / "new"
        digest = attach(self.source, output, self.library, Path("/bin/true"), "test-model", "1")
        check.assert_called_once()
        data = json.loads((output / "cache/model.json").read_text())
        self.assertEqual(data["schema_version"], 1)
        self.assertNotIn("operator_package", data)
        self.assertEqual(data["execution_package"], digest)
        package = output / "cache/packages" / digest
        manifest = json.loads((package / "package.json").read_text())
        self.assertEqual(
            manifest["execution"]["library"]["sha256"],
            hashlib.sha256(self.library.read_bytes()).hexdigest(),
        )
        self.assertEqual((self.package / "package.json").read_bytes(), before)
        self.assertEqual(
            (self.source / "cache/weights/one.safetensors").stat().st_ino,
            (output / "cache/weights/one.safetensors").stat().st_ino,
        )
        self.assertEqual((package / "lib/model.so").read_bytes(), self.library.read_bytes())

    @patch(
        "tools.model.publication.subprocess.run",
        side_effect=RuntimeError("incompatible native ABI"),
    )
    def test_native_validation_failure_never_publishes(self, _):
        output = self.root / "failed"
        with self.assertRaisesRegex(RuntimeError, "incompatible native ABI"):
            attach(self.source, output, self.library, Path("/bin/true"), "test-model", "1")
        self.assertFalse(output.exists())
        self.assertFalse(list(self.root.glob(".failed-*")))
        self.assertTrue((self.package / "package.json").exists())

    def test_identity_assertions_reject_a_mismatched_library_version(self):
        output = self.root / "mismatched"
        with self.assertRaisesRegex(ValueError, "differs from the native library"):
            attach(self.source, output, self.library, Path("/bin/true"), "test-model", "old")
        self.assertFalse(output.exists())

    def test_nested_output_is_rejected_without_changing_source(self):
        for output in (self.source, self.source / "nested"):
            with self.subTest(output=output):
                with self.assertRaisesRegex(ValueError, "outside the immutable source"):
                    attach(self.source, output, self.library, Path("/bin/true"), "test-model", "1")
        self.assertFalse((self.source / "nested").exists())
        self.assertTrue((self.package / "package.json").exists())

    def test_schema_2_is_rejected_without_publication(self):
        path = self.source / "cache/model.json"
        descriptor = json.loads(path.read_text())
        descriptor["schema_version"] = 2
        path.write_text(json.dumps(descriptor))
        with self.assertRaisesRegex(ValueError, "schema 1"):
            attach(
                self.source,
                self.root / "failed",
                self.library,
                Path("/bin/true"),
                "test-model",
                "1",
            )
        self.assertFalse((self.root / "failed").exists())

    def test_bad_library_or_corrupt_source_fails_before_publication(self):
        self.library.write_bytes(b"not an ELF library")
        with self.assertRaisesRegex(ValueError, "ELF"):
            attach(
                self.source,
                self.root / "failed",
                self.library,
                Path("/bin/true"),
                "test-model",
                "1",
            )
        self.assertFalse((self.root / "failed").exists())
