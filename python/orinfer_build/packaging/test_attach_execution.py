"""Attaching compatible bindings preserves source data and publishes atomically."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from orinfer_build.common.files import write_json
from orinfer_build.packaging.attach_execution import attach
from tests.support.model_package import data_package


class ExecutionReplacementTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source = self.root / "source"
        weights = self.source / "cache/weights"
        weights.mkdir(parents=True)
        (weights / "one.safetensors").write_bytes(b"unchanged weights")
        self.descriptor = dict(
            schema_version=1,
            architecture="qwen3_5",
            compute_policy="int8_quality",
            execution_package="0" * 64,
        )
        write_json(self.source / "cache/model.json", self.descriptor)
        (self.source / "config.json").write_text("{}")
        self.package = data_package(self.root)
        self.output = self.root / "new"
        self.engine = Path("/bin/true")

    @patch("orinfer_build.packaging.publication.subprocess.run")
    def test_attach_keeps_source_and_publishes_the_exact_replacement(self, check):
        digest = attach(self.source, self.output, self.package, self.engine)
        check.assert_called_once()
        published = self.output / "cache/packages" / digest
        self.assertEqual(digest, self.package.name)
        self.assertEqual(
            json.loads((self.source / "cache/model.json").read_text()), self.descriptor
        )
        self.assertEqual(
            json.loads((self.output / "cache/model.json").read_text()),
            dict(self.descriptor, execution_package=digest),
        )
        for asset in ("package.json", "bindings.json", "assets.safetensors"):
            self.assertEqual((published / asset).read_bytes(), (self.package / asset).read_bytes())
        weights = "cache/weights/one.safetensors"
        self.assertEqual(
            (self.source / weights).stat().st_ino, (self.output / weights).stat().st_ino
        )

    @patch(
        "orinfer_build.packaging.publication.subprocess.run",
        side_effect=RuntimeError("incompatible config signature"),
    )
    def test_validation_failure_leaves_source_and_destination_intact(self, _):
        with self.assertRaisesRegex(RuntimeError, "incompatible config signature"):
            attach(self.source, self.output, self.package, self.engine)
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob(".new-*")))

    def test_mismatched_model_schema_and_architecture_are_rejected(self):
        for change in (dict(schema_version=2), dict(architecture="flash_next")):
            with self.subTest(change=change):
                write_json(self.source / "cache/model.json", dict(self.descriptor, **change))
                with self.assertRaises(ValueError):
                    attach(self.source, self.output, self.package, self.engine)
                self.assertFalse(self.output.exists())

    def test_nested_output_is_rejected_without_changing_source(self):
        with self.assertRaisesRegex(ValueError, "outside the immutable source"):
            attach(self.source, self.source / "nested", self.package, self.engine)
        self.assertFalse((self.source / "nested").exists())

    def test_corrupt_binding_is_rejected_before_publication(self):
        (self.package / "bindings.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "sha256 mismatch"):
            attach(self.source, self.output, self.package, self.engine)
        self.assertFalse(self.output.exists())
