"""Publication failures must not expose partial models or modify their source."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from orinfer_build.packaging.publication import (
    atomic_model,
    clone_model,
    commit_package,
    load_model,
    link_or_copy,
    seed_compile_cache,
)


class PublicationTests(unittest.TestCase):
    def test_compile_cache_seeding_preserves_platform_and_live_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "source", root / "target"
            target.mkdir()
            for layout in ("v1/kernels", "v2/linux-aarch64/kernels"):
                good = source / layout / "complete"
                good.mkdir(parents=True)
                (good / "kernel.cubin").write_bytes(b"compiled")
                live = target / layout / "live"
                live.mkdir(parents=True)
                (live / "kernel.cubin").write_bytes(b"in progress")
                other = source / layout / "live"
                other.mkdir()
                (other / "kernel.cubin").write_bytes(b"older")
            seed_compile_cache(source, target)
            for layout in ("v1/kernels", "v2/linux-aarch64/kernels"):
                self.assertEqual(
                    (target / layout / "complete/kernel.cubin").read_bytes(), b"compiled"
                )
                self.assertEqual(
                    (target / layout / "live/kernel.cubin").read_bytes(), b"in progress"
                )
                self.assertEqual((source / layout / "live/kernel.cubin").read_bytes(), b"older")

    def test_replacing_a_cloned_asset_does_not_overwrite_the_source_inode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "original"
            clone = root / "clone"
            replacement = root / "replacement"
            source.write_bytes(b"old kernel")
            os.link(source, clone)
            replacement.write_bytes(b"new kernel")
            link_or_copy(replacement, clone)
            self.assertEqual(source.read_bytes(), b"old kernel")
            self.assertEqual(clone.read_bytes(), b"new kernel")
            link_or_copy(clone, clone)
            self.assertEqual(clone.read_bytes(), b"new kernel")

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        seal = patch(
            "orinfer_build.contracts.catalog.seal", side_effect=lambda package, *args: package
        )
        seal.start()
        self.addCleanup(seal.stop)
        self.source = self.root / "source"
        weights = self.source / "cache/weights"
        weights.mkdir(parents=True)
        (weights / "model.safetensors").write_bytes(b"immutable fixture weights")
        (weights / "model.safetensors.index.json").write_text("{}")
        (self.source / "config.json").write_text("{}")
        self.package = dict(kernels=[], revision=1)
        raw = json.dumps(self.package).encode()
        digest = hashlib.sha256(raw).hexdigest()
        self.operator = self.source / "cache/packages" / digest
        self.operator.mkdir(parents=True)
        (self.operator / "package.json").write_bytes(raw)
        self.data = dict(execution_package=digest, metadata={})
        (self.source / "cache/model.json").write_text(json.dumps(self.data))
        self.addCleanup(patch.stopall)
        patch.dict(os.environ, ORINFER_EXECUTION_CACHE=str(self.root / "cache")).start()

    @patch("orinfer_build.packaging.publication.subprocess.run")
    def test_validation_precedes_publication_and_metadata_does_not_alias(self, validate):
        destination = self.root / "output"
        original = (self.operator / "package.json").read_bytes()
        with atomic_model(destination, Path("/usr/bin/true")) as staging:
            data, origin, package = load_model(self.source)
            operator = clone_model(self.source, staging, origin)
            index = staging / "cache/weights/model.safetensors.index.json"
            index.write_text('{"changed":true}')
            package["revision"] = 2
            digest = commit_package(staging, operator, data, package)
            self.assertFalse(destination.exists())
            self.assertEqual((self.operator / "package.json").read_bytes(), original)
        validate.assert_called_once()
        self.assertTrue(destination.exists())
        self.assertEqual(
            (self.source / "cache/weights/model.safetensors.index.json").read_text(), "{}"
        )
        published = destination / "cache/packages" / digest / "package.json"
        self.assertEqual(hashlib.sha256(published.read_bytes()).hexdigest(), digest)
        self.assertEqual(
            (destination / "cache/weights/model.safetensors").stat().st_ino,
            (self.source / "cache/weights/model.safetensors").stat().st_ino,
        )

    @patch(
        "orinfer_build.packaging.publication.subprocess.run",
        side_effect=subprocess.CalledProcessError(1, "validator"),
    )
    def test_failed_validation_cleans_staging_and_leaves_output_absent(self, validate):
        destination = self.root / "output"
        with self.assertRaises(subprocess.CalledProcessError):
            with atomic_model(destination, Path("/usr/bin/true")) as staging:
                clone_model(self.source, staging, self.operator)
        self.assertFalse(destination.exists())
        self.assertEqual(list(self.root.glob(".output-*")), [])

    def test_corrupted_source_and_destination_inside_source_are_rejected(self):
        with self.assertRaises(ValueError):
            clone_model(self.source, self.source / "nested", self.operator)
        (self.operator / "package.json").write_text("{}")
        with self.assertRaises(ValueError):
            load_model(self.source)

    def test_model_clone_retains_cpu_tables_and_independent_metadata(self):
        cpu = self.source / "cache/cpu"
        cpu.mkdir()
        (cpu / "inputs.json").write_text('{"version":1}')
        (cpu / "rows.safetensors").write_bytes(b"immutable CPU rows")
        destination = self.root / "with-cpu"
        clone_model(self.source, destination, self.operator)
        copied = destination / "cache/cpu"
        self.assertEqual(
            (copied / "rows.safetensors").stat().st_ino, (cpu / "rows.safetensors").stat().st_ino
        )
        (copied / "inputs.json").write_text('{"version":2}')
        self.assertEqual((cpu / "inputs.json").read_text(), '{"version":1}')

    def test_compiler_failure_and_existing_output_preserve_ownership(self):
        destination = self.root / "output"
        with self.assertRaises(RuntimeError):
            with atomic_model(destination, Path("/usr/bin/true")) as staging:
                clone_model(self.source, staging, self.operator)
                raise RuntimeError("compilation failed")
        self.assertFalse(destination.exists())
        destination.mkdir()
        with self.assertRaises(ValueError):
            with atomic_model(destination):
                self.fail("Existing output must not be entered")


if __name__ == "__main__":
    unittest.main()
