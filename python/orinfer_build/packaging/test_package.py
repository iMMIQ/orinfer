"""Data-driven schema-2 execution archive identity and atomic installation checks."""

import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from orinfer_build.packaging.package import archive, install
from tests.support.model_package import data_package


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.package = data_package(self.root)
        self.digest = self.package.name

    def test_archive_roundtrip_and_duplicate_install(self):
        path = self.root / "operators.tar.xz"
        archive(self.package, path)
        cache = self.root / "installed"
        self.assertEqual(install(path, cache), self.digest)
        self.assertEqual(
            (cache / self.digest / "bindings.json").read_bytes(),
            (self.package / "bindings.json").read_bytes(),
        )
        with self.assertRaisesRegex(ValueError, "already installed"):
            install(path, cache)
        with self.assertRaisesRegex(ValueError, "already exists"):
            archive(self.package, path)
        self.assertFalse(list(cache.glob(".install-*")))

    def assert_rejected(self, pattern):
        path = self.root / "corrupt.tar.xz"
        archive(self.package, path)
        cache = self.root / "installed"
        with self.assertRaisesRegex(ValueError, pattern):
            install(path, cache)
        self.assertFalse((cache / self.digest).exists())
        self.assertFalse(list(cache.glob(".install-*")))

    def test_modified_bindings_never_install(self):
        (self.package / "bindings.json").write_bytes(b"changed bindings")
        self.assert_rejected("sha256 mismatch")

    def test_corrupt_kernel_never_installs(self):
        path = self.package / "assets.safetensors"
        raw = bytearray(path.read_bytes())
        raw[-1] ^= 1
        path.write_bytes(raw)
        self.assert_rejected("sha256 mismatch")

    def test_modified_manifest_never_installs(self):
        (self.package / "package.json").write_text("{}")
        self.assert_rejected("digest mismatch")

    def test_unsafe_archive_members_are_rejected_before_extraction(self):
        for kind in ("traversal", "symlink"):
            with self.subTest(kind=kind):
                path = self.root / (kind + ".tar.xz")
                with tarfile.open(path, "w:xz") as stream:
                    member = tarfile.TarInfo(
                        self.digest + ("/../outside" if kind == "traversal" else "/link")
                    )
                    if kind == "symlink":
                        member.type = tarfile.SYMTYPE
                        member.linkname = "../../outside"
                        stream.addfile(member)
                    else:
                        member.size = 1
                        stream.addfile(member, io.BytesIO(b"x"))
                cache = self.root / ("cache-" + kind)
                with self.assertRaises(ValueError):
                    install(path, cache)
                self.assertFalse(list(cache.iterdir()))
                self.assertFalse((self.root / "outside").exists())

    def test_invalid_table_index_is_rejected_even_with_matching_hashes(self):
        binding_path = self.package / "bindings.json"
        registry = json.loads(binding_path.read_bytes())
        registry["templates"][0]["choice"] = -1
        raw = json.dumps(registry).encode()
        binding_path.write_bytes(raw)
        manifest_path = self.package / "package.json"
        package = json.loads(manifest_path.read_bytes())
        package["bindings"]["sha256"] = hashlib.sha256(raw).hexdigest()
        raw = json.dumps(package).encode()
        manifest_path.write_bytes(raw)
        self.digest = hashlib.sha256(raw).hexdigest()
        self.package = self.package.rename(self.package.with_name(self.digest))
        self.assert_rejected("Invalid binding table index")


if __name__ == "__main__":
    unittest.main()
