"""Native schema-1 execution archive identity and atomic installation checks."""

import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from orinfer_build.packaging.package import archive, execution_identity, install


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        package = self.root / "package"
        (package / "lib").mkdir(parents=True)
        (package / "kernels").mkdir()
        header = bytearray(20)
        header[:6] = b"\x7fELF\x02\x01"
        header[16:20] = bytes((3, 0, 183, 0))
        (package / "lib/model.so").write_bytes(header)
        cubin = b"fixture cubin bytes"
        (package / "kernels/test.cubin").write_bytes(cubin)
        content = dict(
            schema_version=1,
            runtime_abi=1,
            execution=dict(
                abi_version=1,
                package="test-model",
                version="1",
                library=dict(file="lib/model.so", sha256=hashlib.sha256(header).hexdigest()),
            ),
            catalog=dict(
                assets=[dict(file="kernels/test.cubin", sha256=hashlib.sha256(cubin).hexdigest())]
            ),
        )
        raw = json.dumps(content).encode()
        (package / "package.json").write_bytes(raw)
        self.digest = hashlib.sha256(raw).hexdigest()
        self.package = package.rename(self.root / self.digest)

    def test_archive_roundtrip_and_duplicate_install(self):
        path = self.root / "operators.tar.xz"
        archive(self.package, path)
        cache = self.root / "installed"
        self.assertEqual(install(path, cache), self.digest)
        self.assertEqual(
            (cache / self.digest / "kernels/test.cubin").read_bytes(), b"fixture cubin bytes"
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

    def test_modified_native_library_never_installs(self):
        (self.package / "lib/model.so").write_bytes(b"changed native library")
        self.assert_rejected("Execution library sha256 mismatch")

    def test_corrupt_kernel_never_installs(self):
        (self.package / "kernels/test.cubin").write_bytes(b"corrupt")
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

    def test_library_version_is_read_from_abi_instead_of_workspace(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            library = root / "model.so"
            header = bytearray(20)
            header[:6] = b"\x7fELF\x02\x01"
            header[16:20] = bytes((3, 0, 183, 0))
            library.write_bytes(header)
            info = dict(package="external-model", version="3.7.9", target="sm_87", runtime_abi=1)
            with patch("orinfer_build.packaging.package.subprocess.run") as run:
                run.return_value.stdout = json.dumps(info)
                identity = execution_identity(library, Path("/bin/true"))
                self.assertEqual(identity["version"], "3.7.9")
                self.assertEqual(identity["package"], "external-model")
                self.assertEqual(identity["library"]["sha256"], hashlib.sha256(header).hexdigest())
                for key, value in (("target", "sm_90"), ("runtime_abi", 2), ("version", "")):
                    run.return_value.stdout = json.dumps(dict(info, **{key: value}))
                    with self.assertRaisesRegex(ValueError, "incompatible"):
                        execution_identity(library, Path("/bin/true"))


if __name__ == "__main__":
    unittest.main()
