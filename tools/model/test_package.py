"""Model execution package identity, installation and failed-publication checks."""

import hashlib
import io
import os
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from tools.model.package import archive, install, split


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        header = bytearray(20)
        header[:6] = b"\x7fELF\x02\x01"
        header[16:20] = bytes((3, 0, 183, 0))
        self.library = self.root / "model.so"
        self.library.write_bytes(header)
        env = patch.dict(os.environ, ORINFER_MODEL_LIBRARY=str(self.library))
        env.start()
        self.addCleanup(env.stop)
        self.model = self.root / "source"
        (self.model / "cache/kernels").mkdir(parents=True)
        (self.model / "cache/weights").mkdir()
        config = {
            "model_type": "qwen3_5_text",
            "num_hidden_layers": 1,
            "layer_types": ["linear_attention"],
            "vocab_size": 8,
        }
        (self.model / "config.json").write_text(json.dumps(config))
        for name in ("tokenizer.json", "chat_template.jinja", "generation_config.json"):
            (self.model / name).write_text("{}")
        raw = b"fixture cubin bytes"
        asset = {"file": "kernels/test.cubin", "sha256": hashlib.sha256(raw).hexdigest()}
        (self.model / "cache/kernels/test.cubin").write_bytes(raw)
        buffers = [
            {
                "name": "Weight",
                "dtype": "u8",
                "shape": [4],
                "layout": "packed",
                "access": "read",
                "alignment": 256,
                "data": {"tensor": "Weight", "sha256": "a" * 64},
            },
            {
                "name": "State",
                "dtype": "f32",
                "shape": [4],
                "layout": "contiguous",
                "access": "read_write",
                "alignment": 256,
            },
        ]
        names = [
            "prepare_0",
            "L0_norm_1",
            "prefill_advance_2",
            "head_3",
            "decode_prepare_4",
            "L0_decode_norm_5",
            "decode_advance_6",
        ]
        kernels = [
            dict(
                name=n,
                module=asset,
                source=asset,
                host_abi=asset,
                args=[{"kind": "buffer", "name": "L0_PreWeight"}] if "_norm_" in n else [],
            )
            for n in names
        ]

        def op(n):
            return {"kind": "kernel", "name": n}

        self.manifest = dict(
            schema_version=2,
            target="sm_87",
            vocab=8,
            buffers=buffers,
            reset_buffers=["State"],
            input="Input",
            token="Token",
            status="Status",
            position="Position",
            toolchain={"tilelang": "test", "source_manifest_sha256": "b" * 64},
            kernels=kernels,
            prefill_plans=[
                {"chunk_tokens": 2, "prefill_program": "prefill_m2", "head_program": "head_m2"}
            ],
            programs={
                "prefill_m2": list(map(op, names[:3])),
                "head_m2": [op(names[3])],
                "decode": list(map(op, names[4:])),
            },
        )
        self.save()

    def tearDown(self):
        self.temp.cleanup()

    def save(self):
        (self.model / "cache/manifest.json").write_text(json.dumps(self.manifest))

    @patch("tools.model.package.validate_plan")
    def test_package_identity_is_independent_of_weight_payloads(self, validate):
        first = split(self.model, self.root / "first")
        self.manifest["buffers"][0]["data"]["sha256"] = "c" * 64
        self.manifest["toolchain"]["source_manifest_sha256"] = "d" * 64
        self.save()
        second = split(self.model, self.root / "second")
        self.assertEqual(first["execution_package"], second["execution_package"])
        descriptor = json.loads((self.root / "second/cache/model.json").read_text())
        self.assertNotIn("programs", descriptor["metadata"])
        self.assertNotIn("kernels", descriptor["metadata"])
        self.assertEqual(descriptor["buffer_scopes"]["Weight"], "weights")
        self.assertEqual(descriptor["buffer_scopes"]["State"], "sequence")
        package = self.root / "first/cache/packages" / first["execution_package"]
        content = json.loads((package / "package.json").read_text())
        self.assertTrue(all("data" not in b for b in content["buffer_contracts"]))
        tar = self.root / "operators.tar.gz"
        archive(package, tar)
        cache = self.root / "installed"
        self.assertEqual(install(tar, cache), first["execution_package"])
        self.assertEqual(
            (cache / first["execution_package"] / "kernels/test.cubin").read_bytes(),
            b"fixture cubin bytes",
        )
        with self.assertRaisesRegex(ValueError, "already installed"):
            install(tar, cache)
        self.assertFalse(list(cache.glob(".install-*")))
        self.assertEqual(validate.call_count, 2)

    @patch("tools.model.package.validate_plan")
    def test_modified_native_library_never_installs(self, _):
        prepared = self.root / "prepared"
        report = split(self.model, prepared)
        package = prepared / "cache/packages" / report["execution_package"]
        (package / "lib/model.so").write_bytes(b"changed native library")
        tar = self.root / "tampered-library.tar.gz"
        archive(package, tar)
        cache = self.root / "installed"
        with self.assertRaisesRegex(ValueError, "Execution library sha256 mismatch"):
            install(tar, cache)
        self.assertFalse((cache / report["execution_package"]).exists())
        self.assertFalse(list(cache.glob(".install-*")))

    @patch(
        "tools.model.package.validate_plan",
        side_effect=ValueError("unsupported architecture recipe"),
    )
    def test_failed_plan_never_publishes_directory(self, _):
        output = self.root / "failed"
        with self.assertRaisesRegex(ValueError, "unsupported architecture"):
            split(self.model, output)
        self.assertFalse(output.exists())
        self.assertFalse(list(self.root.glob(".failed-*")))
        self.assertTrue((self.model / "cache/manifest.json").exists())

    @patch("tools.model.package.validate_plan")
    def test_corrupt_assets_and_archive_traversal_are_rejected_atomically(self, _):
        prepared = self.root / "prepared"
        report = split(self.model, prepared)
        package = prepared / "cache/packages" / report["execution_package"]
        (package / "kernels/test.cubin").write_bytes(b"corrupt")
        tar = self.root / "bad.tar.gz"
        archive(package, tar)
        cache = self.root / "installed"
        with self.assertRaisesRegex(ValueError, "sha256"):
            install(tar, cache)
        self.assertFalse((cache / report["execution_package"]).exists())
        for name, kind in [
            (report["execution_package"] + "/../escape", tarfile.REGTYPE),
            (report["execution_package"] + "/link", tarfile.SYMTYPE),
        ]:
            with tarfile.open(tar, "w:gz") as stream:
                member = tarfile.TarInfo(name)
                member.type = kind
                member.size = 1 if kind == tarfile.REGTYPE else 0
                stream.addfile(member, io.BytesIO(b"x"))
            with self.assertRaises(ValueError):
                install(tar, cache)
        self.assertFalse((self.root / "escape").exists())
        self.assertFalse(list(cache.glob(".install-*")))


if __name__ == "__main__":
    unittest.main()
