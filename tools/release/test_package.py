import hashlib
import json
from pathlib import Path
import tempfile
import tarfile
import unittest
from unittest.mock import patch
from tools.release.package import archive, execution_asset_identity, execution_assets, verify


class ReleaseTests(unittest.TestCase):
    def test_reproducible_archive_and_download_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = root / "bundle"
            bundle.mkdir()
            (bundle / "source.txt").write_text("fixed source")
            archive(bundle, root / "one.tar.xz", 100)
            archive(bundle, root / "two.tar.xz", 100)
            self.assertTrue((root / "one.tar.xz").read_bytes().startswith(b"\xfd7zXZ\0"))
            with tarfile.open(root / "one.tar.xz") as stream:
                entry = stream.getmember("bundle/source.txt")
                self.assertEqual((entry.uid, entry.gid, entry.mtime), (0, 0, 100))
            self.assertEqual((root / "one.tar.xz").read_bytes(), (root / "two.tar.xz").read_bytes())
            digest = hashlib.sha256((root / "one.tar.xz").read_bytes()).hexdigest()
            (root / "SHA256SUMS").write_text(f"{digest}  one.tar.xz\n")
            verify(root)
            (root / "one.tar.xz").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksum"):
                verify(root)


class ExecutionAssetsTests(unittest.TestCase):
    def test_native_a2_directory_has_its_own_release_asset_and_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)
            raw = json.dumps(dict(schema_version=1, backend="ascend", plan={})).encode()
            root = cache / hashlib.sha256(raw).hexdigest()
            root.mkdir()
            (root / "package.json").write_bytes(raw)
            assets = execution_assets([root], cache / "unused", Path("/bin/false"), "v0.3.0")
            self.assertEqual(
                assets[0]["metadata"]["file"],
                "orinfer-exec-deepseek-v41-flash-1017-w4a8-ascend910b-v0.3.0.tar.xz",
            )
            self.assertEqual(assets[0]["metadata"]["digest"], root.name)
            (root / "package.json").write_bytes(raw + b" ")
            with self.assertRaisesRegex(ValueError, "digest-named"):
                execution_assets([root], cache / "unused", Path("/bin/false"), "v0.3.0")

    def test_encoder_assets_use_their_pinned_configuration_shapes(self):
        cases = [
            (
                "nemotron_embed",
                {
                    "text_config": dict(
                        hidden_size=2048,
                        intermediate_size=6144,
                        num_hidden_layers=16,
                        num_attention_heads=24,
                        num_key_value_heads=8,
                    )
                },
                "nemotron35-embed-1b",
            ),
            (
                "qwen3_reranker",
                dict(
                    hidden_size=1024,
                    intermediate_size=3072,
                    num_hidden_layers=28,
                    num_attention_heads=16,
                    num_key_value_heads=8,
                ),
                "qwen3-reranker-0-6b",
            ),
        ]
        for architecture, signature, expected in cases:
            with self.subTest(architecture=architecture):
                package = dict(
                    architecture=architecture,
                    compute_policy="int8_quality",
                    target="sm_87",
                    config_signature=signature,
                )
                self.assertEqual(
                    execution_asset_identity(package, "v0.2.2"),
                    (f"orinfer-exec-{expected}-a8-sm87-v0.2.2.tar.xz", expected),
                )
                text = signature.get("text_config", signature)
                text["hidden_size"] += 1
                with self.assertRaisesRegex(ValueError, "Register one"):
                    execution_asset_identity(package, "v0.2.2")

    def test_model_variants_have_distinct_assets_and_tampered_identity_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            header = bytearray(20)
            header[:6] = b"\x7fELF\x02\x01"
            header[16:20] = bytes((3, 0, 183, 0))
            identity = dict(
                abi_version=1,
                package="orinfer-models",
                version="0.1.2",
                library=dict(file="lib/model.so", sha256=hashlib.sha256(header).hexdigest()),
            )
            archives = []
            for index, architecture in enumerate(("qwen3_5", "flash_next", "qwen3_5", "qwen3_5")):
                manifest = dict(
                    schema_version=1,
                    runtime_abi=1,
                    architecture=architecture,
                    config_signature=dict(
                        text=(
                            dict(
                                hidden_size=2560,
                                num_hidden_layers=48,
                                num_experts=512,
                                moe_intermediate_size=640,
                                num_experts_per_tok=10,
                            )
                            if architecture == "flash_next"
                            else dict(
                                hidden_size=2048 if index == 0 else 2560,
                                intermediate_size=6144 if index == 0 else 9216,
                                num_hidden_layers=24 if index == 0 else 32,
                            )
                        )
                    ),
                    compute_policy="int8_quality",
                    target="sm_87",
                    catalog=dict(
                        assets=[], definitions=[], dynamic=[], binding_signature=str(index) * 64
                    ),
                    execution=identity,
                )
                raw = json.dumps(manifest).encode()
                package = root / hashlib.sha256(raw).hexdigest()
                (package / "lib").mkdir(parents=True)
                (package / "package.json").write_bytes(raw)
                (package / "lib/model.so").write_bytes(header)
                suffix = ".tar.gz" if architecture == "qwen3_5" else ".tar.xz"
                path = root / (f"{architecture}-{index}" + suffix)
                from tools.model.package import archive as archive_package

                if suffix == ".tar.gz":
                    with tarfile.open(path, "w:gz") as stream:
                        stream.add(package, arcname=package.name)
                else:
                    archive_package(package, path)
                archives.append(path)
            with patch("tools.release.package.execution_identity", return_value=identity):
                assets = execution_assets(archives[:3], root / "cache", Path("/bin/true"), "v0.1.2")
                names = [a["metadata"]["file"] for a in assets]
                self.assertEqual(len(set(names)), 3)
                self.assertEqual(
                    names,
                    [
                        "orinfer-exec-qwen35-2b-a8-sm87-v0.1.2.tar.xz",
                        "orinfer-exec-flash-next-a8-sm87-v0.1.2.tar.xz",
                        "orinfer-exec-qwen35-4b-a8-sm87-v0.1.2.tar.xz",
                    ],
                )
                self.assertNotIn("27b", names[1])
                self.assertEqual(
                    [a["metadata"]["architecture"] for a in assets],
                    ["qwen3_5", "flash_next", "qwen3_5"],
                )
                from tools.model.package import install

                for asset in assets:
                    output = root / asset["metadata"]["file"]
                    self.assertTrue(output.name.endswith(".tar.xz"))
                    archive(asset["directory"], output, 100)
                    self.assertTrue(output.read_bytes().startswith(b"\xfd7zXZ\0"))
                    self.assertEqual(
                        install(output, root / "reinstalled"), asset["metadata"]["digest"]
                    )
            with patch("tools.release.package.execution_identity", return_value=identity):
                with self.assertRaisesRegex(ValueError, "Duplicate"):
                    execution_assets(
                        archives, root / "collision-cache", Path("/bin/true"), "v0.1.2"
                    )
            with patch(
                "tools.release.package.execution_identity",
                return_value=dict(identity, version="wrong"),
            ):
                with self.assertRaisesRegex(ValueError, "differs from package"):
                    execution_assets(archives, root / "bad-cache", Path("/bin/true"), "v0.1.2")

    def test_unknown_specification_and_unsafe_tag_are_rejected(self):
        package = dict(architecture="qwen3_5", compute_policy="int8_quality", target="sm_87")
        with self.assertRaisesRegex(ValueError, "Register one"):
            execution_asset_identity(package, "v0.2.0")
        with self.assertRaisesRegex(ValueError, "Unsafe release tag"):
            execution_asset_identity(package, "../../v0.2.0")
