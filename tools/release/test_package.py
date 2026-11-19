import hashlib
import json
from pathlib import Path
import tempfile
import tarfile
import unittest
from tests.support.model_package import data_package
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
            assets = execution_assets([root], cache / "unused", "v0.3.0")
            self.assertEqual(
                assets[0]["metadata"]["file"],
                "orinfer-exec-deepseek-v41-flash-1017-w4a8-ascend910b-v0.3.0.tar.xz",
            )
            self.assertEqual(assets[0]["metadata"]["digest"], root.name)
            (root / "package.json").write_bytes(raw + b" ")
            with self.assertRaisesRegex(ValueError, "digest-named"):
                execution_assets([root], cache / "unused", "v0.3.0")

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

    def test_model_variants_have_distinct_assets_and_duplicates_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archives = []
            for index, architecture in enumerate(("qwen3_5", "flash_next", "qwen3_5", "qwen3_5")):
                manifest = dict(
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
                )
                package = data_package(root / str(index), payload=str(index).encode(), **manifest)
                suffix = ".tar.gz" if architecture == "qwen3_5" else ".tar.xz"
                path = root / (f"{architecture}-{index}" + suffix)
                from orinfer_build.packaging.package import archive as archive_package

                if suffix == ".tar.gz":
                    with tarfile.open(path, "w:gz") as stream:
                        stream.add(package, arcname=package.name)
                else:
                    archive_package(package, path)
                archives.append(path)
            assets = execution_assets(archives[:3], root / "cache", "v0.1.2")
            self.assertEqual(
                [a["metadata"]["file"] for a in assets],
                [
                    "orinfer-exec-qwen35-2b-a8-sm87-v0.1.2.tar.xz",
                    "orinfer-exec-flash-next-a8-sm87-v0.1.2.tar.xz",
                    "orinfer-exec-qwen35-4b-a8-sm87-v0.1.2.tar.xz",
                ],
            )
            from orinfer_build.packaging.package import install

            for asset in assets:
                output = root / asset["metadata"]["file"]
                archive(asset["directory"], output, 100)
                self.assertTrue(output.read_bytes().startswith(b"\xfd7zXZ\0"))
                self.assertEqual(install(output, root / "reinstalled"), asset["metadata"]["digest"])
                self.assertEqual(asset["metadata"]["bindings"]["file"], "bindings.json")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                execution_assets(archives, root / "collision-cache", "v0.1.2")

    def test_unknown_specification_and_unsafe_tag_are_rejected(self):
        package = dict(architecture="qwen3_5", compute_policy="int8_quality", target="sm_87")
        with self.assertRaisesRegex(ValueError, "Register one"):
            execution_asset_identity(package, "v0.2.0")
        with self.assertRaisesRegex(ValueError, "Unsafe release tag"):
            execution_asset_identity(package, "../../v0.2.0")
