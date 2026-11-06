import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from safetensors.numpy import save_file

from tools.model.deepseek_v41.floating_head import CompactHeadRelease, HeadRelease, head_calls
from tools.model.safetensors_source import local_header


def fixture(root, reduction=1024, exact=False):
    source, cache = root / "weights", root / "execution"
    package = cache / "original"
    (source / "cache").mkdir(parents=True)
    package.mkdir(parents=True)
    config = dict(model_type="deepseek_v41", text_config=dict(hidden_size=1024, vocab_size=64))
    config["text_config"]["dspark_markov_rank"] = reduction
    (source / "config.json").write_text(json.dumps(config))
    (source / "cache/model.json").write_text(json.dumps(dict(execution_package="original")))
    weight = np.linspace(-0.25, 0.25, 64 * reduction, dtype="f4").reshape(64, reduction)
    if exact:
        weight = (weight.view("u4") & 0xFFFF0000).view("f4")
    save_file(
        {"head": weight, "untouched": np.array([2.0], dtype="f4")}, source / "layer.safetensors"
    )
    (source / "model.safetensors.index.json").write_text(
        json.dumps(
            dict(
                metadata={},
                weight_map=dict(head="layer.safetensors", untouched="layer.safetensors"),
            )
        )
    )
    (source / "release-manifest.json").write_text(
        json.dumps(
            dict(files={"layer.safetensors": dict(tensors=dict(head="old", untouched="old"))})
        )
    )
    buffers = [
        dict(
            name="weight",
            bytes=weight.nbytes,
            prepared_tensor=dict(name="head", dtype="F32", shape=[64, reduction]),
        )
    ]
    packets = []
    for capacity in [4, 8]:
        scratch = "scratch" + str(capacity)
        buffers.append(dict(name=scratch, bytes=capacity * 64 * 8 * 4))
        for op in ["partials", "reduce"]:
            params = dict(rows=capacity, output=64, _row_capacity=capacity)
            if op == "partials":
                params.update(reduction=reduction, activation_dtype="bfloat16")
            (package / f"{op}{capacity}.json").write_text(
                json.dumps(dict(op="dense.fp32_" + op, parameters=params))
            )
        packets.append(
            dict(
                body=[],
                head=[
                    dict(
                        op="kernel",
                        manifest=f"partials{capacity}.json",
                        buffers=["input", "weight", scratch],
                    ),
                    dict(
                        op="kernel", manifest=f"reduce{capacity}.json", buffers=[scratch, "output"]
                    ),
                ],
            )
        )
    model = dict(
        config=dict(file="config.json"),
        index=dict(file="model.safetensors.index.json"),
        ranks=[dict(buffers=buffers, packets=packets)],
    )
    (package / "model.json").write_text(json.dumps(model))
    return source, cache, package, model, weight


def compact_fixture(root, exact=True):
    source, cache, package, _, weight = fixture(root, exact=exact)
    previous = HeadRelease(
        source,
        root / "three-planes",
        source_execution=cache,
        execution_output=root / "unused-cache",
        build_dir=root / "unused-build",
    ).plan()
    group, record = previous.write_shard("layer.safetensors")
    source = previous.output
    (source / "cache").mkdir()
    (source / "cache/model.json").write_text(json.dumps(dict(execution_package="original")))
    (source / "config.json").write_text((root / "weights/config.json").read_text())
    (source / "model.safetensors.index.json").write_text(json.dumps(previous.index))
    (source / "release-manifest.json").write_text(
        json.dumps(dict(source_package="ancestor", files={group: record}))
    )
    (package / "model.json").write_text(json.dumps(previous.model))
    for capacity in [4, 8]:
        (package / f"partials{capacity}.json").write_text(
            json.dumps(
                dict(
                    op="floating.partials",
                    parameters=dict(
                        rows=capacity,
                        output=64,
                        reduction=1024,
                        input_parts=1,
                        weight_parts=3,
                        _row_capacity=capacity,
                    ),
                    operands=[
                        dict(name="activation", dtype="bf16", shape=[1, capacity, 1024]),
                        dict(name="weight", dtype="bf16", shape=[3, 64, 1024]),
                        dict(name="out", dtype="f32", shape=[capacity, 1024]),
                    ],
                )
            )
        )
    return (
        CompactHeadRelease(
            source,
            root / "single-plane",
            source_execution=cache,
            execution_output=root / "compact-cache",
            build_dir=root / "compact-build",
        ),
        source,
        package,
        previous.model,
        weight,
    )


class CompactHeadTests(unittest.TestCase):
    def test_two_parts_preserve_stored_planes_and_label_rounding(self):
        with TemporaryDirectory() as directory:
            job, source, _, _, _ = compact_fixture(Path(directory), exact=False)
            job = CompactHeadRelease(
                source,
                Path(directory) / "two-planes",
                source_execution=job.package.parent,
                execution_output=job.execution_output,
                build_dir=job.build_dir,
                retained_parts=2,
                require_exact=False,
            ).plan()
            self.assertEqual(len(job.calls), 2)
            self.assertEqual(
                job.manifest["head_rounding"], dict(source_parts=3, retained_parts=2, exact=False)
            )
            self.assertEqual(job.index["metadata"]["fp32_head_layout"], "bf16-residual2")
            job.write_shard("layer.safetensors")
            old_start, old = local_header(source / "layer.safetensors")
            new_start, new = local_header(job.output / "layer.safetensors")
            old_bank, new_bank = old["head.bf16_residual3"], new["head.bf16_residual2"]
            self.assertEqual(new_bank["shape"], [2, 64, 1024])
            with (source / "layer.safetensors").open("rb") as stream:
                stream.seek(old_start + old_bank["data_offsets"][0])
                reference = stream.read(2 * 64 * 1024 * 2)
            with (job.output / "layer.safetensors").open("rb") as stream:
                stream.seek(new_start + new_bank["data_offsets"][0])
                self.assertEqual(stream.read(len(reference)), reference)
            self.assertEqual(
                job.model["ranks"][0]["buffers"][0]["prepared_tensor"]["shape"],
                [2, 64, 1024],
            )

    def test_complete_zero_proof_preserves_main_plane_and_other_tensors(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            job, source, _, _, _ = compact_fixture(root)
            job.plan()
            self.assertEqual(len(job.calls), 2)
            self.assertEqual(job.transforms, {"head.bf16_residual3": "head.bf16_exact"})
            self.assertEqual(job.manifest["source_package"], "original")
            bank = job.model["ranks"][0]["buffers"][0]
            self.assertEqual(bank["prepared_tensor"]["shape"], [1, 64, 1024])
            self.assertEqual(bank["bytes"], 64 * 1024 * 2)
            job.write_shard("layer.safetensors")
            old_start, old = local_header(source / "layer.safetensors")
            start, new = local_header(job.output / "layer.safetensors")
            old_value = old["head.bf16_residual3"]
            new_value = new["head.bf16_exact"]
            self.assertEqual(old_value["shape"], [3, 64, 1024])
            self.assertEqual(new_value["shape"], [1, 64, 1024])
            with (source / "layer.safetensors").open("rb") as stream:
                stream.seek(old_start + old_value["data_offsets"][0])
                reference = stream.read(64 * 1024 * 2)
            with (job.output / "layer.safetensors").open("rb") as stream:
                stream.seek(start + new_value["data_offsets"][0])
                self.assertEqual(stream.read(64 * 1024 * 2), reference)
            self.assertIn("untouched", new)

    def test_nonzero_residual_and_nonfinite_main_plane_cannot_be_compacted(self):
        with TemporaryDirectory() as directory:
            job, _, _, _, _ = compact_fixture(Path(directory), exact=False)
            with self.assertRaisesRegex(ValueError, "No redundant"):
                job.plan()
        with TemporaryDirectory() as directory:
            job, source, _, _, _ = compact_fixture(Path(directory))
            start, header = local_header(source / "layer.safetensors")
            offset = start + header["head.bf16_residual3"]["data_offsets"][0]
            with (source / "layer.safetensors").open("r+b") as stream:
                stream.seek(offset)
                stream.write(bytes([0x80, 0x7F]))
            with self.assertRaisesRegex(ValueError, "finite BF16"):
                job.plan()

    def test_other_weight_consumers_require_an_explicit_conversion_contract(self):
        with TemporaryDirectory() as directory:
            job, _, package, model, _ = compact_fixture(Path(directory))
            model["ranks"][0]["packets"][0]["body"].append(dict(op="copy", buffers=["weight"]))
            (package / "model.json").write_text(json.dumps(model))
            job.model = model
            with self.assertRaisesRegex(ValueError, "another consumer"):
                job.plan()


class FloatingHeadTests(unittest.TestCase):
    def test_markov_bank_is_packed_once_without_rounding_or_changing_the_target_layout(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, cache, _, _, weight = fixture(root, reduction=256, exact=True)
            index_path = source / "model.safetensors.index.json"
            index = json.loads(index_path.read_text())
            index["metadata"]["fp32_head_layout"] = "bf16-residual3"
            index_path.write_text(json.dumps(index))
            job = HeadRelease(
                source,
                root / "converted",
                source_execution=cache,
                execution_output=root / "new-execution",
                build_dir=root / "build",
                markov_bias=True,
            ).plan()
            self.assertEqual(len(job.calls), 2)
            self.assertEqual(job.transforms, {"head": "head.bf16_exact"})
            self.assertEqual(job.index["metadata"]["fp32_head_layout"], "bf16-residual3")
            self.assertEqual(job.index["metadata"]["markov_bias_layout"], "bf16-exact")
            binding = job.model["ranks"][0]["buffers"][0]
            self.assertEqual(binding["bytes"], weight.size * 2)
            self.assertEqual(binding["prepared_tensor"]["shape"], [1, 64, 256])
            job.write_shard("layer.safetensors")
            target = root / "converted/layer.safetensors"
            begin, header = local_header(target)
            entry = header["head.bf16_exact"]
            self.assertEqual(entry["shape"], [1, 64, 256])
            with target.open("rb") as stream:
                stream.seek(begin + entry["data_offsets"][0])
                raw = stream.read(entry["data_offsets"][1] - entry["data_offsets"][0])
            restored = (np.frombuffer(raw, dtype="<u2").astype("u4") << 16).reshape(64, 256)
            np.testing.assert_array_equal(restored, weight.view("u4"))
            self.assertEqual(header["untouched"]["shape"], [1])
            self.assertEqual(local_header(source / "layer.safetensors")[1]["head"]["dtype"], "F32")

    def test_markov_export_rejects_weights_that_need_rounding(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, cache, _, _, _ = fixture(root, reduction=256)
            job = HeadRelease(
                source,
                root / "converted",
                source_execution=cache,
                execution_output=root / "new-execution",
                build_dir=root / "build",
                markov_bias=True,
            ).plan()
            with self.assertRaisesRegex(ValueError, "exact finite BF16"):
                job.write_shard("layer.safetensors")

    def test_shared_head_is_converted_once_and_semantic_shard_keeps_other_tensors(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, cache, _, _, weight = fixture(root)
            job = HeadRelease(
                source,
                root / "converted",
                source_execution=cache,
                execution_output=root / "new-execution",
                build_dir=root / "build",
            ).plan()
            self.assertEqual(len(job.calls), 2)
            self.assertEqual(job.transforms, {"head": "head.bf16_residual3"})
            binding = job.model["ranks"][0]["buffers"][0]
            self.assertEqual(
                binding["prepared_tensor"],
                dict(name="head.bf16_residual3", dtype="BF16", shape=[3, 64, 1024]),
            )
            self.assertEqual(binding["bytes"], weight.size * 6)
            self.assertNotIn("head", job.index["weight_map"])
            job.write_shard("layer.safetensors")
            target = root / "converted/layer.safetensors"
            begin, header = local_header(target)
            entry = header["head.bf16_residual3"]
            with target.open("rb") as stream:
                stream.seek(begin + entry["data_offsets"][0])
                raw = stream.read(entry["data_offsets"][1] - entry["data_offsets"][0])
            restored = (
                (np.frombuffer(raw, dtype="<u2").astype("u4") << 16)
                .view("f4")
                .reshape(3, 64, 1024)
                .sum(0)
            )
            np.testing.assert_array_equal(restored, weight)
            self.assertEqual(header["untouched"]["shape"], [1])
            self.assertEqual(local_header(source / "layer.safetensors")[1]["head"]["dtype"], "F32")
            self.assertEqual(job.model["ranks"][0]["buffers"][1]["bytes"], 4 * 128 * 8 * 4)

    def test_head_requires_matching_reducer_and_exclusive_weight_consumers(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, package, model, _ = fixture(root)
            steps = model["ranks"][0]["packets"][0]["head"]
            steps[1]["buffers"][0] = "different"
            with self.assertRaisesRegex(ValueError, "matching FP32 reducer"):
                head_calls(model, package, 64, 1024)
            steps[1]["buffers"][0] = "scratch4"
            model["ranks"][0]["packets"][0]["body"].append(dict(op="embed", buffers=["weight"]))
            with self.assertRaisesRegex(ValueError, "another consumer"):
                head_calls(model, package, 64, 1024)


if __name__ == "__main__":
    unittest.main()
