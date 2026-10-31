import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from safetensors.numpy import save_file

from tools.model.deepseek_v41.floating_head import HeadRelease, head_calls
from tools.model.safetensors_source import local_header


def fixture(root):
    source, cache = root / "weights", root / "execution"
    package = cache / "original"
    (source / "cache").mkdir(parents=True)
    package.mkdir(parents=True)
    config = dict(model_type="deepseek_v41", text_config=dict(hidden_size=1024, vocab_size=64))
    (source / "config.json").write_text(json.dumps(config))
    (source / "cache/model.json").write_text(json.dumps(dict(execution_package="original")))
    weight = np.linspace(-0.25, 0.25, 64 * 1024, dtype="f4").reshape(64, 1024)
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
            prepared_tensor=dict(name="head", dtype="F32", shape=[64, 1024]),
        )
    ]
    packets = []
    for capacity in [4, 8]:
        scratch = "scratch" + str(capacity)
        buffers.append(dict(name=scratch, bytes=capacity * 64 * 8 * 4))
        for op in ["partials", "reduce"]:
            params = dict(rows=capacity, output=64, _row_capacity=capacity)
            if op == "partials":
                params.update(reduction=1024, activation_dtype="bfloat16")
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


class FloatingHeadTests(unittest.TestCase):
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
