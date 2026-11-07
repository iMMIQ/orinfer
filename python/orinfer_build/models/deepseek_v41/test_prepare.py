"""Packed layout, fused banks and immutable offline preparation checks."""

import contextlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

import numpy as np
from safetensors.numpy import save_file

from orinfer_build.common.files import file_hash
from orinfer_build.models.deepseek_v41.prepare import (
    Checkpoint,
    Transpose,
    convert,
    convert_shared,
    prepare,
    tensor_names,
)


class PrepareTests(unittest.TestCase):
    def test_shared_tp8_reassembles_original_banks_without_padding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prefix = "layers.0.ffn.shared_experts"
            random = np.random.default_rng(20261002)
            tensors = {}
            for part in ("w1", "w3"):
                tensors[f"{prefix}.{part}.weight"] = random.integers(
                    -128, 128, (24, 32), dtype=np.int8
                )
                tensors[f"{prefix}.{part}.weight_scale"] = random.random((24, 1), dtype=np.float32)
            tensors[f"{prefix}.w2.weight"] = random.integers(-128, 128, (32, 24), dtype=np.int8)
            save_file(tensors, root / "checkpoint.safetensors")
            (root / "index.json").write_text(
                json.dumps({"weight_map": dict.fromkeys(tensors, "checkpoint.safetensors")})
            )
            checkpoint = Checkpoint(root, "index.json", tensors)
            try:
                for plane in ("up", "scale", "down"):
                    banks = [
                        convert_shared({"prefix": prefix, "rank": rank, "plane": plane}, checkpoint)
                        for rank in range(8)
                    ]
                    if plane == "down":
                        actual = np.concatenate(banks, axis=1)
                        expected = tensors[f"{prefix}.w2.weight"].view(np.uint8)
                    else:
                        suffix = "weight_scale" if plane == "scale" else "weight"
                        actual = np.concatenate(
                            [
                                np.concatenate([b[:3] for b in banks]),
                                np.concatenate([b[3:] for b in banks]),
                            ]
                        )
                        expected = np.concatenate(
                            [tensors[f"{prefix}.{part}.{suffix}"] for part in ("w1", "w3")]
                        )
                        expected = (
                            expected.reshape(-1) if plane == "scale" else expected.view(np.uint8)
                        )
                    np.testing.assert_array_equal(actual, expected)
                with self.assertRaises(ValueError):
                    convert_shared({"prefix": prefix, "rank": 8, "plane": "up"}, checkpoint)
            finally:
                checkpoint.close()

    @classmethod
    def setUpClass(cls):
        cls.compiler = tempfile.TemporaryDirectory()
        cls.transpose = Transpose(cls.compiler.name)

    @classmethod
    def tearDownClass(cls):
        cls.compiler.cleanup()

    def test_simd_tails_and_strided_destination_guards(self):
        random = np.random.default_rng(20261002)
        for rows, columns in [(1, 1), (8, 8), (9, 17), (65, 71), (1152, 5120), (2560, 2304)]:
            with self.subTest(rows=rows, columns=columns):
                source = random.integers(0, 256, (rows, columns), dtype=np.uint8)
                backing = np.full((columns, rows + 16), 179, dtype=np.uint8)
                destination = backing[:, 8 : 8 + rows]
                self.transpose(source, destination)
                np.testing.assert_array_equal(destination, source.T)
                self.assertTrue(np.all(backing[:, :8] == 179))
                self.assertTrue(np.all(backing[:, 8 + rows :] == 179))

    def fixture(self, root):
        random = np.random.default_rng(20261002)
        tensors = {}
        experts = [f"layers.0.ffn.experts.{i}" for i in range(2)]
        for expert in experts:
            for part in ("w1", "w3"):
                prefix = expert + "." + part
                tensors[prefix + ".weight"] = random.integers(-128, 128, (8, 17), dtype=np.int8)
                tensors[prefix + ".weight_scale"] = random.random((16, 1), dtype=np.float32)
                tensors[prefix + ".scale_bias"] = random.integers(-8, 9, (16, 16)).astype("<f4")
                tensors[prefix + ".weight_offset"] = np.zeros((16, 1), dtype="<f4")
        save_file(tensors, root / "checkpoint.safetensors")
        index = root / "index.json"
        index.write_text(
            json.dumps({"weight_map": dict.fromkeys(tensors, "checkpoint.safetensors")})
        )
        return tensors, {
            "kind": "w4",
            "experts": experts,
            "parts": ["w1", "w3"],
            "input": 17,
            "output": 32,
            "plane": "packed",
        }

    def test_fused_packed_scale_and_assist_match_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tensors, weight = self.fixture(root)
            names = [
                n
                for p in ("packed", "scale", "assist")
                for n in tensor_names({**weight, "plane": p})
            ]
            checkpoint = Checkpoint(root, "index.json", names)
            try:
                for plane in ("packed", "scale", "assist"):
                    output = convert({**weight, "plane": plane}, checkpoint, self.transpose)
                    for e, expert in enumerate(weight["experts"]):
                        if plane == "packed":
                            expected = np.concatenate(
                                [
                                    tensors[f"{expert}.{p}.weight"].view(np.uint8)
                                    for p in weight["parts"]
                                ]
                            ).T
                            np.testing.assert_array_equal(output[e], expected)
                        elif plane == "scale":
                            values = np.concatenate(
                                [
                                    tensors[f"{expert}.{p}.weight_scale"].reshape(-1)
                                    for p in weight["parts"]
                                ]
                            )
                            expected = [
                                struct.unpack("<I", struct.pack("<f", v))[0] for v in values
                            ]
                            np.testing.assert_array_equal(output[e, 0], expected)
                        else:
                            expected = np.concatenate(
                                [
                                    tensors[f"{expert}.{p}.scale_bias"].sum(axis=1)
                                    for p in weight["parts"]
                                ]
                            )
                            np.testing.assert_array_equal(output[e], expected)
            finally:
                checkpoint.close()

    def test_prepared_plan_resumes_without_payload_reads_and_rebuilds_truncation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, weight = self.fixture(root)
            plan = root / "model.json"
            original = {
                "index": {"file": "index.json", "sha256": file_hash(root / "index.json")},
                "ranks": [
                    {"rank": 0, "buffers": [{"name": "packed", "bytes": 544, "weight": weight}]}
                ],
            }
            plan.write_text(json.dumps(original))
            output = root / "model-prepared.json"
            with contextlib.redirect_stdout(io.StringIO()):
                prepare(plan, root, output, 2)
                prepared = output.read_bytes()
                with mock.patch.object(
                    Checkpoint, "read", side_effect=AssertionError("payload read on resume")
                ):
                    prepare(plan, root, output, 2)
            self.assertEqual(output.read_bytes(), prepared)
            self.assertEqual(json.loads(plan.read_text()), original)
            definition = json.loads(prepared)["ranks"][0]["buffers"][0]
            self.assertNotIn("weight", definition)
            target = root / definition["prepared_weight"]
            data = target.read_bytes()
            self.assertEqual(len(data), 544)
            target.write_bytes(data[:-1])
            with contextlib.redirect_stdout(io.StringIO()):
                prepare(plan, root, output, 2)
            repaired = json.loads(output.read_bytes())["ranks"][0]["buffers"][0]
            self.assertNotEqual(repaired["prepared_weight"], definition["prepared_weight"])
            self.assertEqual((root / repaired["prepared_weight"]).read_bytes(), data)


if __name__ == "__main__":
    unittest.main()
