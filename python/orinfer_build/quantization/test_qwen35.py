"""Quantized publication preserves row order, source BF16 and shared tables."""

import hashlib
import tempfile
from pathlib import Path
import unittest

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

from orinfer_build.quantization.qwen35 import group, policy, publish, save_original, task_key


class Qwen35Tests(unittest.TestCase):
    def test_precision_policy_is_independent_of_chunk_shape(self):
        matrix = dict(dtype="BF16", shape=[5120, 5120])
        self.assertEqual(
            policy("model.language_model.layers.0.mlp.gate_proj.weight", matrix), "q4-row"
        )
        for name in (
            "model.language_model.embed_tokens.weight",
            "mtp.fc.weight",
            "model.language_model.layers.0.linear_attn.out_proj.weight",
        ):
            self.assertEqual(policy(name, matrix), "int8-row")
        for name in (
            "model.visual.blocks.0.attn.qkv.weight",
            "model.language_model.layers.0.norm.weight",
        ):
            self.assertEqual(policy(name, matrix), "original")
        self.assertEqual(policy("small.weight", dict(dtype="BF16", shape=[48, 5120])), "original")

    def test_component_names_do_not_mix_layers_or_mtp(self):
        self.assertEqual(
            group("model.language_model.layers.12.mlp.gate_proj.weight"), "model-language-layer-012"
        )
        self.assertEqual(group("mtp.layers.0.mlp.gate_proj.weight"), "model-mtp")

    def test_merge_preserves_bf16_bits_and_concatenates_packed_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "chunks").mkdir()
            name = "model.language_model.layers.0.mlp.gate_proj.weight"
            queue, records = [], {}
            expected = np.arange(3 * 64, dtype=np.uint8).reshape(3, 64)
            for first, rows in ((0, 2), (2, 1)):
                task = dict(
                    name=name, kind="q4-row", shape=[3, 128], dtype="BF16", first=first, rows=rows
                )
                file = task_key(task) + ".safetensors"
                save_file(
                    {
                        name + ".packed": expected[first : first + rows].copy(),
                        name + ".scale": np.ones(rows, np.float16),
                    },
                    str(root / "chunks" / file),
                )
                queue.append(task)
                records[task_key(task)] = dict(file=file)
            norm = "model.language_model.layers.0.input_layernorm.weight"
            task = dict(name=norm, kind="original", shape=[3], dtype="BF16", first=0, rows=3)
            file = task_key(task) + ".safetensors"
            raw = np.array([0, 0x8000, 0x3F80], dtype="<u2").tobytes()
            save_original(root / "chunks" / file, norm, dict(dtype="BF16", shape=[3]), raw, "test")
            queue.append(task)
            records[task_key(task)] = dict(file=file)
            signs = {128: np.ones(128, np.int8)}
            total = publish(root, queue, records, signs, "test")
            path = root / "model-language-layer-000.safetensors"
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            with safe_open(path, framework="np") as f:
                np.testing.assert_array_equal(f.get_tensor(name + ".packed"), expected)
                self.assertEqual(f.get_slice(norm).get_dtype(), "BF16")
            from orinfer_build.weights.safetensors_source import local_header

            begin, header = local_header(path)
            lo, hi = header[norm]["data_offsets"]
            self.assertEqual(path.read_bytes()[begin + lo : begin + hi], raw)
            self.assertEqual(publish(root, queue, records, signs, "test"), total)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)


if __name__ == "__main__":
    unittest.main()
