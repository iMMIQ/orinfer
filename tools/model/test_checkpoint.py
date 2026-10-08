"""Real sharded safetensors import, provenance and format rejection without CUDA/Torch."""

import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from safetensors.numpy import save_file
from tools.model.checkpoint import Checkpoint


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = dict(
            model_type="qwen3_5_text",
            hidden_size=5120,
            intermediate_size=17408,
            vocab_size=248320,
            num_hidden_layers=64,
            linear_num_key_heads=16,
            linear_num_value_heads=48,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_conv_kernel_dim=4,
            num_attention_heads=24,
            num_key_value_heads=4,
            head_dim=256,
            rms_norm_eps=1e-6,
            hidden_act="silu",
            tie_word_embeddings=False,
            layer_types=["full_attention" if i % 4 == 3 else "linear_attention" for i in range(64)],
        )
        self.config["rope_parameters"] = dict(
            rope_type="default",
            rope_theta=10000000,
            partial_rotary_factor=0.25,
            mrope_section=[11, 11, 10],
            mrope_interleaved=True,
        )
        (self.root / "config.json").write_text(json.dumps(self.config))
        self.key = "model.language_model.layers.0.mlp.down_proj"
        save_file(
            {self.key + ".weight_packed": np.zeros((5120, 2176), dtype=np.int32)},
            self.root / "a.safetensors",
        )
        save_file(
            {
                self.key + ".weight_scale": np.ones((5120, 136), dtype=np.float16),
                self.key + ".weight_zero_point": np.zeros((640, 136), dtype=np.int32),
            },
            self.root / "b.safetensors",
        )
        self.index = {
            "weight_map": {
                self.key + ".weight_packed": "a.safetensors",
                self.key + ".weight_scale": "b.safetensors",
                self.key + ".weight_zero_point": "b.safetensors",
            }
        }
        (self.root / "model.safetensors.index.json").write_text(json.dumps(self.index))

    def tearDown(self):
        self.temp.cleanup()

    def test_sharded_read_and_source_identity_change(self):
        with Checkpoint(self.root, framework="np") as reader:
            first = reader.identity["sha256"]
            self.assertEqual(reader.get_tensor(self.key + ".weight_scale")[0, 0], 1)
        save_file(
            {
                self.key + ".weight_scale": np.full((5120, 136), 2, dtype=np.float16),
                self.key + ".weight_zero_point": np.zeros((640, 136), dtype=np.int32),
            },
            self.root / "b.safetensors",
        )
        with Checkpoint(self.root, framework="np") as reader:
            self.assertNotEqual(first, reader.identity["sha256"])

    def test_rejects_unsupported_dimensions_and_index_paths(self):
        self.config["hidden_size"] = 1
        (self.root / "config.json").write_text(json.dumps(self.config))
        with self.assertRaisesRegex(ValueError, "dimensions"):
            Checkpoint(self.root, framework="np")
        self.config["hidden_size"] = 5120
        (self.root / "config.json").write_text(json.dumps(self.config))
        self.index["weight_map"][self.key + ".weight_scale"] = "../outside.safetensors"
        (self.root / "model.safetensors.index.json").write_text(json.dumps(self.index))
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            Checkpoint(self.root, framework="np")

    def test_full_build_preflight_rejects_incomplete_backbone(self):
        with Checkpoint(self.root, framework="np") as reader:
            with self.assertRaisesRegex(ValueError, "Missing backbone tensor"):
                reader.validate_backbone()
        self.config["rope_parameters"]["rope_theta"] = 10000
        (self.root / "config.json").write_text(json.dumps(self.config))
        with self.assertRaisesRegex(ValueError, "RoPE"):
            Checkpoint(self.root, framework="np")

    def test_rejects_missing_index_tensor(self):
        self.index["weight_map"]["missing.weight"] = "a.safetensors"
        (self.root / "model.safetensors.index.json").write_text(json.dumps(self.index))
        with self.assertRaisesRegex(ValueError, "missing tensor"):
            Checkpoint(self.root, framework="np")
