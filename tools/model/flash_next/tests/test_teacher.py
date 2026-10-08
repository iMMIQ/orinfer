"""Independent tiny-model reference checks without GPU or remote weights."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from safetensors.numpy import save_file

from tools.model.flash_next.reference.teacher import Teacher
from tools.model.flash_next.reference.inputs import Inputs
from tools.quantization.flash_next import digest


class Fixture:
    def __init__(self):
        self.config = {
            "text_config": {
                "hidden_size": 4,
                "hc_count": 2,
                "hc_lowrank": 2,
                "num_experts": 3,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 2,
                "vocab_size": 16,
                "output_gate_type": "sigmoid",
                "hidden_act": "silu",
                "indexer_budget": 2048,
                "linear_num_key_heads": 1,
                "linear_num_value_heads": 2,
                "linear_key_head_dim": 2,
                "linear_value_head_dim": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 2,
                "rope_parameters": {"partial_rotary_factor": 1.0, "rope_theta": 1e7},
                "rms_norm_eps": 1e-6,
                "layer_types": ["linear_attention", "full_attention"],
                "ple_layer_ids": [2],
                "ngram_size": 2,
                "shared_expert_intermediate_size": 2,
                "linear_conv_kernel_dim": 4,
                "ple_embed_dim": 4,
                "ple_conv_kernel_size": 4,
            }
        }
        self.values = {}
        self.parts = self.values
        self.max_read_bytes = 1024**2
        self.read_counts = []

        def add(name, shape):
            self.values[name] = (torch.randn(shape) * 0.2).bfloat16().float().numpy()

        add("model.language_model.embed_tokens.weight", (16, 4))
        add("lm_head.weight", (16, 4))
        for i in range(2):
            prefix = f"model.language_model.layers.{i}."
            for block in ("attn", "mlp"):
                hc = prefix + block + "_hyper_connection."
                for name, shape in [
                    ("hc_norm.weight", (8,)),
                    ("input_mix_weight_down.weight", (2, 8)),
                    ("input_mix_weight_up.weight", (8, 2)),
                    ("block_inject_weight.weight", (2, 8)),
                ]:
                    add(hc + name, shape)
            for name, shape in [
                ("mlp.gate.weight", (3, 4)),
                ("mlp.shared_expert.gate_proj.weight", (2, 4)),
                ("mlp.shared_expert.up_proj.weight", (2, 4)),
                ("mlp.shared_expert.down_proj.weight", (4, 2)),
                ("mlp.shared_expert_gate.weight", (1, 4)),
                ("mlp.experts.gate_up_proj", (3, 4, 4)),
                ("mlp.experts.down_proj", (3, 4, 2)),
            ]:
                add(prefix + name, shape)
            if i == 0:
                for name, shape in [
                    ("in_proj_qkv.weight", (8, 4)),
                    ("in_proj_z.weight", (4, 4)),
                    ("in_proj_a.weight", (2, 4)),
                    ("in_proj_b.weight", (2, 4)),
                    ("out_proj.weight", (4, 4)),
                    ("A_log", (2,)),
                    ("dt_bias", (2,)),
                    ("conv1d.weight", (8, 1, 4)),
                    ("norm.weight", (2,)),
                ]:
                    add(prefix + "linear_attn." + name, shape)
            else:
                for name, shape in [
                    ("q_proj.weight", (8, 4)),
                    ("k_proj.weight", (2, 4)),
                    ("v_proj.weight", (2, 4)),
                    ("o_proj.weight", (4, 4)),
                    ("q_norm.weight", (2,)),
                    ("k_norm.weight", (2,)),
                ]:
                    add(prefix + "self_attn." + name, shape)
                for name, shape in [
                    ("key_proj.weight", (8, 4)),
                    ("value_proj.weight", (4, 4)),
                    ("conv1d.weight", (8, 1, 4)),
                    ("norm_key.weight", (8,)),
                    ("norm_query.weight", (8,)),
                    ("norm_conv.weight", (8,)),
                ]:
                    add(prefix + "ple." + name, shape)
        for name, shape in [
            ("hc_norm.weight", (8,)),
            ("input_mix_weight_down.weight", (2, 8)),
            ("input_mix_weight_up.weight", (8, 2)),
        ]:
            add("model.language_model.hyper_connection_mixer." + name, shape)

    def shape(self, name):
        return self.values[name].shape

    def info(self, name):
        return {"dtype": "BF16"}

    def tensor(self, name):
        return self.values[name].copy()

    def rows(self, name, first, count):
        if ".experts." in name:
            self.read_counts.append(count)
        return self.values[name][first : first + count].copy()


class TeacherTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261002)
        torch.set_num_threads(2)

    def run_reference(self, source, cases, inputs=None):
        teacher = Teacher(source, cases, chunk_experts=2, device="cpu", inputs=inputs)
        teacher.validate()
        # Tiny owned PLE features; the actual lookup/hash is tested separately.
        if inputs is None:
            teacher.features = torch.tensor(
                [[t * 0.01, t * 0.02, -t * 0.01, t * 0.03] for t in teacher.tokens],
                dtype=torch.bfloat16,
            )
        residual = teacher.initial()
        for layer in range(2):
            residual, stats = teacher.layer(residual, layer)
            self.assertEqual(sum(stats["expert_route_counts"]), len(teacher.tokens) * 2)
            self.assertTrue(bool(torch.isfinite(residual).all()))
        return teacher.head(residual)

    def test_complete_hybrid_reference_and_request_isolation(self):
        source = Fixture()
        cases = [
            {"id": "a", "prompt_ids": [1, 2, 3], "target_ids": [4, 5]},
            {"id": "b", "prompt_ids": [6, 7], "target_ids": [8]},
        ]
        combined = self.run_reference(source, cases)
        isolated = sum((self.run_reference(source, [case]) for case in cases), [])
        self.assertEqual(len(combined), 3)
        for actual, expected in zip(combined, isolated):
            self.assertEqual(actual["context_sha256"], expected["context_sha256"])
            self.assertEqual(
                [v["token_id"] for v in actual["top3"]], [v["token_id"] for v in expected["top3"]]
            )
            self.assertAlmostEqual(
                actual["reference_logprob"], expected["reference_logprob"], places=6
            )
        self.assertTrue(source.read_counts and max(source.read_counts) <= 2)

    def test_original_input_cache_preserves_full_reference_without_embedding_reads(self):
        source = Fixture()
        cases = [
            {"id": "a", "prompt_ids": [1, 2, 3], "target_ids": [4, 5]},
            {"id": "b", "prompt_ids": [6, 7], "target_ids": [8]},
        ]
        expected = self.run_reference(source, cases)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tensors = {}
            histories = {}
            for case in cases:
                tokens = case["prompt_ids"] + case["target_ids"][:-1]
                name = case["id"]
                histories[name] = tokens
                tensors["embedding." + name] = source.values[
                    "model.language_model.embed_tokens.weight"
                ][tokens]
                tensors["ple." + name] = (
                    torch.tensor(
                        [[t * 0.01, t * 0.02, -t * 0.01, t * 0.03] for t in tokens],
                        dtype=torch.bfloat16,
                    )
                    .float()
                    .numpy()
                )
            contract = {
                "source": "owned-bf16-fixture",
                "revision": "0" * 40,
                "config_sha256": "config",
                "scenes_sha256": "scenes",
                "index_sha256": "index",
                "seed": 20261002,
                "format": "orinfer.original-bf16-inputs.v1",
            }
            save_file(tensors, str(root / "inputs.safetensors"))
            (root / "inputs.json").write_text(
                json.dumps(
                    {
                        "contract": contract,
                        "tokens": histories,
                        "sha256": digest(root / "inputs.safetensors"),
                        "source_ranges": [],
                        "complete": True,
                    }
                )
            )
            inputs = Inputs(
                root, **{key: contract[key] for key in contract if key not in ("format", "seed")}
            )
            read = source.rows

            def guarded_rows(name, first, count):
                if name == "model.language_model.embed_tokens.weight":
                    raise AssertionError("Cached input unexpectedly fetched original embeddings")
                return read(name, first, count)

            source.rows = guarded_rows
            actual = self.run_reference(source, cases, inputs)
            self.assertEqual(actual, expected)
            isolated = self.run_reference(source, [cases[1]], inputs)
            self.assertEqual(isolated, [row for row in expected if row["case_id"] == "b"])

    def test_original_expert_reads_exclude_unrouted_neighbours(self):
        source = Fixture()
        cases = [{"id": "a", "prompt_ids": [1], "target_ids": [2]}]
        source.values["model.language_model.layers.0.mlp.gate.weight"] = np.array(
            [[1.0, 1.0, 1.0, 1.0], [-1.0, -1.0, -1.0, -1.0], [2.0, 2.0, 2.0, 2.0]], np.float32
        )
        original_rows = source.rows
        spans = []

        def guarded_rows(name, first, count):
            if ".experts." in name:
                if first <= 1 < first + count:
                    raise AssertionError("Downloaded unrouted expert")
                spans.append((name, first, count))
            return original_rows(name, first, count)

        source.rows = guarded_rows
        teacher = Teacher(source, cases, chunk_experts=2, device="cpu")
        output, counts = teacher.experts(
            torch.ones((2, 4), dtype=torch.bfloat16), teacher.load_weights(0), 0
        )
        self.assertEqual(counts, [2, 0, 2])
        self.assertEqual(
            sorted((first, count) for _, first, count in spans), [(0, 1), (0, 1), (2, 1), (2, 1)]
        )
        self.assertTrue(bool(torch.isfinite(output).all()))

    def test_rejects_quantized_source_and_unbounded_original_read(self):
        source = Fixture()
        case = [{"id": "a", "prompt_ids": [1], "target_ids": [2]}]
        with self.assertRaises(ValueError):
            Teacher(source, case, chunk_experts=100000, device="cpu")
        teacher = Teacher(source, case, chunk_experts=2, device="cpu")
        source.info = lambda name: {"dtype": "I8"}
        with self.assertRaises(ValueError):
            teacher.initial()
        with self.assertRaises(ValueError):
            teacher.validate()


if __name__ == "__main__":
    unittest.main()
