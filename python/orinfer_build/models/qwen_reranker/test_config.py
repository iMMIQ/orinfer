import copy
import unittest
from orinfer_build.models.qwen_reranker.config import EXPECTED, configuration, signature
from orinfer_build.quantization.qwen_e8p import text_configuration, policy, rotated_w8_layers


class Configuration(unittest.TestCase):
    def test_original_qwen3_geometry_and_full_signature(self):
        c = copy.deepcopy(EXPECTED)
        self.assertEqual(text_configuration(c)["num_hidden_layers"], 28)
        q = dict(c, quantization_config=dict(policy="qwen3-reranker-e8p-q4a8-v1"))
        self.assertEqual(signature(c), signature(q))
        configuration(q, quantized=True)
        with self.assertRaises(ValueError):
            configuration(q)

    def test_qk_norm_head_geometry_and_rope_cannot_drift(self):
        for key, value in [
            ("head_dim", 64),
            ("num_key_value_heads", 4),
            ("hidden_size", 2048),
            ("rope_theta", 10000),
            ("sliding_window", 4096),
            ("attention_bias", True),
            ("tie_word_embeddings", False),
        ]:
            c = copy.deepcopy(EXPECTED)
            c[key] = value
            with self.assertRaises(ValueError):
                configuration(c)

    def test_projection_policy_protects_outputs_and_norms(self):
        info = dict(shape=[1024, 2048], dtype="BF16")
        self.assertEqual(policy("model.layers.0.self_attn.o_proj.weight", info), "w8")
        self.assertEqual(policy("model.layers.0.self_attn.q_proj.weight", info), "e8p-q4")
        self.assertEqual(
            policy("model.layers.0.self_attn.q_norm.weight", dict(shape=[128], dtype="BF16")),
            "original",
        )

    def test_protected_qkv_and_ffn_use_the_same_rotated_input_basis(self):
        layers = rotated_w8_layers(EXPECTED)
        self.assertEqual(layers, [0, 1, 2, 3, 8])
        info = dict(shape=[1024, 1024], dtype="BF16")
        for role in [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ]:
            self.assertEqual(
                policy(f"model.layers.3.{role}.weight", info, protected_layers=layers), "w8-rotated"
            )
            self.assertEqual(
                policy(f"model.layers.4.{role}.weight", info, protected_layers=layers), "e8p-q4"
            )
        self.assertEqual(
            policy("model.layers.3.self_attn.o_proj.weight", info, protected_layers=layers), "w8"
        )


if __name__ == "__main__":
    unittest.main()
