"""Reject causal, pooling and geometry drift before encoding new weight values."""

import copy
import unittest

from orinfer_build.models.nemotron_embed.config import TEXT, VISION, ROPE, configuration, signature
from orinfer_build.quantization.qwen_e8p import policy, text_configuration, rotated_w8_layers


def config():
    return dict(
        model_type="mistral3",
        architectures=["Mistral3Model"],
        is_causal=False,
        pooling="avg",
        spatial_merge_size=2,
        image_token_index=10,
        projector_hidden_act="gelu",
        multimodal_projector_bias=False,
        vision_feature_layer=-1,
        text_config=dict(
            TEXT,
            rope_parameters=dict(ROPE),
            llama_4_scaling=dict(beta=0.1, original_max_position_embeddings=16384),
        ),
        vision_config=dict(VISION, rope_parameters=dict(rope_type="default", rope_theta=10000.0)),
    )


class Configuration(unittest.TestCase):
    def test_original_and_signature(self):
        c = config()
        self.assertEqual(configuration(c)["num_attention_heads"], 24)
        self.assertEqual(text_configuration(c)["num_hidden_layers"], 16)
        other = copy.deepcopy(c)
        other["quantization_config"] = dict(quant_method="orinfer_e8p_q4_int8")
        self.assertEqual(signature(c), signature(other))
        configuration(other, quantized=True)
        with self.assertRaises(ValueError):
            configuration(other)

    def test_changed_semantics_fail(self):
        for section, key, value in [
            (None, "is_causal", True),
            (None, "pooling", "last"),
            ("text_config", "num_attention_heads", 16),
            ("vision_config", "patch_size", 16),
            ("text_config", "is_causal", True),
        ]:
            c = config()
            (c if section is None else c[section])[key] = value
            with self.assertRaises(ValueError):
                configuration(c)

    def test_projection_policy_keeps_sensitive_outputs(self):
        matrix = dict(shape=[2048, 2048], dtype="BF16")
        for name in [
            "language_model.embed_tokens.weight",
            "language_model.layers.0.mlp.up_proj.weight",
            "vision_tower.transformer.layers.0.feed_forward.gate_proj.weight",
            "multi_modal_projector.linear_1.weight",
        ]:
            self.assertEqual(policy(name, matrix), "e8p-q4")
        for name in [
            "language_model.layers.0.self_attn.o_proj.weight",
            "vision_tower.transformer.layers.0.attention.o_proj.weight",
            "multi_modal_projector.linear_2.weight",
        ]:
            self.assertEqual(policy(name, matrix), "w8")
        self.assertEqual(
            policy("vision_tower.patch_conv.weight", dict(shape=[1024, 3, 14, 14], dtype="BF16")),
            "original",
        )

    def test_sensitive_block_uses_the_shared_rotated_input_basis(self):
        layers = rotated_w8_layers(config())
        self.assertEqual(layers, [2])
        matrix = dict(shape=[2048, 2048], dtype="BF16")
        for role in [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ]:
            self.assertEqual(
                policy(f"language_model.layers.2.{role}.weight", matrix, protected_layers=layers),
                "w8-rotated",
            )
            self.assertEqual(
                policy(f"language_model.layers.3.{role}.weight", matrix, protected_layers=layers),
                "e8p-q4",
            )
        self.assertEqual(
            policy(
                "language_model.layers.2.self_attn.o_proj.weight", matrix, protected_layers=layers
            ),
            "w8",
        )


if __name__ == "__main__":
    unittest.main()
