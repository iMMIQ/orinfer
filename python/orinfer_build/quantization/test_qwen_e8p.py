"""Precision policy for shared dense Qwen encoders and tied embeddings."""

import unittest
from orinfer_build.quantization.qwen_e8p import canonical_name, policy, text_configuration


class DensePolicyTests(unittest.TestCase):
    def test_protected_projections_keep_the_shared_input_rotation(self):
        info = dict(dtype="BF16", shape=[2048, 2048])
        self.assertEqual(
            policy(
                "model.language_model.layers.3.self_attn.q_proj.weight",
                info,
                protected_layers=[0, 1, 2, 3],
            ),
            "w8-rotated",
        )
        self.assertEqual(
            policy(
                "model.language_model.layers.4.self_attn.q_proj.weight",
                info,
                protected_layers=[0, 1, 2, 3],
            ),
            "e8p-q4",
        )
        self.assertEqual(
            policy(
                "model.language_model.layers.3.self_attn.o_proj.weight",
                info,
                protected_layers=[0, 1, 2, 3],
            ),
            "w8",
        )

    def test_four_billion_preserves_tied_embedding_semantics(self):
        config = dict(
            model_type="qwen3_5",
            tie_word_embeddings=True,
            text_config=dict(hidden_size=2560, intermediate_size=9216, num_hidden_layers=32),
        )
        self.assertTrue(text_configuration(config)["tie_word_embeddings"])
        config["text_config"]["intermediate_size"] = 8192
        with self.assertRaises(ValueError):
            text_configuration(config)

    def test_root_vision_names_use_the_same_sensitive_tensor_policy(self):
        position = canonical_name("visual.pos_embed.weight")
        self.assertEqual(position, "model.visual.pos_embed.weight")
        self.assertEqual(policy(position, dict(dtype="BF16", shape=[2304, 1152])), "original")
        self.assertEqual(canonical_name(position), position)
        self.assertEqual(
            canonical_name("mtp.layers.0.mlp.gate.weight"), "mtp.layers.0.mlp.gate.weight"
        )

    def test_moe_banks_use_q4_without_quantizing_router_or_draft_exceptions(self):
        text = dict(
            hidden_size=2048,
            num_hidden_layers=40,
            num_experts=256,
            num_experts_per_tok=8,
            moe_intermediate_size=512,
            shared_expert_intermediate_size=512,
        )
        config = dict(model_type="qwen3_5_moe", text_config=text, tie_word_embeddings=False)
        self.assertEqual(text_configuration(config)["intermediate_size"], 512)
        for name, shape, expected in (
            ("model.language_model.layers.0.mlp.experts.gate_up_proj", [256, 1024, 2048], "e8p-q4"),
            ("model.language_model.layers.0.mlp.experts.down_proj", [256, 2048, 512], "e8p-q4"),
            ("model.language_model.layers.0.mlp.gate.weight", [256, 2048], "original"),
            ("mtp.layers.0.mlp.experts.0.up_proj.weight", [512, 2048], "w8"),
        ):
            self.assertEqual(policy(name, dict(dtype="BF16", shape=shape)), expected)
        text["num_experts_per_tok"] = 10
        with self.assertRaises(ValueError):
            text_configuration(config)

    def test_medium_dense_preserves_untied_output_head(self):
        config = dict(
            model_type="qwen3_5",
            tie_word_embeddings=False,
            text_config=dict(hidden_size=4096, intermediate_size=12288, num_hidden_layers=32),
        )
        text = text_configuration(config)
        self.assertIs(text, config["text_config"])
        self.assertIs(text["tie_word_embeddings"], False)
        for name in ("model.language_model.embed_tokens.weight", "lm_head.weight"):
            self.assertEqual(policy(name, dict(dtype="BF16", shape=[248320, 4096])), "e8p-q4")
        text["tie_word_embeddings"] = True
        self.assertIs(text_configuration(config)["tie_word_embeddings"], True)
        text["tie_word_embeddings"] = 0
        with self.assertRaises(ValueError):
            text_configuration(config)

    def test_main_projections_use_e8p_and_sensitive_operands_keep_w8(self):
        for name, shape, expected in (
            ("model.language_model.layers.0.mlp.down_proj.weight", [2048, 6144], "e8p-q4"),
            (
                "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
                [6144, 2048],
                "e8p-q4",
            ),
            ("model.language_model.embed_tokens.weight", [248320, 2048], "e8p-q4"),
            ("model.language_model.layers.0.linear_attn.out_proj.weight", [2048, 2048], "w8"),
            ("mtp.layers.0.mlp.up_proj.weight", [6144, 2048], "w8"),
            ("model.language_model.layers.0.linear_attn.in_proj_a.weight", [16, 2048], "original"),
            ("model.visual.pos_embed.weight", [2304, 1024], "original"),
            ("model.visual.blocks.0.attn.qkv.weight", [3072, 1024], "e8p-q4"),
        ):
            with self.subTest(name=name):
                self.assertEqual(policy(name, dict(dtype="BF16", shape=shape)), expected)

    def test_asr_encoder_retains_floating_weights_and_decoder_uses_shared_policy(self):
        self.assertEqual(
            policy("model.audio_tower.layers.0.fc1.weight", dict(dtype="BF16", shape=[4096, 1024])),
            "original",
        )
        self.assertEqual(
            policy(
                "model.language_model.layers.0.mlp.up_proj.weight",
                dict(dtype="BF16", shape=[6144, 2048]),
            ),
            "e8p-q4",
        )
        self.assertEqual(
            policy(
                "model.language_model.layers.0.self_attn.o_proj.weight",
                dict(dtype="BF16", shape=[2048, 2048]),
            ),
            "w8",
        )


if __name__ == "__main__":
    unittest.main()
