"""CPU checks for rejecting incompatible adapters before weight adoption."""

import copy
import unittest
from config import validate_adapter, validate_vision


class AdapterCompatibility(unittest.TestCase):
    def setUp(self):
        self.config = {
            "vision_config": dict(
                hidden_size=1152,
                num_heads=16,
                in_channels=3,
                patch_size=16,
                spatial_merge_size=2,
                temporal_patch_size=2,
                out_hidden_size=5120,
                hidden_act="gelu_pytorch_tanh",
                depth=27,
                intermediate_size=4304,
                num_position_embeddings=2304,
                deepstack_visual_indexes=[],
            ),
            "text_config": dict(
                hidden_size=5120,
                num_attention_heads=24,
                num_key_value_heads=4,
                head_dim=256,
                vocab_size=32,
                rope_parameters=dict(
                    rope_type="default",
                    rope_theta=10_000_000,
                    partial_rotary_factor=0.25,
                    mrope_section=[11, 11, 10],
                    mrope_interleaved=True,
                ),
            ),
        }
        self.manifest = dict(
            vocab=32,
            max_context=1024,
            chunk_tokens=512,
            buffers=[
                dict(name=name, dtype=dtype, shape=shape)
                for name, dtype, shape in [
                    ("Embedding_P", "u8", [32, 2560]),
                    ("Embedding_S", "f16", [32, 40]),
                    ("Embedding_Z", "i8", [32, 40]),
                    ("FullX", "f16", [512, 14336]),
                    ("Hidden", "f16", [512, 5120]),
                    ("Rotary", "f16", [1024, 64]),
                ]
            ],
        )

    def test_flash_vision_uses_the_shared_encoder_with_its_own_merger(self):
        vision = copy.deepcopy(self.config["vision_config"])
        vision["out_hidden_size"] = 2560
        validate_vision(vision, 2560)
        with self.assertRaises(ValueError):
            validate_vision(vision, 5120)

    def test_supported_layout(self):
        validate_adapter(self.config, self.manifest)

    def test_similar_dimensions_do_not_imply_compatible_attention(self):
        for heads, kv, dim in [(40, 8, 128), (24, 8, 256), (24, 4, 128)]:
            config = copy.deepcopy(self.config)
            config["text_config"].update(
                num_attention_heads=heads, num_key_value_heads=kv, head_dim=dim
            )
            with self.assertRaises(ValueError):
                validate_adapter(config, self.manifest)

    def test_incompatible_embedding_and_rotary(self):
        for name, shape in [
            ("Embedding_S", [32, 160]),
            ("Embedding_P", [32, 5120]),
            ("Rotary", [1024, 128]),
        ]:
            manifest = copy.deepcopy(self.manifest)
            next(b for b in manifest["buffers"] if b["name"] == name)["shape"] = shape
            with self.assertRaises(ValueError):
                validate_adapter(self.config, manifest)
        config = copy.deepcopy(self.config)
        config["text_config"]["rope_parameters"]["mrope_interleaved"] = False
        with self.assertRaises(ValueError):
            validate_adapter(config, self.manifest)

    def test_other_vision_variants_need_an_adapter(self):
        for key, value in [
            ("hidden_act", "gelu"),
            ("deepstack_visual_indexes", [8, 16]),
            ("num_position_embeddings", 2305),
            ("in_channels", 4),
        ]:
            config = copy.deepcopy(self.config)
            config["vision_config"][key] = value
            with self.assertRaises(ValueError):
                validate_adapter(config, self.manifest)


if __name__ == "__main__":
    unittest.main()
