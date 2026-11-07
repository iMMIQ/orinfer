"""CPU checks for rejecting incompatible adapters before weight adoption."""

import copy
import unittest
from orinfer_build.contracts.vision import validate_vision


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
        }

    def test_flash_vision_uses_the_shared_encoder_with_its_own_merger(self):
        vision = copy.deepcopy(self.config["vision_config"])
        vision["out_hidden_size"] = 2560
        validate_vision(vision, 2560)
        with self.assertRaises(ValueError):
            validate_vision(vision, 5120)

    def test_supported_layout(self):
        validate_vision(self.config["vision_config"], 5120)

    def test_small_dense_vision_reuses_the_same_adapter(self):
        vision = self.config["vision_config"] | dict(
            hidden_size=1024, intermediate_size=4096, depth=24, out_hidden_size=2048
        )
        validate_vision(vision, 2048)

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
                validate_vision(config["vision_config"], 5120)


if __name__ == "__main__":
    unittest.main()
