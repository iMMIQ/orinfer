import unittest

from orinfer_build.models.flash_next.repack import partitions, shard_name


class RepackTests(unittest.TestCase):
    def test_components_and_layer_boundaries(self):
        examples = {
            "W_blk.2.ffn_down_exps.weight_packed": "experts-l000-l002.safetensors",
            "W_blk.3.ffn_gate_up_exps.weight_scale": "experts-l003-l005.safetensors",
            "W_blk.7.hc_attn_up.weight_value": "dense-l000-l007.safetensors",
            "W_blk.8.attn_qkv.weight_weight": "dense-l008-l015.safetensors",
            "Q4_blk.3.attn_q_packed": "projections-q4.safetensors",
            "Q4_output_packed": "output.safetensors",
            "W_output.weight_weight": "output.safetensors",
            "HC13_DW_output_hc_up.weight_value_Lo": "hc-up-bf16-packed.safetensors",
            "DW_blk.48.ffn_down_exps.weight_packed": "mtp.safetensors",
            "Vision_blocks_0_attn_qkv_weight": "vision.safetensors",
            "W_e8p_nibble_identity": "output.safetensors",
        }
        for tensor, expected in examples.items():
            with self.subTest(tensor=tensor):
                self.assertEqual(shard_name(tensor), expected)
        with self.assertRaisesRegex(ValueError, "Unrecognized"):
            shard_name("unsupported_weight")

    def test_determinism_and_oversized_component_rejection(self):
        tensors = [
            dict(name=name, segments=[("input", 0, 4)])
            for name in ("Q4_output_scale", "W_output.weight_weight", "Q4_blk.0.attn_q_packed")
        ]
        expected = list(partitions(tensors, 8))
        self.assertEqual(expected, list(partitions(list(reversed(tensors)), 8)))
        with self.assertRaisesRegex(ValueError, "exceed shard limit"):
            list(partitions(tensors, 7))


if __name__ == "__main__":
    unittest.main()
