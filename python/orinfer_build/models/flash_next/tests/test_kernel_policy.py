"""Expert tile contracts shared by initial builds and execution upgrades."""

import unittest

from orinfer_build.models.flash_next.kernel_policy import (
    expert_tile_config,
    expert_projection_config,
    expert_implementation,
    projection_policies,
)


class ExpertTilePolicyTests(unittest.TestCase):
    def test_selected_storage_overrides_retained_w8_for_all_shapes(self):
        buffers = {
            "Q4_blk.0.attn_gate_packed": dict(dtype="u16", shape=[6144, 640]),
            "Q4_blk.0.attn_gate_scale": dict(dtype="f16", shape=[6144]),
            "Q4_blk.0.attn_gate_signs": dict(dtype="i8", shape=[2560]),
            "Q4_blk.0.attn_qkv_packed": dict(dtype="u16", shape=[10240, 640]),
            "Q4_blk.0.attn_qkv_scale": dict(dtype="f16", shape=[10240]),
            "W_blk.0.attn_qkv.weight_weight": dict(dtype="i8", shape=[10240, 2560]),
        }
        policy = projection_policies(buffers)["W_blk.0.attn_qkv.weight_weight"]
        self.assertEqual(policy.signs, "Q4_blk.0.attn_gate_signs")
        for rows in (1, 2, 4, 8, 16, 128, 512, 4096):
            self.assertEqual(policy.packed, "Q4_blk.0.attn_qkv_packed")
            self.assertEqual((policy.outputs, policy.inputs), (10240, 2560))
        del buffers[policy.signs]
        with self.assertRaisesRegex(ValueError, "operand"):
            projection_policies(buffers)

    def test_exact_a8_is_not_restricted_to_a_named_single_program(self):
        for rows in (1, 2, 3, 4, 5, 8, 16, 128, 512, 4096):
            self.assertEqual(expert_implementation(rows, 64, nibble_book=True), "exact_a8")
            self.assertEqual(expert_implementation(rows, 64, nibble_book=False), "int8")
            self.assertEqual(expert_implementation(rows, 128, nibble_book=True), "int8")

    def test_reference_tile_keeps_exact_sign_table_at_every_extent(self):
        config = expert_tile_config(1)
        self.assertEqual(config["block_m"], 64)
        self.assertEqual(config["block_n"], 64)
        self.assertTrue(config["shortbook"])
        self.assertTrue(config["byte_permute"])
        self.assertTrue(config["selectorbook"])

    def test_prefill_workspaces_bound_all_routing_distributions(self):
        for rows in range(1, 4097):
            config = expert_tile_config(rows)
            tiles = (rows * 10 + config["block_m"] - 1) // config["block_m"] + 512
            self.assertLessEqual(tiles, 1792)

    def test_reference_configuration_has_no_token_threshold(self):
        for rows in (2, 8, 9, 16, 32, 128, 129, 256, 512, 513, 1024, 4096):
            self.assertEqual(expert_tile_config(rows), expert_tile_config(1))

    def test_invalid_capacity_is_rejected(self):
        for rows in (0, 4097, True, 1.5):
            with self.assertRaises(ValueError):
                expert_tile_config(rows)

    def test_input_width_does_not_choose_an_untuned_implementation(self):
        for rows in (1, 2, 4, 8):
            gate = expert_projection_config(rows, 2560)
            down = expert_projection_config(rows, 640)
            self.assertEqual(gate, down)
            self.assertEqual(gate, expert_tile_config(rows))

    def test_explicit_xor_codec_diagnostic_is_not_a_row_heuristic(self):
        for rows in (1, 3, 8, 512):
            self.assertTrue(expert_projection_config(rows, 2560, parity_xor=True)["parity_xor"])
            self.assertNotIn("parity_xor", expert_projection_config(rows, 2560))
        with self.assertRaises(ValueError):
            expert_projection_config(1, 2560, parity_xor=1)

    def test_prefill_keeps_existing_projection_policy(self):
        for rows in (16, 32, 128, 256, 512, 2048, 4096):
            self.assertEqual(expert_projection_config(rows, 2560), expert_tile_config(rows))
        for inputs in (0, True, 129, 640.0):
            with self.assertRaises(ValueError):
                expert_projection_config(1, inputs)


if __name__ == "__main__":
    unittest.main()
