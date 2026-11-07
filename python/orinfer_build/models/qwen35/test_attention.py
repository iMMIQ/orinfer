"""One measured geometry covers owner counts without model-specific cutoffs."""

import unittest
from orinfer_build.models.qwen35.attention import candidates, causal_selection
from orinfer_build.models.qwen35.parallel import geometry


class OwnerAttentionPolicyTests(unittest.TestCase):
    def test_geometry_is_derived_from_heads_and_reference_is_preserved(self):
        reference = dict(block_m=32, block_n=64, head_group=1, max_splits=8, decoded_kv=True)
        for query, kv in ((8, 2), (16, 4), (24, 4)):
            result = candidates(query, kv, reference)
            self.assertTrue(all(c["block_m"] == 16 for c in result))
            self.assertTrue(all(c["head_group"] == query // kv for c in result))
            self.assertTrue(all(c["independent_queries"] for c in result))
            self.assertEqual(result[0]["max_splits"], 8)
            self.assertTrue(all("decoded_kv" not in c for c in result))
        self.assertTrue(reference["decoded_kv"])
        with self.assertRaises(ValueError):
            candidates(7, 2, reference)

    def test_causal_rows_share_one_choice_and_leave_larger_prefill_choices_intact(self):
        original = dict(
            configs=[dict(name="default"), dict(name="bounded")],
            default="0",
            selections=[dict(min_rows=1, max_rows=17, implementation="1")],
        )
        config = dict(block_m=16, head_group=6, block_n=64, max_splits=8)
        selected = causal_selection(original, config)
        for rows in range(1, 9):
            self.assertEqual(geometry(selected, rows), config | dict(independent_queries=True))
        for rows in (9, 17, 18, 4096):
            self.assertEqual(geometry(selected, rows), geometry(original, rows))
        self.assertNotIn("independent_queries", config)
        self.assertEqual(original["selections"][0]["min_rows"], 1)
