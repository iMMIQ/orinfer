"""Legacy attention geometry and bounded alias adoption contracts."""

import unittest

from orinfer_build.models.qwen35.rebuild_attention import decode_geometry, short_alias


class RebuildAttentionTests(unittest.TestCase):
    def test_measured_geometry_comes_from_compiled_tiles_and_split_contract(self):
        source = """
        void* k = ((void*)((char*)buf_dyn_shmem + 0));
        void* v = ((void*)((char*)buf_dyn_shmem + 32768));
        split_count[0] = attn_split_count(1, 6, 16, 16, 8, owners);
        """
        self.assertEqual(
            decode_geometry(dict(block=[128, 1, 1]), source, 256),
            dict(
                block_m=16,
                block_n=64,
                head_group=6,
                max_splits=8,
                threads=128,
                stages=0,
                independent_queries=True,
            ),
        )
        with self.assertRaisesRegex(ValueError, "compact decode tile"):
            decode_geometry(dict(block=[128, 1, 1]), source.replace("6, 16, 16", "6, 32, 16"), 256)

    def test_small_replacements_leave_existing_large_row_launches_reachable(self):
        prefix = "qwen_rows/layer3/private/attention"
        self.assertIsNone(short_alias(prefix + "/rows/1-2"))
        self.assertIsNone(short_alias(prefix + "_merge/rows/3-8"))
        self.assertEqual(short_alias(prefix + "/rows/6-32"), prefix + "/rows/9-32")
        self.assertEqual(short_alias(prefix + "/rows/33-90"), prefix + "/rows/33-90")
        self.assertEqual(short_alias(prefix), prefix)
        self.assertEqual(
            short_alias("qwen_rows/layer3/pre/q/rows/1-4"), "qwen_rows/layer3/pre/q/rows/1-4"
        )
