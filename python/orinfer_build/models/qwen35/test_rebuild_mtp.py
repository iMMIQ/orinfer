"""Preserve registered geometry and physical buffers during incremental MTP rebuilds."""

import unittest
from orinfer_build.models.qwen35.rebuild_mtp import ExistingWeights, attention_geometry


class ReconstructionTests(unittest.TestCase):
    def test_attention_restores_measured_tiling_and_independent_query_geometry(self):
        kernel = dict(block=[128, 1, 1])
        source = """
        void* k = ((void*)((char*)buf_dyn_shmem + 8192));
        void* v = ((void*)((char*)buf_dyn_shmem + 40960));
        split_count[0] = 8;
        """
        result = attention_geometry(kernel, source, 256, 6)
        self.assertEqual(
            result,
            dict(
                block_m=16,
                target_tiles=16,
                max_splits=8,
                head_group=6,
                independent_queries=True,
                block_n=64,
                threads=128,
                stages=0,
            ),
        )
        source = source.replace(
            "split_count[0] = 8;", "split_count[0] = attn_split_count(m, 1, 32, 16, 8, 1);"
        )
        result = attention_geometry(kernel, source, 256, 6)
        self.assertEqual(
            (result["block_m"], result["head_group"], result["independent_queries"]), (32, 1, False)
        )
        with self.assertRaisesRegex(ValueError, "split target"):
            attention_geometry(kernel, source.replace("32, 16, 8", "32, 32, 8"), 256, 6)

    def test_existing_buffers_keep_scope_payload_and_physical_shape(self):
        source = dict(buffers=[dict(name="Input", dtype="i32", shape=[512], data=None)])
        weights = ExistingWeights(source, {"Input": "sequence"})
        weights.fixed("Input", "i32", [512], "workspace")
        self.assertEqual(weights.scopes["Input"], "sequence")
        with self.assertRaisesRegex(ValueError, "physical buffer"):
            weights.fixed("Input", "i32", [1024])
        with self.assertRaisesRegex(ValueError, "immutable tensor"):
            weights.immutable("missing", cast="float16")
        self.assertEqual(source["buffers"][0]["shape"], [512])


if __name__ == "__main__":
    unittest.main()
