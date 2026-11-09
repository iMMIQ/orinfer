import unittest

from orinfer_build.models.deepseek_v41.candidate_blocks import index_chains


class CandidateChainTests(unittest.TestCase):
    @staticmethod
    def describe(step):
        return {"op": step["manifest"]} if step["op"] == "kernel" else None

    def test_complete_fused_and_unfused_chains_keep_their_boundaries(self):
        def kernel(op):
            return dict(op="kernel", manifest=op)

        body = [
            dict(op="gather"),
            kernel("private_rows.indexer_cube_causal_logits"),
            kernel("topk.local"),
            kernel("topk.merge"),
            kernel("topk.finalize"),
            dict(op="gather"),
            kernel("private_rows.indexer_fused_causal_topk"),
            kernel("topk.merge_tree"),
            kernel("topk.finalize"),
        ]
        result = index_chains(body, self.describe)
        self.assertEqual([(a, b) for a, b, _ in result], [(1, 5), (6, 9)])

    def test_crossing_a_collective_is_rejected(self):
        body = [
            dict(op="kernel", manifest="private_rows.indexer_cube_causal_logits"),
            dict(op="gather"),
            dict(op="kernel", manifest="topk.finalize"),
        ]
        with self.assertRaisesRegex(ValueError, "complete adjacent"):
            index_chains(body, self.describe)

    def test_missing_finalizer_is_rejected(self):
        body = [
            dict(op="kernel", manifest="private_rows.indexer_cube_causal_logits"),
            dict(op="kernel", manifest="topk.local"),
        ]
        with self.assertRaisesRegex(ValueError, "incomplete"):
            index_chains(body, self.describe)


if __name__ == "__main__":
    unittest.main()
