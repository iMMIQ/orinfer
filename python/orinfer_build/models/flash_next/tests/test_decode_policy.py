"""AOT specializations and the symbolic fallback cover rows without gaps."""

import unittest
from orinfer_build.contracts.projection_policy import (
    decode_shapes,
    selection_name,
    q4_configs,
    e8p_q4_configs,
    candidate_scores,
    w8_configs,
)


class DecodeAlignmentTests(unittest.TestCase):
    def test_specializations_and_symbolic_fallback_cover_each_live_row_once(self):
        shapes = decode_shapes()
        self.assertEqual([s.rows for s in shapes if not s.dynamic], [1, 2, 3, 4, 8, 16, 32])
        self.assertEqual(sum(s.dynamic for s in shapes), 1)
        covered = []
        for shape in shapes:
            for first, last in shape.ranges():
                covered.extend(range(first, last + 1))
                self.assertTrue(
                    selection_name("slot", first, last).endswith(f"/rows/{first}-{last}")
                )
        self.assertEqual(sorted(covered), list(range(1, 129)))
        for bounds in [(0, 8), (8, 4), (33, 129)]:
            with self.assertRaises(ValueError):
                selection_name("slot", *bounds)

    def test_one_q4_search_space_preserves_small_and_large_algorithms(self):
        shared = None
        for rows in [1, 3, 8, 32, 128]:
            configs = q4_configs(rows)
            self.assertEqual({c["algorithm"] for c in configs}, {"dp4a", "tensor_core"})
            self.assertTrue(all(c["block_m"] <= 8 for c in configs if c["algorithm"] == "dp4a"))
            tensor_core = [c for c in configs if c["algorithm"] == "tensor_core"]
            self.assertEqual(len(tensor_core), len({tuple(sorted(c.items())) for c in tensor_core}))
            self.assertTrue(any(c["block_n"] == 32 and c["block_k"] == 256 for c in tensor_core))
            if shared is not None:
                self.assertEqual(tensor_core, shared)
            shared = tensor_core

    def test_direct_e8p_is_an_offline_candidate_for_every_row_count(self):
        direct = None
        for rows in (1, 2, 3, 17, 128, 4096):
            regular = e8p_q4_configs(rows)
            self.assertTrue(all(c in regular for c in q4_configs(rows)))
            self.assertTrue(any(c["block_m"] == 256 for c in regular))
            candidates = e8p_q4_configs(rows, direct=True)
            selected = [c for c in candidates if c["algorithm"] == "int4_direct"]
            self.assertEqual({c["block_n"] for c in selected}, {32, 64, 128})
            self.assertEqual({c["split_k"] for c in selected}, {1, 2, 4})
            self.assertTrue(all(c["block_k"] == 128 for c in selected))
            self.assertTrue(all(c["block_n"] * c["split_k"] <= 256 for c in selected))
            self.assertEqual(candidates[: len(q4_configs(rows))], q4_configs(rows))
            if direct is not None:
                self.assertEqual(selected, direct)
            direct = selected

    def test_prefill_objective_minimizes_total_gpu_time_instead_of_short_row_ratios(self):
        measured = [
            dict(rows=5, times_ms=[0.1, 0.3, 0.001]),
            dict(rows=512, times_ms=[4.0, 3.0, 0.001]),
            dict(rows=4096, times_ms=[16.0, 10.0, None]),
        ]
        balanced = candidate_scores(measured, {2})
        prefill = candidate_scores(measured, {2}, "total_latency")
        self.assertLess(balanced[0], balanced[1])
        self.assertLess(prefill[1], prefill[0])
        self.assertEqual(prefill[2], float("inf"))
        self.assertEqual(candidate_scores(measured, {0, 1, 2}), [float("inf")] * 3)
        with self.assertRaises(ValueError):
            candidate_scores(measured, {2}, "unknown")

    def test_w8_search_includes_dp4a_for_large_matrices(self):
        for rows in [1, 4, 8, 32, 128]:
            for outputs in [48, 640, 2560, 6144]:
                candidates = w8_configs(rows, outputs)
                self.assertEqual(
                    {c.get("algorithm", "tensor_core") for c in candidates}, {"dp4a", "tensor_core"}
                )
                self.assertEqual(
                    {c["block_n"] for c in candidates if c.get("algorithm") == "dp4a"}, {4, 8, 16}
                )
