import unittest

from tools.model.deepseek_v41.fuse_attention import (
    selected_chain,
    selected_packet_ranges,
    selected_work_key,
)


class SelectedAttentionChainTests(unittest.TestCase):
    def test_small_target_keeps_original_schedule_without_changing_draft_geometry(self):
        packets = [
            dict(rows=32, min_rows=1, batch_decode=True),
            dict(rows=2048, min_rows=1, max_rows=511, body=[dict(op="gather")]),
            dict(rows=2048, min_rows=512, body=[]),
            dict(rows=512, min_rows=5, max_rows=160, role="draft"),
        ]
        selected = selected_packet_ranges(packets, 32)
        self.assertEqual(
            [(p["min_rows"], p.get("max_rows", p["rows"])) for p in selected],
            [(1, 32), (1, 31), (32, 511), (512, 2048), (5, 160)],
        )
        selected[2]["body"][0]["op"] = "fused"
        self.assertEqual(selected[1]["body"][0]["op"], "gather")
        self.assertEqual(packets[1]["body"][0]["op"], "gather")

    def test_source_and_bounded_decoder_cannot_share_work(self):
        grid = dict(op="rows")
        extent = dict(min_rows=1, max_rows=2048)
        self.assertNotEqual(
            selected_work_key(grid, None, extent, False),
            selected_work_key(grid, None, extent, True),
        )

    def fixture(self):
        names = [
            "kv_int8_gather_attention",
            "attention_cube_qk",
            "attention_cube_softmax",
            "attention_cube_pv",
            "attention_cube_pv",
        ]
        steps = [
            dict(op="kernel", manifest=str(i), buffers=[str(i), "p", "active"]) for i in range(5)
        ]
        specs = {
            str(i): dict(
                op="private_rows." + name,
                parameters=dict(rows=32, heads=8, selected=640, _owner_token_capacity=32),
            )
            for i, name in enumerate(names)
        }
        contract = dict(
            work_ranges={"3": dict(min_rows=1, max_rows=8), "4": dict(min_rows=9, max_rows=32)}
        )
        return steps, specs, contract

    def test_consumes_all_disjoint_pv_variants(self):
        steps, specs, contract = self.fixture()
        self.assertEqual(selected_chain(steps, 0, lambda s: specs[s["manifest"]], contract), steps)

    def test_rejects_gaps_and_overlap(self):
        for low in [8, 10]:
            steps, specs, contract = self.fixture()
            contract["work_ranges"]["4"]["min_rows"] = low
            with self.assertRaisesRegex(ValueError, "without overlap"):
                selected_chain(steps, 0, lambda s: specs[s["manifest"]], contract)

    def test_rejects_distinct_output_view(self):
        steps, specs, contract = self.fixture()
        steps[4]["buffers"][1] = "different"
        with self.assertRaisesRegex(ValueError, "physical output"):
            selected_chain(steps, 0, lambda s: specs[s["manifest"]], contract)

    def test_rejects_missing_upper_range(self):
        steps, specs, contract = self.fixture()
        with self.assertRaisesRegex(ValueError, "do not cover"):
            selected_chain(steps[:4], 0, lambda s: specs[s["manifest"]], contract)
