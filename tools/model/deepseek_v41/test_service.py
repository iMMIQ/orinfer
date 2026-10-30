"""CPU checks for explicit native A2 snapshot geometry."""

import unittest
from tools.model.deepseek_v41.service import enable_prefix_sharing, state_layout


class StateLayoutTests(unittest.TestCase):
    def model(self, capacity=256, storage=256):
        sizes = {
            "cache.swa.0": storage * 512,
            "cache.swa.0.scale": storage * 16,
            "cache.long.2": capacity // 2 * 512,
            "cache.long.2.scale": capacity // 2 * 16,
            "cache.long.20": capacity * 512,
            "cache.long.20.scale": capacity * 16,
            "cache.key.20": capacity * 128,
            "cache.key_scale.20": capacity * 2,
            "cache.ring.2": 262144,
            "kv.int8.work.swa": storage * 1024,
        }
        return dict(
            capacity=capacity,
            batch=dict(private=list(sizes)),
            ranks=[dict(buffers=[dict(name=name, bytes=size) for name, size in sizes.items()])],
        )

    def test_full_rate_and_half_rate_state_keep_their_distinct_row_contracts(self):
        items = {p["buffer"]: p for p in state_layout(self.model())}
        self.assertEqual(items["cache.long.2"]["tokens_per_row"], 2)
        self.assertEqual(items["cache.long.20"]["tokens_per_row"], 1)
        self.assertEqual(items["cache.key.20"]["row_bytes"], 128)
        self.assertEqual(items["cache.key_scale.20"]["row_bytes"], 2)
        self.assertEqual(items["cache.ring.2"]["tokens_per_row"], 0)
        self.assertNotIn("kv.int8.work.swa", items)

    def test_swa_ring_geometry_is_independent_of_logical_context_capacity(self):
        items = {p["buffer"]: p for p in state_layout(self.model(262144, 4096), 4096)}
        self.assertEqual(items["cache.swa.0"]["row_bytes"], 512)
        self.assertEqual(items["cache.swa.0"]["window"], 128)
        self.assertTrue(items["cache.swa.0"]["circular"])
        self.assertEqual(items["cache.long.20"]["tokens_per_row"], 1)
        with self.assertRaisesRegex(ValueError, "SWA storage"):
            state_layout(self.model(262144, 4096))

    def test_only_closed_linear_kv_rows_are_marked_shareable(self):
        model = self.model(262144, 4096)
        model["batch"]["state"] = state_layout(model, 4096)
        self.assertTrue(all("append_only" not in s for s in model["batch"]["state"]))
        enable_prefix_sharing(model)
        states = {s["buffer"]: s for s in model["batch"]["state"]}
        for name in (
            "cache.long.2",
            "cache.long.20",
            "cache.long.2.scale",
            "cache.key.20",
            "cache.key_scale.20",
        ):
            self.assertTrue(states[name]["append_only"])
        for name in ("cache.swa.0", "cache.swa.0.scale", "cache.ring.2"):
            self.assertFalse(states[name].get("append_only", False))


if __name__ == "__main__":
    unittest.main()
