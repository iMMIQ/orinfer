"""Offline schedule candidates preserve the published attention baseline."""

from copy import deepcopy
import unittest

from orinfer_build.models.deepseek_v41.tune_attention import attention_configs, select_schedule


class AttentionTuningTests(unittest.TestCase):
    def test_existing_four_head_schedule_remains_the_baseline(self):
        parameters = dict(softmax_heads=4, tile_slots=16, reuse_long=True)
        original = deepcopy(parameters)
        configs = attention_configs(parameters)
        self.assertEqual(configs["baseline"], {})
        self.assertEqual(set(configs), {"baseline", "head-1", "head-8"})
        self.assertEqual(parameters, original)
        for update in configs.values():
            self.assertTrue(dict(parameters, **update)["reuse_long"])
            self.assertEqual(dict(parameters, **update)["tile_slots"], 16)

    def test_schedule_rejects_an_outlier_even_with_a_large_mean_gain(self):
        records = [
            dict(rows=64, times={"baseline": 10.0, "head-8": 12.0}),
            dict(rows=256, times={"baseline": 20.0, "head-8": 10.0}),
        ]
        self.assertEqual(select_schedule(records), "baseline")

    def test_small_timing_noise_keeps_the_published_schedule(self):
        self.assertEqual(
            select_schedule([dict(rows=64, times={"baseline": 10.0, "head-8": 9.9})]),
            "baseline",
        )

    def test_useful_gain_selects_a_legal_schedule(self):
        self.assertEqual(
            select_schedule([dict(rows=64, times={"baseline": 10.0, "head-8": 8.0})]),
            "head-8",
        )


if __name__ == "__main__":
    unittest.main()
