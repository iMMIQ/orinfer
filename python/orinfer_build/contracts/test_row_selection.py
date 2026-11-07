"""Measured selections cover tails without retaining losing implementations."""

import unittest
import tempfile
import math
from types import SimpleNamespace
from unittest.mock import patch
from orinfer_build.contracts.row_selection import aggregate_winner, ranges
from orinfer_build.contracts.row_selection import (
    choices,
    prefill_fallback,
    project,
    samples,
    split_fixed,
)
from orinfer_build.tuning.row_tuning import RowTuner


class AggregateSelectionTests(unittest.TestCase):
    def test_policy_uses_all_cohorts_and_breaks_ties_deterministically(self):
        records = [dict(times={"0": 1.0, "1": 1.5}), dict(times={"0": 10.0, "1": 2.0})]
        self.assertEqual(aggregate_winner(records), "1")
        self.assertEqual(aggregate_winner(list(reversed(records))), "1")
        self.assertEqual(aggregate_winner([dict(times={"1": 1.0, "0": 1.0})]), "0")

    def test_missing_invalid_and_inconsistent_measurements_are_rejected(self):
        for records in (
            [],
            [dict(times={})],
            [dict(times={"0": 0.0})],
            [dict(times={"0": -1.0})],
            [dict(times={"0": math.nan})],
            [dict(times={"0": math.inf})],
            [dict(times={"0": 1.0}), dict(times={"1": 1.0})],
        ):
            with self.subTest(records=records), self.assertRaises(ValueError):
                aggregate_winner(records)


class ValidationOnlyTuningTests(unittest.TestCase):
    def test_extreme_cases_are_replayed_without_biasing_serving_cost(self):
        checked = set()

        def inputs(rows, tags=("independent",)):
            for tag in tags:
                # The generator deliberately shares this binding. The tuner
                # must finish a case before advancing to the next one.
                current = tag

                def fixture(kernel, config):
                    changed = False

                    def run():
                        self.assertEqual(current, tag)

                    run.ms = (
                        (1.0 if kernel == "fast" else 2.0)
                        if tag == "independent"
                        else (1000.0 if kernel == "fast" else 0.01)
                    )

                    def check():
                        checked.add((tag, kernel, changed))

                    def change():
                        nonlocal changed
                        changed = not changed

                    return run, check, change

                yield fixture

        def benchmark(run, **kwargs):
            run()
            return dict(median_ms=run.ms), SimpleNamespace(replay=run)

        modules = {
            "torch": SimpleNamespace(manual_seed=lambda _: None),
            "orinfer_build.compiler.cuda.common": SimpleNamespace(
                benchmark=benchmark, SEED=20261002
            ),
        }
        with tempfile.TemporaryDirectory() as output, patch.dict("sys.modules", modules):
            result = RowTuner(output, {}).select(
                dict(timed="independent", validation=["four", "hot"]),
                [dict(name="fast"), dict(name="slow")],
                lambda c: c["name"],
                inputs,
                measure_rows=[1],
                validation_inputs=lambda rows: inputs(rows, ("four", "hot")),
            )
        self.assertEqual(result["default"], "0")
        self.assertEqual(result["measurements"][0]["times"], {"0": 1.0, "1": 2.0})
        self.assertEqual(
            checked,
            {
                (tag, name, changed)
                for tag in ("independent", "four", "hot")
                for name in ("fast", "slow")
                for changed in (False, True)
            },
        )


class PrefillTuningTests(unittest.TestCase):
    def test_single_prefill_fallback_preserves_implicit_and_strided_short_choices(self):
        original = dict(
            configs=[dict(tile=16), dict(tile=32), dict(tile=64)],
            default="0",
            selections=[
                dict(min_rows=1, max_rows=3, implementation="1"),
                dict(min_rows=8, max_rows=256, multiple_of=8, implementation="1"),
            ],
        )
        selected = prefill_fallback(original, "2", 16)
        self.assertTrue(all(c["max_rows"] <= 16 for c in selected["selections"]))
        published = list(choices(selected, 4096))
        self.assertEqual(published[0], ("", dict(tile=64)))
        for row in range(1, 4097):
            actual = selected["default"]
            for choice in selected["selections"]:
                if choice["min_rows"] <= row <= choice["max_rows"]:
                    actual = choice["implementation"]
            expected = "2" if row > 16 else "1" if row <= 3 or row % 8 == 0 else "0"
            self.assertEqual(actual, expected)

    def test_attention_staging_boundaries_follow_complete_pipeline_measurements(self):
        original = dict(
            configs=[
                dict(block_m=64, block_n=64, head_group=1, max_splits=8),
                dict(block_m=16, block_n=32, head_group=8, max_splits=16),
            ],
            default="0",
            selections=[dict(min_rows=1, max_rows=32, implementation="1")],
        )
        seen = {}

        def measure(context, spec, configs, *, measure_rows):
            seen.update(configs=configs, rows=measure_rows)
            # Staging wins on both sides of the old 128-row cutoff. It loses
            # again at large extents: either boundary must come from timings.
            records = [
                dict(
                    rows=n,
                    times={"0": 2.0, "1": 3.0, "2": 1.0 if 17 <= n <= 129 else 4.0, "3": 5.0},
                )
                for n in [1, 17, 73, 128, 129, 511, 4096]
            ]
            return dict(configs=configs, measurements=records, **ranges(records))

        with tempfile.TemporaryDirectory() as output:
            tuner = RowTuner(output, {}, dict(num_attention_heads=16, num_key_value_heads=2))
            with patch.object(tuner, "_measure_attention", side_effect=measure):
                result = tuner.staged_attention(262144, original)
                self.assertIs(result, tuner.staged_attention(262144, original))
        self.assertIn(1, seen["rows"])
        self.assertIn(127, seen["rows"])
        self.assertIn(129, seen["rows"])
        published = list(choices(result, 4096, bounded=True))
        for row, expected in [
            (1, False),
            (17, True),
            (73, True),
            (128, True),
            (129, True),
            (4096, False),
        ]:
            config = next(
                c
                for suffix, c in published
                if int(suffix.split("/")[-1].split("-")[0])
                <= row
                <= int(suffix.split("/")[-1].split("-")[1])
            )
            self.assertEqual(config.get("decoded_kv", False), expected)

    def test_one_winner_has_no_other_runtime_implementation(self):
        result = ranges([dict(rows=n, times=dict(a=2.0, b=1.0)) for n in [1, 17, 257, 512]], 512)
        self.assertEqual(result["used"], ["b"])
        self.assertEqual(result["default"], "b")
        self.assertEqual(result["selections"], [])

    def test_boundaries_follow_measurements_instead_of_fixed_tiers(self):
        result = ranges(
            [
                dict(rows=1, times=dict(a=1.0, b=3.0)),
                dict(rows=37, times=dict(a=3.0, b=1.0)),
                dict(rows=512, times=dict(a=4.0, b=1.0)),
            ],
            512,
        )
        self.assertEqual(result["used"], ["a", "b"])
        choices = result["selections"]
        self.assertTrue(choices)
        self.assertFalse(any(c["max_rows"] in [127, 1023] for c in choices))

    def test_aligned_candidate_cannot_escape_its_legal_rows(self):
        result = ranges(
            [
                dict(rows=1, times=dict(general=2.0)),
                dict(rows=128, times=dict(general=3.0, aligned=1.0)),
                dict(rows=512, times=dict(general=3.0, aligned=1.0)),
            ],
            512,
        )
        self.assertEqual(result["default"], "general")
        self.assertEqual(
            result["selections"],
            [dict(min_rows=128, max_rows=512, multiple_of=128, implementation="aligned")],
        )


class DynamicProjectionChoicesTests(unittest.TestCase):
    def test_pipeline_components_share_routing_when_only_codec_changes(self):
        table = dict(
            configs=[dict(block_m=32, expand=False), dict(block_m=32, expand=True)],
            default="1",
            selections=[dict(min_rows=1, max_rows=127, implementation="0")],
        )
        self.assertEqual(list(choices(project(table, ("block_m",)), 512)), [("", dict(block_m=32))])
        self.assertEqual(len(list(choices(table, 512))), 2)

    def test_bounded_choices_cover_only_non_aot_rows_with_measured_configs(self):
        fixed = [1, 2, 3, 4, 8, 16, 32]
        table = dict(
            configs=[{"tile": "large"}, {"tile": "small"}],
            default="0",
            selections=[dict(min_rows=1, max_rows=23, implementation="1")],
        )
        selected = {}
        for suffix, config in choices(table, 64, fixed, bounded=True):
            low, high = map(int, suffix.removeprefix("/rows/").split("-"))
            for row in range(low, high + 1):
                self.assertNotIn(row, selected)
                selected[row] = config
        self.assertEqual(set(selected), set(range(1, 65)) - set(fixed))
        self.assertEqual(selected[23], {"tile": "small"})
        self.assertEqual(selected[24], {"tile": "large"})

    def test_fixed_specializations_are_not_overlapped_by_dynamic_choices(self):
        fixed = [1, 2, 3, 4, 8, 16, 32]
        table = dict(
            configs=[{"tile": "large"}, {"tile": "small"}],
            default="0",
            selections=[dict(min_rows=1, max_rows=64, implementation="1")],
        )
        published = list(choices(table, 128, fixed))
        self.assertEqual(published[0], ("", {"tile": "large"}))
        selected = {}
        for suffix, config in published[1:]:
            low, high = map(int, suffix.removeprefix("/rows/").split("-"))
            for row in range(low, high + 1):
                self.assertNotIn(row, fixed)
                self.assertNotIn(row, selected)
                selected[row] = config
        for row in set(range(1, 129)) - set(fixed):
            self.assertEqual(
                selected.get(row, published[0][1]), {"tile": "small" if row <= 64 else "large"}
            )

    def test_unused_default_is_removed_when_winner_only_covers_aot_points(self):
        table = dict(
            configs=[{"tile": "unused"}, {"tile": "dynamic"}],
            default="0",
            selections=[dict(min_rows=2, max_rows=8, implementation="1")],
        )
        self.assertEqual(list(choices(table, 8, [1])), [("", {"tile": "dynamic"})])
        self.assertEqual(list(choices(table, 8, range(1, 9))), [])

    def test_samples_cover_small_capacities_and_large_tile_boundaries(self):
        self.assertEqual(samples(1), [1])
        for capacity in [8, 128, 4096]:
            rows = samples(capacity)
            self.assertEqual(rows, sorted(set(rows)))
            self.assertEqual(rows[0], 1)
            self.assertEqual(rows[-1], capacity)
            for n in [512, 2048, 4096]:
                for delta in [-1, 0, 1]:
                    if 1 <= n + delta <= capacity:
                        self.assertIn(n + delta, rows)
        self.assertEqual(split_fixed(1, 8, [1, 4, 4, 8]), [(2, 3), (5, 7)])
