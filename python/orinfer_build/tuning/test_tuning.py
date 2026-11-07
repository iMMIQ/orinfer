"""Shared tuning preserves rejection, replay validation and cache identity."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from orinfer_build.common.files import atomic_json
from orinfer_build.tuning.tuning import compile_candidates, measure, read_record, record_path


class TuningTests(unittest.TestCase):
    def test_native_publication_retains_the_measured_choices(self):
        from orinfer_build.models.encoder import validate_row_choices

        choice = "projection/rows/1-64"
        # The binding can exist in a package while the registered row plan omits it.
        plan = dict(kernels=[dict(name=choice)], row_execution={})
        with self.assertRaisesRegex(ValueError, "lost measured row choices"):
            validate_row_choices(plan, [choice])
        plan["row_execution"]["variants"] = {"projection": [dict(kernels=[choice])]}
        validate_row_choices(plan, [choice])
        with self.assertRaisesRegex(ValueError, "lost measured row choices"):
            validate_row_choices(plan, [])

    def test_encoder_subsets_remain_legal_for_short_and_large_packets(self):
        from orinfer_build.contracts.projection_policy import (
            encoder_configs,
            e8p_q4_configs,
            w8_configs,
        )

        for kind in ("e8p_q4", "w8"):
            selected = encoder_configs(kind)
            self.assertEqual(len(selected), len({tuple(sorted(c.items())) for c in selected}))
            self.assertTrue(any(c.get("cache_scales") for c in selected))
            for rows in (1, 32, 2049, 32768):
                legal = e8p_q4_configs(rows) if kind == "e8p_q4" else w8_configs(rows, 6144)
                self.assertTrue(selected and all(c in legal for c in selected))
        with self.assertRaises(ValueError):
            encoder_configs("unknown")

    def test_measured_scale_caching_reaches_the_compiled_projection(self):
        from orinfer_build.tuning.projection_tuning import factory

        for kind in ("e8p_q4", "w8"):
            with patch("orinfer_build.tuning.projection_tuning.make_prim") as build:
                factory(dict(kind=kind))(block_m=256, cache_scales=True)
            spec, config = build.call_args.args
            self.assertEqual(spec["kind"], kind)
            self.assertTrue(config["cache_scales"])

    def test_corrupt_stale_and_incomplete_records_are_recomputed(self):
        with tempfile.TemporaryDirectory() as folder:
            identity = dict(seed=17, geometry=[128, 512])
            path = record_path(Path(folder), identity)
            self.assertEqual(
                path, record_path(Path(folder), dict(reversed(list(identity.items()))))
            )
            self.assertIsNone(read_record(path, identity))
            path.write_text("{truncated")
            self.assertIsNone(read_record(path, identity))
            for value in ([], None, {"identity": {}}, {"identity": identity}):
                atomic_json(path, value)
                self.assertIsNone(read_record(path, identity, lambda r: r["checked_rows"]))
            record = dict(identity=identity, checked_rows=[1, 7, 128])
            atomic_json(path, record)
            self.assertEqual(read_record(path, identity), record)

    def test_failed_candidates_keep_their_config_and_do_not_shift_valid_indices(self):
        def build(config):
            if config == 2:
                raise ValueError("unsupported tile")
            return config * 10

        kernels, configs, rejected = compile_candidates([1, 2, 3], build, tolerate_failures=True)
        self.assertEqual((kernels, configs), ([10, 30], [1, 3]))
        self.assertEqual(rejected[0]["config"], 2)
        with self.assertRaisesRegex(ValueError, "unsupported tile"):
            compile_candidates([2], build)
        with self.assertRaisesRegex(ValueError, "All tile candidates"):
            compile_candidates([2], build, tolerate_failures=True)

    def test_changed_input_is_restored_after_replay_failure(self):
        state, calls = [0], []

        def benchmark(run, **kwargs):
            calls.append(kwargs)
            return {"median_ms": 0.01}, type("Graph", (), {"replay": lambda self: run()})()

        def change():
            state[0] ^= 1

        def check():
            self.assertEqual(state[0], 1)
            raise AssertionError("changed input")

        with self.assertRaisesRegex(AssertionError, "changed input"):
            measure(
                lambda: None,
                budget_ms=0.25,
                max_repetitions=8,
                change=change,
                check=check,
                benchmark=benchmark,
            )
        self.assertEqual(state[0], 0)
        self.assertEqual(calls, [dict(repetitions=1, warmup=1), dict(repetitions=8, warmup=0)])
