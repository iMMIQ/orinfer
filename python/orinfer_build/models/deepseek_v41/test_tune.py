import copy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from safetensors.numpy import save_file

from orinfer_build.models.deepseek_v41.tune import (
    Options,
    tune,
    preserve_small_rows,
    projection_configs,
    read_prepared_bank,
)
from orinfer_build.models.deepseek_v41.tune_auxiliary import (
    canonical_partials,
    head_configs,
    head_row_policy,
)


class AuxiliaryHeadTests(unittest.TestCase):
    def test_row_tile_search_preserves_the_published_arithmetic_schedule(self):
        parameters = dict(
            integer=True,
            output=4096,
            reduction=1280,
            tile_m=256,
            tile_n=128,
            block_k=256,
            pipeline_stages=3,
        )
        configs = projection_configs(parameters, row_tile_only=True)
        self.assertEqual(configs[0]["tile_m"], 256)
        self.assertEqual({c["tile_m"] for c in configs}, {64, 128, 256})
        for config in configs:
            self.assertEqual(
                {k: v for k, v in config.items() if k != "tile_m"},
                {k: parameters[k] for k in config if k != "tile_m"},
            )

    def test_library_search_options_are_checked_before_device_setup(self):
        for options in [
            dict(joint=True, pipeline_only=True),
            dict(compile_workers=0),
            dict(maximum_regression=float("nan")),
        ]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                tune(Options(Path("plan"), Path("model"), Path("output"), **options))

    def test_dynamic_head_keeps_small_rows_and_does_not_mix_compilers(self):
        records = [
            dict(rows=6, times={"0": 1.0, "1": 1.01, "2": 1.18}),
            dict(rows=80, times={"0": 2.0, "1": 2.01, "2": 1.2}),
            dict(rows=192, times={"0": 3.0, "1": 3.01, "2": 2.0}),
        ]
        base = dict(
            op="floating.partials",
            source_sha256="current",
            compiler={"version": "current"},
            parameters=dict(tile_m=64, tile_n=128, rows=544),
        )
        contracts = {
            "0": dict(base, source_sha256="old", compiler={"version": "old"}),
            "1": base,
            "2": dict(base, parameters=dict(base["parameters"], tile_m=128)),
        }
        policy = head_row_policy(records, contracts)
        self.assertEqual(policy, dict(default="1", variant="2", minimum=80))
        contracts["2"]["parameters"]["tile_n"] = 64
        policy = head_row_policy(records, contracts)
        self.assertEqual(policy, dict(default="0", variant=None, minimum=1))

    def test_vocab_tail_and_row_guard_are_excluded_from_logical_partials(self):
        rows, output = 2, 160
        reference = np.arange(rows * output * 8, dtype="f4").reshape(rows, output, 8)
        for tile in (32, 64, 128):
            blocks = (output + tile - 1) // tile
            padded = np.full((rows + 1, blocks * tile, 8), np.nan, dtype="f4")
            padded[:rows, :output] = reference
            physical = (
                padded.reshape(rows + 1, blocks, tile, 8).swapaxes(2, 3).reshape(rows + 1, -1)
            )
            np.testing.assert_array_equal(
                canonical_partials(physical, rows, output, tile), reference
            )

    def test_head_candidates_keep_published_reference_and_partial_layout(self):
        parameters = dict(tile_m=64, tile_n=128, weight_parts=3, output=16160)
        choices = head_configs(parameters)
        self.assertEqual(choices[0], dict(tile_m=64, tile_n=128, pipeline_stages=0))
        self.assertEqual(len(choices), 4)
        for config in choices:
            merged = dict(parameters, **config)
            self.assertEqual(
                (merged["tile_n"], merged["weight_parts"], merged["output"]), (128, 3, 16160)
            )
        choices = head_configs(dict(parameters, tile_m=128))
        self.assertEqual(len({str(sorted(c.items())) for c in choices}), len(choices))


class ReleasedProjectionTests(unittest.TestCase):
    def test_joint_search_covers_aligned_short_k_without_a_special_factory(self):
        source = dict(integer=True, output=5120, reduction=288, block_k=32)
        choices = projection_configs(source, joint=True)
        self.assertEqual(choices[0]["block_k"], 32)
        self.assertEqual({c["block_k"] for c in choices}, {32, 96, 288})
        self.assertGreater(len(choices), 1)
        for config in choices:
            k, stages = config["block_k"], config["pipeline_stages"]
            self.assertEqual(source["reduction"] % k, 0)
            self.assertTrue(stages == 0 or source["reduction"] // k >= stages)

    def test_rectangular_integer_tiles_fit_shared_hardware_bounds(self):
        source = dict(integer=True, output=576, reduction=5120, tile_m=256, tile_n=128, block_k=512)
        choices = projection_configs(source, joint=True)
        self.assertEqual({c["tile_n"] for c in choices} & {96, 192}, {96, 192})
        for config in choices:
            self.assertLessEqual(config["tile_m"] * config["tile_n"] * 4, 131072)
            self.assertLessEqual(config["tile_n"] * 128, 32768)
        source["integer"] = False
        self.assertFalse({c["tile_n"] for c in projection_configs(source, joint=True)} & {96, 192})

    def test_large_row_tuning_preserves_small_fallback_holes_and_specializations(self):
        prior = [dict(min_rows=4, max_rows=8, multiple_of=2, kernels=["small"])]
        variants = preserve_small_rows(prior, [129, 800, 2048], [], "old", "new", 2048)
        for rows in range(1, 129):
            selected = next(
                v["kernels"] for v in variants if v["min_rows"] <= rows <= v["max_rows"]
            )
            self.assertEqual(selected, ["small" if rows in (4, 6, 8) else "old"])
        self.assertEqual(variants[-1]["max_rows"], 128)
        measured = list(range(1, 129))
        new = [dict(min_rows=1, max_rows=8, kernels=["new-small"])]
        self.assertEqual(preserve_small_rows(prior, measured, new, "old", "new", 2048), new)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            preserve_small_rows(prior, [4, 8], new, "old", "new", 2048)

    def test_reads_selected_weight_and_flattens_column_scales(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            weight = np.arange(-6, 6, dtype="i1").reshape(3, 4)
            scale = np.array([[0.25], [0.5], [1.0]], dtype="f4")
            save_file({"weight": weight, "scale": scale}, root / "layer.safetensors")
            index = {"weight": "layer.safetensors", "scale": "layer.safetensors"}
            for name, value, dtype, kind, shape in [
                ("weight", weight, "i8", "I8", [3, 4]),
                ("scale", scale, "f32", "F32", [3]),
            ]:
                definition = dict(
                    bytes=value.nbytes,
                    prepared_tensor=dict(name=name, dtype=kind, shape=list(value.shape)),
                )
                self.assertEqual(
                    read_prepared_bank(root, index, definition, dtype, shape), value.tobytes()
                )

    def test_rejects_mismatched_release_and_projection_contracts(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            save_file({"weight": np.arange(12, dtype="i1").reshape(3, 4)}, root / "w.safetensors")
            index = {"weight": "w.safetensors"}
            definition = dict(
                bytes=12, prepared_tensor=dict(name="weight", dtype="I8", shape=[3, 4])
            )
            invalid = []
            for field, value in [("dtype", "F32"), ("shape", [4, 3]), ("name", "absent")]:
                item = copy.deepcopy(definition)
                item["prepared_tensor"][field] = value
                invalid.append((item, "i8", [3, 4]))
            invalid += [
                (dict(definition, bytes=13), "i8", [3, 4]),
                (definition, "i8", [4, 4]),
                (definition, "f32", [3]),
            ]
            for item, dtype, shape in invalid:
                with self.subTest(item=item, dtype=dtype, shape=shape):
                    with self.assertRaises((KeyError, ValueError)):
                        read_prepared_bank(root, index, item, dtype, shape)


if __name__ == "__main__":
    unittest.main()
