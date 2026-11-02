import copy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from safetensors.numpy import save_file

from tools.model.deepseek_v41.tune import read_prepared_bank
from tools.model.deepseek_v41.tune_auxiliary import (
    canonical_partials,
    head_configs,
    head_row_policy,
)


class AuxiliaryHeadTests(unittest.TestCase):
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
