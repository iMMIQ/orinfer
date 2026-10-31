import copy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from safetensors.numpy import save_file

from tools.model.deepseek_v41.tune import read_prepared_bank


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
