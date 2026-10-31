import unittest
import json
import struct
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from tools.model.deepseek_v41.floating import bf16_parts
from tools.model.deepseek_v41.converge import unit_scale_manifests


class ProjectionScaleTests(unittest.TestCase):
    def test_every_rank_binding_must_have_immutable_unit_scales(self):
        with TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "kernel.json").write_text(
                json.dumps(dict(op="dense.projection", parameters=dict(integer=False)))
            )
            (base / "ones.bin").write_bytes(struct.pack("<ff", 1.0, 1.0))
            (base / "other.bin").write_bytes(struct.pack("<ff", 1.0, 0.0))
            model = dict(
                ranks=[
                    dict(
                        buffers=[
                            dict(name=n, bytes=8, data=dict(file="ones.bin"))
                            for n in ["token", "channel"]
                        ],
                        packets=[
                            dict(
                                body=[
                                    dict(
                                        op="kernel",
                                        manifest="kernel.json",
                                        buffers=["x", "w", "token", "channel", "out", "scratch"],
                                    )
                                ],
                                head=[],
                            )
                        ],
                    )
                    for _ in range(2)
                ]
            )
            self.assertEqual(unit_scale_manifests(model, base), {"kernel.json"})
            bad = model["ranks"][1]["buffers"][0]
            bad["data"]["file"] = "other.bin"
            self.assertFalse(unit_scale_manifests(model, base))
            bad["data"]["file"] = "ones.bin"
            bad["view"] = dict(parent="mutable", offset=0)
            self.assertFalse(unit_scale_manifests(model, base))

    def test_truncated_unit_scale_assets_do_not_enable_fixpipe(self):
        with TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "kernel.json").write_text(
                json.dumps(dict(op="dense.projection", parameters=dict(integer=False)))
            )
            (base / "ones.bin").write_bytes(struct.pack("<f", 1.0))
            model = dict(
                ranks=[
                    dict(
                        buffers=[dict(name="scale", bytes=8, data=dict(file="ones.bin"))],
                        packets=[
                            dict(
                                body=[
                                    dict(
                                        op="kernel",
                                        manifest="kernel.json",
                                        buffers=["x", "w", "scale", "scale", "out", "scratch"],
                                    )
                                ],
                                head=[],
                            )
                        ],
                    )
                ]
            )
            self.assertFalse(unit_scale_manifests(model, base))


class FloatingPartsTests(unittest.TestCase):
    def test_three_parts_reconstruct_finite_model_operands_without_mutation(self):
        rng = np.random.default_rng(20261002)
        source = (
            rng.normal(size=(17, 129)) * np.exp2(rng.integers(-40, 40, size=(17, 129)))
        ).astype("f4")
        before = source.copy()
        parts = bf16_parts(source)
        restored = (parts.astype("u4") << 16).view("f4").sum(0)
        np.testing.assert_array_equal(restored, source)
        np.testing.assert_array_equal(before, source)

    def test_high_part_uses_nearest_even_at_bf16_midpoints(self):
        source = np.array([1.00390625, 1.01171875, -1.00390625], dtype="f4")
        high = (bf16_parts(source)[0].astype("u4") << 16).view("f4")
        np.testing.assert_array_equal(high, [1.0, 1.015625, -1.0])

    def test_invalid_or_excessive_operands_are_rejected(self):
        for value in [np.nan, np.inf, -np.inf, np.finfo("f4").max, 2.0**-140]:
            with self.assertRaises(ValueError):
                bf16_parts(np.array([value], dtype="f4"))


if __name__ == "__main__":
    unittest.main()
