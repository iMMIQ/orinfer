"""Recover reference projections and preserve offline physical export contracts."""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from tools.model.deepseek_v41.export_release import TensorRelease
from tools.model.deepseek_v41.floating import bf16_parts
from tools.model.deepseek_v41.projection_precision import (
    floating_projection_spec,
    reference_projection,
)
from tools.model.deepseek_v41.test_floating_head import fixture
from tools.model.deepseek_v41.test_export_release import read_tensor


class ProjectionPrecisionTests(unittest.TestCase):
    def test_reference_uses_inverse_smoothing_and_shifted_output_basis(self):
        rng = np.random.default_rng(20261002)
        rotation = np.linalg.qr(rng.normal(size=(32, 32)))[0].astype("f4")
        weight = rng.normal(size=(64, 64)).astype("f4")
        gamma = rng.uniform(0.5, 1.5, 64).astype("f4")
        norm = rng.uniform(0.1, 2, 64).astype("f4")
        full = np.kron(np.eye(2, dtype="f4"), rotation)
        query = full @ np.roll(np.eye(64, dtype="f4"), 16, axis=1) @ full
        expected = query.T @ ((weight * gamma) @ full) / norm
        actual = reference_projection(weight, gamma, norm, rotation, query=True)
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(
            reference_projection(weight, gamma, norm, rotation),
            ((weight * gamma) @ full) / norm,
            rtol=2e-5,
            atol=2e-5,
        )
        with self.assertRaises(ValueError):
            reference_projection(weight, gamma, norm * 0, rotation)

    def test_dynamic_bf16_spec_keeps_operand_geometry_and_shared_factory(self):
        original = dict(
            op="dense.projection",
            parameters=dict(rows=256, output=512, reduction=5120, integer=True, _row_capacity=256),
            operands=[dict(dtype="i8") for _ in range(6)],
        )
        copy = deepcopy(original)
        converted = floating_projection_spec(original)
        self.assertEqual(original, copy)
        self.assertEqual(converted["op"], original["op"])
        self.assertFalse(converted["parameters"]["integer"])
        self.assertTrue(converted["parameters"]["unit_scale"])
        self.assertEqual(converted["operands"][1]["dtype"], "bf16")
        self.assertEqual(converted["operands"][5]["dtype"], "f32")
        self.assertEqual(converted["parameters"]["tile_m"], 64)

    def test_shared_tensor_overrides_change_dtype_size_and_keep_other_payloads(self):
        for overlay in [False, True]:
            with self.subTest(overlay=overlay), TemporaryDirectory() as temporary:
                root = Path(temporary)
                source, cache, _, _, weight = fixture(root)
                job = TensorRelease(
                    source,
                    root / "new",
                    source_execution=cache,
                    execution_output=root / "new-cache",
                    overlay=overlay,
                )
                bank = job.model["ranks"][0]["buffers"][0]
                values = bf16_parts(weight)[0]
                group = "override.safetensors" if overlay else "layer.safetensors"
                job.replace_tensor(bank, values, dtype="BF16", suffix=".bf16", group=group)
                job.add_tensor("constant", np.ones(3, dtype="f4"), dtype="F32", group=group)
                job.prepare_overrides()
                job.write_shard(group)
                self.assertEqual(bank["bytes"], values.nbytes)
                self.assertEqual(bank["prepared_tensor"]["dtype"], "BF16")
                self.assertEqual(read_tensor(job.output, job.index, "head.bf16"), values.tobytes())
                self.assertEqual(
                    read_tensor(job.output, job.index, "constant"), np.ones(3, dtype="f4").tobytes()
                )
                if not overlay:
                    self.assertNotIn("head", job.index["weight_map"])
                    self.assertEqual(
                        read_tensor(job.output, job.index, "untouched"),
                        np.array([2], dtype="f4").tobytes(),
                    )
