"""Construction cache must recover without publishing incomplete GPU artifacts."""

import json
from pathlib import Path
import tempfile
import unittest

from orinfer_build.compiler.build_cache import BuildCache
from orinfer_build.models.flash_next.construction import projection
from orinfer_build.tuning.projection_tuning import measurement_rows, profile_rows


class BuildTests(unittest.TestCase):
    def test_hc_fusion_requires_adjacent_decode_operations(self):
        from types import SimpleNamespace
        from orinfer_build.models.flash_next.hc_fusion import decode_pairs

        combine = dict(name="combine", args=[None] * 4)
        norm = dict(name="norm", args=[None] * 3)
        roles = {
            "combine": dict.fromkeys(("Block", "Inject", "Output", "Residual")),
            "norm": dict.fromkeys(("Normed", "Residual", "Weight")),
        }
        compiler = SimpleNamespace(pointers=lambda kernel: roles[kernel["name"]])
        package = dict(kernels=[combine, norm])
        operations = [dict(kind="kernel", name=name) for name in ("combine", "norm")]
        self.assertEqual(
            decode_pairs(dict(programs=dict(decode=operations)), package, compiler),
            [("combine", "norm")],
        )
        barrier = dict(kind="copy", source="Output", destination="Residual", bytes=4)
        self.assertEqual(
            decode_pairs(
                dict(programs=dict(decode=[operations[0], barrier, operations[1]])),
                package,
                compiler,
            ),
            [],
        )
        self.assertEqual(
            decode_pairs(dict(programs=dict(prefill=operations)), package, compiler), []
        )

    def test_dynamic_selection_measurements_keep_declared_non_aot_intervals(self):
        from orinfer_build.models.flash_next.construction import bounded_measurements

        variants = [
            dict(min_rows=first, max_rows=last, kernels=[name])
            for first, last, name in ((8, 8, "aot"), (17, 31, "short"), (33, 127, "medium"))
        ]
        samples = bounded_measurements(dict(row_execution=dict(variants=dict(slot=variants))))
        self.assertEqual(samples, dict(short=[17, 22, 31], medium=[33, 64, 127]))
        self.assertEqual(bounded_measurements({}), {})

    def test_projection_scope_keeps_all_row_contracts_of_selected_geometries(self):
        from orinfer_build.models.flash_next.construction import Projection, select_projections

        groups = {
            Projection(kind, rows, n, k, "float16", rows > 32): [rows]
            for kind, n, k in (("w8", 2560, 640), ("w8", 2560, 6144), ("q4", 2560, 640))
            for rows in (1, 17, 128, 4096)
        }
        selected = select_projections(groups, kinds=["w8"], shapes=[[2560, 640]])
        self.assertEqual({spec.rows for spec in selected}, {1, 17, 128, 4096})
        self.assertTrue(all(spec.kind == "w8" and spec.inputs == 640 for spec in selected))
        self.assertEqual(select_projections(groups), groups)
        self.assertEqual(len(groups), 12)
        with self.assertRaisesRegex(ValueError, "requested scope"):
            select_projections(groups, shapes=[[640, 2560]])

    def test_router_uses_checkpoint_geometry_for_flattened_sliced_workspaces(self):
        from orinfer_build.tuning.auxiliary_tuning import router_geometry

        package = dict(config_signature=dict(text=dict(num_experts=512, num_experts_per_tok=10)))
        operands = {role: dict(name=role) for role in ("Logits", "IDs", "Prob")}
        buffers = {
            role: dict(dtype=dtype, shape=[rows])
            for role, dtype, rows in (
                ("Logits", "f32", 4096 * 512),
                ("IDs", "i32", 4096 * 10),
                ("Prob", "f32", 4096 * 10),
            )
        }
        self.assertEqual(router_geometry(package, operands, buffers, 1024), (512, 10))
        operands["Logits"]["offset"] = 4095 * 512 * 4
        self.assertEqual(router_geometry(package, operands, buffers, 1), (512, 10))
        with self.assertRaisesRegex(ValueError, "physical view"):
            router_geometry(package, operands, buffers, 2)
        operands["Logits"]["offset"] = 1
        with self.assertRaisesRegex(ValueError, "physical view"):
            router_geometry(package, operands, buffers, 1)

    def test_prefill_portfolio_keeps_legal_wide_tiles_at_every_capacity(self):
        from orinfer_build.contracts.projection_policy import q4_configs, w8_configs

        for rows in (1, 8, 128, 4096):
            self.assertIn(
                dict(algorithm="tensor_core", block_m=128, block_n=128, block_k=256, stages=1),
                q4_configs(rows),
            )
            self.assertIn(dict(block_m=128, block_n=128, block_k=128), w8_configs(rows, 48))
            self.assertIn(
                dict(block_m=128, block_n=128, block_k=128, group_m=1, cache_scales=True),
                w8_configs(rows, 48),
            )

    def test_fixed_projection_factory_preserves_scale_cache_and_traversal(self):
        from unittest.mock import patch
        from orinfer_build.tuning.projection_tuning import factory

        spec = dict(kind="w8", rows=32, outputs=2560, inputs=640, dtype="float16", dynamic=False)
        build = factory(spec)
        with patch(
            "orinfer_build.tuning.projection_tuning.make_prim", side_effect=lambda spec, cfg: cfg
        ):
            config = build(block_m=64, group_m=1, cache_scales=True)
            self.assertTrue(config["cache_scales"])
            self.assertEqual(config["group_m"], 1)
            self.assertNotIn("cache_scales", build())

    def test_prefill_workload_does_not_change_dynamic_capacity(self):
        self.assertEqual(measurement_rows(4096, True, [512, 2048, 4096, 512]), [512, 2048, 4096])
        self.assertEqual(measurement_rows(1024, True, None), profile_rows(1024))
        self.assertEqual(measurement_rows(8, False, None), [8])
        for dynamic, values in [
            (False, [8]),
            (True, []),
            (True, [True]),
            (True, [0]),
            (True, [4097]),
        ]:
            with self.assertRaisesRegex(ValueError, "measurement"):
                measurement_rows(4096, dynamic, values)

    def test_resume_corruption_and_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = BuildCache(Path(directory))
            attempts = []

            def failed(folder):
                (folder / "kernel.cubin").write_bytes(b"incomplete")
                raise RuntimeError("interrupted")

            def build(folder):
                attempts.append(1)
                (folder / "kernel.cubin").write_bytes(b"complete")

            contract = dict(source="v1", shape=[1024, 2560], compiler="pinned")
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                cache.get(contract, failed)
            self.assertFalse(list((cache.root / "objects").iterdir()))
            first, hit = cache.get(contract, build)
            self.assertFalse(hit)
            self.assertEqual(cache.get(contract, failed), (first, True))
            (first / "kernel.cubin").write_bytes(b"corrupted")
            self.assertFalse(cache.get(contract, build)[1])
            self.assertEqual(len(attempts), 2)
            second, hit = cache.get(dict(contract, compiler="new"), build)
            self.assertFalse(hit)
            self.assertNotEqual(first, second)

    def test_external_cache_payload_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = BuildCache(Path(directory))
            obj, _ = cache.get({"version": 1}, lambda p: (p / "payload").write_bytes(b"ok"))
            entry = json.loads((obj / "entry.json").read_text())
            entry["files"] = {"../escape": next(iter(entry["files"].values()))}
            (obj / "entry.json").write_text(json.dumps(entry))
            self.assertFalse(cache.valid(obj, {"version": 1}))
            for value in ([], {"contract": {"version": 1}, "files": ["payload"]}):
                (obj / "entry.json").write_text(json.dumps(value))
                self.assertFalse(cache.valid(obj, {"version": 1}))

    def test_typed_rows_and_weight_views_are_checked(self):
        operands = {r: dict(name=r) for r in ("A", "W", "S", "AS", "Out")}
        buffers = {
            "W": dict(dtype="u16", shape=[640, 640], data=dict(sha256="packed")),
            "S": dict(dtype="f16", shape=[640], data=dict(sha256="scale")),
            "AS": dict(dtype="f16"),
            "Out": dict(dtype="f16"),
        }
        kernel = dict(name="slot", args=[dict(value=1024)])
        host = dict(ordered_arguments=[dict(ctype="ctypes.c_int32", value="m")])
        spec, _ = projection(kernel, host, operands, buffers, {"slot": dict(capacity=1024)}, {})
        self.assertEqual(
            (spec.inputs, spec.outputs, spec.rows, spec.dynamic), (2560, 640, 1024, True)
        )
        kernel["args"][0]["value"] = 10240
        with self.assertRaisesRegex(ValueError, "multiplier"):
            projection(kernel, host, operands, buffers, {"slot": dict(capacity=1024)}, {})
        kernel["args"][0]["value"] = 1024
        operands["W"]["offset"] = 128
        with self.assertRaisesRegex(ValueError, "geometry/view"):
            projection(kernel, host, operands, buffers, {"slot": dict(capacity=1024)}, {})


if __name__ == "__main__":
    unittest.main()
