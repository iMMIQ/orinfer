"""Regression checks for native dynamic rows, mixed views and thin publication."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from tools.model.deepseek_v41.converge import compact_buffers, publish_package, validate_rows
from tools.model.row_contract import expression, resolve
from tools.model.row_selection import aggregate_winner, bounded_choices, guarded_winner
from tools.model.deepseek_v41.sequence_parallel import external_reads


class ConvergenceTests(unittest.TestCase):
    def test_shared_projection_checks_consumers_before_the_next_write(self):
        def kernel(access, name="local"):
            return dict(op="kernel", buffers=[name], ports=[dict(access=access)])

        def descriptor(step):
            return dict(operands=step["ports"])

        self.assertEqual(external_reads([kernel("read")], {"local"}, descriptor), {"local"})
        self.assertEqual(
            external_reads([kernel("write"), kernel("read")], {"local"}, descriptor), set()
        )
        self.assertEqual(external_reads([kernel("read_write")], {"local"}, descriptor), {"local"})

    def test_guard_rejects_a_fast_average_with_a_slow_individual_shape(self):
        cases = [
            dict(rows=17, times={"0": 10, "1": 1, "2": 9}),
            dict(rows=129, times={"0": 10, "1": 20, "2": 9}),
        ]
        self.assertEqual(aggregate_winner(cases), "1")
        self.assertEqual(guarded_winner(cases), "2")
        with self.assertRaises(ValueError):
            guarded_winner(cases, maximum_regression=float("nan"))

    def test_local_ranges_require_every_integer_and_do_not_bridge_a_slow_tail(self):
        records = [
            dict(rows=r, times={"0": 10, "1": 8 if r in (2, 3, 5) else 12}) for r in range(1, 7)
        ]
        self.assertEqual(bounded_choices(records[:-1], "0", 6), [])
        self.assertEqual(
            bounded_choices(records, "0", 6),
            [
                dict(min_rows=2, max_rows=3, multiple_of=1, implementation="1"),
                dict(min_rows=5, max_rows=5, multiple_of=1, implementation="1"),
            ],
        )

    def test_row_arithmetic_checks_interior_division_overflow_and_unknown_fields(self):
        value = expression("((m + 7) // 8) * 5120")
        for rows in range(1, 2049):
            self.assertEqual(resolve(value, rows), ((rows + 7) // 8) * 5120)
        for value, rows in [
            (expression("m // (m % 3)"), 3),
            (expression("m * 4294967295"), 2),
            (dict(op="rows", unexpected=1), 2),
        ]:
            with self.assertRaises(ValueError):
                resolve(value, rows)

    def model(self):
        rank = dict(
            rank=7,
            buffers=[
                dict(name="bank", bytes=2048 * 4),
                dict(
                    name="local",
                    bytes=256 * 4,
                    view=dict(
                        parent="bank",
                        offset=7 * 256 * 4,
                        row_capacity=2048,
                        offset_expression=expression("((m + 7) // 8) * 28"),
                    ),
                ),
                dict(name="result", bytes=256 * 4),
                dict(name="unused.score.bank", bytes=4096),
            ],
            packets=[
                dict(
                    rows=2048,
                    min_rows=512,
                    coordinates={},
                    reset=[],
                    best_value="result",
                    best_index="result",
                    logits=None,
                    body=[
                        dict(
                            op="kernel",
                            manifest="kernel.json",
                            buffers=["local", "result"],
                            row_expression=expression("(m + 7) // 8"),
                        ),
                        dict(
                            op="gather",
                            source="result",
                            destination="bank",
                            fp32=True,
                            elements=256,
                            elements_expression=expression("(m + 7) // 8"),
                        ),
                    ],
                    head=[],
                )
            ],
        )
        return dict(ranks=[rank])

    def test_full_domain_keeps_view_parents_and_rejects_an_interior_invalid_offset(self):
        model = compact_buffers(self.model())
        self.assertEqual(
            [b["name"] for b in model["ranks"][0]["buffers"]], ["bank", "local", "result"]
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "kernel.json").write_text(
                json.dumps(
                    dict(
                        parameters={"_row_capacity": 256},
                        operands=[dict(dtype="i32", shape=[256]), dict(dtype="i32", shape=[256])],
                    )
                )
            )
            result = validate_rows(model, root)
            self.assertGreaterEqual(result["checked_rows"], 2048)
            invalid = deepcopy(model)
            invalid["ranks"][0]["buffers"][1]["view"]["offset_expression"] = expression(
                "m // (m % 3)"
            )
            with self.assertRaises(ValueError):
                validate_rows(invalid, root)

    def test_publication_drops_choices_outside_the_programs_real_local_rows(self):
        model = self.model()
        model["row_selections"] = {
            "kernel.json": [
                dict(min_rows=1, max_rows=64, kernels=["small.json"]),
                dict(min_rows=65, max_rows=128, kernels=["medium.json"]),
            ]
        }
        result = compact_buffers(deepcopy(model))["row_selections"]["kernel.json"]
        self.assertEqual(result[0], dict(min_rows=64, max_rows=64, kernels=["small.json"]))
        model["ranks"][0]["packets"][0]["body"].insert(0, dict(op="limit_rows", rows=128))
        result = compact_buffers(model)["row_selections"]["kernel.json"]
        self.assertEqual(result, [dict(min_rows=16, max_rows=16, kernels=["small.json"])])

    def test_owner_work_and_selected_heads_use_their_own_row_axes(self):
        model = self.model()
        model["batch"] = dict(owners=128)
        packet = model["ranks"][0]["packets"][0]
        packet["body"] = [
            dict(
                op="kernel", manifest="private.json", buffers=["result"], owner_work=expression("m")
            )
        ]
        packet["head"] = [dict(op="kernel", manifest="head.json", buffers=["result"], owners=True)]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, capacity in [("private.json", 4096), ("head.json", 128)]:
                (root / name).write_text(
                    json.dumps(
                        dict(
                            parameters={"_row_capacity": capacity},
                            operands=[dict(dtype="i32", shape=[256])],
                        )
                    )
                )
            validate_rows(model, root)
            model["batch"]["owners"] = 127
            with self.assertRaises(ValueError):
                validate_rows(model, root)

    def test_publication_links_only_live_assets_and_rejects_escaping_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "live.bin").write_bytes(b"live")
            (source / "unused.bin").write_bytes(b"unused")
            model = dict(engram=dict(file="live.bin"), rope={}, ranks=[])
            target = root / "package"
            self.assertEqual(publish_package(model, source, target), 1)
            self.assertTrue((target / "live.bin").samefile(source / "live.bin"))
            self.assertFalse((target / "unused.bin").exists())
            with self.assertRaises(ValueError):
                publish_package(
                    dict(model, engram=dict(file="../source/live.bin")), source, root / "bad"
                )

    def test_local_replacements_preserve_math_and_fit_shared_scratch(self):
        model = self.model()
        rank = model["ranks"][0]
        rank["buffers"].append(dict(name="scratch", bytes=256 * 4))
        rank["packets"][0]["body"][0]["buffers"].append("scratch")
        source = dict(
            op="dense.projection",
            parameters=dict(_row_capacity=256, rows=256, reduction=5120, tile_m=256),
            source_sha256="factory",
            compiler={"version": "pinned"},
            operands=[
                dict(name=name, dtype="i32", shape=[256], access=access)
                for name, access in (("input", "read"), ("out", "write"), ("work", "scratch"))
            ],
        )
        source["operands"][2]["shape"] = [32]
        replacement = deepcopy(source)
        replacement["parameters"]["tile_m"] = 128
        replacement["operands"][2]["shape"] = [128]
        model["row_selections"] = {
            "kernel.json": [dict(min_rows=1, max_rows=128, kernels=["local.json"])]
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "kernel.json").write_text(json.dumps(source))
            (root / "local.json").write_text(json.dumps(replacement))
            validate_rows(model, root)
            for key in ("scratch", "output", "math", "factory"):
                invalid = deepcopy(replacement)
                if key == "scratch":
                    invalid["operands"][2]["shape"] = [512]
                elif key == "output":
                    invalid["operands"][1]["shape"] = [128]
                elif key == "math":
                    invalid["parameters"]["reduction"] = 4096
                else:
                    invalid["source_sha256"] = "different"
                (root / "local.json").write_text(json.dumps(invalid))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    validate_rows(model, root)


if __name__ == "__main__":
    unittest.main()
