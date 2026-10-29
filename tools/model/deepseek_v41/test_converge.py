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
from tools.model.deepseek_v41.private_schedule import reusable_long_gathers, select_schedule


class PrivateScheduleSelectionTests(unittest.TestCase):
    def fixture(self):
        rows = list(range(1, 129)) + [129, 512, 800, 2048]
        reports = []
        for selected in (128, 640):
            for tile in (32, 128, 256):
                reports.append(
                    dict(
                        op="attention_cube.pv",
                        source_sha256="source",
                        exact_reference=True,
                        graph_replay=True,
                        changed_owners=True,
                        measurement_identity=dict(measurement_source_sha256="measure"),
                        parameters=dict(rows=2048, heads=8, selected=selected, tile_d=tile),
                        measurements=[
                            dict(
                                rows=r,
                                milliseconds=(
                                    1.3
                                    if selected == 640
                                    and ((tile == 128 and r == 1) or (tile == 256 and r == 17))
                                    else 32 / tile
                                ),
                            )
                            for r in rows
                        ],
                    )
                )
        return reports

    def select(self, reports, **kw):
        return select_schedule(
            reports,
            "attention_cube.pv",
            "tile_d",
            dict(dimension=512, tile_d=32),
            [dict(rows=2048, heads=8, selected=s) for s in (128, 640)],
            "source",
            "measure",
            **kw,
        )

    def test_one_guarded_configuration_and_complete_small_owner_boundary(self):
        result = self.select(self.fixture())
        self.assertEqual(result["parameters"], dict(tile_d=256))
        self.assertEqual(result["first_owner_rows"], 18)
        self.assertEqual(result["geometries"], 2)

    def test_missing_interior_rows_or_capacity_cannot_publish_a_general_schedule(self):
        for missing in (57, 2048):
            reports = self.fixture()
            for report in reports:
                report["measurements"] = [x for x in report["measurements"] if x["rows"] != missing]
            with self.assertRaises(ValueError):
                self.select(reports)

    def test_stale_source_or_measurement_and_missing_validation_are_rejected(self):
        for field in ("source_sha256", "exact_reference", "graph_replay", "changed_owners"):
            reports = self.fixture()
            reports[0][field] = False
            with self.assertRaisesRegex(ValueError, "current-source"):
                self.select(reports)
        reports = self.fixture()
        reports[0]["measurement_identity"]["measurement_source_sha256"] = "stale"
        with self.assertRaisesRegex(ValueError, "current-source"):
            self.select(reports)

    def test_duplicate_and_inconsistent_cases_or_geometries_are_rejected(self):
        reports = self.fixture()
        reports.append(deepcopy(reports[0]))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.select(reports)
        reports = self.fixture()
        reports[0]["measurements"].append(reports[0]["measurements"][0])
        with self.assertRaises(ValueError):
            self.select(reports)
        reports = self.fixture()
        reports[0]["measurements"].append(dict(rows=1024, milliseconds=1))
        with self.assertRaisesRegex(ValueError, "matching"):
            self.select(reports)
        with self.assertRaisesRegex(ValueError, "geometries"):
            self.select(self.fixture()[:3])

    def test_no_measured_safe_improvement_keeps_the_original(self):
        reports = self.fixture()
        for report in reports:
            for item in report["measurements"]:
                item["milliseconds"] = 1 if report["parameters"]["tile_d"] == 32 else 2
        result = self.select(reports)
        self.assertEqual(result["parameters"], dict(tile_d=32))
        self.assertEqual(result["first_owner_rows"], 1)

    def test_nonfinite_timings_and_regression_bounds_are_rejected(self):
        reports = self.fixture()
        reports[0]["measurements"][0]["milliseconds"] = float("nan")
        with self.assertRaises(ValueError):
            self.select(reports)
        with self.assertRaises(ValueError):
            self.select(self.fixture(), maximum_regression=float("nan"))


class LongGatherReuseTests(unittest.TestCase):
    def fixture(self):
        names = ["swa", "swa.scale", "long", "long.scale", "si", "li", "out", "mask"]
        contract = dict(
            columns=[dict(buffer=name, row_bytes=None) for name in names],
            pointers=dict(p=list(range(8))),
            work_ranges=dict(
                short=dict(min_rows=1, max_rows=511), large=dict(min_rows=512, max_rows=2048)
            ),
        )
        specs = dict(
            gather=dict(
                op="private_rows.kv_int8_gather_attention",
                parameters=dict(
                    compressed=True,
                    capacity=262144,
                    compressed_capacity=131072,
                    _owner_token_capacity=2048,
                    _owner_access=[dict(column=i, write=i >= 6) for i in range(8)],
                ),
                operands=[dict(access="read") for _ in range(3)],
            ),
            write=dict(op="write", operands=[dict(access="write")]),
        )
        rank = dict(
            buffers=[dict(name=name, bytes=1024) for name in names]
            + [
                dict(name="work", bytes=1024),
                dict(name="short", bytes=1024),
                dict(name="large", bytes=1024),
                dict(name="p", bytes=1024),
                dict(name="active", bytes=4),
                dict(name="out.view", bytes=1, view=dict(parent="out", offset=0)),
            ],
            packets=[],
        )
        return rank, contract, specs

    def gather(self, work="work", pointers="p"):
        return dict(
            op="kernel", manifest="gather", buffers=[work, pointers, "active"], owner_work={}
        )

    def check(self, steps, expected, change=None):
        rank, contract, specs = self.fixture()
        rank["packets"] = [dict(body=steps)]
        if change:
            change(rank, contract, specs)
        self.assertEqual(
            reusable_long_gathers(rank, contract, specs.__getitem__), {(0, i) for i in expected}
        )

    def test_each_packet_starts_full_and_swa_changes_allow_long_reuse(self):
        steps = [self.gather(), dict(op="kernel", manifest="write", buffers=["swa"]), self.gather()]
        self.check(steps, [2])
        rank, contract, specs = self.fixture()
        rank["packets"] = [dict(body=[self.gather(), self.gather()]) for _ in range(2)]
        self.assertEqual(reusable_long_gathers(rank, contract, specs.__getitem__), {(0, 1), (1, 1)})

    def test_every_source_output_and_alias_write_invalidates(self):
        for name in ("long", "long.scale", "li", "out", "mask", "out.view"):
            with self.subTest(buffer=name):
                self.check(
                    [
                        self.gather(),
                        dict(op="kernel", manifest="write", buffers=[name]),
                        self.gather(),
                        self.gather(),
                    ],
                    [3],
                )
        self.check([self.gather(), dict(op="future_unknown"), self.gather()], [])
        self.check([self.gather(), dict(op="limit_rows", rows=128), self.gather()], [])

    def test_disjoint_owner_variants_require_complete_coverage(self):
        self.check([self.gather("short"), self.gather("large"), self.gather()], [2])
        self.check([self.gather("short"), self.gather(), self.gather("large")], [2])
        self.check([self.gather("short"), self.gather("short"), self.gather("large")], [1])

    def test_pointer_or_row_layout_change_requires_full_gather(self):
        for column in (2, 3, 5, 6, 7):

            def change(rank, contract, specs):
                replacement = dict(contract["columns"][column], row_bytes=128)
                contract["columns"].append(replacement)
                contract["pointers"]["q"] = list(range(8))
                contract["pointers"]["q"][column] = 8

            self.check([self.gather(), self.gather(pointers="q")], [], change)
        other = dict(self.gather(), fixed_rows=17)
        self.check([self.gather(), other], [])

    def test_explicit_stream_boundaries_preserve_only_unmodified_values(self):
        self.check(
            [
                self.gather(),
                dict(op="fork"),
                dict(op="kernel", manifest="write", buffers=["swa"]),
                dict(op="switch"),
                dict(op="snapshot", buffer="out"),
                dict(op="join"),
                self.gather(),
            ],
            [6],
        )
        self.check(
            [self.gather(), dict(op="gather", source="swa", destination="out.view"), self.gather()],
            [],
        )

    def test_aliasing_output_is_rejected(self):
        def change(rank, contract, specs):
            contract["columns"][6]["buffer"] = "long"

        with self.assertRaisesRegex(ValueError, "aliases"):
            self.check([self.gather()], [], change)


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

    def test_indirect_publication_keeps_owner_pointees_and_table_roots(self):
        model = self.model()
        buffers = model["ranks"][0]["buffers"]
        buffers.extend(
            [
                dict(name="owner.bank", bytes=128),
                dict(name="owner.view", bytes=64, view=dict(parent="owner.bank", offset=0)),
                *[dict(name=n, bytes=64) for n in ("pointers", "work", "active", "endpoints")],
            ]
        )
        model["batch"] = dict(
            private=["owner.bank", "owner.view"],
            columns=[dict(buffer="unused.score.bank"), dict(buffer="owner.view")],
            pointers={"pointers": [0, 1]},
            work={"work": expression("m")},
            active="active",
            endpoints="endpoints",
        )
        result = compact_buffers(model)
        names = {b["name"] for b in result["ranks"][0]["buffers"]}
        self.assertTrue(
            {
                "unused.score.bank",
                "owner.bank",
                "owner.view",
                "pointers",
                "work",
                "active",
                "endpoints",
            }
            <= names
        )

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
