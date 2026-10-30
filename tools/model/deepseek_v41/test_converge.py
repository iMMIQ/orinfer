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


class DeviceEmbeddingTests(unittest.TestCase):
    def fixture(self, partitioned=False):
        capacity, local = (2048, 256) if partitioned else (32, 32)
        definitions = [dict(name="embedding", bytes=capacity * 5120 * 2)]
        incoming = "embedding"
        if partitioned:
            incoming = "selected"
            definitions.append(
                dict(
                    name=incoming,
                    bytes=local * 5120 * 2,
                    view=dict(
                        parent="embedding",
                        offset=256 * 3 * 10240,
                        row_capacity=capacity,
                        offset_expression=expression("((m + 7) // 8) * 30720"),
                    ),
                )
            )
        body = [dict(op="embed", destination="embedding")]
        if partitioned:
            body.append(dict(op="kernel", manifest="tail", buffers=["positions", "embedding"]))
        body.append(dict(op="kernel", manifest="expand", buffers=[incoming, "residual"]))
        if partitioned:
            body[-1]["row_expression"] = expression("(m + 7) // 8")
        specs = dict(
            tail=dict(
                op="memory.zero_row_tail",
                parameters=dict(width=5120, dtype="bfloat16", partitions=8),
            ),
            expand=dict(op="vector.mhc_expand", parameters=dict(rows=local, width=5120)),
            extra=dict(op="reader", operands=[dict(access="read"), dict(access="write")]),
        )
        ranks = []
        for rank in range(8):
            buffers = deepcopy(definitions)
            if partitioned:
                buffers[-1]["view"].update(
                    offset=256 * rank * 10240,
                    offset_expression=expression(f"((m + 7) // 8) * {rank * 10240}"),
                )
            ranks.append(
                dict(
                    rank=rank,
                    buffers=buffers,
                    packets=[
                        dict(
                            rows=capacity, coordinates=dict(slots="positions"), body=deepcopy(body)
                        )
                    ],
                )
            )
        model = dict(ranks=ranks)
        return model, lambda step: specs.get(step.get("manifest"))

    def test_embedding_columns_preserve_partition_rows_and_exact_input_spans(self):
        from tools.model.deepseek_v41.embedding_columns import transform

        for partitioned in (False, True):
            model, describe = self.fixture(partitioned)
            compiled = []

            def kernel(name, arguments, specs, capacity):
                compiled.append((name, arguments, specs, capacity))
                return name

            self.assertEqual(transform(model, describe, kernel), 8)
            for rank in model["ranks"]:
                body = rank["packets"][0]["body"]
                self.assertEqual(
                    [s["op"] for s in body],
                    ["embed", "kernel", "exchange", "kernel"]
                    if partitioned
                    else ["embed", "gather", "kernel"],
                )
                self.assertEqual(body[0]["columns"], 640)
                self.assertEqual(body[0]["first_column"], rank["rank"] * 640)
                self.assertEqual(
                    body[-1].get("row_expression"),
                    expression("(m + 7) // 8") if partitioned else None,
                )
                definitions = {b["name"]: b for b in rank["buffers"]}
                self.assertEqual(
                    definitions[body[0]["destination"]]["bytes"],
                    rank["packets"][0]["rows"] * 640 * 2,
                )
                self.assertNotIn("w.embed.weight.device", definitions)
                self.assertEqual(body[-1]["buffers"][-1], "residual")
                restore = compiled[
                    rank["rank"] * (2 if partitioned else 1) + (1 if partitioned else 0)
                ]
                self.assertEqual(restore[0], "gather_columns")
                self.assertEqual(restore[1]["paths"], 4)
                self.assertEqual(restore[2][-1]["shape"], [256 if partitioned else 32, 4, 5120])
                for rows in (1, 7, 8, 17, 31, 32):
                    self.assertEqual(
                        resolve(body[-2]["elements_expression"], rows),
                        ((rows + 7) // 8 if partitioned else rows) * 640,
                    )

    def test_removed_embedding_requires_no_other_consumer(self):
        from tools.model.deepseek_v41.embedding_columns import transform

        model, describe = self.fixture()
        model["ranks"][0]["packets"][0]["body"].append(
            dict(op="kernel", manifest="extra", buffers=["embedding", "other"])
        )
        with self.assertRaisesRegex(ValueError, "still has a reader"):
            transform(model, describe, lambda *_: "fused")

    def test_exchange_layout_matches_the_original_contiguous_prefill_slices(self):
        from tools.model.deepseek_v41.embedding_columns import transform

        model, describe = self.fixture(True)
        transform(model, describe, lambda *_: "fused")
        count = model["ranks"][0]["packets"][0]["body"][-2]["elements_expression"]
        for rows in (17, 513, 1921, 2048):
            local = resolve(count, rows) // 640
            # Small logical columns expose row/peer ordering independently of
            # the physical BF16 width. Each source sends contiguous row blocks.
            source = [
                [
                    [row * 16 + rank * 2 + column for column in range(2)] if row < rows else [0, 0]
                    for row in range(local * 8)
                ]
                for rank in range(8)
            ]
            for destination in range(8):
                first = destination * local
                received = [peer[first : first + local] for peer in source]
                restored = [sum((peer[row] for peer in received), []) for row in range(local)]
                expected = [
                    list(range(row * 16, row * 16 + 16)) if row < rows else [0] * 16
                    for row in range(first, first + local)
                ]
                self.assertEqual(restored, expected)


class AttentionTileScopeTests(unittest.TestCase):
    def test_only_large_family_attention_is_replaced_and_duplicates_are_shared(self):
        from tools.model.deepseek_v41.tune_attention import attention_groups

        def packet(minimum, name, **kw):
            return dict(min_rows=minimum, body=[dict(op="kernel", manifest=name)], **kw)

        model = dict(
            ranks=[
                dict(
                    packets=[
                        packet(1, "small.json"),
                        packet(512, "decode.json", batch_decode=True),
                        packet(512, "large.json"),
                        packet(1024, "large.json"),
                    ]
                )
            ]
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "large.json").write_text(
                json.dumps(dict(op="private_rows.attention_fused_selected_int8"))
            )
            self.assertEqual(list(attention_groups(model, root, 512)), ["large.json"])

    def test_missing_selected_attention_is_rejected(self):
        from tools.model.deepseek_v41.tune_attention import attention_groups

        with self.assertRaises(ValueError):
            attention_groups(dict(ranks=[dict(packets=[])]), Path("unused"), 512)


class TopKTreeTests(unittest.TestCase):
    def fixture(self):
        descriptions = {
            str(i): dict(op="topk.merge", parameters=dict(sources=s, _row_capacity=256))
            for i, s in enumerate([32, 8, 2])
        }
        steps = [
            dict(op="kernel", manifest=str(i), buffers=[f"b{i}", f"b{i + 1}"], fixed_rows=17)
            for i in range(3)
        ]
        producer = dict(op="kernel", manifest="producer", buffers=["work", "pointers", "active"])
        descriptions["producer"] = dict(
            op="private_rows.indexer_fused_causal_topk",
            parameters=dict(
                length=131072,
                _owner_token_capacity=256,
                _owner_aliases=dict(
                    query_owner="int8", indices_owner="int32", candidates_owner="float32"
                ),
            ),
        )
        batch = dict(
            pointers=dict(pointers=[0, 1, 2]),
            columns=[
                {},
                dict(buffer="lengths", row_bytes=4, row_partition=8),
                dict(buffer="b0", row_bytes=131072, row_partition=8),
            ],
        )
        return steps, producer, descriptions, batch, dict(lengths=dict(bytes=1024))

    def test_complete_tree_retains_row_binding(self):
        from tools.model.deepseek_v41.fuse_topk import merge_chain

        steps, _, descriptions, _, _ = self.fixture()

        def describe(step):
            return descriptions[step["manifest"]]

        self.assertEqual(merge_chain(steps, 0, describe), steps)
        self.assertIsNone(merge_chain(steps, 1, describe))
        with self.assertRaises(ValueError):
            merge_chain(steps[:-1], 0, describe)
        for key, value in [("fixed_rows", 18), ("buffers", ["other", "b2"])]:
            bad = deepcopy(steps)
            bad[1][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                merge_chain(bad, 0, describe)

    def test_causal_mask_uses_matching_partitioned_producer_lengths(self):
        from tools.model.deepseek_v41.fuse_topk import causal_lengths

        steps, producer, descriptions, batch, definitions = self.fixture()

        def describe(step):
            return descriptions[step["manifest"]]

        self.assertEqual(causal_lengths(producer, steps, describe, batch, definitions), "lengths")
        for key, value in [("row_bytes", 8), ("row_partition", 4), ("buffer", "other")]:
            bad = deepcopy(batch)
            bad["columns"][1 if key != "buffer" else 2][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                causal_lengths(producer, steps, describe, bad, definitions)
        descriptions["0"]["parameters"]["selected"] = 256
        with self.assertRaises(ValueError):
            causal_lengths(producer, steps, describe, batch, definitions)
        descriptions["0"]["parameters"]["selected"] = 512
        descriptions["producer"]["parameters"]["length"] = 65536
        with self.assertRaises(ValueError):
            causal_lengths(producer, steps, describe, batch, definitions)


class LocalIndexerChainTests(unittest.TestCase):
    def fixture(self):
        kinds = [
            "dense.projection",
            "model_ops.partial_rope",
            "vector.quantize",
            "dense.projection",
        ] + ["memory.partition_rows"] * 3
        bindings = [
            ["qa", "w.layers.2.attn.indexer.wq_b.weight", "scale", "ws", "query", "scratch"],
            ["query", "cos", "sin", "rotated"],
            ["rotated", "quantized", "query_scale"],
            [
                "hidden",
                "w.layers.2.attn.indexer.weights_proj.weight",
                "one",
                "one",
                "weights",
                "scratch",
            ],
            ["quantized", "local_query"],
            ["query_scale", "local_scale"],
            ["weights", "local_weights"],
        ]
        steps = [dict(op="kernel", manifest=str(i), buffers=b) for i, b in enumerate(bindings)]
        return steps, lambda step: dict(op=kinds[int(step["manifest"])])

    def test_only_a_complete_projection_to_partition_chain_is_selected(self):
        from tools.model.deepseek_v41.local_indexer import query_chain

        steps, describe = self.fixture()
        self.assertEqual(query_chain(steps, 0, describe), steps)
        self.assertIsNone(query_chain(steps, 1, describe))
        with self.assertRaisesRegex(ValueError, "complete indexer"):
            query_chain(steps[:-1], 0, describe)

    def test_mismatched_partition_consumer_is_rejected(self):
        from tools.model.deepseek_v41.local_indexer import query_chain

        steps, describe = self.fixture()
        steps[-1]["buffers"][0] = "another_hidden"
        with self.assertRaisesRegex(ValueError, "intermediate buffers"):
            query_chain(steps, 0, describe)


class LocalIndexerLivenessTests(unittest.TestCase):
    def test_direct_and_owner_alias_readers_cannot_be_removed(self):
        from tools.model.deepseek_v41.local_indexer import check_query_liveness

        definitions = {"view": dict(view=dict(parent="query", offset=0))}
        contract = dict(pointers=dict(p=[0]), columns=[dict(buffer="view")])
        for owned in [False, True]:
            step = dict(op="kernel", buffers=["work", "p"] if owned else ["view"])
            if owned:
                step["owner_work"] = {}
            spec = (
                dict(parameters=dict(_owner_access=[dict(column=0, write=False)]))
                if owned
                else dict(operands=[dict(access="read")])
            )
            with self.assertRaisesRegex(ValueError, "still has a reader"):
                check_query_liveness([step], ["query"], lambda _: spec, contract, definitions)

    def test_next_producer_can_overwrite_a_dead_query_buffer(self):
        from tools.model.deepseek_v41.local_indexer import check_query_liveness

        steps = [dict(op="kernel", buffers=["query"], access=a) for a in ["write", "read"]]
        check_query_liveness(
            steps, ["query"], lambda s: dict(operands=[dict(access=s["access"])]), {}, {}
        )


class ProducerBoundaryTests(unittest.TestCase):
    def fixture(self):
        from tools.model.deepseek_v41.decoder_boundary import source_boundary

        def kernel(name, buffers):
            return dict(op="kernel", manifest=name, buffers=buffers)

        specs = {
            "hc": dict(operands=[dict(access="write")]),
            "query": dict(operands=[dict(access="read"), dict(access="write")]),
            "rope": dict(operands=[dict(access="read"), dict(access="write")]),
            "kv": dict(
                operands=[dict(access="read")] * 3,
                parameters=dict(_owner_access=[dict(column=0, write=True)]),
            ),
            "consume": dict(operands=[dict(access="read")]),
        }
        write = dict(kernel("kv", ["work", "pointers", "active"]), owner_work={})
        packet = dict(
            coordinates=dict(slots="positions"),
            body=[
                dict(op="snapshot", name="layer19", buffer="hidden"),
                dict(op="fork"),
                kernel("hc", ["coefficients"]),
                dict(op="switch"),
                kernel("query", ["q.a", "x.q.b"]),
                kernel("rope", ["x.q.b", "x.q.rope"]),
                write,
                kernel("consume", ["x.q.rope"]),
                dict(op="join"),
                dict(op="snapshot", name="layer20", buffer="hidden"),
            ],
        )
        contract = dict(columns=[dict(buffer="cache.long.20")], pointers=dict(pointers=[0]))
        return source_boundary, packet, contract, lambda step: specs[step["manifest"]]

    def test_shared_writes_and_coefficients_complete_before_query_selection(self):
        split, packet, contract, describe = self.fixture()
        original = deepcopy(packet)
        prefix, tail = split(packet, contract, describe)
        self.assertEqual(packet, original)
        self.assertEqual(prefix[-3]["manifest"], "kv")
        self.assertEqual([s["op"] for s in prefix[-2:]], ["join", "snapshot"])
        self.assertEqual([s.get("manifest") for s in tail[:3]], ["query", "rope", "consume"])
        self.assertNotIn("join", [s["op"] for s in tail])

    def test_a_query_consumer_before_shared_write_rejects_reordering(self):
        split, packet, contract, describe = self.fixture()
        packet["body"].insert(6, dict(op="kernel", manifest="consume", buffers=["x.q.rope"]))
        with self.assertRaisesRegex(ValueError, "still required"):
            split(packet, contract, describe)

    def test_a_boundary_inside_nested_branches_is_rejected(self):
        split, packet, contract, describe = self.fixture()
        packet["body"].insert(6, dict(op="fork"))
        with self.assertRaisesRegex(ValueError, "outer mHC"):
            split(packet, contract, describe)


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
    def test_owner_token_capacity_follows_partition_and_work_scalar_stays_bounded(self):
        ports = [
            dict(dtype="i32", shape=[8192, 2]),
            dict(dtype="u64", shape=[8, 2]),
            dict(dtype="i32", shape=[1]),
        ]
        model = dict(
            batch=dict(partitioned_tables={"w": 8, "p": 8}),
            ranks=[
                dict(
                    buffers=[
                        dict(name=name, bytes=size)
                        for name, size in [("w", 8192 * 2 * 4), ("p", 8 * 2 * 8), ("a", 4)]
                    ],
                    packets=[
                        dict(
                            rows=2048,
                            body=[
                                dict(
                                    op="kernel",
                                    manifest="private.json",
                                    buffers=["w", "p", "a"],
                                    owner_work=expression("64"),
                                )
                            ],
                            head=[],
                        )
                    ],
                )
            ],
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "private.json"
            spec = dict(
                parameters=dict(_row_capacity=4096, _owner_token_capacity=256), operands=ports
            )
            path.write_text(json.dumps(spec))
            validate_rows(model, root)
            spec["parameters"]["_owner_token_capacity"] = 2048
            path.write_text(json.dumps(spec))
            with self.assertRaisesRegex(ValueError, "Owner token capacity"):
                validate_rows(model, root)
            spec["parameters"]["_owner_token_capacity"] = 256
            path.write_text(json.dumps(spec))
            for stride in (0, 4):
                model["batch"]["partitioned_tables"]["w"] = stride
                with self.assertRaisesRegex(ValueError, "Owner token capacity"):
                    validate_rows(model, root)

    def test_pruning_owner_tables_releases_dead_workspace_and_keeps_local_roles(self):
        packet = dict(
            coordinates={},
            reset=[],
            body=[
                dict(
                    op="kernel",
                    manifest="local.json",
                    owner_work=expression("64"),
                    buffers=["local.work", "local.pointers", "active"],
                )
            ],
            head=[],
        )
        model = dict(
            batch=dict(
                owners=8,
                private=["keys"],
                active="active",
                endpoints="endpoints",
                columns=[
                    dict(buffer="old.scores"),
                    dict(buffer="keys"),
                    dict(buffer="local.scores", row_partition=8),
                ],
                pointers={"old.pointers": [0, 1], "local.pointers": [1, 2]},
                work={"old.work": expression("64"), "local.work": expression("64")},
                work_ranges={"old.work": dict(min_rows=1, max_rows=2048)},
                partitioned_tables={"local.work": 8, "local.pointers": 8, "old.work": 8},
            ),
            ranks=[
                dict(
                    buffers=[
                        dict(name=name, bytes=8)
                        for name in (
                            "old.scores",
                            "keys",
                            "local.scores",
                            "old.pointers",
                            "local.pointers",
                            "old.work",
                            "local.work",
                            "active",
                            "endpoints",
                        )
                    ],
                    packets=[packet],
                )
            ],
        )
        compact_buffers(model, owner_tables=True)
        batch = model["batch"]
        self.assertEqual(batch["pointers"], {"local.pointers": [0, 1]})
        self.assertEqual([c["buffer"] for c in batch["columns"]], ["keys", "local.scores"])
        self.assertEqual(batch["partitioned_tables"], {"local.work": 8, "local.pointers": 8})
        self.assertEqual(batch["work_ranges"], {})
        self.assertEqual(
            {b["name"] for b in model["ranks"][0]["buffers"]},
            {"keys", "local.scores", "local.pointers", "local.work", "active", "endpoints"},
        )

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

    def test_partitioned_owner_columns_require_an_exact_view_of_shared_workspace(self):
        model = self.model()
        rank = model["ranks"][0]
        rank["buffers"] = [
            dict(name="workspace", bytes=2048 * 4),
            dict(name="local", bytes=256 * 4, view=dict(parent="workspace", offset=0)),
        ]
        rank["packets"][0].update(body=[], best_value="local", best_index="local")
        model["batch"] = dict(
            owners=32,
            columns=[
                dict(buffer="local", row_bytes=4, row_partition=8, decode=False, row_alignment=1)
            ],
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            validate_rows(model, root)
            model["batch"]["columns"][0]["buffer"] = "workspace"
            with self.assertRaisesRegex(ValueError, "physical row stride"):
                validate_rows(model, root)

    def test_exchange_counts_are_per_peer_for_both_read_and_write_spans(self):
        model = self.model()
        rank = model["ranks"][0]
        rank["buffers"] = [
            dict(name="source", bytes=256 * 8 * 4),
            dict(name="dest", bytes=256 * 8 * 4),
        ]
        packet = rank["packets"][0]
        packet.update(
            body=[
                dict(
                    op="exchange",
                    source="source",
                    destination="dest",
                    fp32=False,
                    int32=True,
                    elements=256,
                    elements_expression=expression("(m + 7) // 8"),
                )
            ],
            best_value="dest",
            best_index="dest",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            validate_rows(model, root)
            for name in ["source", "dest"]:
                invalid = deepcopy(model)
                next(b for b in invalid["ranks"][0]["buffers"] if b["name"] == name)["bytes"] -= 4
                with self.assertRaisesRegex(ValueError, "span exceeds"):
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
