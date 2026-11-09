"""Regression checks for native dynamic rows, mixed views and thin publication."""

from copy import deepcopy
from contextlib import redirect_stdout
import io
import json
import math
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch

from orinfer_build.models.deepseek_v41.assembly import (
    compact_buffers,
    publish_package,
    retarget_packet_boundary,
    validate_owner_row_domains,
    validate_rows,
)
from orinfer_build.contracts.row_contract import expression, resolve
from orinfer_build.contracts.row_selection import aggregate_winner, bounded_choices, guarded_winner
from orinfer_build.models.deepseek_v41.sequence_parallel import external_reads
from orinfer_build.models.deepseek_v41.private_schedule import (
    reusable_long_gathers,
    select_schedule,
)


class PipelineTests(unittest.TestCase):
    def test_transforms_and_tuning_preserve_arguments_source_and_stage_order(self):
        from orinfer_build.models.deepseek_v41 import assembly

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "input.json", root / "output.json"
            model = dict(
                schema_version=1,
                layers=40,
                capacity=2048,
                ranks=[
                    dict(
                        buffers=[],
                        packets=[
                            dict(min_rows=0, body=[], head=[], coordinates={}, reset=[]),
                            dict(min_rows=2, rows=2048, body=[], head=[], coordinates={}, reset=[]),
                        ],
                    )
                    for _ in range(8)
                ],
            )
            original = json.dumps(model)
            source.write_text(original)
            arguments = [
                "assembly",
                "--plan",
                str(source),
                "--output",
                str(target),
                "--shared-attention",
                "--fused-attention",
                "--tune-projections",
                "local",
                "--model-dir",
                str(root),
                "--compile-workers",
                "3",
            ]
            order = []

            def transform(name):
                def run(before, after, **options):
                    self.assertEqual(sys.argv, arguments)
                    self.assertEqual(options["workers"], 3)
                    payload = json.loads(before.read_text())
                    payload.setdefault("stages", []).append(name)
                    order.append(name)
                    after.write_text(json.dumps(payload))

                return run

            def tune(options):
                self.assertEqual(sys.argv, arguments)
                self.assertTrue(options.local_only and options.joint)
                self.assertEqual(options.compile_workers, 3)
                after = options.plan.with_name(options.plan.stem + "-tuned.json")
                transform("tune")(options.plan, after, workers=options.compile_workers)
                return after

            with (
                patch.object(sys, "argv", arguments),
                patch(
                    "orinfer_build.models.deepseek_v41.shard_indexer.build",
                    side_effect=transform("index"),
                ),
                patch(
                    "orinfer_build.models.deepseek_v41.sequence_parallel.build",
                    side_effect=transform("sequence"),
                ),
                patch(
                    "orinfer_build.models.deepseek_v41.fuse_attention.build",
                    side_effect=transform("attention"),
                ),
                patch("orinfer_build.models.deepseek_v41.tune.tune", side_effect=tune),
                patch.object(assembly, "validate_rows", return_value={}),
                redirect_stdout(io.StringIO()),
            ):
                assembly.main()
            self.assertEqual(order, ["index", "sequence", "attention", "tune"])
            self.assertEqual(json.loads(target.read_text())["stages"], order)
            self.assertEqual(source.read_text(), original)


class HcFusionTests(unittest.TestCase):
    def test_normalization_and_sinkhorn_require_matching_physical_rows(self):
        from orinfer_build.models.deepseek_v41.fuse_hc import chain

        normalize = dict(op="kernel", manifest="normalize", buffers=["partials", "mix"])
        sinkhorn = dict(op="kernel", manifest="sinkhorn", buffers=["mix", "scale"])
        descriptions = dict(
            normalize=dict(
                op="hc_prefill.reduce",
                parameters=dict(rows=2048, _row_capacity=2048, width=5120, splits=8),
            ),
            sinkhorn=dict(
                op="vector.mhc_split_sinkhorn",
                parameters=dict(rows=2048, _row_capacity=2048, tile_rows=32),
            ),
        )

        def describe(step):
            return descriptions.get(step.get("manifest"))

        self.assertIsNotNone(chain(normalize, sinkhorn, describe))
        self.assertIsNone(chain(sinkhorn, normalize, describe))
        shard = dict(op="divide", lhs=dict(op="rows"), rhs=dict(op="constant", value=8))
        self.assertIsNotNone(
            chain(
                dict(normalize, row_expression=shard),
                dict(sinkhorn, row_expression=shard),
                describe,
            )
        )
        with self.assertRaises(ValueError):
            chain(dict(normalize, row_expression=shard), sinkhorn, describe)
        with self.assertRaises(ValueError):
            chain(dict(normalize, fixed_rows=True), sinkhorn, describe)
        for field, value in [("rows", 192), ("_row_capacity", 192), ("tile_rows", 1)]:
            original = descriptions["sinkhorn"]["parameters"][field]
            descriptions["sinkhorn"]["parameters"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                chain(normalize, sinkhorn, describe)
            descriptions["sinkhorn"]["parameters"][field] = original
        with self.assertRaises(ValueError):
            chain(normalize, dict(sinkhorn, buffers=["other", "scale"]), describe)
        with self.assertRaises(ValueError):
            chain(dict(normalize, decoder_only=True), sinkhorn, describe)


class OwnerRowDomainTests(unittest.TestCase):
    def test_fused_output_and_table_must_inherit_replay_domain_together(self):
        source = dict(buffer="query", row_bytes=4096, decoder=True)
        private = dict(buffer="scratch", row_bytes=None)
        output = dict(buffer="candidates", row_bytes=4096)
        contract = dict(
            columns=[source, private, output],
            pointers={"source": [0, 1], "fused": [0, 1, 2]},
            decoder_replay=dict(tables=["source"]),
        )
        for register, mark in [(False, False), (True, False), (False, True)]:
            candidate = deepcopy(contract)
            if register:
                candidate["decoder_replay"]["tables"].append("fused")
            if mark:
                candidate["columns"][2]["decoder"] = True
            with (
                self.subTest(register=register, mark=mark),
                self.assertRaisesRegex(ValueError, "mixes full and decoder"),
            ):
                validate_owner_row_domains(candidate)
        output["decoder"] = True
        contract["decoder_replay"]["tables"].append("fused")
        validate_owner_row_domains(contract)

    def test_full_rows_and_private_scratch_remain_valid(self):
        contract = dict(
            columns=[dict(buffer="query", row_bytes=4), dict(buffer="state", row_bytes=None)],
            pointers={"full": [0, 1]},
            decoder_replay=dict(tables=[]),
        )
        validate_owner_row_domains(contract)
        contract.pop("decoder_replay")
        contract["columns"][0]["decoder"] = True
        with self.assertRaisesRegex(ValueError, "require a replay contract"):
            validate_owner_row_domains(contract)


class PacketBoundaryTests(unittest.TestCase):
    def test_explicit_verification_capacity_preserves_dispatch_and_checks_outputs(self):
        model = dict(
            batch=dict(owners=32, output_capacity=544, columns=[], pointers={}),
            ranks=[dict(buffers=[], packets=[dict(rows=2048, body=[], head=[])])],
        )
        before = deepcopy(model["ranks"])
        validate_rows(model, Path("."))
        model["batch"]["verification_capacity"] = 192
        validate_rows(model, Path("."))
        self.assertEqual(model["ranks"], before)
        for capacity in (0, True, 545, 2049):
            model["batch"]["verification_capacity"] = capacity
            with (
                self.subTest(capacity=capacity),
                self.assertRaisesRegex(ValueError, "Verification capacity"),
            ):
                validate_rows(model, Path("."))

    def fixture(self):
        packets = [
            dict(rows=32, min_rows=1, batch_decode=True),
            dict(rows=2048, min_rows=1, max_rows=31, body=["short"]),
            dict(rows=2048, min_rows=32, max_rows=511, body=["head"]),
            dict(rows=2048, min_rows=512, body=["query"]),
            dict(rows=512, min_rows=512, role="draft", body=["draft"]),
        ]
        return dict(ranks=[dict(rank=i, packets=deepcopy(packets)) for i in range(8)])

    def test_selection_preserves_coverage_programs_and_other_roles(self):
        original = self.fixture()
        result = retarget_packet_boundary(original, 512, 128)
        for rank in result["ranks"]:
            packets = rank["packets"]
            for rows in range(1, 2049):
                matches = [
                    p
                    for p in packets
                    if not p.get("batch_decode")
                    and p.get("role", "target") == "target"
                    and p["min_rows"] <= rows <= p.get("max_rows", p["rows"])
                ]
                self.assertEqual(len(matches), 1)
                if rows in (127, 128):
                    self.assertEqual(matches[0]["body"], ["head" if rows == 127 else "query"])
            packets[2]["max_rows"], packets[3]["min_rows"] = 511, 512
        self.assertEqual(result, original)
        self.assertEqual(original, self.fixture())

    def test_rejects_ambiguous_domains_without_mutating_input(self):
        for case in ("missing", "capacity", "domain", "duplicate", "empty", "boolean"):
            model = self.fixture()
            packets = model["ranks"][-1]["packets"]
            minimum = 128
            if case == "missing":
                packets.pop(3)
            elif case == "capacity":
                packets[3]["rows"] = 1024
            elif case == "domain":
                packets[3]["max_rows"] = 1024
            elif case == "duplicate":
                packets.append(deepcopy(packets[3]))
            elif case == "empty":
                minimum = 32
            else:
                minimum = True
            before = deepcopy(model)
            with self.subTest(case=case), self.assertRaises(ValueError):
                retarget_packet_boundary(model, 512, minimum)
            self.assertEqual(model, before)


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
        from orinfer_build.models.deepseek_v41.embedding_columns import transform

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
        from orinfer_build.models.deepseek_v41.embedding_columns import transform

        model, describe = self.fixture()
        model["ranks"][0]["packets"][0]["body"].append(
            dict(op="kernel", manifest="extra", buffers=["embedding", "other"])
        )
        with self.assertRaisesRegex(ValueError, "still has a reader"):
            transform(model, describe, lambda *_: "fused")

    def test_exchange_layout_matches_the_original_contiguous_prefill_slices(self):
        from orinfer_build.models.deepseek_v41.embedding_columns import transform

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
        from orinfer_build.models.deepseek_v41.tune_attention import attention_groups

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
        from orinfer_build.models.deepseek_v41.tune_attention import attention_groups

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
        from orinfer_build.models.deepseek_v41.fuse_topk import merge_chain

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
        from orinfer_build.models.deepseek_v41.fuse_topk import causal_lengths

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
        from orinfer_build.models.deepseek_v41.local_indexer import query_chain

        steps, describe = self.fixture()
        self.assertEqual(query_chain(steps, 0, describe), steps)
        self.assertIsNone(query_chain(steps, 1, describe))
        with self.assertRaisesRegex(ValueError, "complete indexer"):
            query_chain(steps[:-1], 0, describe)

    def test_mismatched_partition_consumer_is_rejected(self):
        from orinfer_build.models.deepseek_v41.local_indexer import query_chain

        steps, describe = self.fixture()
        steps[-1]["buffers"][0] = "another_hidden"
        with self.assertRaisesRegex(ValueError, "intermediate buffers"):
            query_chain(steps, 0, describe)


class LocalIndexerLivenessTests(unittest.TestCase):
    def test_direct_and_owner_alias_readers_cannot_be_removed(self):
        from orinfer_build.models.deepseek_v41.local_indexer import check_query_liveness

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
        from orinfer_build.models.deepseek_v41.local_indexer import check_query_liveness

        steps = [dict(op="kernel", buffers=["query"], access=a) for a in ["write", "read"]]
        check_query_liveness(
            steps, ["query"], lambda s: dict(operands=[dict(access=s["access"])]), {}, {}
        )


class ProducerBoundaryTests(unittest.TestCase):
    def fixture(self):
        from orinfer_build.models.deepseek_v41.decoder_boundary import source_boundary

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

    def test_publication_retains_optional_rejection_sampling_assets(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            source.mkdir()
            (source / "data.bin").write_bytes(b"data")
            (source / "kernel.so").write_bytes(b"library")
            rejection = {}
            for name in ("statistics", "merge_statistics", "normalize", "select"):
                manifest = name + ".json"
                (source / manifest).write_text(json.dumps(dict(library=dict(file="kernel.so"))))
                rejection[name] = manifest
            model = dict(
                engram=dict(file="data.bin"), rope={}, ranks=[], sampling=dict(rejection=rejection)
            )
            target = source.parent / "package"
            self.assertEqual(publish_package(model, source, target), 6)
            for name in [*rejection.values(), "kernel.so"]:
                self.assertTrue((target / name).samefile(source / name))

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

    def test_floating_row_choices_keep_partial_layout_and_k_reduction(self):
        model = self.model()
        source = dict(
            op="floating.partials",
            source_sha256="factory",
            compiler={"version": "pinned"},
            parameters=dict(
                rows=256, _row_capacity=256, tile_m=64, tile_n=128, block_k=128, weight_parts=3
            ),
            operands=[
                dict(name=name, dtype="i32", shape=[256], access=access)
                for name, access in (("input", "read"), ("out", "write"))
            ],
        )
        replacement = deepcopy(source)
        replacement["parameters"]["tile_m"] = 128
        model["row_selections"] = {
            "kernel.json": [dict(min_rows=80, max_rows=256, kernels=["large.json"])]
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "kernel.json").write_text(json.dumps(source))
            (root / "large.json").write_text(json.dumps(replacement))
            validate_rows(model, root)
            for key, value in (("tile_n", 64), ("block_k", 256), ("weight_parts", 1)):
                invalid = deepcopy(replacement)
                invalid["parameters"][key] = value
                (root / "large.json").write_text(json.dumps(invalid))
                with (
                    self.subTest(key=key),
                    self.assertRaisesRegex(ValueError, "changes arithmetic"),
                ):
                    validate_rows(model, root)

    def test_direct_bf16_output_requires_unit_scales_for_defaults_and_local_choices(self):
        import struct

        shapes = [(256, 128), (32, 128), (256,), (32,), (256, 32), (256, 32)]
        names = ["x", "weight", "token", "channel", "out", "scratch"]
        dtypes = ["bf16", "bf16", "f32", "f32", "bf16", "f32"]
        definitions = [
            dict(name=name, bytes=math.prod(shape) * (2 if dtype == "bf16" else 4))
            for name, shape, dtype in zip(names, shapes, dtypes, strict=True)
        ]
        source = dict(
            op="dense.projection",
            parameters=dict(rows=256, output=32, reduction=128, integer=False, _row_capacity=256),
            source_sha256="factory",
            compiler={"version": "pinned"},
            operands=[
                dict(
                    name=name,
                    shape=list(shape),
                    dtype=dtype,
                    access="write" if name == "out" else "scratch" if name == "scratch" else "read",
                )
                for name, shape, dtype in zip(names, shapes, dtypes, strict=True)
            ],
        )
        model = dict(
            ranks=[
                dict(
                    buffers=definitions,
                    packets=[
                        dict(
                            rows=256,
                            body=[dict(op="kernel", manifest="kernel.json", buffers=names)],
                            head=[],
                        )
                    ],
                )
            ],
            row_selections={
                "kernel.json": [dict(min_rows=1, max_rows=128, kernels=["local.json"])]
            },
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for i in (2, 3):
                file = names[i] + ".bin"
                definitions[i]["data"] = dict(file=file)
                (root / file).write_bytes(struct.pack("<f", 1.0) * shapes[i][0])
            replacement = deepcopy(source)
            replacement["parameters"]["unit_scale"] = True
            (root / "kernel.json").write_text(json.dumps(source))
            (root / "local.json").write_text(json.dumps(replacement))
            validate_rows(model, root)
            (root / "channel.bin").write_bytes(struct.pack("<f", 0.0) * 32)
            with self.assertRaisesRegex(ValueError, "unit scales"):
                validate_rows(model, root)
            model["row_selections"] = {}
            (root / "kernel.json").write_text(json.dumps(replacement))
            with self.assertRaisesRegex(ValueError, "unit scales"):
                validate_rows(model, root)
            (root / "channel.bin").write_bytes(struct.pack("<f", 1.0) * 32)
            validate_rows(model, root)


class ExpertFusionContractTests(unittest.TestCase):
    def published_fixture(self):
        _, source, specs = self.fixture()
        source[1].update(name="m32.L0.up", workspace="workspace")
        cube = deepcopy(source[1])
        cube.update(op="w4_a4", rows=384, groups="split_groups")
        cube["buffers"][4:] = ["ones", "partials"]
        replacement = [
            dict(
                op="kernel",
                manifest="split",
                buffers=["x", "xs", "ids", "groups", "input", "scale", "errors", "split_groups"],
            ),
            cube,
            dict(
                op="kernel",
                manifest="finish",
                buffers=["partials", "assist", "scale", "groups", "out", "out_scale"],
            ),
        ]
        specs.update(
            split=dict(
                op="expert.gather_split",
                parameters=dict(rows=192, table_rows=32, width=5120, experts=48, _row_capacity=32),
            ),
            finish=dict(
                op="expert.finish",
                parameters=dict(rows=192, width=2304, experts=48, _row_capacity=32),
            ),
        )
        return source, replacement, specs

    def test_published_fusion_requires_identical_weights_counts_and_row_domain(self):
        from orinfer_build.models.deepseek_v41.expert_fusion import published_chain

        source, replacement, specs = self.published_fixture()

        def describe(step):
            return specs.get(step.get("manifest"))

        result = published_chain(source, replacement, describe)
        self.assertEqual(result, replacement)
        self.assertIsNot(result, replacement)
        for slot, port, value in [
            (0, 3, "other_groups"),
            (1, 1, "other_weight"),
            (2, 3, "split_groups"),
        ]:
            candidate = deepcopy(replacement)
            candidate[slot]["buffers"][port] = value
            with (
                self.subTest(slot=slot, port=port),
                self.assertRaisesRegex(ValueError, "routed boundary"),
            ):
                published_chain(source, candidate, describe)
        specs["split"]["parameters"]["_row_capacity"] = 256
        with self.assertRaisesRegex(ValueError, "physical domain"):
            published_chain(source, replacement, describe)

    def test_reuse_preserves_packet_roles_decoder_marker_and_head_readers(self):
        from orinfer_build.models.deepseek_v41.expert_fusion import reuse_published

        source, replacement, specs = self.published_fixture()
        marker = dict(op="decoder_rows", keep=128, prepare=0, select=0)
        packet = dict(rows=32, min_rows=16, body=[marker, *source], head=[])
        model = dict(
            batch={},
            ranks=[
                dict(
                    buffers=[],
                    packets=[
                        dict(rows=32, min_rows=1, max_rows=15, body=replacement, head=[]),
                        packet,
                        dict(packet, role="draft"),
                    ],
                )
            ],
        )
        original = deepcopy(model)

        def describe(step):
            return specs.get(step.get("manifest"))

        self.assertEqual(reuse_published(model, describe, minimum=16, maximum=32), 1)
        self.assertEqual(packet["body"], [marker, *replacement])
        self.assertEqual(model["ranks"][0]["packets"][2], original["ranks"][0]["packets"][2])
        original["ranks"][0]["packets"][1]["head"] = [dict(op="snapshot", buffer="packed")]
        with self.assertRaisesRegex(ValueError, "still has a reader"):
            reuse_published(original, describe, minimum=16, maximum=32)

    def fixture(self):
        from orinfer_build.models.deepseek_v41.expert_fusion import selected_chain

        program = [
            dict(op="kernel", manifest="g", buffers=["x", "xs", "ids", "input", "scale", "errors"]),
            dict(
                op="w4",
                rows=192,
                input=5120,
                output=4608,
                experts=48,
                weight_layout="transposed_nz",
                expected_expert_rows=16,
                groups="groups",
                buffers=["input", "weight", "weight_scale", "assist", "scale", "packed"],
            ),
            dict(op="kernel", manifest="a", buffers=["packed", "out", "out_scale", "active"]),
        ]
        specs = {
            "g": dict(
                op="memory.gather_quantized_rows",
                parameters=dict(rows=192, table_rows=32, width=5120, _row_capacity=32),
            ),
            "a": dict(
                op="model_ops.swiglu_packed_quant_active",
                parameters=dict(rows=192, width=2304, _row_capacity=32),
            ),
        }
        return selected_chain, program, specs

    def test_requires_same_quantization_boundary_and_dataflow(self):
        select, program, specs = self.fixture()
        chain, geometry = select(program, 0, lambda s: specs.get(s.get("manifest")))
        self.assertEqual(chain, program)
        self.assertEqual(geometry, (32, 192, 5120, 2304, 48))
        specs["a"]["parameters"]["limit"] = 8.0
        with self.assertRaisesRegex(ValueError, "geometry or dataflow"):
            select(program, 0, lambda s: specs.get(s.get("manifest")))
        specs["a"]["parameters"]["limit"] = 10.0
        program[2]["buffers"][0] = "other_projection"
        with self.assertRaisesRegex(ValueError, "geometry or dataflow"):
            select(program, 0, lambda s: specs.get(s.get("manifest")))

    def test_rejects_extra_snapshot_and_different_layout(self):
        select, program, specs = self.fixture()
        for field, value in [("weight_layout", "nd"), ("groups", None)]:
            candidate = deepcopy(program)
            candidate[1][field] = value
            with self.assertRaisesRegex(ValueError, "shared routed NZ"):
                select(candidate, 0, lambda s: specs.get(s.get("manifest")))
        program[0]["snapshot"] = "unrelated"
        with self.assertRaisesRegex(ValueError, "shared routed NZ"):
            select(program, 0, lambda s: specs.get(s.get("manifest")))

    def test_draft_route_count_uses_the_same_fusion_contract(self):
        select, program, specs = self.fixture()
        program[1].update(rows=96, experts=16)
        program[1].pop("expected_expert_rows")
        specs["g"]["parameters"]["rows"] = 96
        specs["a"]["parameters"].update(rows=96, selected=3)
        _, geometry = select(program, 0, lambda s: specs.get(s.get("manifest")))
        self.assertEqual(geometry, (32, 96, 5120, 2304, 16))
        specs["a"]["parameters"]["selected"] = 6
        with self.assertRaisesRegex(ValueError, "geometry or dataflow"):
            select(program, 0, lambda s: specs.get(s.get("manifest")))


class CannToolchainTests(unittest.TestCase):
    def test_publication_requires_one_cann92_compiler(self):
        from orinfer_build.compiler.ascend.toolchain import require_cann92

        source = dict(compiler=dict(cann_version="9.2.0", bisheng="pinned", tilelang="pinned"))
        require_cann92([source, deepcopy(source)])
        for compiler in [dict(cann_version="9.1.0"), {}, dict(cann_version="9.2.1")]:
            with self.assertRaisesRegex(ValueError, "every referenced"):
                require_cann92([source, dict(compiler=compiler)])

    def test_compiler_reads_the_installed_version(self):
        from orinfer_build.compiler.ascend.toolchain import cann_version

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "compiler").mkdir()
            path = root / "compiler/version.info"
            path.write_text("Version=9.2.0\nversion_dir=cann\n")
            self.assertEqual(cann_version(root), "9.2.0")
            path.write_text("Version=9.1.0\n")
            with self.assertRaisesRegex(ValueError, "CANN 9.2"):
                cann_version(root)


class ReplayTableCompactionTests(unittest.TestCase):
    def test_retargeted_owner_tables_drop_old_replay_bindings_and_keep_row_roles(self):
        names = [
            "orphan",
            "decoder_rows",
            "stage",
            "live_work",
            "live_pointer",
            "old_work",
            "old_pointer",
            "active",
            "ends",
        ]
        model = dict(
            batch=dict(
                private=["decoder_rows", "stage"],
                active="active",
                endpoints="ends",
                columns=[
                    dict(buffer="orphan"),
                    dict(buffer="decoder_rows", row_bytes=4, decoder=True, decode=False),
                    dict(buffer="stage", row_bytes=None),
                ],
                pointers=dict(live_pointer=[1, 2], old_pointer=[0]),
                work=dict(live_work=dict(op="rows"), old_work=dict(op="rows")),
                decoder_replay=dict(
                    tables=["old_work", "live_pointer", "old_pointer", "live_work"]
                ),
            ),
            ranks=[
                dict(
                    buffers=[dict(name=n, bytes=32) for n in names],
                    packets=[
                        dict(
                            coordinates={},
                            reset=[],
                            head=[],
                            body=[
                                dict(
                                    op="kernel",
                                    manifest="selected",
                                    owner_work=dict(op="rows"),
                                    buffers=["live_work", "live_pointer", "active"],
                                )
                            ],
                        )
                    ],
                )
            ],
        )
        compact_buffers(model, owner_tables=True)
        contract = model["batch"]
        self.assertEqual(contract["decoder_replay"]["tables"], ["live_pointer", "live_work"])
        self.assertEqual(contract["pointers"], dict(live_pointer=[0, 1]))
        self.assertTrue(contract["columns"][0]["decoder"])
        self.assertIsNone(contract["columns"][1]["row_bytes"])
        self.assertNotIn("old_pointer", {b["name"] for b in model["ranks"][0]["buffers"]})


if __name__ == "__main__":
    unittest.main()
