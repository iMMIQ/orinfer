"""CPU contracts for shared DSpark rows, prepared banks and captured features."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from orinfer_build.models.deepseek_v41.dspark import (
    converge_draft_blocks,
    converge_captures,
    converge_markov_steps,
    reuse_prepared_draft,
    validate_block_size,
    write_constant,
)
from orinfer_build.contracts.row_contract import resolve
from orinfer_build.models.deepseek_v41.dspark_state import resize


class HistoryStorageTests(unittest.TestCase):
    def test_history_resize_rebuilds_writers_and_readers_and_preserves_noise(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scatter = dict(
                op="kv_int8.scatter",
                parameters=dict(rows=16, table_rows=128, logical_capacity=4096, retain_tail=True),
                operands=[
                    dict(name=n, shape=[128, 8, 64] if n == "cache" else [128, 8])
                    for n in ("source", "indices", "cache", "scales", "errors")
                ],
            )
            gather = dict(
                op="private_rows.kv_int8_gather_attention",
                parameters=dict(
                    rows=16,
                    capacity=128,
                    compressed_capacity=128,
                    _owner_aliases={"swa_owner": "int8", "swa_scales_owner": "float16"},
                    _owner_access=[dict(column=0, write=False), dict(column=1, write=False)],
                ),
                operands=[],
            )
            for name, spec in [("scatter", scatter), ("gather", gather)]:
                (root / (name + ".json")).write_text(json.dumps(spec))
            state, definitions, columns, pointers, warm, draft = [], [], [], {}, [], []
            for layer in (40, 41, 42):
                name = f"cache.dspark.swa.{layer}"
                for bank, stride in [(name, 512), (name + ".scale", 16)]:
                    state.append(
                        dict(
                            buffer=bank,
                            row_bytes=stride,
                            tokens_per_row=1,
                            window=128,
                            circular=True,
                        )
                    )
                    definitions.append(dict(name=bank, bytes=128 * stride))
                    columns.append(dict(buffer=bank))
                table = f"table{layer}"
                pointers[table] = [len(columns) - 2, len(columns) - 1]
                warm.append(
                    dict(
                        op="kernel",
                        manifest="scatter.json",
                        buffers=["x", "ids", name, name + ".scale", "error"],
                    )
                )
                draft.append(
                    dict(op="kernel", manifest="gather.json", buffers=["work", table, "active"])
                )
            definitions.append(dict(name="cache.dspark.noise.40", bytes=128 * 512))
            model = dict(
                batch=dict(
                    decoder_replay=dict(decoder_state=state), columns=columns, pointers=pointers
                ),
                ranks=[
                    dict(
                        buffers=definitions,
                        packets=[
                            dict(coordinates={}, body=warm, head=[]),
                            dict(
                                role="draft", coordinates=dict(draft_block=5), body=draft, head=[]
                            ),
                        ],
                    )
                ],
            )
            tasks = []

            def rebuild(spec, path):
                return dict(
                    placeholder=path,
                    arguments=deepcopy(spec["parameters"]),
                    specs=deepcopy(spec["operands"]),
                )

            def compile_task(item):
                task, _ = item
                tasks.append(task)
                return task["placeholder"], task["placeholder"] + ".new"

            compiler = SimpleNamespace(map=lambda fn, items: map(fn, items))
            fake = SimpleNamespace(compile_task=compile_task, kernel_rebuild_task=rebuild)
            original = deepcopy(model)
            with patch.dict(sys.modules, {"orinfer_build.models.deepseek_v41.build": fake}):
                result = resize(model, root, 256, compiler)
            self.assertEqual(result, dict(storage_rows=256, kernels=2, bindings=6))
            self.assertEqual(tasks[0]["arguments"]["table_rows"], 256)
            self.assertEqual(tasks[0]["specs"][2]["shape"], [256, 8, 64])
            self.assertEqual(tasks[1]["arguments"]["swa_storage_rows"], 256)
            self.assertEqual(tasks[1]["arguments"]["compressed_capacity"], 128)
            self.assertEqual(definitions[-1]["bytes"], 128 * 512)
            self.assertEqual(definitions[0]["bytes"], 256 * 512)
            self.assertTrue(all(s["window"] == 128 for s in state))
            self.assertEqual(
                model["ranks"][0]["packets"][1]["coordinates"]["draft_swa_storage_rows"], 256
            )
            bad = original
            bad["ranks"][0]["packets"][1]["coordinates"]["draft_swa_storage_rows"] = 256
            with patch.dict(sys.modules, {"orinfer_build.models.deepseek_v41.build": fake}):
                with self.assertRaisesRegex(ValueError, "coordinates differ"):
                    resize(bad, root, 512, compiler)


class CapacityTests(unittest.TestCase):
    def test_legacy_markov_steps_share_operands_and_keep_graph_step_addresses(self):
        descriptions, instructions = {}, []
        for step in range(16):
            for op in ("topk.add_bias_argmax", "topk.reduce_rank_best", "vector.confidence"):
                name = f"{op}.{step}"
                descriptions[name] = dict(
                    op=op, parameters=dict(rows=32, block_size=16, step=step), operands=[]
                )
                instructions.append(dict(op="kernel", manifest=name, buffers=["input"]))
        packet = dict(body=instructions, head=[])
        model = dict(ranks=[dict(buffers=[], packets=[packet])])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertEqual(converge_markov_steps(model, root, descriptions), 48)
            definitions = {b["name"]: b for b in model["ranks"][0]["buffers"]}
            for step in range(16):
                names = {s["buffers"][-1] for s in instructions[step * 3 : (step + 1) * 3]}
                self.assertEqual(len(names), 1)
                view = definitions[names.pop()]["view"]
                self.assertEqual(view["offset"], step * 4)
                parent = definitions[view["parent"]]
                data = (root / parent["data"]["file"]).read_bytes()
                self.assertEqual(int.from_bytes(data[step * 4 : step * 4 + 4], "little"), step)
            self.assertEqual(len({json.dumps(s, sort_keys=True) for s in descriptions.values()}), 3)
            frozen = deepcopy(model)
            self.assertEqual(converge_markov_steps(model, root, descriptions), 0)
            self.assertEqual(model, frozen)

    def test_active_noise_blocks_reuse_body_and_bounded_markov_prefixes(self):
        compiler = SimpleNamespace(map=lambda fn, items: map(fn, items))
        fake = SimpleNamespace(compile_task=lambda item: (item[0]["placeholder"], "padding.json"))
        operands = SimpleNamespace(operand=lambda *args: args)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, op in (
                ("first", "model_ops.block_first"),
                ("logits", "topk.add_bias_argmax"),
                ("confidence", "vector.confidence"),
                ("last", "topk.reduce_rank_best"),
            ):
                (root / f"{name}.json").write_text(json.dumps(dict(op=op)))
            body = [dict(op="embed", destination="body-output")]
            body.append(dict(op="kernel", manifest="first.json", buffers=["ids", "previous"]))
            for _ in range(16):
                for name in ("logits", "confidence"):
                    body.append(
                        dict(
                            op="kernel",
                            manifest=name + ".json",
                            buffers=[name],
                            row_expression=dict(op="rows"),
                        )
                    )
                body.append(dict(op="gather", elements=3 * 16))
                body.append(dict(op="kernel", manifest="last.json", buffers=["tokens"]))
            packet = dict(
                role="draft", rows=48, body=body, head=[], coordinates=dict(draft_block=16)
            )
            model = dict(
                batch=dict(owners=3, endpoints="endpoints"),
                ranks=[dict(buffers=[], packets=[packet])],
            )
            with patch.dict(
                sys.modules,
                {
                    "orinfer_build.models.deepseek_v41.build": fake,
                    "orinfer_build.compiler.ascend.build": operands,
                },
            ):
                converge_draft_blocks(model, root, 1, compiler)
            packets = model["ranks"][0]["packets"]
            self.assertEqual([p["coordinates"]["draft_block"] for p in packets], [5, 8, 16])
            for p, count in zip(packets, (5, 8, 16), strict=True):
                self.assertEqual((p["rows"], p["max_rows"], p["draft_stride"]), (48, 3 * count, 16))
                self.assertEqual(p["body"][0], body[0])
                self.assertEqual(sum(s.get("manifest") == "last.json" for s in p["body"]), count)
                gathers = [s for s in p["body"] if s["op"] == "gather"]
                self.assertTrue(
                    all(s["owners"] and resolve(s["elements_expression"], 2) == 32 for s in gathers)
                )
            # Running convergence again must preserve the already shared aliases.
            with patch.dict(
                sys.modules,
                {
                    "orinfer_build.models.deepseek_v41.build": fake,
                    "orinfer_build.compiler.ascend.build": operands,
                },
            ):
                converge_draft_blocks(model, root, 1, compiler)
            self.assertEqual(len(model["ranks"][0]["packets"]), 3)

    def test_block_capacity_reserves_the_c2_anchor_and_preserves_parent_assets(self):
        for size in (1, 5, 8, 16, 31):
            validate_block_size(size)
        for size in (0, 32):
            with self.assertRaises(ValueError):
                validate_block_size(size)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "constant.bin"
            write_constant(path, b"original")
            write_constant(path, b"original")
            with self.assertRaises(ValueError):
                write_constant(path, b"changed")
            self.assertEqual(path.read_bytes(), b"original")

    def test_physical_tensors_keep_rank_and_unique_semantic_identity(self):
        tensor = dict(name="physical", dtype="I8", shape=[16])
        old = dict(
            rank=0,
            packets=[],
            buffers=[dict(name="dspark.warm.w.x", bytes=16, prepared_tensor=tensor)],
        )
        model = dict(
            ranks=[
                dict(
                    rank=0, packets=[], buffers=[dict(name="w.x", bytes=16, weight=dict(name="x"))]
                )
            ]
        )
        result = deepcopy(model)
        reuse_prepared_draft(result, [old])
        self.assertEqual(result["ranks"][0]["buffers"][0]["prepared_tensor"], tensor)
        self.assertNotIn("weight", result["ranks"][0]["buffers"][0])
        wrong_rank = deepcopy(old)
        wrong_rank["rank"] = 1
        with self.assertRaises(ValueError):
            reuse_prepared_draft(deepcopy(model), [wrong_rank])
        ambiguous = deepcopy(old)
        ambiguous["buffers"].append(
            dict(name="dspark.draft.w.x", bytes=16, prepared_tensor=dict(tensor, name="another"))
        )
        with self.assertRaises(ValueError):
            reuse_prepared_draft(deepcopy(model), [ambiguous])


class CaptureTests(unittest.TestCase):
    def test_trace_only_hc_is_replaced_by_one_gather_of_local_means(self):
        tasks = []

        def rebuild(spec, key):
            return dict(
                placeholder=key,
                arguments=deepcopy(spec["parameters"]),
                specs=deepcopy(spec["operands"]),
            )

        def compile_task(item):
            task, _ = item
            tasks.append(task)
            return task["placeholder"], f"selected-{task['arguments']['section']}.json"

        compiler = SimpleNamespace(map=lambda fn, items: map(fn, items))
        fake = SimpleNamespace(kernel_rebuild_task=rebuild, compile_task=compile_task)
        with tempfile.TemporaryDirectory() as temporary:
            root, body = Path(temporary), []
            for section, layer in enumerate((36, 37, 38)):
                manifest = f"mean-{section}.json"
                (root / manifest).write_text(
                    json.dumps(
                        dict(
                            parameters=dict(rows=16, section=section),
                            operands=[dict(shape=[16, 4, 5120]), dict(shape=[16, 3, 5120])],
                        )
                    )
                )
                body.extend(
                    [
                        dict(
                            op="gather",
                            source="sequence.residual",
                            destination="residual",
                            trace_only=True,
                        ),
                        dict(op="snapshot", name=f"layer{layer}", buffer="residual"),
                        dict(op="kernel", manifest=manifest, buffers=["residual", "features"]),
                    ]
                )
            ordinary = dict(rows=16, body=[dict(op="snapshot", name="layer36", buffer="residual")])
            sharded = dict(rows=16, body=body)
            model = dict(
                ranks=[
                    dict(
                        buffers=[dict(name="features", bytes=16 * 3 * 5120 * 2)],
                        packets=[ordinary, sharded],
                    )
                ]
            )
            with patch.dict(sys.modules, {"orinfer_build.models.deepseek_v41.build": fake}):
                converge_captures(model, root, 1, compiler)
        self.assertEqual(len(tasks), 3)
        self.assertTrue(all(t["arguments"]["rows"] == 2 for t in tasks))
        captures = [s for s in sharded["body"] if s["op"] == "kernel"]
        self.assertTrue(
            all(s["buffers"] == ["sequence.residual", "features.sequence"] for s in captures)
        )
        self.assertTrue(all(resolve(s["row_expression"], 9) == 2 for s in captures))
        gathers = [s for s in sharded["body"] if s["op"] == "gather" and not s.get("trace_only")]
        self.assertEqual(len(gathers), 1)
        self.assertEqual(gathers[0]["destination"], "features")
        self.assertEqual(resolve(gathers[0]["elements_expression"], 9), 2 * 15360)
        self.assertEqual(len(ordinary["body"]), 1)


if __name__ == "__main__":
    unittest.main()
