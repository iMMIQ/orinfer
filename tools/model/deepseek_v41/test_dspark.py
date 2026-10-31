"""CPU contracts for shared DSpark rows, prepared banks and captured features."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.model.deepseek_v41.dspark import (
    converge_draft_blocks,
    converge_captures,
    reuse_prepared_draft,
    validate_block_size,
    write_constant,
)
from tools.model.row_contract import resolve


class CapacityTests(unittest.TestCase):
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
                {"tools.model.deepseek_v41.build": fake, "tools.operators.ascend.build": operands},
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
                {"tools.model.deepseek_v41.build": fake, "tools.operators.ascend.build": operands},
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
            with patch.dict(sys.modules, {"tools.model.deepseek_v41.build": fake}):
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
