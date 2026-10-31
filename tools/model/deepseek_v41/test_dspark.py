"""CPU regression for collecting features from real sequence partitions."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.model.deepseek_v41.dspark import converge_captures
from tools.model.row_contract import resolve


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
