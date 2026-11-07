import unittest

from tools.model.deepseek_v41.fuse_ffn_collectives import (
    activation_readers,
    routed_chain,
    strided_task,
)
from tools.model.row_contract import expression


class RoutedCollectiveTests(unittest.TestCase):
    def test_private_writable_state_is_also_treated_as_a_read(self):
        defs = {n: dict(name=n, bytes=1024) for n in ("global", "local", "scale", "work", "table")}
        step = dict(op="kernel", manifest="private", buffers=["work", "table"])

        def describe(_):
            return dict(
                op="private_rows.update",
                operands=[dict(access="read"), dict(access="read")],
                parameters=dict(_owner_access=[dict(column=0, write=True)]),
            )

        batch = dict(pointers={"table": [0]}, columns=[dict(buffer="global")])
        with self.assertRaisesRegex(ValueError, "Unsupported reader"):
            activation_readers([step], 0, "global", ["local", "scale"], defs, describe, batch)

    def test_strided_consumers_keep_rows_dtype_and_expert_multiplier(self):
        source = dict(
            op="dense.projection",
            parameters=dict(rows=16, output=576, reduction=5120, integer=True, _row_capacity=16),
            operands=[dict(name="activation", dtype="i8", shape=[16, 5120], access="read")],
        )
        task = strided_task(source, 5216)
        self.assertEqual(task["arguments"]["activation_stride"], 5216)
        self.assertEqual(task["specs"][0]["shape"], [16, 5216])
        self.assertEqual(task["dynamic"], dict(rows=1))
        self.assertEqual(source["operands"][0]["shape"], [16, 5120])
        source.update(
            op="memory.gather_quantized_rows",
            parameters=dict(rows=96, table_rows=16, width=5120, _row_capacity=16),
        )
        task = strided_task(source, 5216)
        self.assertEqual(task["dynamic"], dict(rows=6, table_rows=1))
        self.assertEqual(task["arguments"]["table_stride"], 5216)

    def test_activation_scope_stops_at_the_next_aliased_gather_write(self):
        defs = {n: dict(name=n, bytes=1024) for n in ("global", "local", "scale", "out")}
        defs["alias"] = dict(name="alias", bytes=1024, view=dict(parent="global", offset=0))
        descriptions = dict(
            dense=dict(
                op="dense.projection",
                parameters={},
                operands=[dict(access="read"), dict(access="write")],
            ),
            gather=dict(
                op="memory.gather_quantized_rows",
                parameters={},
                operands=[dict(access="read"), dict(access="write")],
            ),
            quant=dict(
                op="vector.quantize",
                parameters={},
                operands=[dict(access="write"), dict(access="write")],
            ),
            later=dict(op="unsupported", parameters={}, operands=[dict(access="read")]),
        )
        steps = [
            dict(op="kernel", manifest="dense", buffers=["global", "out"]),
            dict(op="kernel", manifest="gather", buffers=["global", "out"]),
            dict(op="kernel", manifest="quant", buffers=["local", "scale"]),
            dict(op="gather", source="local", destination="alias"),
            dict(op="kernel", manifest="later", buffers=["global"]),
        ]

        def describe(step):
            return descriptions[step["manifest"]]

        self.assertEqual(
            activation_readers(steps, 0, "global", ["local", "scale"], defs, describe, {}), [0, 1]
        )
        defs["alias"]["view"]["offset"] = 32
        with self.assertRaisesRegex(ValueError, "whole-buffer"):
            activation_readers(steps, 0, "global", ["local", "scale"], defs, describe, {})

    def fixture(self):
        def gather(source, destination, columns, **kind):
            return dict(
                op="gather",
                source=source,
                destination=destination,
                elements=2 * columns,
                elements_expression=expression(f"((m + 7) // 8) * {columns}"),
                **kind,
            )

        steps = [
            gather("li", "gi", 8, int32=True),
            gather("lw", "gw", 8, fp32=True),
            dict(
                op="kernel",
                manifest="q",
                buffers=["norm", "lx", "ls"],
                row_expression=expression("(m + 7) // 8"),
            ),
            gather("lx_view", "gx_view", 1280, int32=True),
            gather("ls", "gs", 1, fp32=True),
        ]
        definitions = {
            n: dict(name=n) for n in ("lx", "gx", "ls", "gs", "li", "gi", "lw", "gw", "norm")
        }
        definitions.update(
            lx_view=dict(view=dict(parent="lx", offset=0)),
            gx_view=dict(view=dict(parent="gx", offset=0)),
        )

        def describe(_):
            return dict(op="vector.quantize", parameters=dict(rows=2, width=5120))

        return steps, definitions, describe

    def test_preserves_original_consumers_and_uses_the_global_dynamic_axis(self):
        steps, definitions, describe = self.fixture()
        chain = routed_chain(steps, 0, describe, definitions, 16, 8)
        self.assertEqual(chain["source"], ["lx", "ls", "li", "lw"])
        self.assertEqual(chain["destination"], ["gx", "gs", "gi", "gw"])
        self.assertEqual(chain["routes"], 8)
        steps[4]["elements_expression"] = expression("m")
        with self.assertRaisesRegex(ValueError, "dynamic extent"):
            routed_chain(steps, 0, describe, definitions, 16, 8)

    def test_rejects_shifted_aliases_and_metadata_dependencies(self):
        steps, definitions, describe = self.fixture()
        definitions["lx_view"]["view"]["offset"] = 4
        with self.assertRaisesRegex(ValueError, "whole-buffer"):
            routed_chain(steps, 0, describe, definitions, 16, 8)
        definitions["lx_view"]["view"]["offset"] = 0
        steps[2]["buffers"][0] = "gi"
        with self.assertRaisesRegex(ValueError, "depends"):
            routed_chain(steps, 0, describe, definitions, 16, 8)
        steps[2]["buffers"][0] = "norm"
        definitions["li"] = dict(view=dict(parent="lx", offset=0))
        with self.assertRaisesRegex(ValueError, "aliases"):
            routed_chain(steps, 0, describe, definitions, 16, 8)
        self.assertIsNone(routed_chain(steps[:-1], 0, describe, definitions, 16, 8))


if __name__ == "__main__":
    unittest.main()
