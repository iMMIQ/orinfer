"""Precision promotion preserves row partitions, BF16 boundaries and shared storage."""

from copy import deepcopy
import unittest

from orinfer_build.models.deepseek_v41.sum_precision import demote, promote
from orinfer_build.contracts.row_contract import expression, resolve


class SumPrecisionTests(unittest.TestCase):
    def fixture(self):
        buffers, packets = [], []
        for name, rows, scatter in (
            ("decode", 32, False),
            ("prefill", 2048, False),
            ("sequence", 2048, True),
            ("draft", 512, False),
        ):
            local = rows // 8 if scatter else rows
            buffers.extend(
                [
                    dict(name=name + ".source", bytes=rows * 64 * 2),
                    dict(name=name + ".output", bytes=local * 64 * 2),
                ]
            )
            step = dict(
                op="reduce",
                source=name + ".source",
                destination=name + ".output",
                elements=local * 64,
                fp32=False,
            )
            if scatter:
                step.update(
                    scatter=True,
                    elements_expression=expression("((rows + 7) // 8) * 64"),
                )
            packets.append(dict(rows=rows, role=name, body=[step], head=[]))
        return dict(
            batch=dict(private=["kv"], state=[dict(buffer="kv", row_bytes=64)]),
            ranks=[
                dict(rank=i, buffers=deepcopy(buffers), packets=deepcopy(packets)) for i in range(8)
            ],
        )

    def transform(self, model):
        calls = set()

        def kernel(rows, width, source, destination):
            calls.add((rows, width, source, destination))
            return f"cast:{rows}:{width}:{source}:{destination}"

        result, changes = promote(model, kernel)
        return result, changes, calls

    def test_all_roles_share_cast_geometry_and_workspace(self):
        original = self.fixture()
        frozen = deepcopy(original)
        result, changes, calls = self.transform(original)
        self.assertEqual(original, frozen)
        self.assertEqual(result["batch"], original["batch"])
        self.assertEqual(changes, 32)
        self.assertEqual(len(calls), 7)
        for rank, before in zip(result["ranks"], original["ranks"], strict=True):
            definitions = {b["name"]: b for b in rank["buffers"]}
            self.assertEqual(rank["buffers"][: len(before["buffers"])], before["buffers"])
            self.assertEqual(definitions["collective.fp32_sum.send"]["bytes"], 2048 * 64 * 4)
            self.assertEqual(definitions["collective.fp32_sum.receive"]["bytes"], 256 * 64 * 4)
            for packet, old in zip(rank["packets"], before["packets"], strict=True):
                up, reduction, down = packet["body"]
                self.assertEqual(up["buffers"][0], old["body"][0]["source"])
                self.assertEqual(down["buffers"][1], old["body"][0]["destination"])
                self.assertTrue(reduction["fp32"])
                self.assertEqual(reduction["elements"], old["body"][0]["elements"])
                if not reduction.get("scatter"):
                    self.assertEqual(reduction["source"], reduction["destination"])

    def test_scatter_keeps_rank_major_padding_for_unaligned_rows(self):
        result, _, _ = self.transform(self.fixture())
        up, reduction, down = result["ranks"][0]["packets"][2]["body"]
        for rows in (1, 2, 7, 8, 9, 31, 127, 128, 2047, 2048):
            local = (rows + 7) // 8
            self.assertEqual(resolve(up["row_expression"], rows), local * 8)
            self.assertEqual(resolve(down["row_expression"], rows), local)
            self.assertEqual(resolve(reduction["elements_expression"], rows), local * 64)
        self.assertEqual(down["manifest"], "cast:256:64:f32:bf16")

    def test_repeated_promotion_is_idempotent(self):
        first, _, _ = self.transform(self.fixture())
        second, changes, calls = self.transform(first)
        self.assertEqual(second, first)
        self.assertEqual(changes, 0)
        self.assertEqual(calls, set())

    def test_limit_rows_restarts_the_collective_domain(self):
        model = self.fixture()
        for rank in model["ranks"]:
            packet = rank["packets"][0]
            packet["rows"] = 2048
            packet["body"].insert(0, dict(op="limit_rows", rows=32))
        result, _, calls = self.transform(model)
        self.assertIn((32, 64, "bf16", "f32"), calls)
        self.assertEqual(
            result["ranks"][0]["packets"][0]["body"][0], dict(op="limit_rows", rows=32)
        )

    def test_invalid_extents_fail_before_mutating_the_input(self):
        for failure in ("bytes", "fixed", "expression"):
            model = self.fixture()
            rank = model["ranks"][-1]
            if failure == "bytes":
                rank["buffers"][0]["bytes"] -= 2
            elif failure == "fixed":
                rank["packets"][0]["body"][0]["fixed_rows"] = True
            else:
                rank["packets"][2]["body"][0]["elements_expression"] = expression("rows")
            frozen = deepcopy(model)
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                self.transform(model)
            self.assertEqual(model, frozen)

    def test_demote_round_trips_promoted_collectives_and_moe_combine(self):
        original = self.fixture()
        promoted, _, calls = self.transform(original)
        specs = {
            f"cast:{rows}:{width}:{src}:{dst}": dict(
                op="model_ops.convert",
                parameters=dict(
                    source_dtype={"bf16": "bfloat16", "f32": "float32"}[src],
                    destination_dtype={"bf16": "bfloat16", "f32": "float32"}[dst],
                ),
            )
            for rows, width, src, dst in calls
        }
        restored, report = demote(promoted, specs)
        self.assertEqual(report, dict(attn_demoted=32, moe_demoted=0))
        self.assertEqual(restored, original)
        again, repeat = demote(restored, specs)
        self.assertEqual(repeat, dict(attn_demoted=0, moe_demoted=0))
        self.assertEqual(again, restored)

        # Also verify MoE combine + zero_row_tail + fp32 reduce + convert demotion.
        specs["moe:combine:f32"] = dict(op="moe.combine", parameters=dict(output_dtype="float32"))
        specs["zrt:f32"] = dict(op="memory.zero_row_tail", parameters=dict(dtype="float32"))
        specs["cast:f32:bf16"] = dict(
            op="model_ops.convert",
            parameters=dict(source_dtype="float32", destination_dtype="bfloat16"),
        )
        moe_model = dict(
            ranks=[
                dict(
                    rank=0,
                    buffers=[
                        dict(name="l0.expert.local", bytes=2048 * 64 * 4),
                        dict(name="l0.expert.reduced", bytes=256 * 64 * 4),
                        dict(name="l0.out", bytes=256 * 64 * 2),
                    ],
                    packets=[
                        dict(
                            rows=2048,
                            role="prefill",
                            body=[
                                dict(
                                    op="kernel",
                                    manifest="moe:combine:f32",
                                    buffers=["a", "b", "c", "d", "l0.expert.local"],
                                ),
                                dict(
                                    op="kernel",
                                    manifest="zrt:f32",
                                    buffers=["r", "l0.expert.local"],
                                ),
                                dict(
                                    op="reduce",
                                    source="l0.expert.local",
                                    destination="l0.expert.reduced",
                                    elements=256 * 64,
                                    fp32=True,
                                    scatter=True,
                                ),
                                dict(
                                    op="kernel",
                                    manifest="cast:f32:bf16",
                                    buffers=["l0.expert.reduced", "l0.out"],
                                ),
                            ],
                            head=[],
                        )
                    ],
                )
            ]
        )
        demoted_moe, moe_report = demote(
            moe_model,
            specs,
            combine_replacements={"moe:combine:f32": "moe:combine:bf16"},
            zero_row_tail_manifest="zrt:bf16",
        )
        self.assertEqual(moe_report, dict(attn_demoted=0, moe_demoted=1))
        body = demoted_moe["ranks"][0]["packets"][0]["body"]
        self.assertEqual([s["op"] for s in body], ["kernel", "kernel", "reduce"])
        self.assertEqual(body[0]["manifest"], "moe:combine:bf16")
        self.assertEqual(body[1]["manifest"], "zrt:bf16")
        self.assertFalse(body[2]["fp32"])
        self.assertEqual(body[2]["destination"], "l0.out")
        bufs = {b["name"]: b["bytes"] for b in demoted_moe["ranks"][0]["buffers"]}
        self.assertEqual(bufs, {"l0.expert.local": 2048 * 64 * 2, "l0.out": 256 * 64 * 2})


if __name__ == "__main__":
    unittest.main()
