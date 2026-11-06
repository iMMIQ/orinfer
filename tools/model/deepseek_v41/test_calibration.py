from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from tools.model.deepseek_v41.calibration import collect_routed, instrument


class ActivationTracePlanTests(unittest.TestCase):
    def fixture(self):
        definitions = [
            dict(name="x", bytes=2048 * 5120),
            dict(name="s", bytes=256 * 4),
            dict(name="y", bytes=2048 * 512),
            dict(name="w.layers.0.attn.wkv.weight.0-512.0-5120.i8", bytes=512 * 5120),
            dict(name="ws", bytes=512 * 4),
            dict(name="scratch", bytes=2048 * 512 * 4),
        ]
        projection = dict(
            op="kernel",
            manifest="projection.json",
            buffers=["x", definitions[3]["name"], "s", "ws", "y", "scratch"],
        )
        body = [
            dict(op="fork"),
            projection,
            dict(op="switch"),
            dict(op="gather", source="x", destination="y", elements=16, fp32=False),
            dict(op="join"),
            dict(op="snapshot", name="layer00", buffer="y"),
            dict(op="decoder_rows", keep=128, prepare=0, select=0),
        ]
        packet = dict(rows=2048, body=body, head=[])
        model = dict(ranks=[dict(rank=0, buffers=definitions, packets=[packet])])
        return model, lambda _name: dict(op="dense.projection")

    def test_preserves_numerical_instructions_and_replay_boundary_without_mutating_source(self):
        model, reader = self.fixture()
        original = deepcopy(model)
        result, metadata = instrument(model, reader)
        self.assertEqual(model, original)
        body = result["ranks"][0]["packets"][0]["body"]
        retained = [s for s in body if not s.get("diagnostic")]
        expected = [
            s
            for s in original["ranks"][0]["packets"][0]["body"]
            if s["op"] not in ("fork", "switch", "join")
        ]
        self.assertEqual(retained, expected)
        scale = next(p for p in metadata["points"] if p["name"].endswith("token_scale"))
        self.assertTrue(scale["fixed_rows"])
        output = next(p for p in metadata["points"] if p["name"].endswith("output"))
        self.assertFalse(output["fixed_rows"])
        self.assertEqual(output["projection"]["weight"], original["ranks"][0]["buffers"][3])

    def test_draft_programs_are_preserved_and_duplicate_instrumentation_is_rejected(self):
        model, reader = self.fixture()
        draft = deepcopy(model["ranks"][0]["packets"][0])
        draft["role"] = "draft"
        model["ranks"][0]["packets"].append(draft)
        result, _ = instrument(model, reader)
        self.assertEqual(result["ranks"][0]["packets"][1], draft)
        with self.assertRaises(ValueError):
            instrument(result, reader)

    def test_bounded_expert_views_keep_math_and_layer_markers(self):
        model, reader = self.fixture()
        rank = model["ranks"][0]
        rank["buffers"].extend(
            [
                dict(name="ex", bytes=2048 * 6 * 5120),
                dict(name="es", bytes=2048 * 6 * 4),
                dict(name="ey", bytes=2048 * 6 * 4608 * 2),
                dict(name="counts", bytes=48 * 8),
            ]
        )
        step = dict(
            op="w4",
            name="P3.L00.up",
            rows=2048 * 6,
            output=4608,
            groups="counts",
            buffers=["ex", "x", "ws", "ws", "es", "ey"],
        )
        rank["packets"][0]["body"].insert(-2, step)
        result, metadata = instrument(
            model, reader, layers=[0], labels=r"expert\.up\..*|routing\.counts", expert_rows=17
        )
        points = metadata["points"]
        self.assertEqual(len(points), 4)
        bounded = [p for p in points if "captured_rows" in p]
        self.assertEqual(len(bounded), 3)
        for point in bounded:
            self.assertEqual(point["captured_rows"], 17)
            self.assertEqual(point["buffer"], point["definition"]["name"])
            self.assertTrue(point["fixed_rows"])
        sizes = {p["name"].split(".")[-1]: p["definition"]["bytes"] for p in bounded}
        self.assertEqual(sizes, dict(input=17 * 5120, token_scale=17 * 4, output=17 * 4608 * 2))
        body = result["ranks"][0]["packets"][0]["body"]
        self.assertIn(step, body)
        marker = next(s for s in body if s.get("name") == "layer00")
        definition = next(b for b in result["ranks"][0]["buffers"] if b["name"] == marker["buffer"])
        self.assertEqual(definition["bytes"], 2)
        self.assertEqual(definition["view"]["parent"], "y")

    def test_draft_role_keeps_namespace_and_physical_markov_endpoints(self):
        model, reader = self.fixture()
        rank = model["ranks"][0]
        packet = rank["packets"][0]
        packet.update(role="draft", coordinates=dict(draft_block=5), confidence="s", best_index="x")
        weight = rank["buffers"][3]
        old = weight["name"]
        weight["name"] = "dspark.draft.w.mtp.0.attn.wkv.weight"
        packet["body"][1]["buffers"][1] = weight["name"]
        result, metadata = instrument(model, reader, roles=["draft"])
        points = metadata["points"]
        self.assertTrue(all(p["role"] == "draft" and p["draft_block"] == 5 for p in points))
        self.assertTrue(all("L40" in p["name"] for p in points))
        endpoints = [
            p for p in points if p["name"].endswith((".draft.confidence", ".draft.tokens"))
        ]
        self.assertEqual(len(endpoints), 2)
        self.assertTrue(all(p["fixed_rows"] for p in endpoints))
        self.assertEqual(
            [s["buffer"] for s in result["ranks"][0]["packets"][0]["body"][-2:]], ["s", "x"]
        )
        self.assertNotIn(old, str(result))
        with self.assertRaises(ValueError):
            instrument(model, reader, roles=["unknown"])

    def test_norm_roles_do_not_overwrite_the_packet_role(self):
        model, _ = self.fixture()
        packet = model["ranks"][0]["packets"][0]
        packet.update(role="draft", confidence="s", coordinates=dict(draft_block=5))
        packet["body"][1].update(buffers=["x", "dspark.w.mtp.0.ffn_norm.weight", "y", "s"])
        _, points = instrument(model, lambda _: dict(op="model_ops.norm"), roles=["draft"])
        self.assertTrue(all(p["role"] == "draft" for p in points["points"]))
        self.assertTrue(any(p["name"].endswith("draft.confidence") for p in points["points"]))

    def test_routed_collection_keeps_reverse_owners_and_conversation_folds(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            trace = root / "trace"
            frame = trace / "7-100-5_2-100-5"
            frame.mkdir(parents=True)
            arrays = {
                "expert.table.i8": np.arange(80, dtype="i1").reshape(10, 8),
                "expert.table.scale": np.ones(10, dtype="<f4"),
                "routing.indices": np.tile(np.array([1, 3, 0, 0], dtype="<i4"), (10, 1)),
                "routing.weights": np.ones((10, 4), dtype="<f4"),
            }
            points = []
            for label, values in arrays.items():
                name = "P0.L40." + label
                values.tofile(frame / ("p0-" + name + ".bin"))
                points.append(
                    dict(
                        rank=0,
                        role="draft",
                        packet=0,
                        name=name,
                        capacity=20,
                        definition=dict(bytes=160),
                        routing=dict(selected=2, experts=4),
                    )
                )
            metadata = root / "points.json"
            metadata.write_text(json.dumps(dict(points=points)))
            samples = root / "samples.json"
            values = [
                dict(owner=7, conversation="a", split="fit"),
                dict(owner=2, conversation="b", split="validation"),
            ]
            samples.write_text(json.dumps(dict(samples=values)))
            output = collect_routed(trace, metadata, samples, root / "inputs")
            with np.load(output / "layer40.npz") as archive:
                np.testing.assert_array_equal(archive["owner"], [7] * 5 + [2] * 5)
                np.testing.assert_array_equal(archive["fold"], [0] * 5 + [1] * 5)
                np.testing.assert_array_equal(archive["indices"], arrays["routing.indices"][:, :2])
                np.testing.assert_array_equal(archive["inputs"], arrays["expert.table.i8"])
            values[1]["conversation"] = "a"
            samples.write_text(json.dumps(dict(samples=values)))
            with self.assertRaises(ValueError):
                collect_routed(trace, metadata, samples, root / "invalid")
