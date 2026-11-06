from copy import deepcopy
import unittest

from tools.model.deepseek_v41.calibration import instrument


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
