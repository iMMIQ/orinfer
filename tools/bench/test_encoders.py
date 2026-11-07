"""Reject incomplete or incomparable encoder performance claims."""

import copy
import unittest

from tools.bench.encoders import compare, outputs, prompt_tokens, request


class EncoderBenchTests(unittest.TestCase):
    def test_suite_average_does_not_hide_a_slow_case(self):
        baseline = dict(
            task="rerank",
            requests_sha256="same",
            identity={},
            cases=[
                dict(
                    id="single",
                    items=1,
                    median_s=4,
                    outputs=[0.9],
                    response=dict(usage=dict(prompt_tokens=170)),
                ),
                dict(
                    id="batch",
                    items=2,
                    median_s=4,
                    outputs=[0.9, 0.1],
                    response=dict(usage=dict(prompt_tokens=340)),
                ),
            ],
        )
        candidate = copy.deepcopy(baseline)
        candidate["cases"][0]["median_s"] = 1
        candidate["cases"][1]["median_s"] = 3
        result = compare(baseline, candidate)
        self.assertEqual(result["suite_speedup"], 2)
        self.assertFalse(result["all_cases_reach_2x"])
        candidate["requests_sha256"] = "changed"
        with self.assertRaises(ValueError):
            compare(baseline, candidate)

    def test_template_or_truncation_mismatch_rejects_a_speedup(self):
        case = dict(
            id="a", items=1, median_s=1, outputs=[0.9], response=dict(usage=dict(prompt_tokens=170))
        )
        baseline = dict(task="rerank", requests_sha256="same", identity={}, cases=[case])
        candidate = copy.deepcopy(baseline)
        candidate["cases"][0]["response"]["usage"]["prompt_tokens"] = 99
        with self.assertRaisesRegex(ValueError, "Input token count differs"):
            compare(baseline, candidate)
        self.assertEqual(prompt_tokens(dict(meta=dict(billed_units=dict(input_tokens=89)))), 89)
        for response in [{}, dict(usage=dict(prompt_tokens=0))]:
            with self.assertRaises(ValueError):
                prompt_tokens(response)

    def test_missing_case_or_document_cannot_inflate_throughput(self):
        with self.assertRaises(ValueError):
            outputs("rerank", "vllm", {"results": [{"index": 0, "relevance_score": 0.9}]}, 2)
        baseline = dict(task="rerank", requests_sha256="same", identity={}, cases=[dict(id="a")])
        with self.assertRaises(ValueError):
            compare(baseline, dict(baseline, cases=[]))

    def test_embedding_nan_and_short_dimensions_are_rejected(self):
        for vector in [[float("nan")] + [0] * 2047, [1] + [0] * 1023]:
            with self.assertRaises(ValueError):
                outputs("embedding", "vllm", {"embeddings": {"float": [vector]}}, 1)

    def test_cohere_input_keeps_image_order_text_and_role(self):
        path, body = request(
            "embedding",
            "vllm",
            "model",
            dict(
                input_type="query",
                input=[
                    dict(
                        text="memory?",
                        images=["data:image/png;base64,a", "data:image/png;base64,b"],
                    )
                ],
            ),
        )
        self.assertEqual(path, "/v2/embed")
        self.assertEqual(body["input_type"], "query")
        self.assertEqual(body["truncate"], "NONE")
        content = body["inputs"][0]["content"]
        self.assertEqual(
            [x["image_url"]["url"] for x in content[:2]],
            ["data:image/png;base64,a", "data:image/png;base64,b"],
        )
        self.assertEqual(content[-1], dict(type="text", text="memory?"))


if __name__ == "__main__":
    unittest.main()
