import unittest

from tools.bench.code_generation import summarize


class SummaryTests(unittest.TestCase):
    def test_empty_or_truncated_output_is_reported_not_hidden(self):
        row = {
            "case": "code",
            "warmup": False,
            "output_tps": 20,
            "ttft_s": 0.01,
            "first_content_s": None,
            "content": "",
            "finish_reason": "length",
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 512,
                "prompt_tokens_details": {"cached_tokens": 100},
            },
        }
        result = summarize([dict(row, warmup=True, output_tps=1), row])["code"]
        self.assertEqual(result["median_output_tps"], 20)
        self.assertEqual(result["prefix_token_hit_rate"], 1)
        self.assertEqual(result["empty_content_requests"], 1)
        self.assertEqual(result["truncated_requests"], 1)
        self.assertIsNone(result["median_first_content_s"])


if __name__ == "__main__":
    unittest.main()
