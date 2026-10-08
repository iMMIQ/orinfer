import unittest
from tools.bench.scheduling import summarize, validate_trace


class SchedulingTests(unittest.TestCase):
    def test_trace_keeps_arrivals_independent_of_completion(self):
        rows = [dict(id=str(i), at_s=t, request={}) for i, t in enumerate([0.0, 0.0, 0.1])]
        self.assertEqual(validate_trace({"requests": rows}), rows)
        for time in [float("nan"), float("inf"), -1, True, "1"]:
            with self.assertRaises(ValueError):
                validate_trace({"requests": [dict(id="x", at_s=time, request={})]})
        with self.assertRaises(ValueError):
            validate_trace({"requests": [rows[0], rows[0]]})
        with self.assertRaises(ValueError):
            validate_trace(rows)

    def test_failures_are_reported_and_output_is_counted_from_usage(self):
        result = summarize(
            [
                dict(
                    usage={
                        "completion_tokens": 8,
                        "prompt_tokens": 16,
                        "prompt_tokens_details": {"cached_tokens": 8},
                    },
                    wall_s=2.0,
                    ttft_s=1.0,
                    chunk_gap_max_s=0.5,
                    arrival_lateness_s=0.01,
                ),
                dict(error="queue full", wall_s=0.1),
            ],
            4.0,
        )
        self.assertFalse(result["complete"])
        self.assertEqual(result["failed"], 1)
        self.assertEqual(result["committed_output_tps"], 2.0)
        self.assertEqual(result["chunk_gap_max"]["p99_s"], 0.5)
        self.assertEqual(result["prefix_cache"]["request_hit_rate"], 1.0)
        self.assertEqual(result["prefix_cache"]["token_hit_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
