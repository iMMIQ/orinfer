import io
import unittest
import urllib.error
from unittest.mock import patch
from tools.bench.concurrency import (
    RequestError,
    counter_delta,
    decode_window,
    events,
    request,
    percentile,
)


class StreamingTests(unittest.TestCase):
    def test_queue_deadlines_remain_retryable_only_before_any_output(self):
        error = b'data: {"error":{"code":"request_queue_timeout","retryable":true,"retry_after_ms":1500}}\n\n'
        prefix = b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
        for started in [False, True]:
            stream = io.BytesIO((prefix if started else b"") + error)
            with patch("urllib.request.urlopen", return_value=stream):
                with self.assertRaises(RequestError) as caught:
                    request("http://localhost/v1", {})
            self.assertEqual(caught.exception.code, "request_queue_timeout")
            self.assertEqual(caught.exception.retryable, not started)
            self.assertEqual(caught.exception.retry_after_s, 1.5)

    def test_http_overload_keeps_status_and_backoff_hint(self):
        error = urllib.error.HTTPError(
            "http://localhost/v1",
            429,
            "full",
            {"Retry-After": "2"},
            io.BytesIO(b'{"error":{"code":"queue_full"}}'),
        )
        with patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaises(RequestError) as caught:
                request("http://localhost/v1", {}, timeout_s=3)
        self.assertEqual(caught.exception.status, 429)
        self.assertTrue(caught.exception.retryable)
        self.assertEqual(caught.exception.retry_after_s, 2)

    def test_decode_window_excludes_prefill_and_queueing_but_keeps_idle_samples(self):
        def sample(at, prefill, tokens, active=4, queued=0):
            return dict(
                observed_at_s=at,
                active_requests=active,
                queued_requests=queued,
                scheduler_statistics=dict(prefill_tokens=prefill, decode_tokens=tokens),
            )

        samples = [
            sample(0, 100, 0, queued=1),
            sample(1, 200, 10),
            sample(2, 200, 10),
            sample(3, 200, 18),
            sample(4, 200, 22, active=3),
        ]
        self.assertEqual(
            decode_window(samples, 200, 4),
            dict(wall_s=2, committed_tokens=8, committed_tps=4),
        )
        self.assertIsNone(decode_window(samples[:1], 200, 4))

    def test_counter_delta_handles_new_histogram_keys_and_omits_maxima(self):
        self.assertEqual(
            counter_delta(
                {"iterations": 5, "peak_active": 2, "batch_histogram": {"1": 5}},
                {"iterations": 8, "peak_active": 4, "batch_histogram": {"1": 5, "4": 3}},
            ),
            {"iterations": 3, "batch_histogram": {"1": 0, "4": 3}},
        )
        self.assertIsNone(counter_delta(None, {}))
        with self.assertRaises(ValueError):
            counter_delta({"iterations": 5}, {"iterations": 0})

    def test_small_sample_tail_percentile_uses_nearest_rank(self):
        self.assertEqual(percentile([2.0, 1.0], 0.95), 2.0)
        self.assertEqual(percentile([2.0, 1.0], 0.5), 1.0)

    def test_multiline_events_and_incomplete_stream(self):
        self.assertEqual(
            list(events(io.BytesIO(b"data: a\ndata: b\n\ndata: [DONE]\n\n"))), ["a\nb", "[DONE]"]
        )
        with self.assertRaises(ValueError):
            list(events(io.BytesIO(b"data: incomplete\n")))

    def test_throughput_counts_usage_instead_of_chunks(self):
        stream = b'data: {"choices":[{"delta":{"content":"several tokens"}}]}\n\n'
        stream += b'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n\n'
        stream += b'data: {"choices":[],"usage":{"completion_tokens":7,"prompt_tokens":5}}\n\n'
        stream += b"data: [DONE]\n\n"
        with patch("urllib.request.urlopen", return_value=io.BytesIO(stream)):
            row = request("http://localhost/v1", {})
        self.assertEqual(row["usage"]["completion_tokens"], 7)
        self.assertEqual(row["content"], "several tokens")
        self.assertIsNone(row["chunk_gap_p50_s"])
        self.assertIsNotNone(row["ttft_s"])

    def test_stream_trigger_ignores_empty_role_and_retains_usage_count(self):
        stream = b'data: {"choices":[{"delta":{"role":"assistant","content":""}}]}\n\n'
        stream += b'data: {"choices":[{"delta":{"content":"many tokens"}}]}\n\n'
        stream += b'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n\n'
        stream += b'data: {"choices":[],"usage":{"completion_tokens":9,"prompt_tokens":5}}\n\n'
        stream += b"data: [DONE]\n\n"
        observed = []
        with patch("urllib.request.urlopen", return_value=io.BytesIO(stream)):
            row = request(
                "http://localhost/v1", {}, on_content=observed.append, collect_arrivals=True
            )
        self.assertEqual(observed, row["arrivals_s"])
        self.assertEqual(len(observed), 1)
        self.assertEqual(row["usage"]["completion_tokens"], 9)
        self.assertGreater(row["started_monotonic_s"], 0)
        self.assertEqual(row["first_content_s"], observed[0])
        self.assertIsNone(row["first_reasoning_s"])


if __name__ == "__main__":
    unittest.main()
