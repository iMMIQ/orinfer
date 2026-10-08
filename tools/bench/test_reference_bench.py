"""Check token accounting and actual SSE event framing independently of vLLM."""

import io
import json
import unittest
from unittest.mock import patch

from reference_bench import sse_events, stream_request


class TimingProtocolTests(unittest.TestCase):
    def test_sse_multi_line_and_keepalive(self):
        source = io.BytesIO(b': keepalive\n\ndata: {"a":\ndata: 1}\n\ndata: [DONE]\n\n')
        self.assertEqual(list(sse_events(source)), ['{"a":\n1}', "[DONE]"])

    def test_truncated_sse_rejected(self):
        with self.assertRaises(ValueError):
            list(sse_events(io.BytesIO(b'data: {"a":1}\n')))

    def test_first_multi_token_chunk_excluded_from_decode_interval(self):
        chunks = [
            {"choices": [{"index": 0, "token_ids": [10, 11], "text": "a"}]},
            {"choices": [{"index": 0, "token_ids": [12], "text": "b"}]},
            {"choices": [{"index": 0, "token_ids": [13], "text": "c", "finish_reason": "length"}]},
            {"choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": 4}},
        ]
        wire = "".join("data: " + json.dumps(x) + "\n\n" for x in chunks) + "data: [DONE]\n\n"
        with (
            patch("reference_bench.urllib.request.urlopen", return_value=io.BytesIO(wire.encode())),
            patch("reference_bench.time.perf_counter", side_effect=[0, 1, 4, 7, 7.1, 7.2, 7.3]),
        ):
            result = stream_request(
                {"url": "http://127.0.0.1:1", "served_model": "test"}, [1, 2], 4
            )
        self.assertEqual(result["first_chunk_tokens"], 2)
        self.assertAlmostEqual(result["client_decode_tps"], 2 / 6)
        self.assertEqual(result["input_tokens_per_client_ttft"], 2)


if __name__ == "__main__":
    unittest.main()
