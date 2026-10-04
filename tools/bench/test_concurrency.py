import io
import unittest
from unittest.mock import patch
from tools.bench.concurrency import counter_delta, events, request, percentile


class StreamingTests(unittest.TestCase):
    def test_counter_delta_handles_new_histogram_keys_and_omits_maxima(self):
        self.assertEqual(counter_delta(
            {'iterations': 5, 'peak_active': 2, 'batch_histogram': {'1': 5}},
            {'iterations': 8, 'peak_active': 4, 'batch_histogram': {'1': 5, '4': 3}}),
            {'iterations': 3, 'batch_histogram': {'1': 0, '4': 3}})
        self.assertIsNone(counter_delta(None, {}))
        with self.assertRaises(ValueError):
            counter_delta({'iterations': 5}, {'iterations': 0})
    def test_small_sample_tail_percentile_uses_nearest_rank(self):
        self.assertEqual(percentile([2.,1.],.95),2.)
        self.assertEqual(percentile([2.,1.],.5),1.)
    def test_multiline_events_and_incomplete_stream(self):
        self.assertEqual(list(events(io.BytesIO(b'data: a\ndata: b\n\ndata: [DONE]\n\n'))),
                         ['a\nb', '[DONE]'])
        with self.assertRaises(ValueError):
            list(events(io.BytesIO(b'data: incomplete\n')))

    def test_throughput_counts_usage_instead_of_chunks(self):
        stream = b'data: {"choices":[{"delta":{"content":"several tokens"}}]}\n\n'
        stream += b'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n\n'
        stream += b'data: {"choices":[],"usage":{"completion_tokens":7,"prompt_tokens":5}}\n\n'
        stream += b'data: [DONE]\n\n'
        with patch('urllib.request.urlopen', return_value=io.BytesIO(stream)):
            row = request('http://localhost/v1', {})
        self.assertEqual(row['usage']['completion_tokens'], 7)
        self.assertEqual(row['content'], 'several tokens')
        self.assertIsNone(row['chunk_gap_p50_s'])
        self.assertIsNotNone(row['ttft_s'])


if __name__ == '__main__':
    unittest.main()
