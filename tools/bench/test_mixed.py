import unittest

from tools.bench.mixed import overlapping_gaps


class MixedTimingTests(unittest.TestCase):
    def test_gap_spanning_injection_is_included(self):
        row = dict(started_monotonic_s=10.0, arrivals_s=[0.1, 0.2, 1.0, 1.1])
        # The stall starts before injection and ends after cold prefill.
        self.assertAlmostEqual(overlapping_gaps(row, 10.3, 10.8)[0], 0.8)
        self.assertEqual(len(overlapping_gaps(row, 10.3, 10.8)), 1)
        self.assertEqual(overlapping_gaps(row, 11.2, 12.0), [])

    def test_request_start_is_used_to_align_multiple_streams(self):
        row = dict(started_monotonic_s=20.0, arrivals_s=[0.0, 0.5, 2.0])
        self.assertEqual(overlapping_gaps(row, 20.6, 21.9), [1.5])
        self.assertEqual(overlapping_gaps(row, 19.0, 20.0), [])


if __name__ == "__main__":
    unittest.main()
