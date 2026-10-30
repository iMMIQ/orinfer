"""Physical router spans restore each logical output column exactly once."""

import unittest

from tools.model.deepseek_v41.router_sharding import column_spans


class RouterShardingTest(unittest.TestCase):
    def test_aligned_and_overlapping_tails_cover_the_original_columns(self):
        for width, ranks in ((384, 8), (256, 8), (1024, 8)):
            physical, starts = column_spans(width, ranks)
            logical = width // ranks
            restored = []
            for rank, first in enumerate(starts):
                self.assertGreaterEqual(first, 0)
                self.assertLessEqual(first + physical, width)
                values = list(range(first, first + physical))
                begin = rank * logical - first
                restored.extend(values[begin : begin + logical])
            self.assertEqual(restored, list(range(width)))

    def test_invalid_matrix_or_rank_geometry_is_rejected(self):
        for width, ranks in ((383, 8), (16, 8), (384, 0)):
            with self.assertRaises(ValueError):
                column_spans(width, ranks)


if __name__ == "__main__":
    unittest.main()
