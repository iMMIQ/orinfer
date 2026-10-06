import unittest

from tools.model.flash_chunks import chunks


class ChunkTests(unittest.TestCase):
    def test_exact_input_and_bounded_tail(self):
        for maximum in (1, 2, 4, 8, 16, 32, 64, 128):
            for length in (1, 7, 8, 9, 127, 128, 129, 2049, 8191):
                tokens = list(range(length))
                batches = list(chunks(tokens, maximum))
                self.assertEqual([token for batch in batches for token in batch], tokens)
                self.assertTrue(all(len(batch) in {maximum, 8, 1} and len(batch) <= maximum for batch in batches))
                if maximum >= 8:
                    self.assertLess(sum(len(batch) == 1 for batch in batches), 8)

    def test_invalid_input(self):
        for maximum in (0, 129, True, 8.0):
            with self.assertRaises(ValueError):
                list(chunks([1], maximum))
        with self.assertRaises(ValueError):
            list(chunks([], 128))
