import unittest

from tools.model.flash_next.chunks import chunks, index_capacity


class ChunkTests(unittest.TestCase):
    def test_exact_input_and_bounded_tail(self):
        for maximum in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096):
            for length in (1, 7, 8, 9, 127, 128, 129, 511, 512, 513,
                           2047, 2048, 2049, 4095, 4096, 4097, 8191, 8192, 8193):
                tokens = list(range(length))
                batches = list(chunks(tokens, maximum))
                self.assertEqual([token for batch in batches for token in batch], tokens)
                self.assertEqual(len(batches), (length + maximum - 1) // maximum)
                self.assertTrue(all(len(batch) == maximum for batch in batches[:-1]))
                self.assertTrue(1 <= len(batches[-1]) <= maximum)

    def test_invalid_input(self):
        for maximum in (0, 4097, True, 8.0):
            with self.assertRaises(ValueError):
                list(chunks([1], maximum))
        with self.assertRaises(ValueError):
            list(chunks([], 128))

    def test_live_index_boundaries_and_context_tail(self):
        for capacity in (128, 2051, 262144):
            for end in (1, 127, 128, 511, 512, 513, 2048, 2049, 8192, 8193, 262144):
                if end > capacity:continue
                bucket = index_capacity(end, capacity)
                self.assertTrue(end <= bucket <= capacity)
                self.assertEqual(bucket, min(capacity, max(512, 1 << (end - 1).bit_length())))
        for end,capacity in ((0,512),(513,512),(True,512),(512,262145)):
            with self.assertRaises(ValueError):index_capacity(end,capacity)
