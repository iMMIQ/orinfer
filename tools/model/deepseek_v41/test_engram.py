import unittest

import numpy as np

from tools.model.deepseek_v41.export_engram import layers, next_prime, validate_hadamard32


class EngramMetadataTests(unittest.TestCase):
    def test_fast_rotation_requires_exact_signed_walsh_bank(self):
        walsh = np.array(
            [[(-1) ** ((i & j).bit_count()) for j in range(32)] for i in range(32)],
            dtype=np.float32,
        )
        signs = np.where(np.arange(32) % 3 == 0, -1, 1).astype(np.float32)
        signs[0] = 1
        bank = walsh * signs[:, None] * np.float32(0.17679559)
        validate_hadamard32(bank)
        for bad in (bank.T, bank * np.linspace(1, 2, 32, dtype=np.float32), bank[::-1]):
            with self.assertRaises(ValueError):
                validate_hadamard32(bad)
        bank[1, 3] = np.nextafter(bank[1, 3], np.float32(0))
        with self.assertRaises(ValueError):
            validate_hadamard32(bank)

    def test_checkpoint_bucket_sizes_and_layer_rng(self):
        text = dict(
            engram_layer_ids=[1, 14],
            engram_num_embeddings=[384006168, 384016682],
            engram_max_ngram_size=4,
            engram_vocab_size=16000000,
            engram_n_heads=8,
        )
        result = layers(text, 99092)
        self.assertEqual([sum(layer["primes"]) for layer in result], text["engram_num_embeddings"])
        self.assertNotEqual(result[0]["multipliers"], result[1]["multipliers"])
        self.assertEqual(len(set(p for layer in result for p in layer["primes"])), 48)
        self.assertTrue(all(m % 2 == 1 for layer in result for m in layer["multipliers"]))
        text["engram_num_embeddings"][0] += 1
        with self.assertRaises(ValueError):
            layers(text, 99092)

    def test_prime_selection_is_strict_and_excludes_used_buckets(self):
        self.assertEqual(next_prime(10, set()), 11)
        self.assertEqual(next_prime(10, {11}), 13)


if __name__ == "__main__":
    unittest.main()
