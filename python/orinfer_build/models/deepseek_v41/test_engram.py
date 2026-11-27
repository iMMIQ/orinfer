import unittest

from orinfer_build.models.deepseek_v41.export_engram import layers, next_prime


class EngramMetadataTests(unittest.TestCase):
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
        self.assertEqual(len({p for layer in result for p in layer["primes"]}), 48)
        self.assertTrue(all(m % 2 == 1 for layer in result for m in layer["multipliers"]))
        text["engram_num_embeddings"][0] += 1
        with self.assertRaises(ValueError):
            layers(text, 99092)

    def test_prime_selection_is_strict_and_excludes_used_buckets(self):
        self.assertEqual(next_prime(10, set()), 11)
        self.assertEqual(next_prime(10, {11}), 13)


if __name__ == "__main__":
    unittest.main()
