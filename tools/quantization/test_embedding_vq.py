import unittest

import numpy as np

from tools.quantization.e8p import basis
from tools.quantization.embedding_vq import pack, decode
from tools.quantization.rotation import transform
from tools.quantization.vq import e8p_decode


class EmbeddingVQTests(unittest.TestCase):
    def test_native160_row_packets_and_inverse(self):
        rng = np.random.default_rng(20261002)
        codes = rng.integers(0,65536,(7,20),dtype=np.uint16)
        scales = np.linspace(.001,.007,7).astype(np.float16)
        signs = rng.choice(np.array([-1,1],np.int8),160)
        packets = pack(codes,scales)
        self.assertEqual(packets.shape,(7,42))
        restored = decode(packets,basis(),signs)
        rotated = e8p_decode(codes,basis()).reshape(7,160)*scales.astype(np.float32)[:,None]
        np.testing.assert_allclose(transform(restored,signs,mode='full'),rotated,atol=1e-7,rtol=1e-5)

    def test_reject_nonfinite_scale(self):
        with self.assertRaises(ValueError):pack(np.zeros((1,20),np.uint16),np.array([np.nan],np.float16))


if __name__ == '__main__':unittest.main()
