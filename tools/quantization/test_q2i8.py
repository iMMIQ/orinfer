import tempfile
from pathlib import Path
import unittest

import numpy as np
from safetensors.numpy import save_file

from tools.quantization.q2i8 import Weights, activation_importance, quantize, save, load


class Q2I8Tests(unittest.TestCase):
    def test_all_packed_quartets_and_signed_palette(self):
        p = np.arange(256,dtype=np.uint8).reshape(1,256)
        c = np.tile(np.array([-128,-7,1,127],np.int8),(1,8,1))
        w = Weights(p,c,np.array([.25],np.float16),128)
        # Scalar byte oracle independent of the vectorized reader.
        expected = [int(c[0,0,(int(byte)>>shift)&3]) for byte in p[0] for shift in (0,2,4,6)]
        np.testing.assert_array_equal(w.integer_weights()[0],expected)
        a,b,s = w.gpu_layout()
        self.assertEqual(a.shape,(8,1,32))
        self.assertEqual(int(b[0,0]),int.from_bytes(bytes([128,249,1,127]),'little'))
        self.assertEqual(float(s[0]),.25)

    def test_zero_rows_and_exact_representable_levels(self):
        x = np.tile(np.array([-3,-1,1,3],np.float32),32)[None,:]
        x = np.concatenate([x,np.zeros_like(x)],axis=0)
        w = quantize(x)
        np.testing.assert_array_equal(w.integer_weights()[1],0)
        self.assertEqual(float(w.scales[1]),1)
        self.assertLess(float(np.abs(w.dequantize()-x).max()),.025)

    def test_real_activation_second_moment_and_row_block_determinism(self):
        rng = np.random.default_rng(20261002)
        x = rng.normal(size=(17,128)).astype(np.float32)
        x[:,0] *= 10
        h = activation_importance(x)
        self.assertGreater(h[0],h[1]*20)
        w = rng.normal(size=(3,128)).astype(np.float32)
        a,b = quantize(w,importance=h,row_block=1),quantize(w,importance=h,row_block=3)
        for name in ('indices','codebooks','scales'):
            np.testing.assert_array_equal(getattr(a,name),getattr(b,name))
        np.testing.assert_array_equal(activation_importance(np.zeros((2,128),np.float32)),1)

    def test_container_roundtrip_and_no_overwrite(self):
        w = quantize(np.arange(384,dtype=np.float32).reshape(3,128)-192)
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'weights.safetensors'
            save(p,w)
            before = p.read_bytes()
            restored = load(p)
            np.testing.assert_array_equal(restored.dequantize(),w.dequantize())
            with self.assertRaises(FileExistsError):
                save(p,w)
            self.assertEqual(p.read_bytes(),before)
            save_file({'indices':w.indices,'codebooks':w.codebooks,'scales':w.scales},str(Path(directory)/'bad.safetensors'),metadata={'format':'other'})
            with self.assertRaises(ValueError):
                load(Path(directory)/'bad.safetensors')

    def test_invalid_source_calibration_and_scale(self):
        for w in [np.zeros((1,127),np.float32),np.zeros((1,128),np.int8),np.full((1,128),np.nan,np.float32)]:
            with self.assertRaises(ValueError):
                quantize(w)
        for x in [np.empty((0,128)),np.full((1,128),np.inf)]:
            with self.assertRaises(ValueError):
                activation_importance(x)
        with self.assertRaises(ValueError):
            quantize(np.ones((1,128),np.float32),importance=np.zeros(128))
        with self.assertRaises(ValueError):
            Weights(np.zeros((1,32),np.uint8),np.zeros((1,1,4),np.int8),np.array([0],np.float16),128).validate()


if __name__ == '__main__':
    unittest.main()
