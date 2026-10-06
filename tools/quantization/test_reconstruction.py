import tempfile
from pathlib import Path
import unittest

import numpy as np

from tools.quantization.q2i8 import quantize
from tools.quantization.reconstruction import quantize_reconstruction
from tools.quantization.q2i8_ffn import a8, samples


class ReconstructionTests(unittest.TestCase):
    def test_correlated_rank_deficient_samples_and_no_source_mutation(self):
        rng = np.random.default_rng(20261002)
        x = np.tile(rng.normal(size=(256,8)),(1,16)).astype(np.float32)
        w = rng.normal(0,.03,size=(3,128)).astype(np.float32)
        original = w.copy()
        independent = quantize(w)
        fitted = quantize_reconstruction(w,x,row_block=1)
        reference_error = np.linalg.norm(x@(w-independent.dequantize()).T)
        actual_error = np.linalg.norm(x@(w-fitted.dequantize()).T)
        self.assertLess(actual_error,reference_error*.4)
        np.testing.assert_array_equal(w,original)
        fitted.validate()
        for damping in (0,-1,float('nan')):
            with self.assertRaises(ValueError):
                quantize_reconstruction(w,x,damping=damping)
        with self.assertRaises(ValueError):
            quantize_reconstruction(w,x[:,:64])

    def test_zero_calibration_falls_back_to_weight_only(self):
        w = np.linspace(-1,1,128,dtype=np.float32).reshape(1,128)
        a = quantize(w)
        b = quantize_reconstruction(w,np.zeros((2,128),np.float32))
        np.testing.assert_array_equal(a.indices,b.indices)
        np.testing.assert_array_equal(a.codebooks,b.codebooks)
        np.testing.assert_array_equal(a.scales,b.scales)

    def test_a8_zero_rows_subnormal_scale_and_rne(self):
        # Scale is exactly one; ties exercise even rounding rather than truncation.
        x = np.array([[127,1.5,2.5,-2.5],[0,0,0,0],[2**-24,0,0,0]],np.float32)
        q,s = a8(x)
        np.testing.assert_array_equal(q[0],[127,2,2,-2])
        np.testing.assert_array_equal(q[1],0)
        self.assertEqual(float(s[1,0]),1)
        self.assertEqual(float(s[2,0]),2**-24)
        self.assertEqual(int(q[2,0]),1)

    def test_reject_prompt_leakage_and_duplicate_expert_routes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'samples.npz'
            data = dict(activations=np.ones((4,128),np.float32),
                        routed_ids=np.tile(np.array([0,1],np.int32),(4,1)),
                        prompt_ids=np.array([0,0,1,1],np.int32),
                        calibration_mask=np.array([True,True,False,False]),
                        expert_ids=np.array([0,1],np.int32))
            np.savez(path,**data)
            self.assertEqual(samples(path)[0].shape,(4,128))
            data['calibration_mask'] = np.array([True,False,True,False])
            np.savez(path,**data)
            with self.assertRaises(ValueError):
                samples(path)
            data['calibration_mask'] = np.array([True,True,False,False])
            data['routed_ids'][0] = 1
            np.savez(path,**data)
            with self.assertRaises(ValueError):
                samples(path)


if __name__ == '__main__':
    unittest.main()
