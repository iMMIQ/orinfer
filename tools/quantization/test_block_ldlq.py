import unittest

import numpy as np

from tools.quantization.block_ldlq import factor, round_blocks
from tools.quantization.rotation import hadamard20, transform


class BlockLDLTests(unittest.TestCase):
    def test_factor_reconstructs_regularized_covariance_with_identity_blocks(self):
        rng = np.random.default_rng(20261002)
        x = rng.normal(size=(7,32))  # intentionally rank deficient
        h,l,blocks = factor(x.T@x,block_size=8)
        d = np.zeros_like(h)
        for index,block in enumerate(blocks):
            start = index*8;d[start:start+8,start:start+8] = block
            np.testing.assert_allclose(l[start:start+8,start:start+8],np.eye(8),atol=2e-14)
        np.testing.assert_allclose(l@d@l.T,h,atol=2e-14,rtol=2e-14)

    def test_reverse_feedback_reduces_a_correlated_integer_rounding_error(self):
        w = np.array([[.49,.49]])
        h,l,_ = factor(np.array([[1.,.9],[.9,1.]]),block_size=1)
        q = round_blocks(w,l,np.rint,block_size=1)
        naive = np.rint(w)
        np.testing.assert_array_equal(q,[[1.,0.]])
        self.assertLess(((w-q)@h@(w-q).T).item(),((w-naive)@h@(w-naive).T).item())

    def test_invalid_covariance_and_quantizer_are_rejected(self):
        for h in [np.zeros((8,8)),np.ones((3,4)),np.array([[1.,.2],[.4,1.]]),np.array([[np.nan]])]:
            with self.assertRaises(ValueError):factor(h,block_size=1)
        with self.assertRaises(ValueError):factor(np.eye(8),damping=0)
        with self.assertRaises(ValueError):round_blocks(np.ones((1,8)),np.eye(8),lambda x:np.full_like(x,np.nan))


class FullRotationTests(unittest.TestCase):
    def test_paley_matrix_and_full_transform_preserve_inner_products_and_inverse(self):
        h = hadamard20()
        np.testing.assert_array_equal(h@h.T,20*np.eye(20))
        rng = np.random.default_rng(20261002)
        for k in (640,1280,2560):
            x = rng.normal(size=(3,k)).astype(np.float32)
            w = rng.normal(size=(5,k)).astype(np.float32)
            s = rng.choice(np.array([-1,1],np.int8),k)
            tx,tw = transform(x,s),transform(w,s)
            np.testing.assert_allclose(tx@tw.T,x@w.T,rtol=3e-5,atol=3e-5)
            np.testing.assert_allclose(transform(tx,s,inverse=True),x,rtol=5e-5,atol=1e-6)

    def test_two_sided_weight_transform_preserves_projection(self):
        rng = np.random.default_rng(20261002)
        x = rng.normal(size=(3,640)).astype(np.float32)
        w = rng.normal(size=(1280,640)).astype(np.float32)
        si = rng.choice(np.array([-1,1],np.int8),640)
        so = rng.choice(np.array([-1,1],np.int8),1280)
        transformed = transform(transform(w,si).T,so).T
        y = transform(transform(x,si)@transformed.T,so,inverse=True)
        np.testing.assert_allclose(y,x@w.T,rtol=1e-3,atol=7e-5)


if __name__ == '__main__':unittest.main()
