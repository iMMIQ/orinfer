"""Damped second-order error feedback for Q2I8's fixed row-scale format.

Uses the sequential reconstruction principle from GPTQ
(https://arxiv.org/abs/2210.17323), with local four-entry integer codebooks.
Physical column order is preserved; no runtime permutation or float K scales.
Scratch is one KxK covariance/factor plus a bounded output-row block.
"""
import numpy as np

from tools.quantization.q2i8 import Weights, activation_importance, quantize, _labels, _centers


def quantize_reconstruction(weight, activations, *, group_size=128, damping=.01, row_block=128):
    x = np.asarray(activations,dtype=np.float64)
    hdiag = activation_importance(x)
    if not np.isfinite(damping) or damping <= 0:
        raise ValueError('Reconstruction damping must be finite and positive')
    w = np.asarray(weight)
    if w.ndim != 2 or w.shape[1] != x.shape[1]:
        raise ValueError('Reconstruction samples must match weight channels')
    base = quantize(w,group_size=group_size,importance=hdiag,row_block=row_block)
    n,k = base.validate()
    h = x.T@x/len(x)
    average = np.diag(h).mean()
    if average == 0:
        return base
    h /= average
    h.flat[::k+1] += damping
    # FP64 factorization stabilizes sparse/rank-deficient routed samples.
    lower = np.linalg.cholesky(h)
    inverse_lower = np.linalg.solve(lower,np.eye(k,dtype=np.float64))
    inverse = inverse_lower.T@inverse_lower
    inverse = (inverse+inverse.T)*.5
    upper = np.linalg.cholesky(inverse).T.astype(np.float32)
    del h,lower,inverse_lower,inverse
    packed = np.empty_like(base.indices)
    books = np.empty_like(base.codebooks)
    g = group_size
    for row in range(0,n,row_block):
        work = w[row:row+row_block].astype(np.float32).copy()
        scale = base.scales[row:row+len(work)].astype(np.float32)
        for start in range(0,k,g):
            end = start+g
            block = work[:,start:end].copy()
            importance = hdiag[start:end].reshape(1,1,g)
            target = block[:,None,:]
            mean = (target*importance).sum(-1)/importance.sum(-1)
            sigma = np.sqrt((np.square(target-mean[...,None])*importance).sum(-1)/importance.sum(-1))
            centers = mean[...,None]+sigma[...,None]*np.array([-1.5,-.5,.5,1.5],np.float32)
            for _ in range(10):
                centers = _centers(target,_labels(target,centers),centers,importance)
            for _ in range(4):
                table = np.clip(np.rint(centers/scale[:,None,None]),-127,127).astype(np.int8)
                centers = table.astype(np.float32)*scale[:,None,None]
                centers = _centers(target,_labels(target,centers),centers,importance)
            table = np.clip(np.rint(centers[:,0]/scale[:,None]),-127,127).astype(np.int8)
            levels = table.astype(np.float32)*scale[:,None]
            q = np.empty((len(work),g),np.uint8)
            feedback = np.empty_like(block)
            for col in range(g):
                mids = (levels[:,:-1]+levels[:,1:])*.5
                code = (block[:,col,None] > mids).sum(-1).astype(np.uint8)
                value = np.take_along_axis(levels,code[:,None],axis=1)[:,0]
                delta = (block[:,col]-value)/upper[start+col,start+col]
                block[:,col:] -= delta[:,None]*upper[start+col,start+col:end][None,:]
                feedback[:,col] = delta
                q[:,col] = code
            work[:,end:] -= feedback@upper[start:end,end:]
            quartet = q.reshape(len(work),g//4,4)
            packed[row:row+len(work),start//4:end//4] = quartet[...,0] | (quartet[...,1]<<2) | (quartet[...,2]<<4) | (quartet[...,3]<<6)
            books[row:row+len(work),start//g] = table
    result = Weights(packed,books,base.scales.copy(),g)
    result.validate()
    return result
