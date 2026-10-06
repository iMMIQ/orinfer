"""Bounded-memory GPU weight-only fitting for our integer E8P representation."""
import numpy as np
import torch

from kernels.model.e8p_encode import encode
from tools.quantization.e8p import basis
from tools.quantization.rotation import hadamard20
from tools.quantization.vq import Weights, e8p_decode


class Encoder:
    def __init__(self):
        self.table = basis()
        self.book = torch.from_numpy(self.table).cuda()
        self.grid = torch.from_numpy(e8p_decode(np.arange(65536,dtype=np.uint16),self.table).astype(np.float32)).cuda()
        self.kernels = {}

    def nearest(self, values):
        count = len(values)
        if count not in self.kernels:self.kernels[count] = encode(count)
        codes = torch.empty(count,device='cuda',dtype=torch.uint16)
        self.kernels[count](values.contiguous(),self.book,codes)
        return codes

    def fit_arrays(self, raw, signs, *, iterations=2, rotation='block128'):
        source = np.asarray(raw)
        if source.ndim != 2 or min(source.shape) <= 0 or not np.isfinite(source).all():
            raise ValueError('Expected finite [N,K] matrix')
        n,k = source.shape
        if type(iterations) is not int or not 0 <= iterations <= 8:
            raise ValueError('Invalid fitting iteration count')
        if rotation == 'block128':
            if k%128:raise ValueError('Block rotation requires K divisible by 128')
            width = 128
        elif rotation == 'full':
            width = k//20
            if k%20 or width <= 0 or width & (width-1):
                raise ValueError('Full rotation requires K = 20 * power of two')
        else:raise ValueError('Unsupported rotation')
        if signs.dtype != np.int8 or signs.shape != (k,) or not np.isin(signs,[-1,1]).all():
            raise ValueError('Invalid input rotation signs')
        w = torch.from_numpy(source.astype(np.float32)).cuda()*torch.from_numpy(signs).cuda()
        for stage in range(width.bit_length()-1):
            h = 1 << stage
            view = w.reshape(n,-1,2*h)
            left,right = view[...,:h].clone(),view[...,h:].clone()
            view[...,:h],view[...,h:] = left+right,left-right
        w *= width**-.5
        if rotation == 'full':
            h20 = torch.from_numpy(hadamard20()).cuda()*(20**-.5)
            w = torch.einsum('ij,rjp->rip',h20,w.reshape(n,20,width)).reshape(n,k)
        # The stored F16 scale is used during fitting, just as in online decode.
        scale = (w.square().mean(1).sqrt()/4).clamp(2**-24,65504).half()
        for iteration in range(iterations+1):
            codes = self.nearest((w/scale.float()[:,None]).reshape(-1,8))
            if iteration == iterations:break
            q = self.grid[codes.long()].reshape(n,k)
            scale = ((w*q).sum(1)/q.square().sum(1).clamp_min(1)).clamp(2**-24,65504).half()
        return codes.cpu().numpy().reshape(n,k//8),scale.cpu().numpy()

    def fit(self, raw, signs, *, iterations=2):
        codes,scale = self.fit_arrays(raw,signs,iterations=iterations)
        fitted = Weights('e8p',codes,self.table.copy(),scale,signs.copy())
        fitted.validate()
        return fitted
