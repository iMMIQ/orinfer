"""Offline BF16/F16/F32 -> two-bit indices and local INT8 codebooks.

Codebooks use one common FP16 scale per output row. Group-local amplitude is
encoded by four arbitrary integer entries, not a floating inner-K scale.
The base fitter optionally uses diagonal activation second moments. The CLI
also supports covariance-based reconstruction from actual routed samples.
Neither local reconstruction metric establishes model-quality acceptance.
"""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file


FORMAT = 'orinfer.q2i8.v1'


@dataclass(frozen=True)
class Weights:
    indices: np.ndarray       # [N,K/4], adjacent low-to-high two-bit fields
    codebooks: np.ndarray     # [N,K/G,4], signed integer entries
    scales: np.ndarray        # [N], positive FP16 output-row scale
    group_size: int

    def validate(self):
        p, c, s, g = self.indices, self.codebooks, self.scales, self.group_size
        if type(g) is not int or g not in (64, 128):
            raise ValueError('Q2I8 group size must be 64 or 128')
        if p.dtype != np.uint8 or p.ndim != 2 or min(p.shape) <= 0:
            raise ValueError('Indices must be nonempty U8[N,K/4]')
        n, k = p.shape[0], p.shape[1]*4
        if k % g or c.dtype != np.int8 or c.shape != (n,k//g,4):
            raise ValueError('Codebooks must be I8[N,K/G,4]')
        if s.dtype != np.float16 or s.shape != (n,) or not np.isfinite(s).all() or (s <= 0).any():
            raise ValueError('Row scales must be finite positive F16[N]')
        return n, k

    @property
    def nbytes(self):
        return self.indices.nbytes+self.codebooks.nbytes+self.scales.nbytes

    def integer_weights(self):
        n,k = self.validate()
        q = ((self.indices[...,None] >> np.arange(0,8,2,dtype=np.uint8)) & 3).reshape(n,k//self.group_size,self.group_size)
        return np.take_along_axis(self.codebooks,q.astype(np.intp),axis=-1).reshape(n,k)

    def dequantize(self):
        return self.integer_weights().astype(np.float32)*self.scales.astype(np.float32)[:,None]

    def gpu_layout(self):
        """Lossless group-major codes and little-endian packed integer tables.

        Caller adds a bank dimension, uploads once and releases CPU/transient
        representations. No expanded persistent W8 is part of this format.
        """
        n,k = self.validate()
        g = self.group_size
        p = np.ascontiguousarray(self.indices.reshape(n,k//g,g//4).transpose(1,0,2))
        c = np.ascontiguousarray(self.codebooks.transpose(1,0,2)).view('<u4').reshape(k//g,n)
        return p,c,self.scales.copy()


def activation_importance(activations):
    """Second moment per input channel from real, already routed activations.

    Never generate fake samples for calibration. A caller without samples can
    explicitly use weight-only fitting. Preserve a floor for unseen channels.
    """
    x = np.asarray(activations)
    if x.ndim != 2 or min(x.shape) <= 0 or not np.isfinite(x).all():
        raise ValueError('Calibration activations must be finite nonempty [T,K]')
    h = np.zeros(x.shape[1],np.float64)
    for start in range(0,len(x),256):
        block = x[start:start+256].astype(np.float64)
        h += np.square(block).sum(0)
    h /= len(x)
    average = h.mean()
    return np.maximum(h/average,1e-6).astype(np.float32) if average > 0 else np.ones(x.shape[1],np.float32)


def _labels(x, centers):
    mids = (centers[...,:-1]+centers[...,1:])*np.float32(.5)
    q = np.zeros(x.shape,np.uint8)
    for j in range(3):
        q += (x > mids[...,j,None]).astype(np.uint8)
    return q


def _centers(x, q, old, h):
    result = old.copy()
    for j in range(4):
        masked = (q == j)*h
        total = (x*masked).sum(-1)
        count = masked.sum(-1)
        np.divide(total,count,out=result[...,j],where=count > 0)
    return np.sort(result,axis=-1)


def quantize(weight, *, group_size=128, importance=None, iterations=10, row_block=128):
    """Fit the final integer representation directly from original weights.

    Row-blocked scratch; finite floats only. Updating codebooks includes the
    common integer-grid constraint. There is no execution-time requantizer.
    """
    w = np.asarray(weight)
    if w.ndim != 2 or min(w.shape) <= 0 or w.dtype.kind != 'f':
        raise ValueError('Weights must be a nonempty floating matrix [N,K]')
    n,k = w.shape
    if group_size not in (64,128) or k % group_size:
        raise ValueError('K must be a multiple of group size 64 or 128')
    if type(iterations) is not int or iterations < 1 or type(row_block) is not int or row_block < 1:
        raise ValueError('Iterations and row block must be positive integers')
    h = np.ones(k,np.float32) if importance is None else np.asarray(importance,dtype=np.float32)
    if h.shape != (k,) or not np.isfinite(h).all() or (h <= 0).any():
        raise ValueError('Importance must contain K finite positive weights')
    # Normalization keeps the objective unchanged and avoids avoidable overflow.
    h = (h.astype(np.float64)/h.astype(np.float64).mean()).astype(np.float32).reshape(1,k//group_size,group_size)
    packed = np.empty((n,k//4),np.uint8)
    tables = np.empty((n,k//group_size,4),np.int8)
    scales = np.empty(n,np.float16)
    for start in range(0,n,row_block):
        x = w[start:start+row_block].astype(np.float32).reshape(-1,k//group_size,group_size)
        if not np.isfinite(x).all():
            raise ValueError('Weights contain nonfinite or unsupported values')
        mean = (x*h).sum(-1)/h.sum(-1)
        sigma = np.sqrt((np.square(x-mean[...,None])*h).sum(-1)/h.sum(-1))
        centers = mean[...,None]+sigma[...,None]*np.array([-1.5,-.5,.5,1.5],np.float32)
        for _ in range(iterations):
            q = _labels(x,centers)
            centers = _centers(x,q,centers,h)
        maximum = np.abs(centers).max(axis=(1,2))
        s = np.where(maximum > 0,np.maximum(maximum/127,2**-24),1).astype(np.float16)
        if not np.isfinite(s).all():
            raise ValueError('Weights exceed the FP16 row-scale range')
        for _ in range(4):
            c = np.clip(np.rint(centers/s.astype(np.float32)[:,None,None]),-127,127).astype(np.int8)
            centers = c.astype(np.float32)*s.astype(np.float32)[:,None,None]
            q = _labels(x,centers)
            centers = _centers(x,q,centers,h)
        c = np.clip(np.rint(centers/s.astype(np.float32)[:,None,None]),-127,127).astype(np.int8)
        q = _labels(x,c.astype(np.float32)*s.astype(np.float32)[:,None,None]).reshape(-1,k//4,4)
        packed[start:start+len(x)] = q[...,0] | (q[...,1]<<2) | (q[...,2]<<4) | (q[...,3]<<6)
        tables[start:start+len(x)] = c
        scales[start:start+len(x)] = s
    result = Weights(packed,tables,scales,group_size)
    result.validate()
    return result


def save(path, weights, *, provenance=None):
    n,k = weights.validate()
    path = Path(path)
    metadata = {'format':FORMAT,'group_size':str(weights.group_size),'shape':json.dumps([n,k]),
                'provenance':json.dumps(provenance or {},sort_keys=True)}
    # Exclusive creation: never replace an accepted artifact on a failed fit.
    with path.open('xb'):
        pass
    try:
        save_file({'indices':weights.indices,'codebooks':weights.codebooks,'scales':weights.scales},str(path),metadata=metadata)
    except BaseException:
        path.unlink()
        raise


def load(path):
    with safe_open(path,framework='np') as reader:
        metadata = reader.metadata() or {}
        if metadata.get('format') != FORMAT or set(reader.keys()) != {'indices','codebooks','scales'}:
            raise ValueError('Unsupported Q2I8 tensor container')
        try:
            result = Weights(reader.get_tensor('indices'),reader.get_tensor('codebooks'),reader.get_tensor('scales'),int(metadata['group_size']))
            shape = json.loads(metadata['shape'])
        except (KeyError,ValueError,TypeError,json.JSONDecodeError) as error:
            raise ValueError('Malformed Q2I8 metadata') from error
        if list(result.validate()) != shape:
            raise ValueError('Q2I8 shape metadata mismatch')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights',type=Path,required=True,help='Original floating [N,K] .npy; never a decoded Q2 reference')
    p.add_argument('--activations',type=Path,help='Real routed [T,K] .npy calibration samples')
    p.add_argument('--calibration-method',choices=('diagonal','reconstruction'),default='reconstruction')
    p.add_argument('--group-size',type=int,choices=(64,128),default=128)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    w = np.load(a.weights,mmap_mode='r',allow_pickle=False)
    x = np.load(a.activations,mmap_mode='r',allow_pickle=False) if a.activations else None
    if x is not None and a.calibration_method == 'reconstruction':
        from tools.quantization.reconstruction import quantize_reconstruction
        fitted = quantize_reconstruction(w,x,group_size=a.group_size)
    else:
        fitted = quantize(w,group_size=a.group_size,importance=activation_importance(x) if x is not None else None)
    save(a.output,fitted,provenance={'weights':str(a.weights),'activations':str(a.activations) if a.activations else None,
        'calibration':a.calibration_method if x is not None else 'weight-only; not model-quality acceptance'})
    print(json.dumps({'shape':list(w.shape),'bytes':fitted.nbytes,'bits_per_weight':fitted.nbytes*8/w.size,
                      'calibrated':x is not None,'output':str(a.output)}))


if __name__ == '__main__':
    main()
