"""Our E8P PLE row packets, decoded before the ordinary PLE projections.

One row is F16 scale followed by little-endian U16 vector codes. A full
Paley20 x Walsh transform permits the native 160-channel embedding width
without padding. Its inverse is applied after lookup, never to model inputs.
"""
import numpy as np

from tools.quantization.rotation import transform
from tools.quantization.vq import e8p_decode


def pack(codes, scales):
    if codes.dtype != np.uint16 or codes.ndim != 2 or scales.dtype != np.float16 or scales.shape != (len(codes),):
        raise ValueError('Invalid embedding indices/scales')
    if not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError('Invalid embedding scales')
    out = np.empty((len(codes),2+codes.shape[1]*2),np.uint8)
    out[:,:2] = scales.astype('<f2').view(np.uint8).reshape(-1,2)
    out[:,2:] = codes.astype('<u2').view(np.uint8).reshape(len(codes),-1)
    return out


def decode(packets, table, signs):
    if packets.dtype != np.uint8 or packets.ndim != 2 or packets.shape[1] != 2+len(signs)//4:
        raise ValueError('Invalid embedding packets')
    scale = np.ascontiguousarray(packets[:,:2]).view('<f2').reshape(-1).astype(np.float32)
    if not np.isfinite(scale).all() or (scale <= 0).any():raise ValueError('Invalid embedding row scale')
    codes = np.ascontiguousarray(packets[:,2:]).view('<u2')
    rotated = e8p_decode(codes,table).reshape(len(packets),len(signs)).astype(np.float32)*scale[:,None]
    return transform(rotated,signs,mode='full',inverse=True)
