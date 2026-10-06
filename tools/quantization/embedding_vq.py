"""Our E8P PLE row packets, decoded before the ordinary PLE projections.

One row is F16 scale followed by little-endian U16 vector codes. A full
Paley20 x Walsh transform permits the native 160-channel embedding width
without padding. Its inverse is applied after lookup, never to model inputs.
"""
from collections import OrderedDict

import numpy as np

from tools.quantization.rotation import transform, hadamard20, _walsh
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


class EmbeddingDecoder:
    """Bounded, exact lookup decoder owned by one checkpoint reader.

    Keep one 512 KiB integer codebook and at most 128 validated sign vectors.
    Keys use contents, so mutated metadata cannot reuse stale validation or a
    stale codebook. FP32 operation order matches the reference row decoder.
    """
    def __init__(self):
        self._table_key = None
        self._book = None
        self._signs = OrderedDict()
        self._hadamard = hadamard20()*np.float32(20**-.5)
        self._hadamard.setflags(write=False)

    def decode(self, packets, table, signs):
        s = np.asarray(signs)
        if s.ndim != 1 or not len(s) or len(s)%20:
            raise ValueError('Invalid embedding transform dimension')
        width = len(s)//20
        if width & (width-1):raise ValueError('Invalid embedding transform dimension')
        sign_key = (s.dtype.str,s.tobytes())
        if sign_key not in self._signs:
            if not np.isin(s,[-1,1]).all():raise ValueError('Transform requires one +/-1 sign per channel')
            if len(self._signs) == 128:self._signs.popitem(last=False)
            self._signs[sign_key] = None
        self._signs.move_to_end(sign_key)
        if packets.dtype != np.uint8 or packets.ndim != 2 or packets.shape[1] != 2+len(s)//4:
            raise ValueError('Invalid embedding packets')
        if table.dtype != np.int8 or table.shape != (256,8):raise ValueError('Invalid E8P embedding table')
        table_key = table.tobytes()
        if table_key != self._table_key:
            book = e8p_decode(np.arange(65536,dtype=np.uint16),table)
            book.setflags(write=False)
            self._book,self._table_key = book,table_key
        scale = np.ascontiguousarray(packets[:,:2]).view('<f2').reshape(-1).astype(np.float32)
        if not np.isfinite(scale).all() or (scale <= 0).any():raise ValueError('Invalid embedding row scale')
        codes = np.ascontiguousarray(packets[:,2:]).view('<u2')
        rotated = self._book[codes].reshape(len(packets),len(s)).astype(np.float32)*scale[:,None]
        value = np.einsum('ji,...jp->...ip',self._hadamard,rotated.reshape(len(packets),20,width))
        return _walsh(value).reshape(rotated.shape)*s
