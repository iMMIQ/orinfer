"""Experimental four-entry I8 LUT + integer coarse step, NumPy offline.

For U4 q, reconstructed I8 is base[q % 4] + (q // 4)*step - 128.
Five bytes per group are additional metadata if original decode S/Z stay.
This approximates current row-W8 values; no model-quality acceptance claim.
"""
import numpy as np


def prepare(scale, zero, row_scale):
    assert scale.ndim == 2 and zero.shape == scale.shape
    assert row_scale.shape == (scale.shape[0],)
    assert np.isfinite(scale).all() and (row_scale > 0).all()
    codes=np.arange(16,dtype=np.float16)
    # Preserve the current half weight materialization before row quantization.
    dequant=(codes-zero[...,None].astype(np.float16))*scale[...,None]
    reference=np.clip(np.rint(dequant.astype(np.float32)/row_scale[:,None,None].astype(np.float32)),-127,127).astype(np.int16)
    base=reference[...,:4]+128
    proposed=np.rint(4*scale.astype(np.float32)/row_scale[:,None].astype(np.float32)).astype(np.int16)
    bound=(255-base.max(axis=-1))//3
    coarse=np.arange(16,dtype=np.int16)//4
    best_error=np.full(scale.shape,np.iinfo(np.int32).max,dtype=np.int32)
    best=np.zeros(scale.shape,dtype=np.int16)
    for change in (0,-1,1):
        step=np.minimum(np.maximum(proposed+change,0),bound)
        approximate=base[...,np.arange(16)%4]+coarse*step[...,None]-128
        cost=((approximate-reference).astype(np.int32)**2).sum(axis=-1)
        choose=cost<best_error
        best=np.where(choose,step,best);best_error=np.minimum(cost,best_error)
    approximate=base[...,np.arange(16)%4]+coarse*best[...,None]-128
    assert approximate.min()>=-127 and approximate.max()<=127
    table=np.zeros(scale.shape,dtype=np.uint32)
    for i in range(4):table|=base[...,i].astype(np.uint32)<<(8*i)
    return table,best.astype(np.uint8),reference,approximate
