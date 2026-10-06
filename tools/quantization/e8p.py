"""Weight-only integer E8P encoder, independent of external quantizer code.

The 256 basis vectors are generated from lattice geometry. The 29 outer-shell
vectors use a deterministic spread over the 56 possible five-coordinate sets;
their order is our own and must travel with the indices in safetensors.
"""
import itertools

import numpy as np

from tools.quantization.vq import SIGN_BITS, e8p_decode


def basis():
    rows = np.array(list(itertools.product((2,6,10),repeat=8)),np.int16)
    rows = rows[(rows*rows).sum(1) <= 160]
    combinations = list(itertools.combinations(range(8),5))
    outer = np.full((29,8),2,np.int16)
    for i in range(29):
        outer[i,list(combinations[i*len(combinations)//29])] = 6
    rows = np.concatenate((rows,outer))
    rows[:,-1] *= 1-2*((rows.sum(1)//4)%2)
    assert rows.shape == (256,8)
    return rows.astype(np.int8)


def nearest(values, table):
    """Small CPU oracle; production conversion uses the fused GPU encoder."""
    x = np.asarray(values,np.float32)
    table = np.asarray(table)
    if x.ndim != 2 or x.shape[1] != 8 or not np.isfinite(x).all():
        raise ValueError('Expected finite [vectors,8] inputs')
    if table.shape != (256,8) or table.dtype != np.int8 or (table%2).any():
        raise ValueError('Expected even integer E8P basis')
    absolute = np.abs(table.astype(np.float32))
    bp = (table < 0).sum(1)%2
    candidates = []
    for p in (0,1):
        shifted = x-(1-2*p)
        negative = shifted < 0
        value = np.abs(shifted)
        corrections = (negative.sum(1)%2)[:,None] != bp[None,:]
        costs = 4*value[:,None,:]*absolute[None,:,:]
        flip = costs.argmin(2)
        score = ((value[:,None,:]-absolute[None,:,:])**2).sum(2)
        score += corrections*costs.min(2)
        index = score.argmin(1)
        orientation = negative.copy()
        row = np.arange(len(x))
        orientation[row,flip[row,index]] ^= corrections[row,index]
        mask = orientation ^ (table[index] < 0)
        low = (mask.astype(np.uint16)*(1 << SIGN_BITS)[None,:]).sum(1)^p
        code = ((index.astype(np.uint16) << 8)|low.astype(np.uint16)).astype(np.uint16)
        # Compare actual reconstructed distances, including parity shift.
        distance = ((x-e8p_decode(code,table).astype(np.float32))**2).sum(1)
        candidates.append((code,distance))
    return np.where(candidates[0][1] <= candidates[1][1],candidates[0][0],candidates[1][0])
