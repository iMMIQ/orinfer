"""Experimental integer VQ fixtures; no full-model acceptance implied.

VQ4: one U8 index selects four signed bytes. E8P: one U16 index selects
eight lattice coordinates, represented exactly on an integer grid. Rotation
is an input-side normalized block128 Hadamard with fixed signs, not a transform
that can be moved through SwiGLU. Codebook data is supplied by the caller.
"""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file


SIGN_BITS = np.array([0,4,1,5,2,6,3,7],np.uint16)


def rotate(x, signs):
    x = np.asarray(x,dtype=np.float32)
    signs = np.asarray(signs)
    if x.shape[-1] % 128 or signs.shape != (x.shape[-1],) or not np.isin(signs,[-1,1]).all():
        raise ValueError('Block128 rotation needs one +/-1 sign per input channel')
    y = (x*signs).copy()
    for stage in range(7):
        h = 1 << stage
        block = y.reshape(*y.shape[:-1],-1,2*h)
        left,right = block[...,:h].copy(),block[...,h:].copy()
        block[...,:h],block[...,h:] = left+right,left-right
    return y*np.float32(128**-.5)


def e8p_decode(codes, table):
    c = np.asarray(codes,dtype=np.uint16)
    signs = c & 255
    parity = np.zeros(c.shape,np.uint16)
    for i in range(8):
        parity ^= (signs >> i) & 1
    adjusted = signs ^ parity
    negative = ((adjusted[...,None] >> SIGN_BITS) & 1).astype(np.int16)
    shift = (1-2*parity.astype(np.int16))[...,None]
    return (table[c >> 8].astype(np.int16)*(1-2*negative)+shift).astype(np.int8)


@dataclass(frozen=True)
class Weights:
    kind: str
    indices: np.ndarray
    table: np.ndarray
    scales: np.ndarray
    signs: np.ndarray
    patches: np.ndarray | None = None  # [N,K/128] U16: valid:1, signed value:8, position:7

    def validate(self):
        d,dtype = {'vq4':(4,np.uint8),'e8p':(8,np.uint16)}.get(self.kind,(0,None))
        if not d or self.indices.ndim != 2 or min(self.indices.shape) <= 0 or self.indices.dtype != dtype:
            raise ValueError('Invalid vector indices')
        n,k = self.indices.shape[0],self.indices.shape[1]*d
        if k % 128 or self.table.shape != (256,d) or self.table.dtype != np.int8:
            raise ValueError('Invalid vector table/geometry')
        if self.kind == 'e8p' and (np.abs(self.table.astype(np.int16)).max() > 126 or (self.table.astype(np.int16)%2).any()):
            raise ValueError('E8P basis must be even signed integers without shift overflow')
        if self.scales.dtype != np.float16 or self.scales.shape != (n,) or not np.isfinite(self.scales).all() or (self.scales <= 0).any():
            raise ValueError('Invalid vector row scales')
        if self.signs.dtype != np.int8 or self.signs.shape not in ((0,),(k,)) or (len(self.signs) and not np.isin(self.signs,[-1,1]).all()):
            raise ValueError('Invalid vector rotation signs')
        if self.patches is not None and (self.patches.dtype != np.uint16 or self.patches.shape != (n,k//128)):
            raise ValueError('Invalid integer patch slots')
        return n,k

    @property
    def nbytes(self):
        return sum(a.nbytes for a in (self.indices,self.table,self.scales,self.signs,self.patches) if a is not None)

    def integer_weights(self):
        n,k = self.validate()
        q = self.table[self.indices] if self.kind == 'vq4' else e8p_decode(self.indices,self.table)
        q = q.reshape(n,k).copy()
        if self.patches is not None:
            p = self.patches
            rows,groups = np.where((p & 32768) != 0)
            positions = groups*128+(p[rows,groups]&127)
            values = ((p[rows,groups] >> 7)&255).astype(np.uint8).view(np.int8)
            q[rows,positions] = values
        return q

    def dequantize(self):
        return self.integer_weights().astype(np.float32)*self.scales.astype(np.float32)[:,None]

    def gpu_layout(self):
        n,k = self.validate()
        d = 4 if self.kind == 'vq4' else 8
        p = np.ascontiguousarray(self.indices.reshape(n,k//128,128//d).transpose(1,0,2))
        table = np.ascontiguousarray(self.table).view('<u4').reshape(256,d//4)
        return p,table,self.scales.copy()


def save(path,w,provenance=None):
    shape = w.validate()
    path = Path(path)
    with path.open('xb'):
        pass
    try:
        tensors = {name:getattr(w,name) for name in ('indices','table','scales','signs')}
        if w.patches is not None:tensors['patches'] = w.patches
        save_file(tensors,str(path),
                  metadata={'format':'orinfer.integer_vq.v1','kind':w.kind,'shape':json.dumps(shape),
                            'provenance':json.dumps(provenance or {},sort_keys=True)})
    except BaseException:
        path.unlink()
        raise


def load(path):
    with safe_open(path,framework='np') as f:
        m = f.metadata() or {}
        required = {'indices','table','scales','signs'}
        if m.get('format') != 'orinfer.integer_vq.v1' or set(f.keys()) not in (required,required|{'patches'}):
            raise ValueError('Unsupported vector fixture')
        w = Weights(m.get('kind',''),*[f.get_tensor(n) for n in ('indices','table','scales','signs')],
                    f.get_tensor('patches') if 'patches' in f.keys() else None)
        if list(w.validate()) != json.loads(m.get('shape','null')):
            raise ValueError('Vector fixture shape mismatch')
        return w
