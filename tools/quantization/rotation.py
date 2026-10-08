"""Offline signed Hadamard transforms for the Flash Next fixture dimensions.

Full dimensions use the Kronecker product of a Paley order20 Hadamard
and a power-of-two Hadamard, avoiding padding for 640/1280/2560 channels.
The order20 factor is nonsymmetric, so inverse and forward differ.
"""

import numpy as np

from tools.quantization.vq import rotate


def hadamard20():
    residues = {i * i % 19 for i in range(1, 19)}
    h = np.ones((20, 20), np.float32)
    h[1:, 0] = -1
    for i in range(19):
        for j in range(19):
            delta = (i - j) % 19
            h[i + 1, j + 1] = 1 if delta == 0 or delta in residues else -1
    return h


def _walsh(x):
    y = x.copy()
    size = y.shape[-1]
    for stage in range(size.bit_length() - 1):
        half = 1 << stage
        view = y.reshape(*y.shape[:-1], -1, 2 * half)
        left, right = view[..., :half].copy(), view[..., half:].copy()
        view[..., :half], view[..., half:] = left + right, left - right
    return y * np.float32(size**-0.5)


def transform(x, signs, *, mode="full", inverse=False):
    x = np.asarray(x, dtype=np.float32)
    s = np.asarray(signs)
    if not x.ndim or s.shape != (x.shape[-1],) or not np.isin(s, [-1, 1]).all():
        raise ValueError("Transform requires one +/-1 sign per channel")
    if mode == "block128":
        return rotate(x, np.ones_like(s)) * s if inverse else rotate(x, s)
    size = x.shape[-1] // 20
    if mode != "full" or x.shape[-1] % 20 or size <= 0 or size & (size - 1):
        raise ValueError("Full transform dimension must be 20 times a power of two")
    h = hadamard20() * np.float32(20**-0.5)
    if inverse:
        y = np.einsum("ji,...jp->...ip", h, x.reshape(*x.shape[:-1], 20, size))
        return _walsh(y).reshape(x.shape) * s
    y = _walsh((x * s).reshape(*x.shape[:-1], 20, size))
    return np.einsum("ij,...jp->...ip", h, y).reshape(x.shape)
