"""Offline block-LDL factors and an independent sequential rounding oracle.

For H=L D L.T with unit block-lower L, reverse-order rounding feeds
the already rounded tail error through L. This is the BlockLDLQ principle
from https://arxiv.org/abs/2402.04396; code is independently implemented.
No codebook, runtime layout, output-row scale or activation precision is
implied by this mathematical helper.
"""

import numpy as np


def factor(covariance, *, block_size=8, damping=0.01):
    h = np.asarray(covariance, dtype=np.float64)
    if h.ndim != 2 or h.shape[0] != h.shape[1] or not len(h) or not np.isfinite(h).all():
        raise ValueError("Covariance must be a finite nonempty square matrix")
    if type(block_size) is not int or block_size <= 0 or len(h) % block_size:
        raise ValueError("Block size must divide the covariance dimension")
    if not np.isfinite(damping) or damping <= 0:
        raise ValueError("Damping must be finite and positive")
    if not np.allclose(h, h.T, rtol=1e-8, atol=1e-10):
        raise ValueError("Covariance must be symmetric")
    mean = float(np.trace(h) / len(h))
    if mean <= 0:
        raise ValueError("Covariance needs positive average diagonal")
    regularized = (h + h.T) * (0.5 / mean)
    regularized.flat[:: len(h) + 1] += damping
    cholesky = np.linalg.cholesky(regularized)
    diagonal = np.stack(
        [cholesky[j : j + block_size, j : j + block_size] for j in range(0, len(h), block_size)]
    )
    lower = np.einsum(
        "nbi,bij->nbj", cholesky.reshape(len(h), -1, block_size), np.linalg.inv(diagonal)
    ).reshape(h.shape)
    blocks = diagonal @ diagonal.transpose(0, 2, 1)
    return regularized, lower, blocks


def round_blocks(weight, lower, quantizer, *, block_size=8):
    """Small-shape CPU oracle; quantizer returns reconstructed [N,block] values."""
    w, l = np.asarray(weight, dtype=np.float64), np.asarray(lower, dtype=np.float64)
    if (
        w.ndim != 2
        or l.shape != (w.shape[1], w.shape[1])
        or not np.isfinite(w).all()
        or not np.isfinite(l).all()
    ):
        raise ValueError("Weight and factor geometry must match and be finite")
    if type(block_size) is not int or block_size <= 0 or w.shape[1] % block_size:
        raise ValueError("Block size must divide the weight dimension")
    q = np.empty_like(w)
    for start in range(w.shape[1] - block_size, -1, -block_size):
        end = start + block_size
        target = w[:, start:end] + (w[:, end:] - q[:, end:]) @ l[end:, start:end]
        rounded = np.asarray(quantizer(target), dtype=np.float64)
        if rounded.shape != target.shape or not np.isfinite(rounded).all():
            raise ValueError("Quantizer must return finite reconstructed blocks")
        q[:, start:end] = rounded
    return q
