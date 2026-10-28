"""Offline BF16 high/residual parts of finite FP32 projection operands."""

import numpy as np


def bf16_parts(values):
    residual = np.asarray(values, dtype=np.float32).copy()
    if not np.isfinite(residual).all() or np.abs(residual).max(initial=0) > 1e30:
        raise ValueError("Floating projection operands must be finite and within the BF16 range")
    parts = []
    for _ in range(3):
        bits = residual.view(np.uint32)
        rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
        restored = (rounded.astype(np.uint32) << 16).view(np.float32)
        parts.append(rounded)
        residual -= restored
    if np.any(residual != 0):
        raise ValueError("FP32 weight cannot be represented exactly by three BF16 parts")
    return np.stack(parts)
