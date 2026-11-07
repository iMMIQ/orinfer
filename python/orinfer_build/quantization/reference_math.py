"""Floating/A8 numerical oracles shared by packed-weight validation tools."""

import numpy as np


def a8(x, group=None):
    """Match the online FP16 boundary, FP16 scale and round-to-nearest-even."""
    x = np.asarray(x).astype(np.float16).astype(np.float32)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("A8 inputs must be finite FP16-representable matrices")
    group = x.shape[1] if group is None else group
    if type(group) is not int or group <= 0 or x.shape[1] % group:
        raise ValueError("A8 group must divide the channel dimension")
    grouped = x.reshape(len(x), -1, group)
    maximum = np.abs(grouped).max(-1)
    scale = np.where(maximum > 0, np.maximum(maximum / 127, 2**-24), 1).astype(np.float16)
    code = np.clip(np.rint(grouped / scale.astype(np.float32)[..., None]), -127, 127).astype(
        np.int8
    )
    return code.reshape(x.shape), scale


def a8_value(x, group=None):
    q, s = a8(x, group)
    return (
        q.reshape(len(q), s.shape[1], -1).astype(np.float32) * s.astype(np.float32)[..., None]
    ).reshape(q.shape)


def swiglu(gu):
    g, u = np.split(np.asarray(gu, dtype=np.float32), 2, axis=-1)
    e = np.exp(-np.abs(g))
    sigmoid = np.where(g >= 0, 1 / (1 + e), e / (1 + e))
    return (g * sigmoid) * u


def floating_ffn(x, gate_up, down, *, activation_group=None):
    if activation_group is None:
        return swiglu(x.astype(np.float32) @ gate_up.T) @ down.T
    # This floating oracle models activation rounding separately; it does not
    # certify native-kernel performance or bitwise accumulation order.
    group = None if activation_group == "row" else activation_group
    gu = (a8_value(x, group) @ gate_up.T).astype(np.float16)
    return (a8_value(swiglu(gu), group) @ down.T).astype(np.float16).astype(np.float32)
