"""Shared FP16 SwiGLU rounding boundary for fused A8 projections."""

import tilelang.language as T


@T.macro
def swiglu_fp16(gate, up):
    """Fusion expression; the returned value has the required FP16 rounding.

    A downstream A8 quantizer must cast this FP16 value back to FP32 before its
    amax/scale/code calculation to match an unfused FP16 SwiGLU -> A8 path.
    """
    g = T.cast(gate, T.float32)
    u = T.cast(up, T.float32)
    e = T.exp(-T.abs(g))
    sigmoid = T.if_then_else(g >= 0, 1.0 / (1.0 + e), e / (1.0 + e))
    return T.cast((g * sigmoid) * u, T.float16)
