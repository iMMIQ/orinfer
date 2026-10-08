"""Experimental FP16-domain FP32 swish lookup and fused A8.

The immutable 65536-entry table is produced with the same TileLang expression
as op04; it costs 256KiB and must be accounted for if ever selected by a model.
This candidate is not registered in the production builder.
"""

import tilelang.language as T
from tools.operators.common import orin_jit


@orin_jit
def initialize_swish_lut():
    @T.prim_func
    def main(Codes: T.Tensor((65536,), T.int32), LUT: T.Tensor((65536,), T.float32)):
        with T.Kernel(256, threads=256) as bx:
            for j in T.Parallel(256):
                code = bx * 256 + j
                g = T.cast(T.reinterpret(T.float16, T.cast(Codes[code], T.uint16)), T.float32)
                e = T.exp(-T.abs(g))
                sigmoid = T.if_then_else(g >= 0, 1.0 / (1.0 + e), e / (1.0 + e))
                LUT[code] = g * sigmoid

    return main


@orin_jit
def swiglu_lut_activation_quantization(K: int, threads=512, guarded=True):
    assert K > 0 and threads in (128, 256, 512)
    rows = T.dynamic("rows")
    tile = T.ceildiv(K, threads) * threads

    @T.prim_func
    def main(
        X: T.Tensor((rows, 2 * K), T.float16),
        LUT: T.Tensor((65536,), T.float32),
        Q: T.Tensor((rows, K), T.int8),
        S: T.Tensor((rows,), T.float16),
    ):
        with T.Kernel(rows, threads=threads) as row:
            value = T.alloc_fragment((tile,), T.float32)
            absolute = T.alloc_fragment((tile,), T.float32)
            maximum = T.alloc_fragment((1,), T.float32)
            scale = T.alloc_fragment((1,), T.float16)
            reciprocal = T.alloc_fragment((1,), T.float32)
            for j in T.Parallel(tile):
                value[j] = 0.0
                if j < K:
                    code = T.cast(T.reinterpret(T.uint16, X[row, j]), T.int32)
                    value[j] = T.cast(
                        T.cast(LUT[code] * T.cast(X[row, K + j], T.float32), T.float16), T.float32
                    )
                absolute[j] = T.abs(value[j])
            T.reduce_max(absolute, maximum, dim=0)
            scale[0] = T.if_then_else(
                maximum[0] > 0.0,
                T.max(T.call_extern("float32", "__fdiv_rn", maximum[0], 127.0), 2**-24),
                1.0,
            )
            reciprocal[0] = T.call_extern("float32", "__fdiv_rn", 1.0, T.cast(scale[0], T.float32))
            S[row] = scale[0]
            for j in T.Parallel(tile):
                if j < K:
                    ratio = value[j] * reciprocal[0]
                    corrected = (
                        T.if_then_else(
                            T.abs(ratio - T.round(ratio)) >= 0.5 - 2**-14,
                            T.call_extern(
                                "float32", "__fdiv_rn", value[j], T.cast(scale[0], T.float32)
                            ),
                            ratio,
                        )
                        if guarded
                        else ratio
                    )
                    Q[row, j] = T.cast(T.max(-127.0, T.min(127.0, T.round(corrected))), T.int8)

    return main
