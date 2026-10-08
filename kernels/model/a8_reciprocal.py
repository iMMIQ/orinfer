"""A8 candidate: row reciprocal, strict division close to RNE half ties.

FP16 inputs/output-scale and SwiGLU materialization boundary are unchanged.
For the actual per-row |ratio|<=128, FP32 reciprocal/multiply error is much
smaller than the 2^-14 tie guard. Test the code contract independently before
selecting this export; no default operator or model quality policy changes.
"""

import tilelang.language as T
from kernels.operators.op04_swiglu import swiglu_fp16
from tools.operators.common import orin_jit


@orin_jit
def activation_quantization_reciprocal(K: int, fused=False, threads=256):
    assert K > 0 and threads in (128, 256, 512)
    rows = T.dynamic("rows")
    tile = T.ceildiv(K, threads) * threads

    @T.prim_func
    def main(
        X: T.Tensor((rows, 2 * K if fused else K), T.float16),
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
                    value[j] = (
                        T.cast(swiglu_fp16(X[row, j], X[row, K + j]), T.float32)
                        if fused
                        else T.cast(X[row, j], T.float32)
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
                    corrected = T.if_then_else(
                        T.abs(ratio - T.round(ratio)) >= 0.5 - 2**-14,
                        T.call_extern(
                            "float32", "__fdiv_rn", value[j], T.cast(scale[0], T.float32)
                        ),
                        ratio,
                    )
                    Q[row, j] = T.cast(T.max(-127.0, T.min(127.0, T.round(corrected))), T.int8)

    return main
