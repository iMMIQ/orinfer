"""Signed block128 rotation and row A8 with exact FP16 materialization boundaries."""

import tilelang.language as T
from tools.operators.common import orin_jit
from kernels.model.rows import row_count


@orin_jit
def rotate_activation_a8(M: int, K: int, swiglu=False, reciprocal=True, dynamic_rows: bool = False):
    assert M > 0 and K % 128 == 0
    width = 2 * K if swiglu else K
    M = row_count(M, dynamic_rows)

    @T.prim_func
    def main(
        X: T.Tensor((M, width), T.float16),
        Signs: T.Tensor((K,), T.int8),
        Q: T.Tensor((M, K), T.int8),
        S: T.Tensor((M, 1), T.float16),
    ):
        with T.Kernel(M, threads=128) as row:
            current = T.alloc_shared((K,), T.float32)
            temporary = T.alloc_shared((K,), T.float32)
            values = T.alloc_fragment((K,), T.float32)
            absolute = T.alloc_fragment((K,), T.float32)
            maximum = T.alloc_fragment((1,), T.float32)
            scale = T.alloc_fragment((1,), T.float16)
            inv = T.alloc_fragment((1,), T.float32)
            for k in T.Parallel(K, coalesced_width=T.int32(1)):
                if swiglu:
                    gate = T.cast(X[row, k], T.float32)
                    e = T.exp(-T.abs(gate))
                    sigmoid = T.if_then_else(gate >= 0, 1 / (1 + e), e / (1 + e))
                    value = T.cast(
                        T.cast((gate * sigmoid) * T.cast(X[row, k + K], T.float32), T.float16),
                        T.float32,
                    )
                    current[k] = value * T.cast(Signs[k], T.float32)
                else:
                    current[k] = T.cast(X[row, k], T.float32) * T.cast(Signs[k], T.float32)
            T.sync_threads()
            for stage in T.unroll(7):
                for k in T.Parallel(K):
                    temporary[k] = current[k ^ (1 << stage)] + current[k] * (
                        1 - 2 * ((k >> stage) & 1)
                    )
                T.sync_threads()
                T.copy(temporary, current)
                T.sync_threads()
            for k in T.Parallel(K):
                values[k] = T.cast(T.cast(current[k] * 0.08838834764831845, T.float16), T.float32)
                absolute[k] = T.abs(values[k])
            T.reduce_max(absolute, maximum, dim=0)
            scale[0] = T.if_then_else(
                maximum[0] > 0,
                T.max(T.call_extern("float32", "__fdiv_rn", maximum[0], 127.0), 2**-24),
                1.0,
            )
            S[row, 0] = scale[0]
            if reciprocal:
                inv[0] = T.call_extern("float32", "__fdiv_rn", 1.0, T.cast(scale[0], T.float32))
            for k in T.Parallel(K):
                if reciprocal:
                    ratio = values[k] * inv[0]
                    ratio = T.if_then_else(
                        T.abs(ratio - T.round(ratio)) >= 0.5 - 2**-14,
                        T.call_extern(
                            "float32", "__fdiv_rn", values[k], T.cast(scale[0], T.float32)
                        ),
                        ratio,
                    )
                else:
                    ratio = T.call_extern(
                        "float32", "__fdiv_rn", values[k], T.cast(scale[0], T.float32)
                    )
                Q[row, k] = T.cast(T.max(-127.0, T.min(127.0, T.round(ratio))), T.int8)

    return main
