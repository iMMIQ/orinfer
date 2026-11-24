"""Portable dense W8A8 Tensor Core projection for non-Orin discrete CUDA GPUs.

Matches Orinfer's exact `int8-row` representation and FP32 rescaling order
`(float32(acc) * float32(WS[n])) * float32(AS[m])` via standard portable
TileLang `T.gemm` (`int8 x int8 -> int32`).
"""

import tilelang.language as T

from kernels.shared.rows import row_count
from orinfer_build.compiler.cuda.common import cuda_jit


@cuda_jit
def w8a8_gemm(
    M,
    N,
    K,
    *,
    out_dtype="float16",
    block_m=64,
    block_n=64,
    block_k=128,
    num_stages=2,
    dynamic_rows=False,
):
    if (
        any(type(v) is not int or v <= 0 for v in (M, N, K))
        or K % block_k
        or block_m not in (16, 32, 64, 128)
        or block_n not in (32, 64, 128)
        or block_k not in (64, 128)
        or num_stages not in (1, 2)
        or out_dtype not in ("float16", "bfloat16", "float32")
        or type(dynamic_rows) is not bool
    ):
        raise ValueError("Invalid portable W8A8 GEMM geometry")

    rows_sym = row_count(M, dynamic_rows)

    @T.prim_func
    def main(
        A: T.Tensor((rows_sym, K), T.int8),
        W: T.Tensor((N, K), T.int8),
        WS: T.Tensor((N,), T.float16),
        AS: T.Tensor((rows_sym,), T.float16),
        Out: T.Tensor((rows_sym, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_n), T.ceildiv(rows_sym, block_m), threads=128) as (
            bx,
            by,
        ):
            A_s = T.alloc_shared((block_m, block_k), T.int8)
            W_s = T.alloc_shared((block_n, block_k), T.int8)
            C_f = T.alloc_fragment((block_m, block_n), T.int32)
            C_s = T.alloc_shared((block_m, block_n), T.int32)
            AS_s = T.alloc_shared((block_m,), T.float32)
            WS_s = T.alloc_shared((block_n,), T.float32)

            T.clear(C_f)
            for kb in T.Pipelined(K // block_k, num_stages=num_stages):
                T.copy(A[by * block_m, kb * block_k], A_s)
                T.copy(W[bx * block_n, kb * block_k], W_s)
                T.gemm(A_s, W_s, C_f, transpose_B=True)

            T.copy(C_f, C_s)
            for i in T.Parallel(block_m):
                r = by * block_m + i
                AS_s[i] = T.if_then_else(r < rows_sym, T.cast(AS[r], T.float32), 0.0)
            for j in T.Parallel(block_n):
                col = bx * block_n + j
                WS_s[j] = T.if_then_else(col < N, T.cast(WS[col], T.float32), 0.0)
            for i, j in T.Parallel(block_m, block_n):
                if by * block_m + i < rows_sym and bx * block_n + j < N:
                    Out[by * block_m + i, bx * block_n + j] = T.cast(
                        (T.cast(C_s[i, j], T.float32) * WS_s[j]) * AS_s[i], out_dtype
                    )

    return main
