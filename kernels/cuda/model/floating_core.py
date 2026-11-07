"""Shared floating MMA loop with explicit operand and reduction contracts.

Grouping and operand conversion are compile-time choices. Sequential group
sums preserve HC's numerical boundary; no split-K or parallel sum is introduced.
"""

import tilelang.language as T


BF16_SOURCE = r"""
__device__ __forceinline__ float orinfer_round_bf16(float x) {
    unsigned short h;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(h) : "f"(x));
    return __uint_as_float(static_cast<unsigned>(h) << 16);
}
__device__ __forceinline__ void orinfer_store_bf16(void* p, float x) {
    unsigned short h;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(h) : "f"(x));
    *static_cast<unsigned short*>(p) = h;
}
"""


def rounded_float(value, dtype):
    """Explicit stream rounding, without a vectorized BF16 alias round-trip."""
    if dtype == "bfloat16":
        return T.call_pure_extern("float32", "orinfer_round_bf16", value)
    return T.cast(T.cast(value, dtype), T.float32)


def mma_loop(
    m,
    k,
    bm,
    bn,
    bk,
    *,
    dtype="bfloat16",
    stages=2,
    weight_left=False,
    reduction_groups=1,
    grouped_input=False,
    weight_first=False,
    cast_input=False,
):
    @T.macro
    def product(A, W, acc, by, bx, group):
        a = T.alloc_shared((bm, bk), dtype)
        w = T.alloc_shared((bn, bk), dtype)
        if reduction_groups != 1:
            total = T.alloc_fragment((bn, bm) if weight_left else (bm, bn), T.float32)
            T.clear(total)
        for branch in T.serial(reduction_groups):
            T.clear(acc)
            for kg in T.Pipelined(T.ceildiv(k // reduction_groups, bk), num_stages=stages):
                offset = branch * (k // reduction_groups) + kg * bk
                if weight_first:
                    T.copy(W[bx * bn, group * k + offset], w)
                if cast_input:
                    for i, j in T.Parallel(bm, bk, coalesced_width=T.int32(1)):
                        a[i, j] = T.if_then_else(by * bm + i < m, A[by * bm + i, offset + j], 0.0)
                elif grouped_input:
                    for i, j in T.Parallel(bm, bk):
                        a[i, j] = 0
                        if by * bm + i < m:
                            a[i, j] = A[by * bm + i, group, offset + j]
                else:
                    T.copy(A[by * bm, offset], a)
                if not weight_first:
                    T.copy(W[bx * bn, group * k + offset], w)
                if weight_left:
                    T.gemm(w, a, acc, transpose_B=True)
                else:
                    T.gemm(a, w, acc, transpose_B=True)
            if reduction_groups != 1:
                for i, j in T.Parallel(bm, bn):
                    if weight_left:
                        total[j, i] += acc[j, i]
                    else:
                        total[i, j] += acc[i, j]
        if reduction_groups != 1:
            T.copy(total, acc)

    return product
