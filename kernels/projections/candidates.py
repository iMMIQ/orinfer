"""SM87 projection candidates adapted from local prefill_layout/kernels.py.
Logical weights P[N,K/2] are adjacent low/high U4 pairs; scales and unsigned
zeros are N x K/128. Dynamic M masks load/store tails; temporary W8 is per call.
"""

from functools import wraps
import torch
import tilelang
import tilelang.language as T


def orin_jit(function):
    compiled = tilelang.jit(
        out_idx=[], execution_backend="nvrtc", target={"kind": "cuda", "arch": "sm_87"}
    )(function)

    @wraps(function)
    def build(*args, **kwargs):
        kernel = compiled(*args, **kwargs)
        kernel.adapter.kernels = dict(kernel.adapter.kernels)

        def launch(*inputs, stream=None):
            if stream is None:
                stream = torch.cuda.current_stream(inputs[0].device).cuda_stream
            return kernel.adapter.func(*inputs, stream=stream)

        kernel.torch_function = launch
        return kernel

    return build


PAIR_SOURCE = r"""
#include <cuda_fp16.h>
#include <tl_templates/cuda/instruction/mma.h>
__device__ __forceinline__ unsigned int deq_u4_pair(unsigned char x, half_t scale,
                                                  signed char zero) {
    unsigned int bits = 0x64006400u | (x & 15u) | ((x & 240u) << 12);
    __half2 values = *reinterpret_cast<__half2*>(&bits);
    __half2 offset = __half2half2(__int2half_rn(1024 + int(zero)));
    __half native_scale = *reinterpret_cast<__half*>(&scale);
    __half2 result = __hmul2(__hsub2(values, offset), __half2half2(native_scale));
    return *reinterpret_cast<unsigned int*>(&result);
}
"""


@orin_jit
def fp16_gemm(
    M: int,
    N: int,
    K: int,
    BM: int = 64,
    BN: int = 64,
    BK: int = 64,
    stages: int = 3,
    threads: int = 128,
):
    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), T.float16),
        B: T.Tensor((N, K), T.float16),
        C: T.Tensor((M, N), T.float16),
    ):
        with T.Kernel(N // BN, T.ceildiv(M, BM), threads=threads) as (bx, by):
            a = T.alloc_shared((BM, BK), T.float16)
            b = T.alloc_shared((BN, BK), T.float16)
            accum = T.alloc_fragment((BM, BN), T.float32)
            T.clear(accum)
            for ko in T.Pipelined(K // BK, num_stages=stages):
                T.copy(A[by * BM, ko * BK], a)
                T.copy(B[bx * BN, ko * BK], b)
                T.gemm(a, b, accum, transpose_B=True)
            T.copy(accum, C[by * BM, bx * BN])

    return kernel


@orin_jit
def int8_gemm(
    M: int,
    N: int,
    K: int,
    BM: int = 128,
    BN: int = 128,
    BK: int = 128,
    stages: int = 3,
    threads: int = 256,
    min_blocks: int = 1,
    grid_order: str = "nfirst",
    cache_policy: str = "default",
):
    """Compute ceiling: per-channel W8 permits scaling only in the epilogue."""
    assert grid_order in (
        "nfirst",
        "mfirst",
        "grouped4",
        "grouped8",
        "n1m2",
        "n1m4",
        "n1m8",
        "n2m2",
        "n2m4",
    )
    assert cache_policy in (
        "default",
        "a-last-b-first",
        "a-last",
        "b-first",
        "b-last",
        "a-first-b-last",
    )
    grouped = grid_order in ("grouped4", "grouped8", "n1m2", "n1m4", "n1m8", "n2m2", "n2m4")
    nm, nn = (M + BM - 1) // BM, N // BN
    gn = 1 if grid_order.startswith("n1m") else 2 if grid_order.startswith("n2m") else 4
    requested_m = (
        int(grid_order[-1])
        if grid_order.startswith(("n1m", "n2m"))
        else 4
        if grid_order == "grouped4"
        else 8
    )
    gm = min(requested_m, nm) if grouped else 1
    if grouped:
        assert nn % gn == 0
    gx, gy = (nn * nm, 1) if grouped else (nn, nm) if grid_order == "nfirst" else (nm, nn)

    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), T.int8),
        B: T.Tensor((N, K), T.int8),
        AS: T.Tensor((M,), T.float16),
        BS: T.Tensor((N,), T.float16),
        C: T.Tensor((M, N), T.float16),
    ):
        with T.Kernel(gx, gy, threads=threads) as (blockx, blocky):
            # The final M group can contain fewer tiles. Keep every valid
            # (M,N) tile exactly once, including non-aligned shape tails.
            if grouped:
                first_m = (blockx // (nn * gm)) * gm
                active_m = T.min(nm - first_m, gm)
                within = blockx % (nn * gm)
                bx = (within // (gn * active_m)) * gn + within % gn
                by = first_m + (within // gn) % active_m
            else:
                bx = blockx if grid_order == "nfirst" else blocky
                by = blocky if grid_order == "nfirst" else blockx
            T.annotate_min_blocks_per_sm(min_blocks)
            a = T.alloc_shared((BM, BK), T.int8)
            b = T.alloc_shared((BN, BK), T.int8)
            accum = T.alloc_fragment((BM, BN), T.int32)
            T.clear(accum)
            for ko in T.Pipelined(K // BK, num_stages=stages):
                T.copy(
                    A[by * BM, ko * BK],
                    a,
                    eviction_policy="evict_last"
                    if cache_policy in ("a-last", "a-last-b-first")
                    else "evict_first"
                    if cache_policy == "a-first-b-last"
                    else None,
                )
                T.copy(
                    B[bx * BN, ko * BK],
                    b,
                    eviction_policy="evict_first"
                    if cache_policy in ("b-first", "a-last-b-first")
                    else "evict_last"
                    if cache_policy in ("b-last", "a-first-b-last")
                    else None,
                )
                T.gemm(a, b, accum, transpose_B=True)
            for i, j in T.Parallel(BM, BN):
                if by * BM + i < M:
                    C[by * BM + i, bx * BN + j] = (
                        T.cast(accum[i, j], T.float32)
                        * T.cast(AS[by * BM + i], T.float32)
                        * T.cast(BS[bx * BN + j], T.float32)
                    )

    return kernel


@orin_jit
def w4_splitk(M: int, N: int, K: int, SPLIT: int = 8, BM: int = 16, BN: int = 64, BK: int = 128):
    assert K % (BK * SPLIT) == 0

    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), T.float16),
        P: T.Tensor((N, K // 2), T.uint8),
        S: T.Tensor((N, K // 128), T.float16),
        Z: T.Tensor((N, K // 128), T.int8),
        O: T.Tensor((SPLIT, M, N), T.float32),
    ):
        with T.Kernel(N // BN, T.ceildiv(M, BM), SPLIT, threads=128) as (bx, by, sk):
            T.import_source(PAIR_SOURCE)
            a = T.alloc_shared((BM, BK), T.float16)
            p = T.alloc_shared((BN, BK // 2), T.uint8)
            b = T.alloc_shared((BN, BK), T.float16)
            scale = T.alloc_shared((BN,), T.float16)
            zero = T.alloc_shared((BN,), T.int8)
            accum = T.alloc_fragment((BM, BN), T.float32)
            T.clear(accum)
            for ki in T.Pipelined(K // BK // SPLIT, num_stages=2):
                ko = sk * (K // BK // SPLIT) + ki
                T.copy(A[by * BM, ko * BK], a)
                T.copy(P[bx * BN, ko * (BK // 2)], p)
                for i in T.Parallel(BN):
                    scale[i] = S[bx * BN + i, ko]
                    zero[i] = Z[bx * BN + i, ko]
                for i, j in T.Parallel(BN, BK // 2):
                    pair = T.call_pure_extern("uint32", "deq_u4_pair", p[i, j], scale[i], zero[i])
                    b[i, j * 2] = T.reinterpret(T.float16, T.cast(pair & 65535, T.uint16))
                    b[i, j * 2 + 1] = T.reinterpret(T.float16, T.cast(pair >> 16, T.uint16))
                T.gemm(a, b, accum, transpose_B=True)
            T.copy(accum, O[sk, by * BM, bx * BN])

    return kernel


@orin_jit
def splitk_reduce(M: int, N: int, SPLIT: int = 8):
    @T.prim_func
    def kernel(P: T.Tensor((SPLIT, M, N), T.float32), O: T.Tensor((M, N), T.float16)):
        with T.Kernel(T.ceildiv(M * N, 256), threads=256) as bx:
            thread = T.get_thread_binding()
            index = bx * 256 + thread
            accum = T.alloc_local((1,), T.float32)
            accum[0] = 0
            if index < M * N:
                for sk in T.serial(SPLIT):
                    accum[0] += P[sk, index // N, index % N]
                O[index // N, index % N] = accum[0]

    return kernel


@orin_jit
def expand_weight_q8_inline_lut(N: int, K: int, BN: int = 64, BK: int = 256):
    """Adapt mature Marlin LUT method to explicitly checked logical NK U4 ABI."""

    @T.prim_func
    def kernel(
        P: T.Tensor((N, K // 2), T.uint8),
        S: T.Tensor((N, K // 128), T.float16),
        Z: T.Tensor((N, K // 128), T.int8),
        BS: T.Tensor((N,), T.float16),
        B: T.Tensor((N, K), T.int8),
    ):
        with T.Kernel(N // BN, K // BK, threads=256) as (bx, bk):
            packed = T.alloc_shared((BN, BK // 2), T.uint8)
            scale = T.alloc_shared((BN, BK // 128), T.float16)
            zero = T.alloc_shared((BN, BK // 128), T.int8)
            rowscale = T.alloc_shared((BN,), T.float16)
            table = T.alloc_shared((BK // 128, BN, 4), T.int32)
            T.copy(P[bx * BN, bk * (BK // 2)], packed)
            T.copy(S[bx * BN, bk * (BK // 128)], scale)
            T.copy(Z[bx * BN, bk * (BK // 128)], zero)
            T.copy(BS[bx * BN], rowscale)
            for g, i, word_idx in T.Parallel(BK // 128, BN, 4):
                word = T.alloc_var(T.int32)
                word = 0
                for c in T.unroll(4):
                    weight = T.cast(
                        T.cast(word_idx * 4 + c - T.cast(zero[i, g], T.int32), T.float16)
                        * scale[i, g],
                        T.float32,
                    )
                    code8 = T.cast(
                        T.min(
                            127.0, T.max(-127.0, T.round(weight / T.cast(rowscale[i], T.float32)))
                        ),
                        T.int32,
                    )
                    word = word | ((code8 & 255) << (c * 8))
                table[g, i, word_idx] = word
            for i, j in T.Parallel(BN, BK):
                code4 = (T.cast(packed[i, j // 2], T.int32) >> ((j % 2) * 4)) & 15
                word8 = table[j // 128, i, code4 // 4]
                B[bx * BN + i, bk * BK + j] = T.cast((word8 >> ((code4 % 4) * 8)) & 255, T.int8)

    return kernel
