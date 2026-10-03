"""Untied full-vocabulary W4A16 LM head, SM87, explicit FP32 logits.

Offline group-asymmetric quantization is a separate runner concern. The physical
format and register half2 decoder are shared with op03; no full expanded weight,
activation quantization, selection or normalization is hidden in this operator.
"""
import tilelang.language as T
from kernels.operators.op03_ffn_gate_up import _orin_jit, _PAIR_SOURCE

VOCAB = 248320
HIDDEN = 5120


@_orin_jit
def lm_head(M, N: int = VOCAB, K: int = HIDDEN,
            implementation: str = "register", BM: int = 16,
            BN: int = 64, BK: int = 128, stages: int = 2,
            threads: int = 128):
    """Build kernel(A,P,S,Z,Logits,stream=...), dynamic M allowed.

    A[M,K] half, P[N,K/2] adjacent low/high U4, S[N,K/128] half,
    Z[N,K/128] numeric int8 0..15, Logits[M,N] FP32. The reference
    weight is half(half(q-z)*s); FP32 accumulation is stored without an
    extra FP16 output rounding. M is scoring rows, not prompt length.
    """
    assert implementation in ("register", "shared")
    assert K % 128 == 0 and K % BK == 0 and BK == 128
    assert N % BN == 0 and BM >= 16
    register = implementation == "register"

    @T.prim_func
    def kernel(A: T.Tensor((M, K), T.float16),
               P: T.Tensor((N, K // 2), T.uint8),
               S: T.Tensor((N, K // 128), T.float16),
               Z: T.Tensor((N, K // 128), T.int8),
               Logits: T.Tensor((M, N), T.float32)):
        with T.Kernel(N // BN, T.ceildiv(M, BM), threads=threads) as (bx, by):
            T.import_source(_PAIR_SOURCE)
            a = T.alloc_shared((BM, BK), T.float16)
            b = T.alloc_shared((BN, BK), T.float16)
            packed = T.alloc_fragment((BN, BK // 2), T.uint8) if register else T.alloc_shared((BN, BK // 2), T.uint8)
            scale = T.alloc_fragment((BN,), T.float16) if register else T.alloc_shared((BN,), T.float16)
            zero = T.alloc_fragment((BN,), T.int8) if register else T.alloc_shared((BN,), T.int8)
            accum = T.alloc_fragment((BM, BN), T.float32)
            T.clear(accum)
            for ko in T.Pipelined(K // BK, num_stages=stages):
                T.copy(A[by * BM, ko * BK], a)
                T.copy(P[bx * BN, ko * BK // 2], packed)
                for i in T.Parallel(BN):
                    scale[i] = S[bx * BN + i, ko]
                    zero[i] = Z[bx * BN + i, ko]
                for i, j in T.Parallel(BN, BK // 2):
                    pair = T.call_pure_extern("uint32", "op03_deq_pair", packed[i, j], scale[i], zero[i])
                    b[i, j * 2] = T.reinterpret(T.float16, T.cast(pair & 65535, T.uint16))
                    b[i, j * 2 + 1] = T.reinterpret(T.float16, T.cast(pair >> 16, T.uint16))
                T.gemm(a, b, accum, transpose_B=True)
            T.copy(accum, Logits[by * BM, bx * BN])
    return kernel
