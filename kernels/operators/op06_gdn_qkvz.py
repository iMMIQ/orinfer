"""SM87 W4A16 GDN projection with tight, separate QKV and Z outputs.

Losslessly repacked P[16384,2560] adjacent U4, S[16384,40] FP16,
Z[16384,40] int8 numeric zero-points. W=half((q-Z)*S), accumulation
FP32, outputs FP16. No workspace or post-projection copies.
"""

import tilelang.language as T
from kernels.operators.op03_ffn_gate_up import _orin_jit, _PAIR_SOURCE

K_HIDDEN, N_QKV, N_Z = 5120, 10240, 6144


@_orin_jit
def gdn_qkvz(
    M,
    implementation: str = "register",
    BM: int = 16,
    BN: int = 64,
    BK: int = 128,
    stages: int = 2,
    threads: int = 128,
):
    """Build (A,P,S,Z,QKV,ZOUT); M may be T.dynamic('M').

    QKV[M,10240] is tight Q2048/K2048/V6144 token-major; ZOUT[M,6144]
    is tight, viewable as [M,48,128]. QKV can view as [B,T,10240]
    for op08. All caller-owned allocations must be contiguous and disjoint.
    Explicit stream is supported; omitted stream is resolved at each call.
    Reuses op03's validated FP16 pair dequantization/JIT, not its output API.
    """
    assert implementation in ("shared", "register")
    assert BM in (16, 64) and BN == 64 and BK == 128
    register = implementation == "register"

    @T.prim_func
    def kernel(
        A: T.Tensor((M, K_HIDDEN), T.float16),
        P: T.Tensor((N_QKV + N_Z, K_HIDDEN // 2), T.uint8),
        S: T.Tensor((N_QKV + N_Z, K_HIDDEN // 128), T.float16),
        Z: T.Tensor((N_QKV + N_Z, K_HIDDEN // 128), T.int8),
        QKV: T.Tensor((M, N_QKV), T.float16),
        ZOUT: T.Tensor((M, N_Z), T.float16),
    ):
        with T.Kernel((N_QKV + N_Z) // BN, T.ceildiv(M, BM), threads=threads) as (bx, by):
            T.import_source(_PAIR_SOURCE)
            a = T.alloc_shared((BM, BK), T.float16)
            b = T.alloc_shared((BN, BK), T.float16)
            packed = (
                T.alloc_fragment((BN, BK // 2), T.uint8)
                if register
                else T.alloc_shared((BN, BK // 2), T.uint8)
            )
            scale = (
                T.alloc_fragment((BN,), T.float16) if register else T.alloc_shared((BN,), T.float16)
            )
            zero = T.alloc_fragment((BN,), T.int8) if register else T.alloc_shared((BN,), T.int8)
            accum = T.alloc_fragment((BM, BN), T.float32)
            T.clear(accum)
            for ko in T.Pipelined(K_HIDDEN // BK, num_stages=stages):
                T.copy(A[by * BM, ko * BK], a)
                T.copy(P[bx * BN, ko * BK // 2], packed)
                for i in T.Parallel(BN):
                    scale[i] = S[bx * BN + i, ko]
                    zero[i] = Z[bx * BN + i, ko]
                for i, j in T.Parallel(BN, BK // 2):
                    pair = T.call_pure_extern(
                        "uint32", "op03_deq_pair", packed[i, j], scale[i], zero[i]
                    )
                    b[i, j * 2] = T.reinterpret(T.float16, T.cast(pair & 65535, T.uint16))
                    b[i, j * 2 + 1] = T.reinterpret(T.float16, T.cast(pair >> 16, T.uint16))
                T.gemm(a, b, accum, transpose_B=True)
            if bx < N_QKV // BN:
                T.copy(accum, QKV[by * BM, bx * BN])
            else:
                T.copy(accum, ZOUT[by * BM, bx * BN - N_QKV])

    return kernel
