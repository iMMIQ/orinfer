"""INT8 GDN input projection with separate tight QKV/Z epilogues.

Weights are per-call W8 workspace expanded from resident W4. A8 and row/group
scale preparation are explicit neighboring kernels; output is FP16.
"""

import tilelang.language as T
from kernels.operators.op03_ffn_gate_up import _orin_jit


@_orin_jit
def gdn_qkvz_int8(M, BM=256, BN=128, BK=128, stages=2, threads=256, grid_order="nfirst"):
    assert grid_order in ("nfirst", "mfirst")
    gx, gy = (
        (16384 // BN, T.ceildiv(M, BM))
        if grid_order == "nfirst"
        else (T.ceildiv(M, BM), 16384 // BN)
    )

    @T.prim_func
    def kernel(
        A: T.Tensor((M, 5120), T.int8),
        B: T.Tensor((16384, 5120), T.int8),
        AS: T.Tensor((M,), T.float16),
        BS: T.Tensor((16384,), T.float16),
        QKV: T.Tensor((M, 10240), T.float16),
        ZOUT: T.Tensor((M, 6144), T.float16),
    ):
        with T.Kernel(gx, gy, threads=threads) as (blockx, blocky):
            bx = blockx if grid_order == "nfirst" else blocky
            by = blocky if grid_order == "nfirst" else blockx
            a = T.alloc_shared((BM, BK), T.int8)
            b = T.alloc_shared((BN, BK), T.int8)
            accum = T.alloc_fragment((BM, BN), T.int32)
            T.clear(accum)
            for ko in T.Pipelined(5120 // BK, num_stages=stages):
                T.copy(A[by * BM, ko * BK], a)
                T.copy(B[bx * BN, ko * BK], b)
                T.gemm(a, b, accum, transpose_B=True)
            for i, j in T.Parallel(BM, BN):
                if by * BM + i < M:
                    value = (
                        T.cast(accum[i, j], T.float32)
                        * T.cast(AS[by * BM + i], T.float32)
                        * T.cast(BS[bx * BN + j], T.float32)
                    )
                    if bx < 10240 // BN:
                        QKV[by * BM + i, bx * BN + j] = value
                    else:
                        ZOUT[by * BM + i, bx * BN + j - 10240] = value

    return kernel
