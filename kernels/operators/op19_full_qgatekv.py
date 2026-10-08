"""SM87 W4A16 full-attention projection with op20-compatible tight output.

A[M,5120], adjacent-U4 P[14336,2560], FP16 S[14336,40], numeric
int8 Z[14336,40] -> X[M,14336] FP16. First 12288 output channels
are 24 heads each (Q256, raw_gate256), followed by K1024 and V1024.
No sigmoid, output shuffle, allocation, or expanded global weight workspace.
"""

from kernels.operators.op03_ffn_gate_up import ffn_gate_up

K_HIDDEN, N_QGATEKV = 5120, 14336


def full_qgatekv(M, implementation="register", BM=16, BN=64, BK=128, stages=2, threads=128):
    """Build (A,P,S,Z,X), explicit output, optional explicit CUDA stream.

    M may be TileLang T.dynamic('M'); current stream is resolved per call.
    Reuses op03's real TileLang shared/register dequantization+FP32 MMA.
    Physical checkpoint q_proj rows already have the required head ordering;
    offline packer concatenates q_proj/k_proj/v_proj without permuting rows.
    Weight math is W=half((q-z)*s), accumulation FP32, output FP16.
    All buffers must be contiguous, disjoint, and graph addresses stable.
    """
    assert BM in (16, 64) and BN == 64 and BK == 128
    return ffn_gate_up(
        M,
        N=N_QGATEKV,
        K=K_HIDDEN,
        implementation=implementation,
        BM=BM,
        BN=BN,
        BK=BK,
        stages=stages,
        threads=threads,
    )
