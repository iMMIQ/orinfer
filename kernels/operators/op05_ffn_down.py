"""W4A16 FFN down 17408->5120, explicit-output SM87 TileLang API.

Weights are P[N,K/2] low/high adjacent U4, S[N,K/128] FP16,
Z[N,K/128] numeric int8 0..15. W=half((q-z)*s). No residual is added.
The caller owns all storage and supplies the active CUDA stream per call.
"""

from dataclasses import dataclass

import tilelang.language as T

from kernels.projections.candidates import orin_jit, PAIR_SOURCE, w4_splitk
from kernels.operators.op03_ffn_gate_up import ffn_gate_up
from kernels.operators.op32_split_k_merge import split_k_merge

N_DOWN = 5120
K_INTERMEDIATE = 17408


@orin_jit
def _register_partial(M, N: int, K: int, SPLIT: int, BM: int, BN: int):
    """Packed/metadata registers -> FP16 shared B -> FP32 partial MMA.

    This preserves the existing shared split-K mapping while removing packed
    bytes and metadata shared stores/loads, as in op03's register dataflow.
    """

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
            a = T.alloc_shared((BM, 128), T.float16)
            b = T.alloc_shared((BN, 128), T.float16)
            packed = T.alloc_fragment((BN, 64), T.uint8)
            scale = T.alloc_fragment((BN,), T.float16)
            zero = T.alloc_fragment((BN,), T.int8)
            accum = T.alloc_fragment((BM, BN), T.float32)
            T.clear(accum)
            for ki in T.Pipelined(K // 128 // SPLIT, num_stages=2):
                ko = sk * (K // 128 // SPLIT) + ki
                T.copy(A[by * BM, ko * 128], a)
                T.copy(P[bx * BN, ko * 64], packed)
                for i in T.Parallel(BN):
                    scale[i] = S[bx * BN + i, ko]
                    zero[i] = Z[bx * BN + i, ko]
                for i, j in T.Parallel(BN, 64):
                    pair = T.call_pure_extern(
                        "uint32", "deq_u4_pair", packed[i, j], scale[i], zero[i]
                    )
                    b[i, j * 2] = T.reinterpret(T.float16, T.cast(pair & 65535, T.uint16))
                    b[i, j * 2 + 1] = T.reinterpret(T.float16, T.cast(pair >> 16, T.uint16))
                T.gemm(a, b, accum, transpose_B=True)
            T.copy(accum, O[sk, by * BM, bx * BN])

    return kernel


def ffn_down_partial(M, N=N_DOWN, K=K_INTERMEDIATE, SPLIT=8, implementation="shared", BM=16, BN=64):
    """Build (X,P,S,Z,partialF32), M integer or T.dynamic('M').

    Equal 128-column groups per split are required; reject uneven K/split
    instead of silently dropping a tail. Dynamic M masks row tails.
    """
    if implementation not in ("shared", "register"):
        raise ValueError("implementation must be shared or register")
    if SPLIT <= 0 or K <= 0 or K % (128 * SPLIT) or N <= 0 or N % BN or BM < 16:
        raise ValueError("positive dimensions, N%BN=0 and K%(128*SPLIT)=0 required")
    if isinstance(M, int) and M <= 0:
        raise ValueError("M must be positive")
    if implementation == "shared":
        return w4_splitk(M, N, K, SPLIT=SPLIT, BM=BM, BN=BN, BK=128)
    return _register_partial(M, N, K, SPLIT, BM, BN)


def ffn_down_merge(M, N=N_DOWN, SPLIT=8):
    """Build (partialF32,YF16), FP32 serial split sum, FP16 final cast.

    No residual argument, so it cannot be added per split accidentally.
    """
    if SPLIT <= 0 or N <= 0 or isinstance(M, int) and M <= 0:
        raise ValueError("positive dimensions and SPLIT required")
    return split_k_merge(M, N=N, SPLIT=SPLIT)


def ffn_down_full(M, implementation="register", BM=64):
    """Build (X,P,S,Z,YF16), W4 full-K FP32 MMA, no global workspace.

    Reuses op03's dimension-generic production implementation, with N/K
    explicitly rebound to down. This is the quality-preserving prefill path.
    """
    if isinstance(M, int) and M <= 0:
        raise ValueError("M must be positive")
    return ffn_gate_up(M, N=N_DOWN, K=K_INTERMEDIATE, implementation=implementation, BM=BM)


@dataclass(frozen=True)
class FFNDown:
    """Allocation-free launch plan; partial workspace is caller-owned FP32."""

    route: str
    projection: object
    merge: object = None
    splits: int = 0

    def workspace_shape(self, M):
        if M <= 0:
            raise ValueError("M must be positive")
        return (self.splits, M, N_DOWN) if self.merge is not None else None

    def __call__(self, X, P, S, Z, Y, partial=None, *, stream=None):
        # Each kernel resolves current stream at invocation if stream omitted.
        if self.merge is None:
            if partial is not None:
                raise ValueError("full route takes no partial workspace")
            self.projection(X, P, S, Z, Y, stream=stream)
        else:
            if partial is None:
                raise ValueError("split-K requires caller-owned partialF32")
            self.projection(X, P, S, Z, partial, stream=stream)
            self.merge(partial, Y, stream=stream)


def build_ffn_down(M, route="splitk_register", SPLIT=8):
    """Reusable plan, specialized or one symbolic-M kernel per route.

    Default selects the validated register split-K8 decode route. Full routes are
    the explicit BM64 prefill alternatives. No weight preparation is hidden.
    """
    if route not in ("splitk_shared", "splitk_register", "full_shared", "full_register"):
        raise ValueError("unknown FFN down route")
    implementation = route.rsplit("_", 1)[1]
    if route.startswith("splitk"):
        return FFNDown(
            route,
            ffn_down_partial(M, SPLIT=SPLIT, implementation=implementation),
            ffn_down_merge(M, SPLIT=SPLIT),
            SPLIT,
        )
    return FFNDown(route, ffn_down_full(M, implementation=implementation))
