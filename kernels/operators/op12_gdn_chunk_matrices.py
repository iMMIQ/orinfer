"""TileLang SM87 weighted GDN KK/QK matrices; no allocations or Torch math.

Q/K share 16 heads; the 48 value heads map through hv//3. q_scale is a
required specialization: normalized unscaled Q uses 128**-0.5, already-scaled
FP32 Q uses 1. All buffers are contiguous, disjoint and caller-owned.
"""

import math

import tilelang
import tilelang.language as T

HK, HV, DK = 16, 48, 128


@tilelang.jit(out_idx=[], execution_backend="nvrtc", target={"kind": "cuda", "arch": "sm_87"})
def _compile(bt: int, q_scale: float, qk_dtype: str):
    batch, chunks = T.dynamic("batch"), T.dynamic("chunks")

    @T.prim_func
    def main(
        Q: T.Tensor((batch, HK, chunks, bt, DK), qk_dtype),
        K: T.Tensor((batch, HK, chunks, bt, DK), qk_dtype),
        G: T.Tensor((batch, HV, chunks, bt), "float32"),
        Beta: T.Tensor((batch, HV, chunks, bt), "float32"),
        L: T.Tensor((batch, HV, chunks, bt, bt), "float32"),
        QK: T.Tensor((batch, HV, chunks, bt, bt), "float32"),
    ):
        if qk_dtype == "float16":
            with T.Kernel(HV, chunks, batch, threads=64 if bt == 16 else 128) as (h, c, b):
                qs = T.alloc_shared((bt, DK), "float16")
                ks = T.alloc_shared((bt, DK), "float16")
                kk = T.alloc_fragment((bt, bt), "float32")
                qk = T.alloc_fragment((bt, bt), "float32")
                T.copy(Q[b, h // 3, c, :, :], qs)
                T.copy(K[b, h // 3, c, :, :], ks)
                T.clear(kk)
                T.clear(qk)
                T.gemm(ks, ks, kk, transpose_B=True)
                T.gemm(qs, ks, qk, transpose_B=True)
                for i, j in T.Parallel(bt, bt):
                    # Clamp before exp: unused upper positive differences cannot
                    # overflow and poison masked outputs. Gates are nonpositive.
                    decay = T.exp(T.min(G[b, h, c, i] - G[b, h, c, j], 0.0))
                    if i > j:
                        L[b, h, c, i, j] = kk[i, j] * Beta[b, h, c, i] * decay
                    elif i == j:
                        L[b, h, c, i, j] = 1.0
                    else:
                        L[b, h, c, i, j] = 0.0
                    if i >= j:
                        QK[b, h, c, i, j] = qk[i, j] * q_scale * decay
                    else:
                        QK[b, h, c, i, j] = 0.0
        else:
            # Explicit FP32 SIMT fallback. It preserves already-scaled FP32 Q
            # without a hidden FP16/TF32 rounding step; correctness before speed.
            with T.Kernel(HV * bt * bt, chunks, batch, threads=128) as (p, c, b):
                h, i, j = p // (bt * bt), (p // bt) % bt, p % bt
                kk = T.alloc_fragment((DK,), "float32")
                qk = T.alloc_fragment((DK,), "float32")
                sumkk = T.alloc_fragment((1,), "float32")
                sumqk = T.alloc_fragment((1,), "float32")
                for d in T.Parallel(DK):
                    kk[d] = K[b, h // 3, c, i, d] * K[b, h // 3, c, j, d]
                    qk[d] = Q[b, h // 3, c, i, d] * K[b, h // 3, c, j, d]
                T.reduce_sum(kk, sumkk, dim=0)
                T.reduce_sum(qk, sumqk, dim=0)
                decay = T.exp(T.min(G[b, h, c, i] - G[b, h, c, j], 0.0))
                if i > j:
                    L[b, h, c, i, j] = sumkk[0] * Beta[b, h, c, i] * decay
                elif i == j:
                    L[b, h, c, i, j] = 1.0
                else:
                    L[b, h, c, i, j] = 0.0
                if i >= j:
                    QK[b, h, c, i, j] = sumqk[0] * q_scale * decay
                else:
                    QK[b, h, c, i, j] = 0.0

    return main


def gdn_chunk_matrices(*, q_scale, bt=64, qk_dtype="float16"):
    """Compile explicit-output L/QK from op11 cumulative G and padded beta.

    Q/K: [B,16,C,BT,128], G/Beta: FP32[B,48,C,BT],
    L/QK: FP32[B,48,C,BT,BT]. Default Q/K are unscaled normalized FP16.
    Tail Q/K=0, g=0 and beta=0 are supplied by the caller; G is cumulative
    and retains the last valid prefix in the tail. No lengths or state owned.
    """
    if type(bt) is not int or bt not in (16, 32, 64):
        raise ValueError("BT must be 16, 32 or 64")
    if qk_dtype not in ("float16", "float32"):
        raise ValueError("qk_dtype must be float16 or float32")
    if (
        isinstance(q_scale, bool)
        or not isinstance(q_scale, (int, float))
        or not math.isfinite(q_scale)
        or q_scale <= 0
    ):
        raise ValueError("q_scale must be an explicit finite positive number")
    kernel = _compile(bt, float(q_scale), qk_dtype)
    kernel.adapter.kernels = dict(kernel.adapter.kernels)
    return kernel


def launch(kernel, q, k, cumulative_g, beta, system, causal_qk, *, stream):
    """Explicit caller stream; capture-safe with stable preallocated buffers."""
    return kernel.adapter.func(q, k, cumulative_g, beta, system, causal_qk, stream=stream)
