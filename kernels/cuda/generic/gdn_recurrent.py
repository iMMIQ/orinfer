"""Portable single-launch FP32 Gated DeltaNet recurrence for non-Orin GPUs.

Keeps each recurrent state column `S[0..dk-1, v_col]` resident in on-chip
registers across the entire token sequence (both prefill and stateful decode),
eliminating intermediate state HBM traffic without architecture-specific PTX.
"""

import math
import tilelang.language as T

from orinfer_build.compiler.cuda.common import cuda_jit


@cuda_jit
def gdn_recurrent(
    B,
    L,
    hk,
    hv,
    dk=128,
    dv=128,
    *,
    block_v=64,
    dtype="bfloat16",
    dynamic_shape=True,
):
    if (
        any(type(v) is not int or v <= 0 for v in (B, L, hk, hv, dk, dv, block_v))
        or hv % hk
        or dv % block_v
        or dk % block_v
        or dtype not in ("float16", "bfloat16")
        or type(dynamic_shape) is not bool
    ):
        raise ValueError("Invalid portable GDN recurrence geometry")

    repeat = hv // hk
    inv_sqrt_dk = 1.0 / math.sqrt(dk)
    batch_dim = T.dynamic("b") if dynamic_shape else B
    seq_dim = T.dynamic("l") if dynamic_shape else L

    @T.prim_func
    def main(
        Q: T.Tensor((batch_dim, seq_dim, hk, dk), dtype),
        K: T.Tensor((batch_dim, seq_dim, hk, dk), dtype),
        V: T.Tensor((batch_dim, seq_dim, hv, dv), dtype),
        Decay: T.Tensor((batch_dim, seq_dim, hv), T.float32),
        Beta: T.Tensor((batch_dim, seq_dim, hv), T.float32),
        State: T.Tensor((batch_dim, hv, dk, dv), T.float32),
        Out: T.Tensor((batch_dim, seq_dim, hv, dv), dtype),
    ):
        with T.Kernel(dv // block_v, hv, batch_dim, threads=block_v) as (bv, h, b):
            tid = T.get_thread_binding()
            v_col = bv * block_v + tid
            hk_idx = h // repeat

            S_col = T.alloc_local((dk,), T.float32)
            k_s = T.alloc_shared((dk,), T.float32)
            q_s = T.alloc_shared((dk,), T.float32)
            norm_s = T.alloc_shared((2,), T.float32)

            for i in T.serial(dk):
                S_col[i] = State[b, h, i, v_col]

            for t in T.serial(seq_dim):
                for r in T.unroll(dk // block_v):
                    idx = r * block_v + tid
                    k_s[idx] = T.cast(K[b, t, hk_idx, idx], T.float32)
                    q_s[idx] = T.cast(Q[b, t, hk_idx, idx], T.float32)
                T.sync_threads()

                if tid == 0:
                    kn2 = T.alloc_local((1,), T.float32)
                    qn2 = T.alloc_local((1,), T.float32)
                    kn2[0] = 0.0
                    qn2[0] = 0.0
                    for i in T.serial(dk):
                        kn2[0] = kn2[0] + k_s[i] * k_s[i]
                        qn2[0] = qn2[0] + q_s[i] * q_s[i]
                    norm_s[0] = T.rsqrt(kn2[0] + 1e-6)
                    norm_s[1] = T.rsqrt(qn2[0] + 1e-6) * inv_sqrt_dk
                T.sync_threads()

                kn = norm_s[0]
                qn = norm_s[1]
                for r in T.unroll(dk // block_v):
                    idx = r * block_v + tid
                    k_s[idx] = k_s[idx] * kn
                    q_s[idx] = q_s[idx] * qn
                T.sync_threads()

                exp_d = T.exp(Decay[b, t, h])
                beta_t = Beta[b, t, h]
                v_val = T.cast(V[b, t, h, v_col], T.float32)

                pred = T.alloc_local((1,), T.float32)
                pred[0] = 0.0
                for i in T.unroll(dk, unroll_factor=4):
                    s_i = S_col[i] * exp_d
                    S_col[i] = s_i
                    pred[0] = pred[0] + s_i * k_s[i]

                upd = beta_t * (v_val - pred[0])

                o_acc = T.alloc_local((1,), T.float32)
                o_acc[0] = 0.0
                for i in T.unroll(dk, unroll_factor=4):
                    s_i = S_col[i] + k_s[i] * upd
                    S_col[i] = s_i
                    o_acc[0] = o_acc[0] + s_i * q_s[i]

                Out[b, t, h, v_col] = T.cast(o_acc[0], dtype)

            for i in T.serial(dk):
                State[b, h, i, v_col] = S_col[i]

    return main
