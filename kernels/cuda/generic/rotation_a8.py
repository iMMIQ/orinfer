"""Portable signed block-128 Hadamard rotation and symmetric row-A8 quantization.

Uses double-buffered shared-memory Walsh-Hadamard butterflies in pure TileLang
with zero warp-shuffle inline PTX, preserving Orinfer's exact FP16 materialization
boundaries across Ampere, Ada, Hopper, and Blackwell GPUs.
"""

import tilelang.language as T

from kernels.shared.rows import row_count
from orinfer_build.compiler.cuda.common import cuda_jit


@cuda_jit
def rotate_activation_a8(M, K, *, swiglu=False, in_dtype="float16", dynamic_rows=False):
    """Fused signed block-128 rotation (+ optional SwiGLU) and symmetric row INT8."""
    if (
        any(type(v) is not int or v <= 0 for v in (M, K))
        or K % 128
        or in_dtype not in ("float16", "bfloat16")
        or type(swiglu) is not bool
        or type(dynamic_rows) is not bool
    ):
        raise ValueError("Invalid portable rotation-A8 geometry")

    in_cols = 2 * K if swiglu else K
    rows_sym = row_count(M, dynamic_rows)
    inv_sqrt_128 = 128.0**-0.5

    @T.prim_func
    def main(
        X: T.Tensor((rows_sym, in_cols), in_dtype),
        Signs: T.Tensor((K,), T.int8),
        Q: T.Tensor((rows_sym, K), T.int8),
        S: T.Tensor((rows_sym,), T.float16),
    ):
        with T.Kernel(rows_sym, threads=128) as row:
            buf0 = T.alloc_shared((K,), T.float32)
            buf1 = T.alloc_shared((K,), T.float32)
            abs_f = T.alloc_fragment((K,), T.float32)
            max_f = T.alloc_fragment((1,), T.float32)
            scale_s = T.alloc_shared((1,), T.float16)

            for k in T.Parallel(K):
                if swiglu:
                    g = T.cast(X[row, k], T.float32)
                    u = T.cast(X[row, k + K], T.float32)
                    eg = T.exp(-T.abs(g))
                    sig = T.if_then_else(g >= 0.0, 1.0 / (1.0 + eg), eg / (1.0 + eg))
                    val = T.cast(T.cast((g * sig) * u, T.float16), T.float32)
                else:
                    val = T.cast(T.cast(X[row, k], T.float16), T.float32)
                buf0[k] = val * T.cast(Signs[k], T.float32)

            for stage in T.unroll(7):
                step = 1 << stage
                if stage % 2 == 0:
                    for k in T.Parallel(K):
                        partner = k ^ step
                        is_hi = (k >> stage) & 1
                        buf1[k] = T.if_then_else(
                            is_hi == 0, buf0[k] + buf0[partner], buf0[partner] - buf0[k]
                        )
                else:
                    for k in T.Parallel(K):
                        partner = k ^ step
                        is_hi = (k >> stage) & 1
                        buf0[k] = T.if_then_else(
                            is_hi == 0, buf1[k] + buf1[partner], buf1[partner] - buf1[k]
                        )

            for k in T.Parallel(K):
                rotated = T.cast(T.cast(buf1[k] * inv_sqrt_128, T.float16), T.float32)
                buf0[k] = rotated
                abs_f[k] = T.abs(rotated)

            T.reduce_max(abs_f, max_f, dim=0)
            if T.get_thread_binding() == 0:
                mx = max_f[0]
                sc = T.cast(
                    T.if_then_else(mx > 0.0, T.max(mx / 127.0, 2.0**-24), 1.0),
                    T.float16,
                )
                scale_s[0] = sc
                S[row] = sc
            T.sync_threads()

            sc_f32 = T.cast(scale_s[0], T.float32)
            for k in T.Parallel(K):
                q_val = T.round(buf0[k] / sc_f32)
                Q[row, k] = T.cast(T.max(-127.0, T.min(127.0, q_val)), T.int8)

    return main


@cuda_jit
def plain_activation_a8(M, K, *, in_dtype="float16", dynamic_rows=False):
    """Portable symmetric per-row INT8 quantization for unrotated W8A8 inputs."""
    if (
        any(type(v) is not int or v <= 0 for v in (M, K))
        or in_dtype not in ("float16", "bfloat16")
        or type(dynamic_rows) is not bool
    ):
        raise ValueError("Invalid portable plain-A8 geometry")

    rows_sym = row_count(M, dynamic_rows)
    width = ((K + 127) // 128) * 128

    @T.prim_func
    def main(
        X: T.Tensor((rows_sym, K), in_dtype),
        Q: T.Tensor((rows_sym, K), T.int8),
        S: T.Tensor((rows_sym,), T.float16),
    ):
        with T.Kernel(rows_sym, threads=128) as row:
            buf = T.alloc_shared((K,), T.float32)
            abs_f = T.alloc_fragment((width,), T.float32)
            max_f = T.alloc_fragment((1,), T.float32)
            scale_s = T.alloc_shared((1,), T.float16)

            for k in T.Parallel(width):
                if k < K:
                    v = T.cast(T.cast(X[row, k], T.float16), T.float32)
                    buf[k] = v
                    abs_f[k] = T.abs(v)
                else:
                    abs_f[k] = 0.0

            T.reduce_max(abs_f, max_f, dim=0)
            if T.get_thread_binding() == 0:
                mx = max_f[0]
                sc = T.cast(
                    T.if_then_else(mx > 0.0, T.max(mx / 127.0, 2.0**-24), 1.0),
                    T.float16,
                )
                scale_s[0] = sc
                S[row] = sc
            T.sync_threads()

            sc_f32 = T.cast(scale_s[0], T.float32)
            for k in T.Parallel(width):
                if k < K:
                    q_val = T.round(buf[k] / sc_f32)
                    Q[row, k] = T.cast(T.max(-127.0, T.min(127.0, q_val)), T.int8)

    return main
