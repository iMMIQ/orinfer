"""Qwen3 per-head RMSNorm/RoPE and an exact two-row last-position head."""

import tilelang.language as T
from orinfer_build.compiler.cuda.common import orin_jit
from kernels.shared.rows import row_count


@orin_jit
def qk_rope(capacity, dynamic_rows=True, head_group=1):
    if head_group not in (1, 2, 4, 8, 16, 32):
        raise ValueError("Invalid Qwen3 head group")
    m = row_count(capacity, dynamic_rows)

    @T.prim_func
    def main(
        Ranges: T.Tensor((m, 2), T.int32),
        X: T.Tensor((m, 4096), T.float16),
        QNorm: T.Tensor((128,), T.float16),
        KNorm: T.Tensor((128,), T.float16),
        Cos: T.Tensor((capacity, 128), T.float16),
        Sin: T.Tensor((capacity, 128), T.float16),
        Q: T.Tensor((m, 2048), T.float16),
        K: T.Tensor((m, 1024), T.float16),
        V: T.Tensor((m, 1024), T.float16),
    ):
        with T.Kernel(m, 32 // head_group, threads=128) as (row, group):
            values = T.alloc_fragment((head_group, 128), T.float32)
            squares = T.alloc_fragment((head_group, 128), T.float32)
            total = T.alloc_fragment((head_group,), T.float32)
            normalized = T.alloc_shared((head_group, 128), T.float16)
            for h, d in T.Parallel(head_group, 128):
                values[h, d] = T.cast(X[row, (group * head_group + h) * 128 + d], T.float32)
                squares[h, d] = values[h, d] * values[h, d]
            T.reduce_sum(squares, total, dim=1)
            for h, d in T.Parallel(head_group, 128):
                head = group * head_group + h
                v = T.cast(values[h, d] * T.rsqrt(total[h] / 128 + 1e-6), T.float16)
                normalized[h, d] = v * T.if_then_else(head < 16, QNorm[d], KNorm[d])
            T.sync_threads()
            for h, d in T.Parallel(head_group, 128):
                head = group * head_group + h
                if head < 24:
                    a = T.cast(normalized[h, d] * Cos[row - Ranges[row, 0], d], T.float16)
                    b = T.cast(
                        normalized[h, (d + 64) % 128]
                        * T.if_then_else(
                            d < 64, -Sin[row - Ranges[row, 0], d], Sin[row - Ranges[row, 0], d]
                        ),
                        T.float16,
                    )
                    value = T.cast(T.cast(a, T.float32) + T.cast(b, T.float32), T.float16)
                    if head < 16:
                        Q[row, head * 128 + d] = value
                    else:
                        K[row, (head - 16) * 128 + d] = value
                else:
                    V[row, (head - 24) * 128 + d] = X[row, head * 128 + d]

    return main


@orin_jit
def classifier(capacity, dynamic_rows=True):
    m = row_count(capacity, dynamic_rows)

    @T.prim_func
    def main(
        X: T.Tensor((m, 1024), T.float16),
        W: T.Tensor((2, 1024), T.float16),
        Y: T.Tensor((2,), T.float32),
    ):
        with T.Kernel(2, threads=256) as label:
            product = T.alloc_fragment((1024,), T.float32)
            total = T.alloc_fragment((1,), T.float32)
            for d in T.Parallel(1024):
                product[d] = T.cast(X[m - 1, d], T.float32) * T.cast(W[label, d], T.float32)
            T.reduce_sum(product, total, dim=0)
            Y[label] = total[0]

    return main
