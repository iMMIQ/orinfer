"""Bidirectional embedding boundaries shared by text and visual encoders."""

import tilelang.language as T

from orinfer_build.compiler.cuda.common import orin_jit
from kernels.shared.rows import row_count
from kernels.cuda.operators.op30_activation_quantization import quantized_code


@orin_jit
def quantize(capacity, hidden, dynamic_rows=True):
    m = row_count(capacity, dynamic_rows)

    @T.prim_func
    def main(
        X: T.Tensor((m, hidden), T.float16),
        Q: T.Tensor((m, hidden), T.int8),
        S: T.Tensor((m,), T.float16),
    ):
        with T.Kernel(m, threads=256) as row:
            absolute = T.alloc_fragment((hidden,), T.float32)
            maximum = T.alloc_fragment((1,), T.float32)
            scale = T.alloc_fragment((1,), T.float16)
            for d in T.Parallel(hidden):
                absolute[d] = T.abs(T.cast(X[row, d], T.float32))
            T.reduce_max(absolute, maximum, dim=0)
            scale[0] = T.if_then_else(maximum[0] > 0, T.max(maximum[0] / 127, 2**-24), 1.0)
            S[row] = scale[0]
            for d in T.Parallel(hidden):
                Q[row, d] = quantized_code(T.cast(X[row, d], T.float32), scale[0])

    return main


@orin_jit
def rms_norm(capacity, hidden, epsilon=1e-5, dynamic_rows=True):
    m = row_count(capacity, dynamic_rows)

    @T.prim_func
    def main(
        X: T.Tensor((m, hidden), T.float16),
        W: T.Tensor((hidden,), T.float16),
        Y: T.Tensor((m, hidden), T.float16),
    ):
        with T.Kernel(m, threads=256) as row:
            squares = T.alloc_fragment((hidden,), T.float32)
            total = T.alloc_fragment((1,), T.float32)
            for d in T.Parallel(hidden):
                value = T.cast(X[row, d], T.float32)
                squares[d] = value * value
            T.reduce_sum(squares, total, dim=0)
            for d in T.Parallel(hidden):
                value = T.cast(
                    T.cast(X[row, d], T.float32) * T.rsqrt(total[0] / hidden + epsilon),
                    T.float16,
                )
                Y[row, d] = value * W[d]

    return main


@orin_jit
def add(capacity, hidden, dynamic_rows=True, block_size=256):
    if block_size not in (128, 256, 512, 1024):
        raise ValueError("Invalid encoder elementwise tile")
    m = row_count(capacity, dynamic_rows)

    @T.prim_func
    def main(
        X: T.Tensor((m, hidden), T.float16),
        Residual: T.Tensor((m, hidden), T.float16),
        Y: T.Tensor((m, hidden), T.float16),
    ):
        with T.Kernel(m, T.ceildiv(hidden, block_size), threads=128) as (row, block):
            for d in T.Parallel(block_size):
                col = block * block_size + d
                if col < hidden:
                    Y[row, col] = T.cast(X[row, col], T.float32) + T.cast(
                        Residual[row, col], T.float32
                    )

    return main


@orin_jit
def activation(capacity, hidden, kind="swiglu", dynamic_rows=True, block_size=256):
    if block_size not in (128, 256, 512, 1024):
        raise ValueError("Invalid encoder elementwise tile")
    if kind not in ("swiglu", "gelu"):
        raise ValueError("Invalid embedding activation")
    m = row_count(capacity, dynamic_rows)

    @T.prim_func
    def main(
        X: T.Tensor((m, hidden * (2 if kind == "swiglu" else 1)), T.float16),
        Y: T.Tensor((m, hidden), T.float16),
    ):
        with T.Kernel(m, T.ceildiv(hidden, block_size), threads=128) as (row, block):
            for d in T.Parallel(block_size):
                col = block * block_size + d
                if col < hidden:
                    value = T.cast(X[row, col], T.float32)
                    if kind == "swiglu":
                        # Match SiLU's materialization before multiplying the up branch.
                        silu = T.cast(value / (1 + T.exp(-value)), T.float16)
                        Y[row, col] = T.cast(silu, T.float32) * T.cast(
                            X[row, col + hidden], T.float32
                        )
                    else:
                        Y[row, col] = (
                            0.5
                            * value
                            * (1 + T.call_extern("float32", "erff", value * 0.7071067811865476))
                        )

    return main


@orin_jit
def text_rope(capacity, heads=24, kv_heads=8, dim=128, dynamic_rows=True, head_group=1):
    if head_group not in (1, 2, 4, 8):
        raise ValueError("Invalid encoder RoPE head group")
    m = row_count(capacity, dynamic_rows)
    width = (heads + 2 * kv_heads) * dim

    @T.prim_func
    def main(
        Ranges: T.Tensor((m, 2), T.int32),
        X: T.Tensor((m, width), T.float16),
        Cos: T.Tensor((capacity, dim), T.float16),
        Sin: T.Tensor((capacity, dim), T.float16),
        QScale: T.Tensor((capacity,), T.float16),
        Q: T.Tensor((m, heads * dim), T.float16),
        K: T.Tensor((m, kv_heads * dim), T.float16),
        V: T.Tensor((m, kv_heads * dim), T.float16),
    ):
        with T.Kernel(m, T.ceildiv(heads + 2 * kv_heads, head_group), threads=128) as (row, group):
            for h, d in T.Parallel(head_group, dim):
                head = group * head_group + h
                if head < heads + kv_heads:
                    other = (d + dim // 2) % dim
                    # HF materializes both products in the activation dtype.
                    a = T.cast(X[row, head * dim + d] * Cos[row - Ranges[row, 0], d], T.float16)
                    b = T.cast(
                        X[row, head * dim + other]
                        * T.if_then_else(
                            d < dim // 2,
                            -Sin[row - Ranges[row, 0], d],
                            Sin[row - Ranges[row, 0], d],
                        ),
                        T.float16,
                    )
                    value = T.cast(T.cast(a, T.float32) + T.cast(b, T.float32), T.float16)
                    if head < heads:
                        Q[row, head * dim + d] = value * QScale[row - Ranges[row, 0]]
                    else:
                        K[row, (head - heads) * dim + d] = value
                elif head < heads + 2 * kv_heads:
                    V[row, (head - heads - kv_heads) * dim + d] = X[row, head * dim + d]

    return main


@orin_jit
def pool(capacity, hidden, dynamic_rows=True):
    """Mean all real tokens, including the prompt, then L2-normalize in FP32."""
    m = row_count(capacity, dynamic_rows)

    @T.prim_func
    def main(
        X: T.Tensor((m, hidden), T.float16),
        Mean: T.Tensor((hidden,), T.float32),
    ):
        with T.Kernel(T.ceildiv(hidden, 256), threads=256) as block:
            total = T.alloc_fragment((256,), T.float32)
            T.clear(total)
            for row in T.serial(m):
                for d in T.Parallel(256):
                    if block * 256 + d < hidden:
                        total[d] += T.cast(X[row, block * 256 + d], T.float32)
            for d in T.Parallel(256):
                if block * 256 + d < hidden:
                    Mean[block * 256 + d] = total[d] / m

    return main


@orin_jit
def normalize(hidden):
    @T.prim_func
    def main(
        Mean: T.Tensor((hidden,), T.float32),
        Output: T.Tensor((hidden,), T.float32),
    ):
        with T.Kernel(1, threads=256):
            squares = T.alloc_fragment((hidden,), T.float32)
            total = T.alloc_fragment((1,), T.float32)
            for d in T.Parallel(hidden):
                squares[d] = Mean[d] * Mean[d]
            T.reduce_sum(squares, total, dim=0)
            for d in T.Parallel(hidden):
                Output[d] = Mean[d] * T.rsqrt(T.max(total[0], 1e-24))

    return main


@orin_jit
def overlay(capacity, hidden, features, dynamic_rows=True):
    m = row_count(capacity, dynamic_rows)

    @T.prim_func
    def main(
        Hidden: T.Tensor((m, hidden), T.float16),
        Features: T.Tensor((features, hidden), T.float16),
        Index: T.Tensor((capacity,), T.int32),
    ):
        with T.Kernel(m, T.ceildiv(hidden, 256), threads=128) as (row, block):
            for d in T.Parallel(256):
                col = block * 256 + d
                if col < hidden and Index[row] >= 0 and Index[row] < features:
                    Hidden[row, col] = Features[Index[row], col]

    return main


@orin_jit
def vision_rope(capacity, hidden=1024, heads=16, dynamic_rows=True):
    m = row_count(capacity, dynamic_rows)
    dim = hidden // heads

    @T.prim_func
    def main(
        X: T.Tensor((m, 3 * hidden), T.float16),
        Grid: T.Tensor((2,), T.int32),
        Q: T.Tensor((m, hidden), T.float16),
        K: T.Tensor((m, hidden), T.float16),
        V: T.Tensor((m, hidden), T.float16),
    ):
        with T.Kernel(m, heads, threads=128) as (row, head):
            for d in T.Parallel(dim):
                freq = d % (dim // 2)
                axis = T.if_then_else(freq < dim // 4, row // Grid[1], row % Grid[1])
                exponent = T.if_then_else(freq < dim // 4, 4 * freq, 4 * (freq - dim // 4) + 2)
                angle = T.cast(axis, T.float32) * T.exp(
                    -9.210340371976184 * T.cast(exponent, T.float32) / dim
                )
                cos = T.cast(T.cos(angle), T.float16)
                sin = T.cast(T.sin(angle), T.float16)
                other = (d + dim // 2) % dim
                for part in T.unroll(2):
                    a = T.cast(X[row, part * hidden + head * dim + d] * cos, T.float16)
                    b = T.cast(
                        X[row, part * hidden + head * dim + other]
                        * T.if_then_else(d < dim // 2, -sin, sin),
                        T.float16,
                    )
                    value = T.cast(T.cast(a, T.float32) + T.cast(b, T.float32), T.float16)
                    if part == 0:
                        Q[row, head * dim + d] = value
                    else:
                        K[row, head * dim + d] = value
                V[row, head * dim + d] = X[row, 2 * hidden + head * dim + d]

    return main


@orin_jit
def merge(capacity, hidden=1024, dynamic_rows=True):
    """HF unfold order is channel, patch y, patch x; merge rows are real rows/4."""
    m = row_count(capacity, dynamic_rows)

    @T.prim_func
    def main(
        X: T.Tensor((m * 4, hidden), T.float16),
        Grid: T.Tensor((2,), T.int32),
        Y: T.Tensor((m, hidden * 4), T.float16),
    ):
        with T.Kernel(m, T.ceildiv(hidden * 4, 256), threads=128) as (row, block):
            for d in T.Parallel(256):
                col = block * 256 + d
                if col < hidden * 4:
                    y = row // (Grid[1] // 2) * 2 + (col % 4) // 2
                    x = row % (Grid[1] // 2) * 2 + col % 2
                    Y[row, col] = X[y * Grid[1] + x, col // 4]

    return main
