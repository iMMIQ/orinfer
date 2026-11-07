"""SM87 ViT kernels: BF16/FP16 tensors, FP32 reductions/accumulators.

Online execution is through exported AOT CUDA kernels. Padding is workspace
only: Length masks every attention key; it never adds model input tokens.
"""

import math
import tilelang.language as T
from orinfer_build.compiler.cuda.common import orin_jit
from kernels.cuda.model.floating_core import mma_loop, rounded_float, BF16_SOURCE


def activation_value(x, activation):
    if activation == "gelu_tanh":
        return 0.5 * x * (1 + T.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
    if activation == "gelu":
        return 0.5 * x * (1 + T.call_extern("float32", "erff", x * 0.7071067811865476))
    return x


@orin_jit
def bias_activation(hidden: int, activation: str = "none"):
    """Preserve the Linear FP16 bias boundary after a shared Q4A8 product."""
    assert activation in ("none", "gelu", "gelu_tanh")
    rows = T.dynamic("rows")

    @T.prim_func
    def kernel(
        X: T.Tensor((rows, hidden), T.float32),
        Bias: T.Tensor((hidden,), T.float16),
        Y: T.Tensor((rows, hidden), T.float16),
    ):
        with T.Kernel(rows, T.ceildiv(hidden, 256), threads=128) as (row, block):
            for j in T.Parallel(256):
                col = block * 256 + j
                if col < hidden:
                    value = T.cast(
                        T.cast(X[row, col] + T.cast(Bias[col], T.float32), T.float16), T.float32
                    )
                    Y[row, col] = activation_value(value, activation)

    return kernel


@orin_jit
def linear(
    n: int,
    k: int,
    activation: str = "none",
    bm=64,
    bn=64,
    bk=64,
    dtype: str = "float16",
    output_dtype: str | None = None,
    separate_bias: bool = False,
    stages: int = 2,
):
    assert activation in ("none", "gelu", "gelu_tanh")
    output_dtype = output_dtype or dtype
    rows = T.dynamic("rows")
    project = mma_loop(rows, k, bm, bn, bk, dtype=dtype, stages=stages)

    @T.macro
    def store(Y, i, j, value):
        if output_dtype == "bfloat16":
            T.evaluate(
                T.call_extern("void", "orinfer_store_bf16", T.access_ptr(Y[i, j], "w", 1), value)
            )
        else:
            Y[i, j] = T.cast(value, output_dtype)

    @T.prim_func
    def kernel(
        X: T.Tensor((rows, k), dtype),
        W: T.Tensor((n, k), dtype),
        Bias: T.Tensor((n,), dtype),
        Y: T.Tensor((rows, n), output_dtype),
    ):
        with T.Kernel(T.ceildiv(rows, bm), T.ceildiv(n, bn), threads=128) as (by, bx):
            if dtype == "bfloat16" or output_dtype == "bfloat16":
                T.import_source(BF16_SOURCE)
            accum = T.alloc_fragment((bm, bn), T.float32)
            project(X, W, accum, by, bx, 0)
            for i, j in T.Parallel(bm, bn):
                if by * bm + i < rows and bx * bn + j < n:
                    # Conv3D patch embedding stores the dot product in the
                    # native dtype before adding bias; Linear fuses the bias
                    # into its FP32 accumulator. Preserve both semantics.
                    if separate_bias:
                        product = rounded_float(accum[i, j], dtype)
                    else:
                        product = accum[i, j]
                    value = rounded_float(product + T.cast(Bias[bx * bn + j], T.float32), dtype)
                    store(
                        Y,
                        by * bm + i,
                        bx * bn + j,
                        activation_value(T.cast(value, T.float32), activation),
                    )

    return kernel


@orin_jit
def layer_norm(
    hidden: int, padded: int | None = None, dtype: str = "float16", epsilon: float = 1e-6
):
    padded = padded or 1 << (hidden - 1).bit_length()
    assert hidden <= padded
    rows = T.dynamic("rows")

    @T.prim_func
    def kernel(
        X: T.Tensor((rows, hidden), dtype),
        W: T.Tensor((hidden,), dtype),
        B: T.Tensor((hidden,), dtype),
        Y: T.Tensor((rows, hidden), dtype),
    ):
        with T.Kernel(rows, threads=128) as row:
            x = T.alloc_fragment((padded,), T.float32)
            square = T.alloc_fragment((padded,), T.float32)
            mean = T.alloc_fragment((1,), T.float32)
            variance = T.alloc_fragment((1,), T.float32)
            for i in T.Parallel(padded):
                x[i] = T.if_then_else(i < hidden, T.cast(X[row, i], T.float32), 0)
            T.reduce_sum(x, mean, dim=0)
            for i in T.Parallel(padded):
                square[i] = T.if_then_else(
                    i < hidden, (x[i] - mean[0] / hidden) * (x[i] - mean[0] / hidden), 0
                )
            T.reduce_sum(square, variance, dim=0)
            for i in T.Parallel(padded):
                if i < hidden:
                    Y[row, i] = (x[i] - mean[0] / hidden) * T.rsqrt(
                        variance[0] / hidden + epsilon
                    ) * T.cast(W[i], T.float32) + T.cast(B[i], T.float32)

    return kernel


@orin_jit
def add(hidden: int, dtype: str = "float16"):
    rows = T.dynamic("rows")

    @T.prim_func
    def kernel(
        X: T.Tensor((rows, hidden), dtype),
        Residual: T.Tensor((rows, hidden), dtype),
        Y: T.Tensor((rows, hidden), dtype),
    ):
        with T.Kernel(T.ceildiv(rows * hidden, 1024), threads=128) as block:
            for i in T.Parallel(1024):
                flat = block * 1024 + i
                if flat < rows * hidden:
                    Y[flat // hidden, flat % hidden] = T.cast(
                        X[flat // hidden, flat % hidden], T.float32
                    ) + T.cast(Residual[flat // hidden, flat % hidden], T.float32)

    return kernel


@orin_jit
def position(
    hidden: int,
    grid_side: int = 48,
    merge: int = 2,
    dtype: str = "float16",
    fp32_interpolation: bool = False,
):
    rows = T.dynamic("rows")

    @T.prim_func
    def kernel(
        X: T.Tensor((rows, hidden), dtype),
        W: T.Tensor((grid_side * grid_side, hidden), dtype),
        Grid: T.Tensor((2,), T.int32),
        Length: T.Tensor((1,), T.int32),
        Y: T.Tensor((rows, hidden), dtype),
    ):
        with T.Kernel(rows, threads=128) as row:
            if row < Length[0]:
                h = (row // (merge * merge) // (Grid[1] // merge)) * merge + (
                    row % (merge * merge)
                ) // merge
                w = (row // (merge * merge) % (Grid[1] // merge)) * merge + row % merge
                hf = T.cast(h, T.float32) * (grid_side - 1) / T.max(Grid[0] - 1, 1)
                wf = T.cast(w, T.float32) * (grid_side - 1) / T.max(Grid[1] - 1, 1)
                h0 = T.cast(T.floor(hf), T.int32)
                w0 = T.cast(T.floor(wf), T.int32)
                h1 = T.min(h0 + 1, grid_side - 1)
                w1 = T.min(w0 + 1, grid_side - 1)
                dh = hf - T.cast(h0, T.float32)
                dw = wf - T.cast(w0, T.float32)
                for j in T.Parallel(hidden):
                    total = T.alloc_var(dtype)
                    if fp32_interpolation:
                        # Current Qwen4Exp interpolates the learned table in FP32,
                        # then rounds once before the residual addition.
                        total = (
                            T.cast(W[h0 * grid_side + w0, j], T.float32) * (1 - dh) * (1 - dw)
                            + T.cast(W[h0 * grid_side + w1, j], T.float32) * (1 - dh) * dw
                            + T.cast(W[h1 * grid_side + w0, j], T.float32) * dh * (1 - dw)
                            + T.cast(W[h1 * grid_side + w1, j], T.float32) * dh * dw
                        )
                    else:
                        a = T.cast(
                            T.cast(W[h0 * grid_side + w0, j], T.float32)
                            * T.cast(T.cast((1 - dh) * (1 - dw), dtype), T.float32),
                            dtype,
                        )
                        b = T.cast(
                            T.cast(W[h0 * grid_side + w1, j], T.float32)
                            * T.cast(T.cast((1 - dh) * dw, dtype), T.float32),
                            dtype,
                        )
                        c = T.cast(
                            T.cast(W[h1 * grid_side + w0, j], T.float32)
                            * T.cast(T.cast(dh * (1 - dw), dtype), T.float32),
                            dtype,
                        )
                        d = T.cast(
                            T.cast(W[h1 * grid_side + w1, j], T.float32)
                            * T.cast(T.cast(dh * dw, dtype), T.float32),
                            dtype,
                        )
                        ab = T.cast(T.cast(a, T.float32) + T.cast(b, T.float32), dtype)
                        abc = T.cast(T.cast(ab, T.float32) + T.cast(c, T.float32), dtype)
                        total = T.cast(T.cast(abc, T.float32) + T.cast(d, T.float32), dtype)
                    Y[row, j] = T.cast(X[row, j], T.float32) + T.cast(total, T.float32)
            else:
                for j in T.Parallel(hidden):
                    Y[row, j] = 0

    return kernel


@orin_jit
def qkv_rope(hidden: int = 1152, heads: int = 16, merge: int = 2, dtype: str = "float16"):
    dim = hidden // heads
    half = dim // 2
    quarter = dim // 4
    assert dim % 4 == 0
    rows = T.dynamic("rows")

    @T.prim_func
    def kernel(
        X: T.Tensor((rows, 3 * hidden), dtype),
        Grid: T.Tensor((2,), T.int32),
        Length: T.Tensor((1,), T.int32),
        Q: T.Tensor((rows, hidden), dtype),
        K: T.Tensor((rows, hidden), dtype),
        V: T.Tensor((rows, hidden), dtype),
    ):
        with T.Kernel(rows, heads, threads=128) as (row, head):
            h = (row // (merge * merge) // (Grid[1] // merge)) * merge + (
                row % (merge * merge)
            ) // merge
            w = (row // (merge * merge) % (Grid[1] // merge)) * merge + row % merge
            for j in T.Parallel(dim):
                if row < Length[0]:
                    idx = j % half
                    pos = T.if_then_else(idx < quarter, h, w)
                    angle = T.cast(pos, T.float32) * T.pow(
                        10000.0, -T.cast(idx % quarter, T.float32) / quarter
                    )
                    cosine = T.cos(angle)
                    sine = T.sin(angle)
                    partner = T.if_then_else(j < half, j + half, j - half)
                    sign = T.if_then_else(j < half, -1.0, 1.0)
                    Q[row, head * dim + j] = (
                        T.cast(X[row, head * dim + j], T.float32) * cosine
                        + sign * T.cast(X[row, head * dim + partner], T.float32) * sine
                    )
                    K[row, head * dim + j] = (
                        T.cast(X[row, hidden + head * dim + j], T.float32) * cosine
                        + sign * T.cast(X[row, hidden + head * dim + partner], T.float32) * sine
                    )
                    V[row, head * dim + j] = X[row, 2 * hidden + head * dim + j]
                else:
                    Q[row, head * dim + j] = 0
                    K[row, head * dim + j] = 0
                    V[row, head * dim + j] = 0

    return kernel


@orin_jit
def attention(
    hidden: int = 1152,
    heads: int = 16,
    bm: int = 32,
    bn: int = 32,
    dtype: str = "float16",
    window: int = 0,
    windowed_lengths: bool = False,
    kv_heads: int | None = None,
    causal: bool = False,
    segmented: bool = False,
    head_group: int = 1,
    stages: int = 0,
    base2: bool = False,
    full_row: bool = False,
):
    if segmented and (window or windowed_lengths):
        raise ValueError("Segmented attention uses independent complete sequences")
    dim = hidden // heads
    kv_heads = heads if kv_heads is None else kv_heads
    if (
        hidden % heads
        or heads % kv_heads
        or head_group < 1
        or (heads // kv_heads) % head_group
        or stages not in (0, 1)
    ):
        raise ValueError("Invalid grouped-query attention geometry")
    if segmented and (bm > 32 or 32 % bm):
        raise ValueError("Segmented query tiles must divide the packet alignment")
    head_stride = head_group
    query_rows = bm * head_stride
    policy = T.GemmWarpPolicy.FullRow if full_row else T.GemmWarpPolicy.Square
    padded = ((dim + 15) // 16) * 16
    rows = T.dynamic("rows")
    length_count = T.ceildiv(rows, window) if windowed_lengths else 1

    @T.prim_func
    def kernel(
        Q: T.Tensor((rows, hidden), dtype),
        K: T.Tensor((rows, kv_heads * dim), dtype),
        V: T.Tensor((rows, kv_heads * dim), dtype),
        Length: T.Tensor((rows, 2) if segmented else (length_count,), T.int32),
        Y: T.Tensor((rows, hidden), dtype),
    ):
        with T.Kernel(
            T.ceildiv(window, bm) if window else T.ceildiv(rows, bm),
            heads // head_group,
            T.ceildiv(rows, window) if window else 1,
            threads=128,
        ) as (qb, work_head, segment):
            base = Length[qb * bm, 0] if segmented else segment * window
            query = qb * bm if segmented else base + qb * bm
            end = (
                Length[qb * bm, 1]
                if segmented
                else (
                    base + T.min(Length[segment], window)
                    if windowed_lengths
                    else (T.min(Length[0], base + window) if window else Length[0])
                )
            )
            q = T.alloc_shared((query_rows, padded), dtype)
            k = T.alloc_shared((bn, padded), dtype)
            v = T.alloc_shared((bn, padded), dtype)
            p = T.alloc_shared((query_rows, bn), dtype)
            score = T.alloc_fragment((query_rows, bn), T.float32)
            out = T.alloc_fragment((query_rows, padded), T.float32)
            maximum = T.alloc_fragment((query_rows,), T.float32)
            previous = T.alloc_fragment((query_rows,), T.float32)
            denom = T.alloc_fragment((query_rows,), T.float32)
            correction = T.alloc_fragment((query_rows,), T.float32)
            rowsum = T.alloc_fragment((query_rows,), T.float32)
            T.fill(maximum, -1e30)
            T.clear(denom)
            T.clear(out)
            for i, d in T.Parallel(query_rows, padded):
                token = query + i // head_stride
                head = work_head * head_group + i % head_stride
                q[i, d] = T.if_then_else(
                    token < end and i % head_stride < head_group and d < dim,
                    Q[token, head * dim + d],
                    0,
                )
            if stages:
                live_k = T.decl_buffer((end, kv_heads * dim), dtype, data=K.data)
                live_v = T.decl_buffer((end, kv_heads * dim), dtype, data=V.data)
            for kb in T.Pipelined(
                T.ceildiv(T.min(end - base, query - base + bm) if causal else end - base, bn),
                num_stages=stages,
            ):
                if stages:
                    T.copy(
                        live_k[
                            base + kb * bn, (work_head // (heads // kv_heads // head_group)) * dim
                        ],
                        k,
                    )
                    T.copy(
                        live_v[
                            base + kb * bn, (work_head // (heads // kv_heads // head_group)) * dim
                        ],
                        v,
                    )
                else:
                    for j, d in T.Parallel(bn, padded):
                        k[j, d] = T.if_then_else(
                            base + kb * bn + j < end and d < dim,
                            K[
                                base + kb * bn + j,
                                (work_head // (heads // kv_heads // head_group)) * dim + d,
                            ],
                            0,
                        )
                        v[j, d] = T.if_then_else(
                            base + kb * bn + j < end and d < dim,
                            V[
                                base + kb * bn + j,
                                (work_head // (heads // kv_heads // head_group)) * dim + d,
                            ],
                            0,
                        )
                T.sync_threads()
                T.clear(score)
                T.gemm(q, k, score, transpose_B=True, policy=policy)
                for i in T.Parallel(query_rows):
                    previous[i] = maximum[i]
                for i, j in T.Parallel(query_rows, bn):
                    score[i, j] = T.if_then_else(
                        base + kb * bn + j < end
                        and (not causal or base + kb * bn + j <= query + i // head_stride),
                        score[i, j] * math.sqrt(1.0 / dim),
                        -1e30,
                    )
                T.reduce_max(score, maximum, dim=1, clear=False)
                for i, j in T.Parallel(query_rows, bn):
                    score[i, j] = T.if_then_else(
                        base + kb * bn + j < end
                        and (not causal or base + kb * bn + j <= query + i // head_stride),
                        (
                            T.exp2((score[i, j] - maximum[i]) * 1.4426950408889634)
                            if base2
                            else T.exp(score[i, j] - maximum[i])
                        ),
                        0,
                    )
                    p[i, j] = score[i, j]
                T.reduce_sum(score, rowsum, dim=1)
                for i in T.Parallel(query_rows):
                    correction[i] = (
                        T.exp2((previous[i] - maximum[i]) * 1.4426950408889634)
                        if base2
                        else T.exp(previous[i] - maximum[i])
                    )
                    denom[i] = denom[i] * correction[i] + rowsum[i]
                for i, d in T.Parallel(query_rows, padded):
                    out[i, d] *= correction[i]
                T.gemm(p, v, out, policy=policy)
            for i, d in T.Parallel(query_rows, padded):
                token = query + i // head_stride
                head = work_head * head_group + i % head_stride
                if (
                    token < rows
                    and (not window or qb * bm + i // head_stride < window)
                    and d < dim
                    and i % head_stride < head_group
                ):
                    Y[token, head * dim + d] = T.if_then_else(
                        token < end, out[i, d] / T.max(denom[i], 1e-30), 0
                    )

    return kernel
