"""Qwen4 gated residual streams with explicit low-precision boundaries.

Norm weights are zero-centered FP32. Linear projections retain BF16 weights;
state/output dtype is chosen by the caller. FP32 reductions do not change the
materialization boundaries of norm, SiLU, sigmoid, product or residual updates.

Elementwise loops use scalar coalescing: the pinned TileLang/NVRTC combination
can lower vectorized BF16 conversions to trap-only cubins on SM87. Keeping
coalesced_width as an explicit IntImm avoids that lowering; the projections
still use BF16 Tensor Cores.
"""

import tilelang
import tilelang.language as T

from orinfer_build.compiler.cuda.common import orin_jit
from kernels.shared.rows import row_count
from kernels.cuda.model.half_math import SOURCE as HALF_MATH, stream_sigmoid
from kernels.cuda.model.floating_core import mma_loop


def _dtype(value):
    if value not in ("float16", "bfloat16"):
        raise ValueError("HC stream dtype must be FP16 or BF16")
    return value


def combine_fragment(H, streams, dtype):
    """Residual injection and its FP32 norm reduction, retaining stream rounding."""
    width = 1 << (H - 1).bit_length()
    vec_fp16 = dtype == "float16"

    @T.macro
    def combine(Block, Residual, Inject, Output, values, square, total, row, branch):
        raw = T.cast(T.cast(T.cast(Inject[row, branch], T.float32) / streams, dtype), T.float32)
        gate_val = T.alloc_local((1,), T.float32)
        gate_val[0] = T.cast(
            T.cast(2.0 * T.cast(stream_sigmoid(raw, dtype), T.float32), dtype), T.float32
        )
        if vec_fp16:
            for j in T.Parallel(width):
                values[j] = 0.0
                if j < H:
                    injection = T.cast(T.cast(Block[row, j], T.float32) * gate_val[0], dtype)
                    values[j] = T.cast(
                        T.cast(
                            T.cast(Residual[row, branch, j], T.float32)
                            + T.cast(injection, T.float32),
                            dtype,
                        ),
                        T.float32,
                    )
                    Output[row, branch, j] = T.cast(values[j], dtype)
                square[j] = values[j] * values[j]
        else:
            for j in T.Parallel(width, coalesced_width=T.int32(1)):
                values[j] = 0.0
                if j < H:
                    injection = T.cast(T.cast(Block[row, j], T.float32) * gate_val[0], dtype)
                    values[j] = T.cast(
                        T.cast(
                            T.cast(Residual[row, branch, j], T.float32)
                            + T.cast(injection, T.float32),
                            dtype,
                        ),
                        T.float32,
                    )
                    Output[row, branch, j] = T.cast(values[j], dtype)
                square[j] = values[j] * values[j]
        T.reduce_sum(square, total, dim=0)

    return combine


def norm_fragment(H, eps, dtype):
    """Keep the HC norm tree and stream rounding when consumed by a fusion."""
    width = 1 << (H - 1).bit_length()

    @T.macro
    def normalize(Residual, Weight, Normed, row, branch):
        values = T.alloc_fragment((width,), T.float32)
        square = T.alloc_fragment((width,), T.float32)
        total = T.alloc_fragment((1,), T.float32)
        for j in T.Parallel(width, coalesced_width=T.int32(1)):
            values[j] = T.if_then_else(j < H, T.cast(Residual[row, branch, j], T.float32), 0.0)
            square[j] = values[j] * values[j]
        T.reduce_sum(square, total, dim=0)
        for j in T.Parallel(width, coalesced_width=T.int32(1)):
            Normed[j] = 0
            if j < H:
                Normed[j] = T.cast(
                    values[j] * T.rsqrt(total[0] / H + eps) * (1.0 + Weight[branch, j]), dtype
                )

    return normalize


@orin_jit
def hc_norm(
    M: int,
    H: int,
    streams: int = 4,
    eps: float = 1e-6,
    dtype: str = "bfloat16",
    dynamic_rows: bool = False,
):
    """Build (Residual[M,streams,H], Weight[streams,H], Normed), dtype/F32/dtype."""
    _dtype(dtype)
    if any(type(x) is not int or x <= 0 for x in (M, H, streams)) or not 0 < eps < 1:
        raise ValueError("Invalid HC normalization dimensions/epsilon")
    width = 1 << (H - 1).bit_length()
    M = row_count(M, dynamic_rows)
    normalize = norm_fragment(H, eps, dtype)

    @T.prim_func
    def main(
        Residual: T.Tensor((M, streams, H), dtype),
        Weight: T.Tensor((streams, H), T.float32),
        Normed: T.Tensor((M, streams, H), dtype),
    ):
        with T.Kernel(M, streams, threads=min(256, width)) as (row, branch):
            normalized = T.alloc_fragment((width,), dtype)
            normalize(Residual, Weight, normalized, row, branch)
            for j in T.Parallel(H, coalesced_width=T.int32(1)):
                Normed[row, branch, j] = normalized[j]

    return main


@orin_jit
def hc_silu(
    M: int, rank: int, streams: int = 4, dtype: str = "bfloat16", dynamic_rows: bool = False
):
    """Build (Down[M,rank], Activated), dividing by stream count before SiLU."""
    _dtype(dtype)
    if any(type(x) is not int or x <= 0 for x in (M, rank, streams)):
        raise ValueError("Invalid HC low-rank dimensions")
    M = row_count(M, dynamic_rows)

    @T.prim_func
    def main(Down: T.Tensor((M, rank), dtype), Activated: T.Tensor((M, rank), dtype)):
        with T.Kernel(M, threads=128) as row:
            for j in T.Parallel(rank, coalesced_width=T.int32(1)):
                v = T.cast(T.cast(T.cast(Down[row, j], T.float32) / streams, dtype), T.float32)
                Activated[row, j] = v / (1.0 + T.exp(-v))

    return main


@orin_jit
def hc_mix(M: int, H: int, streams: int = 4, dtype: str = "bfloat16", dynamic_rows: bool = False):
    """Build (Normed[M,streams,H], Up[M,streams,H], Mixed[M,H])."""
    _dtype(dtype)
    if any(type(x) is not int or x <= 0 for x in (M, H, streams)):
        raise ValueError("Invalid HC mixing dimensions")
    M = row_count(M, dynamic_rows)

    @T.prim_func
    def main(
        Normed: T.Tensor((M, streams, H), dtype),
        Up: T.Tensor((M, streams, H), dtype),
        Mixed: T.Tensor((M, H), dtype),
    ):
        with T.Kernel(M, T.ceildiv(H, 256), threads=128) as (row, block):
            T.import_source(HALF_MATH)
            total = T.alloc_fragment((256,), T.float32)
            T.clear(total)
            for branch in T.serial(streams):
                for j in T.Parallel(256, coalesced_width=T.int32(1)):
                    col = block * 256 + j
                    if col < H:
                        gate = stream_sigmoid(T.cast(Up[row, branch, col], T.float32), dtype)
                        product = T.cast(
                            T.cast(gate, T.float32) * T.cast(Normed[row, branch, col], T.float32),
                            dtype,
                        )
                        total[j] += T.cast(product, T.float32)
            for j in T.Parallel(256, coalesced_width=T.int32(1)):
                if block * 256 + j < H:
                    Mixed[row, block * 256 + j] = total[j] / streams

    return main


@orin_jit
def hc_combine(
    M: int, H: int, streams: int = 4, dtype: str = "bfloat16", dynamic_rows: bool = False
):
    """Build (Block[M,H], Residual[M,streams,H], Inject[M,streams], Output).

    Inject is the linear projection of normed residuals before division/sigmoid.
    Output may alias Residual: each element has exactly one owner and is read
    before its write. The Block and Inject inputs must remain disjoint.
    """
    _dtype(dtype)
    if any(type(x) is not int or x <= 0 for x in (M, H, streams)):
        raise ValueError("Invalid HC combination dimensions")
    M = row_count(M, dynamic_rows)

    @T.prim_func
    def main(
        Block: T.Tensor((M, H), dtype),
        Residual: T.Tensor((M, streams, H), dtype),
        Inject: T.Tensor((M, streams), dtype),
        Output: T.Tensor((M, streams, H), dtype),
    ):
        with T.Kernel(M, streams, T.ceildiv(H, 256), threads=128) as (row, branch, block):
            T.import_source(HALF_MATH)
            raw = T.cast(T.cast(T.cast(Inject[row, branch], T.float32) / streams, dtype), T.float32)
            gate = T.cast(2.0 * T.cast(stream_sigmoid(raw, dtype), T.float32), dtype)
            for j in T.Parallel(256, coalesced_width=T.int32(1)):
                col = block * 256 + j
                if col < H:
                    injection = T.cast(
                        T.cast(Block[row, col], T.float32) * T.cast(gate, T.float32), dtype
                    )
                    Output[row, branch, col] = T.cast(
                        Residual[row, branch, col], T.float32
                    ) + T.cast(injection, T.float32)

    return main


@orin_jit
def hc_combine_norm(
    M: int,
    H: int,
    streams: int = 4,
    eps: float = 1e-6,
    dtype: str = "bfloat16",
    dynamic_rows: bool = False,
):
    """Combine residuals and normalize, retaining the rounded residual output.

    Output may alias Residual. Normed must remain disjoint from all inputs and
    Output. Each CTA owns one whole row/stream, including the FP32 reduction.
    The intervening stream-dtype rounding matches hc_combine then hc_norm.
    """
    _dtype(dtype)
    if any(type(x) is not int or x <= 0 for x in (M, H, streams)) or not 0 < eps < 1:
        raise ValueError("Invalid HC combination/normalization dimensions")
    width = 1 << (H - 1).bit_length()
    M = row_count(M, dynamic_rows)
    combine = combine_fragment(H, streams, dtype)
    vec_fp16 = dtype == "float16"

    @T.macro
    def body(Block, Residual, Inject, Weight, Output, Normed):
        with T.Kernel(M, streams, threads=min(256, width)) as (row, branch):
            T.import_source(HALF_MATH)
            values = T.alloc_fragment((width,), T.float32)
            square = T.alloc_fragment((width,), T.float32)
            total = T.alloc_fragment((1,), T.float32)
            combine(Block, Residual, Inject, Output, values, square, total, row, branch)
            inverse = T.rsqrt(total[0] / H + eps)
            if vec_fp16:
                for j in T.Parallel(H):
                    Normed[row, branch, j] = values[j] * inverse * (1.0 + Weight[branch, j])
            else:
                for j in T.Parallel(H, coalesced_width=T.int32(1)):
                    Normed[row, branch, j] = values[j] * inverse * (1.0 + Weight[branch, j])

    if dynamic_rows:

        @T.prim_func
        def hc_combine_norm_rows(
            Block: T.Tensor((M, H), dtype),
            Residual: T.Tensor((M, streams, H), dtype),
            Inject: T.Tensor((M, streams), dtype),
            Weight: T.Tensor((streams, H), T.float32),
            Output: T.Tensor((M, streams, H), dtype),
            Normed: T.Tensor((M, streams, H), dtype),
        ):
            body(Block, Residual, Inject, Weight, Output, Normed)

        return hc_combine_norm_rows

    @T.prim_func
    def hc_combine_norm(
        Block: T.Tensor((M, H), dtype),
        Residual: T.Tensor((M, streams, H), dtype),
        Inject: T.Tensor((M, streams), dtype),
        Weight: T.Tensor((streams, H), T.float32),
        Output: T.Tensor((M, streams, H), dtype),
        Normed: T.Tensor((M, streams, H), dtype),
    ):
        body(Block, Residual, Inject, Weight, Output, Normed)

    return hc_combine_norm


def _prepare_hc_projection(
    M: int,
    N: int,
    K: int,
    *,
    block_m: int,
    dtype: str,
    block_n: int,
    silu: bool,
    streams: int,
    group_m: int,
    dynamic_rows: bool,
    stages: int,
    cast_input: bool,
    cast_silu: bool,
    coalesced_silu: bool,
    reduction_groups: int,
):
    _dtype(dtype)
    if (
        any(type(x) is not int or x <= 0 for x in (M, N, K))
        or K % 64
        or block_m not in (16, 32, 64, 128)
    ):
        raise ValueError("Invalid HC projection dimensions")
    if block_n not in (16, 32, 64, 128):
        raise ValueError("Invalid HC output tile")
    if type(group_m) is not int or group_m not in (0, 1, 2, 4, 8):
        raise ValueError("Invalid HC row grouping")
    if (
        type(reduction_groups) is not int
        or reduction_groups not in (1, 4)
        or K % (64 * reduction_groups)
    ):
        raise ValueError("Invalid HC reduction grouping")
    threads = 64 if block_n == 16 else 128
    M = row_count(M, dynamic_rows)
    accumulate = mma_loop(
        M,
        K,
        block_m,
        block_n,
        64,
        stages=stages,
        reduction_groups=reduction_groups,
        cast_input=cast_input,
    )

    @T.macro
    def project(A, W, C, by, bx):
        acc = T.alloc_fragment((block_m, block_n), T.float32)
        accumulate(A, W, acc, by, bx, 0)
        if silu:
            if coalesced_silu:
                for i, j in T.Parallel(block_m, block_n, coalesced_width=T.int32(1)):
                    if by * block_m + i < M and bx * block_n + j < N:
                        rounded = T.cast(acc[i, j], dtype)
                        v = T.cast(T.cast(T.cast(rounded, T.float32) / streams, dtype), T.float32)
                        value = v / (1 + T.exp(-v))
                        C[by * block_m + i, bx * block_n + j] = (
                            T.cast(value, dtype) if cast_silu else value
                        )
            else:
                for i, j in T.Parallel(block_m, block_n):
                    if by * block_m + i < M and bx * block_n + j < N:
                        rounded = T.cast(acc[i, j], dtype)
                        v = T.cast(T.cast(T.cast(rounded, T.float32) / streams, dtype), T.float32)
                        value = v / (1 + T.exp(-v))
                        C[by * block_m + i, bx * block_n + j] = (
                            T.cast(value, dtype) if cast_silu else value
                        )
        else:
            T.copy(acc, C[by * block_m, bx * block_n])

    return M, threads, project


@orin_jit
def hc_projection(
    M: int,
    N: int,
    K: int,
    block_m: int = 16,
    dtype: str = "bfloat16",
    block_n: int = 64,
    silu: bool = False,
    streams: int = 4,
    group_m: int = 0,
    dynamic_rows: bool = False,
    reduction_groups: int = 1,
):
    """Build (A[M,K], W[N,K], C[M,N]), stream dtype/BF16/stream dtype.

    FP32 accumulation, BF16 Tensor Core arithmetic, explicit output rounding.
    This preserves the original BF16 HC weights. N tails are supported for
    the four-stream injection projection; K must be a multiple of 64.
    Four reduction groups match the decode HC down partials: accumulate each
    contiguous stream in FP32, then add streams in order before FP16 rounding.
    """
    M, threads, project = _prepare_hc_projection(
        M,
        N,
        K,
        block_m=block_m,
        dtype=dtype,
        block_n=block_n,
        silu=silu,
        streams=streams,
        group_m=group_m,
        dynamic_rows=dynamic_rows,
        stages=2,
        cast_input=True,
        cast_silu=False,
        coalesced_silu=False,
        reduction_groups=reduction_groups,
    )
    if group_m:
        mt, nt = (M + block_m - 1) // block_m, (N + block_n - 1) // block_n

        @T.prim_func
        def grouped(
            A: T.Tensor((M, K), dtype), W: T.Tensor((N, K), T.bfloat16), C: T.Tensor((M, N), dtype)
        ):
            with T.Kernel(mt * nt, threads=threads) as pid:
                first = (pid // (group_m * nt)) * group_m
                actual = T.min(mt - first, group_m)
                by = first + (pid % (group_m * nt)) % actual
                bx = (pid % (group_m * nt)) // actual
                project(A, W, C, by, bx)

        return grouped

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype), W: T.Tensor((N, K), T.bfloat16), C: T.Tensor((M, N), dtype)
    ):
        with T.Kernel(T.ceildiv(M, block_m), T.ceildiv(N, block_n), threads=threads) as (
            by,
            bx,
        ):
            project(A, W, C, by, bx)

    return main


@orin_jit
def hc_up_mix(
    M: int,
    H: int,
    rank: int,
    streams: int = 4,
    dynamic_rows: bool = False,
    block_m: int | None = None,
):
    """BF16 up projection and branch mix, preserving every FP16 boundary."""
    assert M >= 1 and H % 32 == 0 and rank % 64 == 0 and streams == 4
    small = M <= 8
    bm, bn = (8 if small else 16), 32
    if block_m is not None:
        if small or block_m not in (16, 32, 64, 128):
            raise ValueError("Unsupported HC up row tile")
        bm = block_m
    M = row_count(M, dynamic_rows)

    @T.prim_func
    def main(
        A: T.Tensor((M, rank), T.float16),
        W: T.Tensor((streams * H, rank), T.bfloat16),
        Normed: T.Tensor((M, streams, H), T.float16),
        Out: T.Tensor((M, H), T.float16),
    ):
        with T.Kernel(T.ceildiv(M, bm), T.ceildiv(H, bn), threads=128) as (by, bx):
            T.import_source(HALF_MATH)
            a = T.alloc_shared((bm, 64), T.bfloat16)
            w = T.alloc_shared((streams * bn, 64), T.bfloat16)
            shape = (streams * bn, bm) if small else (bm, streams * bn)
            acc = T.alloc_fragment(shape, T.float32)
            rounded_up = T.alloc_shared(shape, T.float16)
            total = T.alloc_fragment((bm, bn), T.float32)
            T.annotate_layout(
                {
                    total: tilelang.Fragment(
                        (bm, bn),
                        forward_thread_fn=lambda i, j: (i % 4) * 32 + j,
                        forward_index_fn=lambda i, j: i // 4,
                    )
                }
            )
            T.clear(acc)
            T.clear(total)
            for kg in T.Pipelined(rank // 64, num_stages=2):
                for i, j in T.Parallel(bm, 64):
                    a[i, j] = 0
                    if by * bm + i < M:
                        a[i, j] = A[by * bm + i, kg * 64 + j]
                for n, j in T.Parallel(streams * bn, 64):
                    w[n, j] = W[(n // bn) * H + bx * bn + n % bn, kg * 64 + j]
                # N x 8 MMA avoids the sixteen-row activation tile for small
                # decode/verification batches. Preserve the same K traversal.
                if small:
                    T.gemm(w, a, acc, transpose_B=True)
                else:
                    T.gemm(a, w, acc, transpose_B=True)
            T.copy(acc, rounded_up)
            for branch in T.serial(streams):
                for i, j in T.Parallel(bm, bn, coalesced_width=T.int32(1)):
                    if by * bm + i < M:
                        up = (
                            rounded_up[branch * bn + j, i]
                            if small
                            else rounded_up[i, branch * bn + j]
                        )
                        gate = stream_sigmoid(T.cast(up, T.float32), "float16")
                        product = T.cast(
                            T.cast(gate, T.float32)
                            * T.cast(Normed[by * bm + i, branch, bx * bn + j], T.float32),
                            T.float16,
                        )
                        total[i, j] += T.cast(product, T.float32)
            for i, j in T.Parallel(bm, bn, coalesced_width=T.int32(1)):
                if by * bm + i < M:
                    Out[by * bm + i, bx * bn + j] = total[i, j] / streams

    return main


@orin_jit
def hc_injection(
    M: int, N: int, K: int, dtype: str = "float16", dynamic_rows: bool = False, block_m: int = 1
):
    """Narrow BF16 MMA with the same K16 accumulation as bulk projection.

    Cache a token tile and one weight row; each output channel owns a CTA.
    No split-K or floating sum changes with the logical token count.
    """
    from kernels.cuda.model.flash_next import scalar_bf16_projection

    if N != 4:
        raise ValueError("HC injection requires four stream gates")
    return scalar_bf16_projection.__wrapped__(
        M, K, dynamic_rows, outputs=N, block_m=block_m, dtype=dtype
    )


@orin_jit
def hc_down_partial(M: int, N: int, K: int, splits: int = 4, dynamic_rows: bool = False):
    """BF16 Tensor Core HC Down slices with FP32 partial sums.

    Partials are transient plan workspace, overwritten completely on every
    call. Do not round partials or apply SiLU before the final reduction.
    """
    if (
        any(type(x) is not int or x <= 0 for x in (M, N, K))
        or splits not in (2, 4, 8, 16)
        or K % (64 * splits)
    ):
        raise ValueError("Invalid HC split-K dimensions")
    # Small verification profiles repeatedly convert padded activation rows
    # within each K tile. Cache only live rows, retaining the four split sums
    # and the original BF16 MMA traversal. Large dynamic capacities keep the
    # streaming plan instead of reserving an oversized activation cache.
    cached = M <= 8
    live_rows = M if cached else 0
    bm, bn = 16, 64 if cached else 32
    M = row_count(M, dynamic_rows)

    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.float16),
        W: T.Tensor((N, K), T.bfloat16),
        Partials: T.Tensor((splits, M, N), T.float32),
    ):
        with T.Kernel(T.ceildiv(M, bm), T.ceildiv(N, bn), splits, threads=128) as (by, bx, part):
            a = T.alloc_shared((bm, 64), T.bfloat16)
            w = T.alloc_shared((bn, 64), T.bfloat16)
            acc = T.alloc_fragment((bm, bn), T.float32)
            if cached:
                live = T.alloc_shared((live_rows, K // splits), T.bfloat16)
                for i, j in T.Parallel(live_rows, K // splits):
                    live[i, j] = 0
                    if by * bm + i < M:
                        live[i, j] = A[by * bm + i, part * (K // splits) + j]
            T.clear(acc)
            for kg in T.Pipelined(K // (64 * splits), num_stages=2):
                for i, j in T.Parallel(bm, 64, coalesced_width=T.int32(1)):
                    a[i, j] = 0
                    if by * bm + i < M:
                        if cached:
                            a[i, j] = live[i, kg * 64 + j]
                        else:
                            a[i, j] = A[by * bm + i, part * (K // splits) + kg * 64 + j]
                T.copy(W[bx * bn, part * (K // splits) + kg * 64], w)
                T.gemm(a, w, acc, transpose_B=True)
            for i, j in T.Parallel(bm, bn):
                if by * bm + i < M and bx * bn + j < N:
                    Partials[part, by * bm + i, bx * bn + j] = acc[i, j]

    return main


@orin_jit
def hc_down_finish(M: int, N: int, splits: int = 4, streams: int = 4, dynamic_rows: bool = False):
    """Reduce FP32 HC Down slices, then round/divide/SiLU in original order."""
    if any(type(x) is not int or x <= 0 for x in (M, N, streams)) or splits not in (2, 4, 8, 16):
        raise ValueError("Invalid HC split-K epilogue dimensions")
    M = row_count(M, dynamic_rows)

    @T.prim_func
    def main(Partials: T.Tensor((splits, M, N), T.float32), Out: T.Tensor((M, N), T.float16)):
        with T.Kernel(T.ceildiv(M * N, 128), threads=128) as block:
            acc = T.alloc_fragment((128,), T.float32)
            T.clear(acc)
            for part in T.serial(splits):
                for i in T.Parallel(128):
                    index = block * 128 + i
                    if index < M * N:
                        acc[i] += Partials[part, index // N, index % N]
            for i in T.Parallel(128):
                index = block * 128 + i
                if index < M * N:
                    rounded = T.cast(acc[i], T.float16)
                    value = T.cast(
                        T.cast(T.cast(rounded, T.float32) / streams, T.float16), T.float32
                    )
                    Out[index // N, index % N] = value / (1 + T.exp(-value))

    return main


@orin_jit
def bf16_projection_small(
    M: int,
    N: int,
    K: int,
    *,
    splits: int = 1,
    block_m: int = 8,
    block_n: int = 16,
    block_k: int = 128,
    stages: int = 2,
    threads: int = 64,
    dynamic_rows: bool = False,
):
    """Weight-left BF16 Tensor Core projection for small decode/lookup batches.

    Preserves the exact m16n8k16 K-traversal and 4-stream split layout of
    bf16_rows_kernel_kernel while raising SM occupancy on small M (3..15).
    """
    if (
        any(
            type(x) is not int or x <= 0
            for x in (M, N, K, splits, block_m, block_n, block_k, stages, threads)
        )
        or splits not in (1, 4)
        or block_m not in (8, 16)
        or block_n not in (16, 32, 64)
        or block_k not in (64, 128)
        or N % block_n
        or K % (block_k * splits)
        or threads not in (32, 64, 128)
    ):
        raise ValueError("Invalid small BF16 projection tile")
    sub_k = K // splits
    M = row_count(M, dynamic_rows)

    if splits == 1:

        @T.prim_func
        def bf16_small_router(
            A: T.Tensor((M, K), T.float16),
            Weight: T.Tensor((N, K), T.bfloat16),
            Output: T.Tensor((M, N), T.float32),
        ):
            with T.Kernel(T.ceildiv(M, block_m), N // block_n, 1, threads=threads) as (by, bx, _):
                w = T.alloc_shared((block_n, block_k), T.bfloat16)
                a = T.alloc_shared((block_m, block_k), T.bfloat16)
                acc = T.alloc_fragment((block_n, block_m), T.float32)
                T.clear(acc)
                for kg in T.Pipelined(sub_k // block_k, num_stages=stages):
                    T.copy(Weight[bx * block_n, kg * block_k], w)
                    for i, j in T.Parallel(block_m, block_k, coalesced_width=T.int32(1)):
                        a[i, j] = 0
                        if by * block_m + i < M:
                            a[i, j] = A[by * block_m + i, kg * block_k + j]
                    T.gemm(w, a, acc, transpose_B=True)
                for n, m_idx in T.Parallel(block_n, block_m, coalesced_width=T.int32(1)):
                    if by * block_m + m_idx < M:
                        Output[by * block_m + m_idx, bx * block_n + n] = acc[n, m_idx]

        return bf16_small_router

    @T.prim_func
    def bf16_small_split(
        A: T.Tensor((M, K), T.float16),
        Weight: T.Tensor((N, K), T.bfloat16),
        Output: T.Tensor((splits, M, N), T.float32),
    ):
        with T.Kernel(T.ceildiv(M, block_m), N // block_n, splits, threads=threads) as (
            by,
            bx,
            part,
        ):
            w = T.alloc_shared((block_n, block_k), T.bfloat16)
            a = T.alloc_shared((block_m, block_k), T.bfloat16)
            acc = T.alloc_fragment((block_n, block_m), T.float32)
            T.clear(acc)
            for kg in T.Pipelined(sub_k // block_k, num_stages=stages):
                offset = part * sub_k + kg * block_k
                T.copy(Weight[bx * block_n, offset], w)
                for i, j in T.Parallel(block_m, block_k, coalesced_width=T.int32(1)):
                    a[i, j] = 0
                    if by * block_m + i < M:
                        a[i, j] = A[by * block_m + i, offset + j]
                T.gemm(w, a, acc, transpose_B=True)
            for n, m_idx in T.Parallel(block_n, block_m, coalesced_width=T.int32(1)):
                if by * block_m + m_idx < M:
                    Output[part, by * block_m + m_idx, bx * block_n + n] = acc[n, m_idx]

    return bf16_small_split


_HC_INJECT_DOT8 = r"""
#include <cuda_fp16.h>
__device__ __forceinline__ float hc_dot8_f16_bf16(const void* a_ptr, const void* w_ptr) {
    uint4 av = *reinterpret_cast<const uint4*>(a_ptr);
    uint4 wv = *reinterpret_cast<const uint4*>(w_ptr);
    const __half* ah = reinterpret_cast<const __half*>(&av);
    const unsigned short* wh = reinterpret_cast<const unsigned short*>(&wv);
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float fa = __half2float(ah[i]);
        unsigned short ab;
        asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(ab) : "f"(fa));
        float af = __uint_as_float(static_cast<unsigned>(ab) << 16);
        float wf = __uint_as_float(static_cast<unsigned>(wh[i]) << 16);
        sum = fmaf(af, wf, sum);
    }
    return sum;
}
"""


@orin_jit
def hc_injection_fast(
    M: int,
    K: int = 10240,
    outputs: int = 4,
    dynamic_rows: bool = False,
):
    """Full-SM 128-bit vectorized HC injection dot product with BF16 input rounding."""
    if any(type(x) is not int or x <= 0 for x in (M, K, outputs)) or K % 1024:
        raise ValueError("Invalid fast HC injection geometry")
    M = row_count(M, dynamic_rows)
    chunks = K // 1024  # 128 threads * 8 elements = 1024 elements per step

    @T.prim_func
    def hc_inject_vec(
        A: T.Tensor((M, K), T.float16),
        Weight: T.Tensor((outputs, K), T.bfloat16),
        Output: T.Tensor((M, outputs), T.float16),
    ):
        with T.Kernel(M, outputs, threads=128) as (row, col):
            T.import_source(_HC_INJECT_DOT8)
            partial = T.alloc_fragment((128,), T.float32)
            total = T.alloc_fragment((1,), T.float32)
            T.clear(partial)
            for step in T.unroll(chunks):
                for t in T.Parallel(128):
                    offset = step * 1024 + t * 8
                    partial[t] += T.call_pure_extern(
                        "float32",
                        "hc_dot8_f16_bf16",
                        T.access_ptr(A[row, offset], "r", 8),
                        T.access_ptr(Weight[col, offset], "r", 8),
                    )
            T.reduce_sum(partial, total, dim=0)
            if T.get_thread_binding() == 0:
                Output[row, col] = T.cast(total[0], T.float16)

    return hc_inject_vec
