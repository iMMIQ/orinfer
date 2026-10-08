"""SM87 last-row gather and zero-centered final RMSNorm.

Only B selected rows are read/normalized; no full-prompt norm intermediate.
Outputs are caller-owned and immutable inputs make graph repetitions safe.
"""

import tilelang.language as T
from tools.operators.common import orin_jit


def validate_last_indices(indices, rows: int, batch: int | None = None):
    """Validate CPU scheduler indices before uploading/capturing the raw kernel.

    Raw device launch requires this contract, including after indices change.
    Duplicate and unordered indices are legal; empty batches are not launched.
    """
    if type(rows) is not int or not 0 < rows <= 2147483647:
        raise ValueError("rows must be a positive int32 dimension")
    if not isinstance(indices, (list, tuple)) or not indices:
        raise ValueError("indices must be a nonempty CPU list/tuple")
    if batch is not None and (type(batch) is not int or batch != len(indices)):
        raise ValueError("batch must equal the number of indices")
    if len(indices) > 2147483647:
        raise ValueError("batch exceeds int32")
    if any(type(i) is not int or not 0 <= i < rows for i in indices):
        raise ValueError("every last-row index must be an integer in [0, rows)")
    return tuple(indices)


@orin_jit
def final_norm(
    M: int | None = None,
    B: int | None = None,
    hidden: int = 5120,
    epsilon: float = 1e-6,
    threads: int = 256,
    last_row: bool = False,
):
    """Build (X, R, I, W, Y): FP16 hidden plus FP32 residual, FP16 Y.

    Contiguous X/R[M,H], int32 I[B], zero-centered FP16 W[H], Y[B,H].
    Each CTA gathers one row I[b], adds in FP32, norms the unrounded sum,
    and rounds only the final weighted value to FP16. No residual output.
    All tensor buffers must be distinct; I in [0,M), finite inputs, explicit
    stream, stable addresses during graph replay; see validate_last_indices.
    """
    assert hidden > 0 and epsilon > 0 and threads in (128, 256, 512)
    assert M is None or M > 0
    assert B is None or B > 0
    assert not last_row or (M is not None and B == 1)
    rows = T.dynamic("rows") if M is None else M
    batch = T.dynamic("batch") if B is None else B
    columns = ((hidden + threads - 1) // threads) * threads

    @T.prim_func
    def kernel(
        X: T.Tensor((rows, hidden), T.float16),
        R: T.Tensor((rows, hidden), T.float32),
        I: T.Tensor((batch,), T.int32),
        W: T.Tensor((hidden,), T.float16),
        Y: T.Tensor((batch, hidden), T.float16),
    ):
        with T.Kernel(batch, threads=threads) as b:
            value = T.alloc_fragment((columns,), T.float32)
            square = T.alloc_fragment((columns,), T.float32)
            total = T.alloc_fragment((1,), T.float32)
            for j in T.Parallel(columns):
                value[j] = 0.0
                if j < hidden:
                    value[j] = (
                        T.cast(X[rows - 1 if last_row else I[b], j], T.float32)
                        + R[rows - 1 if last_row else I[b], j]
                    )
                square[j] = value[j] * value[j]
            T.reduce_sum(square, total, dim=0)
            for j in T.Parallel(columns):
                if j < hidden:
                    Y[b, j] = (value[j] * T.rsqrt(total[0] / hidden + epsilon)) * (
                        T.cast(W[j], T.float32) + 1.0
                    )

    return kernel


@orin_jit
def final_norm_presummed(
    M: int | None = None,
    B: int | None = None,
    hidden: int = 5120,
    epsilon: float = 1e-6,
    input_dtype: str = "float32",
    threads: int = 256,
):
    """Build (U, I, W, Y), normalizing selected already-summed rows.

    FP32 U preserves the FP32 residual sum. FP16 U is a separate no-residual
    hidden mode; pre-rounding a residual sum to FP16 changes the contract.
    Y is always FP16, including FP32 input (unlike native orig_dtype output).
    Layout/index/alias/stream contracts are the same as final_norm.
    """
    assert hidden > 0 and epsilon > 0 and threads in (128, 256, 512)
    assert input_dtype in ("float16", "float32")
    assert M is None or M > 0
    assert B is None or B > 0
    rows = T.dynamic("rows") if M is None else M
    batch = T.dynamic("batch") if B is None else B
    columns = ((hidden + threads - 1) // threads) * threads

    @T.prim_func
    def kernel(
        U: T.Tensor((rows, hidden), input_dtype),
        I: T.Tensor((batch,), T.int32),
        W: T.Tensor((hidden,), T.float16),
        Y: T.Tensor((batch, hidden), T.float16),
    ):
        with T.Kernel(batch, threads=threads) as b:
            value = T.alloc_fragment((columns,), T.float32)
            square = T.alloc_fragment((columns,), T.float32)
            total = T.alloc_fragment((1,), T.float32)
            for j in T.Parallel(columns):
                value[j] = 0.0
                if j < hidden:
                    value[j] = T.cast(U[I[b], j], T.float32)
                square[j] = value[j] * value[j]
            T.reduce_sum(square, total, dim=0)
            for j in T.Parallel(columns):
                if j < hidden:
                    Y[b, j] = (value[j] * T.rsqrt(total[0] / hidden + epsilon)) * (
                        T.cast(W[j], T.float32) + 1.0
                    )

    return kernel


@orin_jit
def last_hidden_gather(dtype: str = "float16", hidden: int = 5120, threads: int = 256):
    """Exact-copy diagnostic/cache helper (X,I,G); not needed by fused norm."""
    assert dtype in ("float16", "float32") and hidden > 0
    rows, batch = T.dynamic("rows"), T.dynamic("batch")

    @T.prim_func
    def kernel(
        X: T.Tensor((rows, hidden), dtype),
        I: T.Tensor((batch,), T.int32),
        G: T.Tensor((batch, hidden), dtype),
    ):
        with T.Kernel(batch, threads=threads) as b:
            for j in T.Parallel(hidden):
                G[b, j] = X[I[b], j]

    return kernel
