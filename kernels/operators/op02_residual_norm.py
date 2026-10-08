"""SM87 residual + zero-centered RMSNorm with explicit output buffers.

The matched Qwen3.5/vLLM Gemma norm sums in FP32, normalizes that unrounded
sum, and returns FP32 residuals. FP16 residual output is an explicit alternate
storage contract, never an implicit change to the normalization input.
"""

import tilelang.language as T
from tools.operators.common import orin_jit


@orin_jit
def residual_norm(
    M: int | None = None,
    hidden: int = 5120,
    epsilon: float = 1e-6,
    residual_dtype: str = "float32",
    output_residual_dtype: str = "float32",
    threads: int = 256,
):
    """Build (X, R, W, Y, RO); M=None builds one runtime-row cubin.

    X/Y and zero-centered W are FP16, row-major contiguous. R and RO dtype
    are explicit. RO=cast(X.float()+R.float()); Y=half(sum*rsqrt(mean(sum²)
    +epsilon)*(1+W.float())). Inputs and outputs must not alias. The compiled
    adapter accepts an explicit CUDA stream via kernel(..., stream=handle).
    """
    assert hidden > 0 and epsilon > 0 and threads in (128, 256, 512)
    assert residual_dtype in ("float16", "float32")
    assert output_residual_dtype in ("float16", "float32")
    rows = T.dynamic("rows") if M is None else M
    columns = ((hidden + threads - 1) // threads) * threads

    @T.prim_func
    def kernel(
        X: T.Tensor((rows, hidden), T.float16),
        R: T.Tensor((rows, hidden), residual_dtype),
        W: T.Tensor((hidden,), T.float16),
        Y: T.Tensor((rows, hidden), T.float16),
        RO: T.Tensor((rows, hidden), output_residual_dtype),
    ):
        with T.Kernel(rows, threads=threads) as row:
            value = T.alloc_fragment((columns,), T.float32)
            square = T.alloc_fragment((columns,), T.float32)
            total = T.alloc_fragment((1,), T.float32)
            for j in T.Parallel(columns):
                value[j] = 0.0
                if j < hidden:
                    value[j] = T.cast(X[row, j], T.float32) + T.cast(R[row, j], T.float32)
                    RO[row, j] = value[j]
                square[j] = value[j] * value[j]
            T.reduce_sum(square, total, dim=0)
            for j in T.Parallel(columns):
                if j < hidden:
                    Y[row, j] = (value[j] * T.rsqrt(total[0] / hidden + epsilon)) * (
                        T.cast(W[j], T.float32) + 1.0
                    )

    return kernel


@orin_jit
def first_norm(M: int | None = None, hidden: int = 5120, epsilon: float = 1e-6, threads: int = 256):
    """Build (X, W, Y, RO), with exact FP16 RO=X for absent residual.

    This initializes the first layer residual independently of normalized Y.
    Subsequent calls to residual_norm add the mixer/down output exactly once.
    """
    assert hidden > 0 and epsilon > 0 and threads in (128, 256, 512)
    rows = T.dynamic("rows") if M is None else M
    columns = ((hidden + threads - 1) // threads) * threads

    @T.prim_func
    def kernel(
        X: T.Tensor((rows, hidden), T.float16),
        W: T.Tensor((hidden,), T.float16),
        Y: T.Tensor((rows, hidden), T.float16),
        RO: T.Tensor((rows, hidden), T.float16),
    ):
        with T.Kernel(rows, threads=threads) as row:
            value = T.alloc_fragment((columns,), T.float32)
            square = T.alloc_fragment((columns,), T.float32)
            total = T.alloc_fragment((1,), T.float32)
            for j in T.Parallel(columns):
                value[j] = 0.0
                if j < hidden:
                    value[j] = T.cast(X[row, j], T.float32)
                    RO[row, j] = X[row, j]
                square[j] = value[j] * value[j]
            T.reduce_sum(square, total, dim=0)
            for j in T.Parallel(columns):
                if j < hidden:
                    Y[row, j] = (value[j] * T.rsqrt(total[0] / hidden + epsilon)) * (
                        T.cast(W[j], T.float32) + 1.0
                    )

    return kernel
