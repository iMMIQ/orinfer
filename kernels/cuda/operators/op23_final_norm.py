"""SM87 last-row gather and zero-centered final RMSNorm.

Only B selected rows are read/normalized; no full-prompt norm intermediate.
Outputs are caller-owned and immutable inputs make graph repetitions safe.
"""

import tilelang.language as T
from orinfer_build.compiler.cuda.common import orin_jit


@orin_jit
def final_norm(
    M: int | None = None,
    B: int | None = None,
    hidden: int = 5120,
    epsilon: float = 1e-6,
    threads: int = 256,
    last_row: bool = False,
    zero_centered: bool = True,
):
    """Build (X, R, I, W, Y): FP16 hidden plus FP32 residual, FP16 Y.

    Contiguous X/R[M,H], int32 I[B], zero-centered FP16 W[H], Y[B,H].
    Each CTA gathers one row I[b], adds in FP32, norms the unrounded sum,
    and rounds only the final weighted value to FP16. No residual output.
    All tensor buffers must be distinct; I in [0,M), finite inputs, explicit
    stream and stable addresses during graph replay.
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
                        T.cast(W[j], T.float32) + (1.0 if zero_centered else 0.0)
                    )

    return kernel
