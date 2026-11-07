"""GDN ordinary per-value-head RMSNorm then SiLU gate; SM87 TileLang.

X[M,48,128] FP16/FP32, Z same shape FP16, shared ordinary W[128]
FP16, Y same shape FP16. All arrays contiguous, nonaliasing; eps=1e-6.
FP32 arithmetic throughout; only the final output is rounded to FP16.
"""

import tilelang
import tilelang.language as T

HEADS, WIDTH = 48, 128


@T.macro
def gated_norm_epilogue(x, z, weight, rstd):
    """Fusion expression after the owner CTA computes full-head FP32 rstd.

    rstd=rsqrt(sum(x.float()**2)/128 + 1e-6). Do not round recurrent
    output/norm/gate intermediates; this expression explicitly rounds Y.
    """
    xf = T.cast(x, T.float32)
    zf = T.cast(z, T.float32)
    wf = T.cast(weight, T.float32)
    sigmoid = 1.0 / (1.0 + T.exp(-zf))
    return T.cast(((xf * rstd) * wf) * (zf * sigmoid), T.float16)


@tilelang.jit(out_idx=[], execution_backend="nvrtc", target={"kind": "cuda", "arch": "sm_87"})
def _compile_gated_norm(
    M: int | None = None,
    x_dtype: str = "float16",
    heads_per_block: int = 4,
    threads: int = 128,
    heads: int = 48,
):
    HEADS = heads
    if type(HEADS) is not int or HEADS <= 0:
        raise ValueError("Invalid GDN value heads")
    assert x_dtype in ("float16", "float32")
    assert heads_per_block in (1, 2, 4, 8)
    assert threads in (32, 64, 128, 256)
    assert M is None or M > 0
    rows = T.dynamic("rows") if M is None else M

    @T.prim_func
    def main(
        X: T.Tensor((rows, HEADS, WIDTH), x_dtype),
        Z: T.Tensor((rows, HEADS, WIDTH), T.float16),
        W: T.Tensor((WIDTH,), T.float16),
        Y: T.Tensor((rows, HEADS, WIDTH), T.float16),
    ):
        with T.Kernel(T.ceildiv(rows * HEADS, heads_per_block), threads=threads) as bx:
            values = T.alloc_fragment((heads_per_block, WIDTH), T.float32)
            square = T.alloc_fragment((heads_per_block, WIDTH), T.float32)
            sums = T.alloc_fragment((heads_per_block,), T.float32)
            for h, j in T.Parallel(heads_per_block, WIDTH):
                index = bx * heads_per_block + h
                values[h, j] = 0.0
                if index < rows * HEADS:
                    values[h, j] = T.cast(X[index // HEADS, index % HEADS, j], T.float32)
                square[h, j] = values[h, j] * values[h, j]
            T.reduce_sum(square, sums, dim=1)
            for h, j in T.Parallel(heads_per_block, WIDTH):
                index = bx * heads_per_block + h
                if index < rows * HEADS:
                    rstd = T.rsqrt(sums[h] / WIDTH + 1e-6)
                    Y[index // HEADS, index % HEADS, j] = gated_norm_epilogue(
                        values[h, j], Z[index // HEADS, index % HEADS, j], W[j], rstd
                    )

    return main
