"""NHWC stride-2 convolution, CNN layout and sinusoidal positions."""

import tilelang.language as T
from orinfer_build.compiler.cuda.common import orin_jit
from kernels.cuda.vision.encoder import activation_value


@orin_jit
def convolution(channels, height, width, outputs=480, bm=64, bn=64, bk=64, stages=2):
    chunks = T.dynamic("rows")
    oh, ow = (height + 1) // 2, (width + 1) // 2
    reduction = channels * 9

    @T.prim_func
    def main(
        X: T.Tensor((chunks, height, width, channels), T.float16),
        W: T.Tensor((outputs, 3, 3, channels), T.float16),
        Bias: T.Tensor((outputs,), T.float16),
        Y: T.Tensor((chunks, oh, ow, outputs), T.float16),
    ):
        with T.Kernel(T.ceildiv(oh * ow, bm), T.ceildiv(outputs, bn), chunks, threads=128) as (
            by,
            bx,
            batch,
        ):
            x = T.alloc_shared((bm, bk), T.float16)
            w = T.alloc_shared((bn, bk), T.float16)
            acc = T.alloc_fragment((bm, bn), T.float32)
            T.clear(acc)
            for kg in T.Pipelined(T.ceildiv(reduction, bk), num_stages=stages):
                for i, j in T.Parallel(bm, bk):
                    at = by * bm + i
                    k = kg * bk + j
                    iy = (at // ow) * 2 + (k // channels) // 3 - 1
                    ix = (at % ow) * 2 + (k // channels) % 3 - 1
                    x[i, j] = T.if_then_else(
                        at < oh * ow
                        and k < reduction
                        and iy >= 0
                        and iy < height
                        and ix >= 0
                        and ix < width,
                        X[batch, iy, ix, k % channels],
                        0,
                    )
                for i, j in T.Parallel(bn, bk):
                    k = kg * bk + j
                    w[i, j] = T.if_then_else(
                        bx * bn + i < outputs and k < reduction,
                        W[bx * bn + i, (k // channels) // 3, (k // channels) % 3, k % channels],
                        0,
                    )
                T.gemm(x, w, acc, transpose_B=True)
            for i, j in T.Parallel(bm, bn):
                at = by * bm + i
                channel = bx * bn + j
                if at < oh * ow and channel < outputs:
                    value = T.cast(
                        T.cast(acc[i, j] + T.cast(Bias[channel], T.float32), T.float16), T.float32
                    )
                    Y[batch, at // ow, at % ow, channel] = activation_value(value, "gelu")

    return main


@orin_jit
def flatten(channels=480, frequency=16, time=13):
    chunks = T.dynamic("rows")
    width = channels * frequency

    @T.prim_func
    def main(
        X: T.Tensor((chunks, frequency, time, channels), T.float16),
        Y: T.Tensor((chunks * time, width), T.float16),
    ):
        with T.Kernel(chunks * time, T.ceildiv(width, 256), threads=128) as (row, block):
            for j in T.Parallel(256):
                col = block * 256 + j
                if col < width:
                    Y[row, col] = X[row // time, col % frequency, row % time, col // frequency]

    return main


@orin_jit
def position(hidden=1024):
    rows = T.dynamic("rows")

    @T.prim_func
    def main(X: T.Tensor((rows, hidden), T.float16), PE: T.Tensor((13, hidden), T.float16)):
        with T.Kernel(rows, T.ceildiv(hidden, 256), threads=128) as (row, block):
            for j in T.Parallel(256):
                col = block * 256 + j
                if col < hidden:
                    X[row, col] = T.cast(X[row, col], T.float32) + T.cast(
                        PE[row % 13, col], T.float32
                    )

    return main


@orin_jit
def split_qkv(hidden=1024):
    rows = T.dynamic("rows")

    @T.prim_func
    def main(
        X: T.Tensor((rows, hidden * 3), T.float16),
        Q: T.Tensor((rows, hidden), T.float16),
        K: T.Tensor((rows, hidden), T.float16),
        V: T.Tensor((rows, hidden), T.float16),
    ):
        with T.Kernel(rows, T.ceildiv(hidden, 256), threads=128) as (row, block):
            for j in T.Parallel(256):
                col = block * 256 + j
                if col < hidden:
                    Q[row, col] = X[row, col]
                    K[row, col] = X[row, hidden + col]
                    V[row, col] = X[row, 2 * hidden + col]

    return main
