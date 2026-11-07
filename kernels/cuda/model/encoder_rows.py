"""Per-sequence outputs for independently owned, aligned encoder rows."""

import tilelang.language as T
from orinfer_build.compiler.cuda.common import orin_jit


@orin_jit
def pool(hidden, sequences=128):
    rows = T.dynamic("rows")

    @T.prim_func
    def main(
        X: T.Tensor((rows, hidden), T.float16),
        Segments: T.Tensor((sequences, 2), T.int32),
        Count: T.Tensor((1,), T.int32),
        Mean: T.Tensor((sequences, hidden), T.float32),
    ):
        with T.Kernel(T.ceildiv(hidden, 256), sequences, threads=256) as (block, owner):
            if owner < Count[0]:
                total = T.alloc_fragment((256,), T.float32)
                T.clear(total)
                for row in T.serial(Segments[owner, 0], Segments[owner, 1]):
                    for d in T.Parallel(256):
                        if block * 256 + d < hidden:
                            total[d] += T.cast(X[row, block * 256 + d], T.float32)
                for d in T.Parallel(256):
                    if block * 256 + d < hidden:
                        Mean[owner, block * 256 + d] = total[d] / (
                            Segments[owner, 1] - Segments[owner, 0]
                        )

    return main


@orin_jit
def normalize(hidden, sequences=128):
    @T.prim_func
    def main(
        Mean: T.Tensor((sequences, hidden), T.float32),
        Count: T.Tensor((1,), T.int32),
        Output: T.Tensor((sequences, hidden), T.float32),
    ):
        with T.Kernel(sequences, threads=256) as owner:
            if owner < Count[0]:
                squares = T.alloc_fragment((hidden,), T.float32)
                total = T.alloc_fragment((1,), T.float32)
                for d in T.Parallel(hidden):
                    squares[d] = Mean[owner, d] * Mean[owner, d]
                T.reduce_sum(squares, total, dim=0)
                for d in T.Parallel(hidden):
                    Output[owner, d] = Mean[owner, d] * T.rsqrt(T.max(total[0], 1e-24))

    return main


@orin_jit
def classifier(sequences=128):
    rows = T.dynamic("rows")

    @T.prim_func
    def main(
        X: T.Tensor((rows, 1024), T.float16),
        W: T.Tensor((2, 1024), T.float16),
        Segments: T.Tensor((sequences, 2), T.int32),
        Count: T.Tensor((1,), T.int32),
        Y: T.Tensor((sequences, 2), T.float32),
    ):
        with T.Kernel(2, sequences, threads=256) as (label, owner):
            if owner < Count[0]:
                product = T.alloc_fragment((1024,), T.float32)
                total = T.alloc_fragment((1,), T.float32)
                for d in T.Parallel(1024):
                    product[d] = T.cast(X[Segments[owner, 1] - 1, d], T.float32) * T.cast(
                        W[label, d], T.float32
                    )
                T.reduce_sum(product, total, dim=0)
                Y[owner, label] = total[0]

    return main
