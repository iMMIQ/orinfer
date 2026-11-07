"""Qwen row controls and row-scaled embedding; arithmetic kernels are shared."""

import tilelang.language as T
from orinfer_build.compiler.cuda.common import orin_jit
from kernels.shared.rows import row_count


@orin_jit
def prepare(capacity: int, hidden: int, context: int, mrope_input: bool = False):
    m = row_count(capacity, True)

    @T.prim_func
    def main(
        Step: T.Tensor((1,), T.int32),
        Positions: T.Tensor((m,), T.int32),
        Length: T.Tensor((1,), T.int32),
        SegmentLength: T.Tensor((1,), T.int32),
        Last: T.Tensor((1,), T.int32),
        Requests: T.Tensor((m,), T.int32),
        MRope: T.Tensor((context, 3), T.int32),
        Residual: T.Tensor((m, hidden), T.float32),
    ):
        with T.Kernel(m, threads=256) as row:
            for column in T.Parallel(hidden):
                Residual[row, column] = 0
            if T.get_thread_binding() == 0:
                Positions[row] = Step[0] + row
                Requests[row] = 0
                if not mrope_input:
                    for axis in T.unroll(3):
                        MRope[Step[0] + row, axis] = Step[0] + row
                if row == 0:
                    Length[0] = Step[0] + m
                    SegmentLength[0] = m
                    Last[0] = m - 1

    return main


@orin_jit
def embedding(capacity: int, vocab: int, hidden: int):
    m = row_count(capacity, True)

    @T.prim_func
    def main(
        IDs: T.Tensor((m,), T.int32),
        Weight: T.Tensor((vocab, hidden), T.int8),
        Scale: T.Tensor((vocab,), T.float16),
        Output: T.Tensor((m, hidden), T.float16),
    ):
        with T.Kernel(m, T.ceildiv(hidden, 256), threads=128) as (row, block):
            for i in T.Parallel(256):
                column = block * 256 + i
                if column < hidden:
                    token = IDs[row]
                    Output[row, column] = T.if_then_else(
                        0 <= token and token < vocab,
                        T.cast(Weight[token, column], T.float32) * T.cast(Scale[token], T.float32),
                        0,
                    )

    return main
