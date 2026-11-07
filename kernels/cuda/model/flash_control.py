"""Publish native Flash selection into the runtime's separate controls."""

import tilelang.language as T
from orinfer_build.compiler.cuda.common import orin_jit
from kernels.shared.rows import row_count


@orin_jit
def selection():
    @T.prim_func
    def kernel(
        Pair: T.Tensor((2,), T.int32),
        Token: T.Tensor((1,), T.int32),
        Status: T.Tensor((1,), T.int32),
    ):
        with T.Kernel(1, threads=32):
            if T.get_thread_binding() == 0:
                Token[0] = Pair[0]
                Status[0] = Pair[1]

    return kernel


@orin_jit
def selections(rows: int, dynamic_rows: bool = False):
    """Split row-wise greedy results into the runtime's independent controls."""

    rows = row_count(rows, dynamic_rows)

    @T.prim_func
    def kernel(
        Pair: T.Tensor((rows, 2), T.int32),
        Tokens: T.Tensor((rows,), T.int32),
        Status: T.Tensor((rows,), T.int32),
    ):
        with T.Kernel(T.ceildiv(rows, 32), threads=32) as block:
            row = block * 32 + T.get_thread_binding()
            if row < rows:
                Tokens[row] = Pair[row, 0]
                Status[row] = Pair[row, 1]

    return kernel


@orin_jit
def capture(
    rows: int,
    width: int,
    ring: int = 512,
    dynamic_rows: bool = False,
):
    """Preserve raw Flash HC branches after advancing the target cursor."""
    if rows < 1 or ring < 1 or width <= 0 or (rows > ring and not dynamic_rows):
        raise ValueError("Target capture must fit the hidden ring")

    rows = count = row_count(rows, dynamic_rows)
    stored = T.min(count, ring) if dynamic_rows else rows
    start = count - stored if dynamic_rows else 0

    @T.prim_func
    def kernel(
        Hidden: T.Tensor((rows, width), T.float16),
        Position: T.Tensor((1,), T.int32),
        Ring: T.Tensor((ring, width), T.float16),
    ):
        with T.Kernel(stored, T.ceildiv(width, 256), threads=128) as (local_row, block):
            row = start + local_row
            for j in T.Parallel(256):
                col = block * 256 + j
                if col < width and row < count:
                    Ring[(Position[0] - count + row) % ring, col] = Hidden[row, col]

    return kernel


@orin_jit
def last_hidden(rows: int, width: int, dynamic_rows: bool = False):

    rows = count = row_count(rows, dynamic_rows)

    @T.prim_func
    def kernel(
        Hidden: T.Tensor((rows, width), T.float16), Condition: T.Tensor((1, width), T.float16)
    ):
        with T.Kernel(T.ceildiv(width, 256), threads=128) as block:
            for j in T.Parallel(256):
                col = block * 256 + j
                if col < width:
                    Condition[0, col] = Hidden[count - 1, col]

    return kernel
