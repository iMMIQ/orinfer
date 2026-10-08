"""Publish native Flash selection into the runtime's separate controls."""

import tilelang.language as T
from tools.operators.common import orin_jit


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
def selections(rows: int):
    """Split row-wise greedy results into the runtime's independent controls."""

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
def capture(rows: int, width: int, ring: int = 512):
    """Preserve raw Flash HC branches after advancing the target cursor."""
    if not 1 <= rows <= ring or width <= 0:
        raise ValueError("Target capture must fit the hidden ring")

    @T.prim_func
    def kernel(
        Hidden: T.Tensor((rows, width), T.float16),
        Position: T.Tensor((1,), T.int32),
        Ring: T.Tensor((ring, width), T.float16),
    ):
        with T.Kernel(rows, T.ceildiv(width, 256), threads=128) as (row, block):
            for j in T.Parallel(256):
                col = block * 256 + j
                if col < width:
                    Ring[(Position[0] - rows + row) % ring, col] = Hidden[row, col]

    return kernel


@orin_jit
def last_hidden(rows: int, width: int):
    @T.prim_func
    def kernel(
        Hidden: T.Tensor((rows, width), T.float16), Condition: T.Tensor((1, width), T.float16)
    ):
        with T.Kernel(T.ceildiv(width, 256), threads=128) as block:
            for j in T.Parallel(256):
                col = block * 256 + j
                if col < width:
                    Condition[0, col] = Hidden[rows - 1, col]

    return kernel


@orin_jit
def length(rows: int):
    @T.prim_func
    def kernel(Length: T.Tensor((1,), T.int32)):
        with T.Kernel(1, threads=32):
            if T.get_thread_binding() == 0:
                Length[0] = rows

    return kernel
