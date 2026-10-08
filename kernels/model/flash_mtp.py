"""Flash Next MTP fusion and bounded speculative prefix-state snapshots."""

import tilelang.language as T
from tools.operators.common import orin_jit


@orin_jit
def norm(rows: int, width: int):
    """Gemma RMS across the whole input; Gamma includes the zero-centered offset."""
    if rows < 1 or width < 1:
        raise ValueError("Invalid MTP norm shape")
    size = 1 << (width - 1).bit_length()

    @T.prim_func
    def main(
        X: T.Tensor((rows, width), T.float16),
        Gamma: T.Tensor((width,), T.float32),
        Out: T.Tensor((rows, width), T.float16),
    ):
        with T.Kernel(rows, threads=256) as row:
            square = T.alloc_fragment((size,), T.float32)
            total = T.alloc_fragment((1,), T.float32)
            for j in T.Parallel(size):
                square[j] = 0
                if j < width:
                    square[j] = T.cast(X[row, j], T.float32) * T.cast(X[row, j], T.float32)
            T.reduce_sum(square, total, dim=0)
            for j in T.Parallel(width):
                Out[row, j] = (
                    T.cast(X[row, j], T.float32) * T.rsqrt(total[0] / width + 1e-6) * Gamma[j]
                )

    return main


@orin_jit
def fuse(rows: int, hidden: int, branches: int = 4):
    @T.prim_func
    def main(
        Embedding: T.Tensor((rows, hidden), T.float16),
        Hidden: T.Tensor((rows, branches, hidden), T.float16),
        Out: T.Tensor((rows, branches, hidden), T.float16),
    ):
        with T.Kernel(rows, branches, T.ceildiv(hidden, 256), threads=128) as (row, branch, block):
            for j in T.Parallel(256):
                col = block * 256 + j
                if col < hidden:
                    Out[row, branch, col] = T.cast(Embedding[row, col], T.float32) + T.cast(
                        Hidden[row, branch, col], T.float32
                    )

    return main


@orin_jit
def history_prefix(rows: int, width: int, history: int):
    """Save every chronological raw/PLE history before its final in-place shift."""

    @T.prim_func
    def main(
        X: T.Tensor((rows, width), T.float16),
        State: T.Tensor((history, width), T.float16),
        Prefix: T.Tensor((rows, history, width), T.float16),
    ):
        with T.Kernel(rows, T.ceildiv(width, 256), threads=128) as (row, block):
            for j in T.Parallel(256):
                col = block * 256 + j
                if col < width:
                    for h in T.serial(history):
                        index = row + 1 - history + h
                        if index >= 0:
                            Prefix[row, h, col] = X[index, col]
                        else:
                            Prefix[row, h, col] = State[history + index, col]

    return main


@orin_jit
def pending_prefix(rows: int):
    """Preserve all four physical raw index slots, including inactive old slots."""

    @T.prim_func
    def main(
        QK: T.Tensor((rows, 5, 128), T.float16),
        State: T.Tensor((4, 128), T.float16),
        Position: T.Tensor((1,), T.int32),
        Prefix: T.Tensor((rows, 4, 128), T.float16),
    ):
        with T.Kernel(rows, 4, threads=128) as (row, slot):
            last = Position[0] + row
            token = last - (last - slot) % 4
            for d in T.Parallel(128):
                if token >= Position[0]:
                    Prefix[row, slot, d] = QK[token - Position[0], 4, d]
                else:
                    Prefix[row, slot, d] = State[slot, d]

    return main
