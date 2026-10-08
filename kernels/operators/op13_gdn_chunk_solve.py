"""FP32 SM87 unit-lower GDN inverse; caller-owned contiguous buffers.

Only strict-lower input entries are read. Diagonal is implicit one and upper
input garbage is ignored. Default parallel elimination retains A in registers,
publishing the next solved pivot row through shared memory. An independent
column recurrence and all-shared elimination remain measured alternatives.
"""

import tilelang
import tilelang.language as T

HEADS = 48


@tilelang.jit(out_idx=[], execution_backend="nvrtc", target={"kind": "cuda", "arch": "sm_87"})
def _compile(bt: int = 64, matrices_per_block: int = 1):
    assert bt in (16, 32, 64)
    assert matrices_per_block in (1, 2, 4)
    batch, chunks = T.dynamic("batch"), T.dynamic("chunks")
    threads = max(32, bt * matrices_per_block)

    @T.prim_func
    def main(
        L: T.Tensor((batch, HEADS, chunks, bt, bt), "float32"),
        A: T.Tensor((batch, HEADS, chunks, bt, bt), "float32"),
    ):
        with T.Kernel(
            T.ceildiv(batch * HEADS * chunks, matrices_per_block), threads=threads
        ) as block:
            ls = T.alloc_shared((matrices_per_block, bt, bt), "float32")
            inv = T.alloc_shared((matrices_per_block, bt, bt), "float32")
            value = T.alloc_local((1,), "float32")
            tx = T.get_thread_binding()
            for m, i, j in T.Parallel(matrices_per_block, bt, bt):
                matrix = block * matrices_per_block + m
                if matrix < batch * HEADS * chunks and j < i:
                    ls[m, i, j] = L[
                        matrix // (HEADS * chunks),
                        (matrix // chunks) % HEADS,
                        matrix % chunks,
                        i,
                        j,
                    ]
                else:
                    ls[m, i, j] = 0.0
                inv[m, i, j] = T.if_then_else(i == j, 1.0, 0.0)
            T.sync_threads()
            m, column = tx // bt, tx % bt
            if tx < matrices_per_block * bt:
                for row in T.serial(1, bt):
                    if column < row:
                        value[0] = -ls[m, row, column]
                        for k in T.serial(column + 1, row):
                            value[0] = value[0] - ls[m, row, k] * inv[m, k, column]
                        inv[m, row, column] = value[0]
                matrix = block * matrices_per_block + m
                if matrix < batch * HEADS * chunks:
                    for row in T.serial(bt):
                        A[
                            matrix // (HEADS * chunks),
                            (matrix // chunks) % HEADS,
                            matrix % chunks,
                            row,
                            column,
                        ] = inv[m, row, column]

    return main


@tilelang.jit(out_idx=[], execution_backend="nvrtc", target={"kind": "cuda", "arch": "sm_87"})
def _compile_elimination(bt: int = 64, threads: int = 256, registers: bool = False):
    assert bt in (16, 32, 64) and threads in (128, 256)
    batch, chunks = T.dynamic("batch"), T.dynamic("chunks")

    @T.prim_func
    def main(
        L: T.Tensor((batch, HEADS, chunks, bt, bt), "float32"),
        A: T.Tensor((batch, HEADS, chunks, bt, bt), "float32"),
    ):
        with T.Kernel(batch * HEADS * chunks, threads=threads) as matrix:
            ls = T.alloc_shared((bt, bt), "float32")
            inv = T.alloc_shared((bt, bt), "float32")
            reg = T.alloc_fragment((bt, bt), "float32")
            T.annotate_layout(
                {
                    reg: T.Fragment(
                        (bt, bt),
                        forward_thread_fn=lambda i, j: (i * bt + j) % threads,
                        forward_index_fn=lambda i, j: (i * bt + j) // threads,
                    )
                }
            )
            for i, j in T.Parallel(bt, bt):
                if j < i:
                    ls[i, j] = L[
                        matrix // (HEADS * chunks),
                        (matrix // chunks) % HEADS,
                        matrix % chunks,
                        i,
                        j,
                    ]
                    inv[i, j] = -ls[i, j]
                else:
                    ls[i, j] = 0.0
                    inv[i, j] = T.if_then_else(i == j, 1.0, 0.0)
                if registers:
                    reg[i, j] = inv[i, j]
            T.sync_threads()
            for pivot in T.serial(1, bt - 1):
                for i, j in T.Parallel(bt, bt):
                    if i > pivot and j < pivot:
                        if registers:
                            reg[i, j] = reg[i, j] - ls[i, pivot] * inv[pivot, j]
                        else:
                            inv[i, j] = inv[i, j] - ls[i, pivot] * inv[pivot, j]
                    if registers:
                        if i == pivot + 1:
                            inv[i, j] = reg[i, j]
                T.sync_threads()
            for i, j in T.Parallel(bt, bt):
                A[matrix // (HEADS * chunks), (matrix // chunks) % HEADS, matrix % chunks, i, j] = (
                    reg[i, j] if registers else inv[i, j]
                )

    return main


def gdn_chunk_solve(bt=64, matrices_per_block=1, algorithm="registers", threads=256):
    """Compile dynamic B/C inverse; diagonal ignored, strict lower must be finite.

    L/A are distinct contiguous FP32 [B,48,C,BT,BT], B/C positive. Caller
    supplies stable allocations and explicit stream; no workspace/state/alloc.
    Tail input must have strict-lower padded rows/columns zero, yielding exact
    identity in padding. Invalid lower entries propagate; no gate validation.
    """
    assert algorithm in ("columns", "elimination", "registers")
    assert algorithm == "columns" or matrices_per_block == 1
    kernel = (
        _compile(bt, matrices_per_block)
        if algorithm == "columns"
        else _compile_elimination(bt, threads, algorithm == "registers")
    )
    kernel.adapter.kernels = dict(kernel.adapter.kernels)
    return kernel


def launch(kernel, system, transform, *, stream):
    return kernel.adapter.func(system, transform, stream=stream)
