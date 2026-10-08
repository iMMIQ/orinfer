"""Finite-logit greedy selection, matching the lowest-index argmax tie rule."""

import tilelang
import tilelang.language as T

from tools.operators.common import orin_jit
from kernels.model.rows import row_count


@orin_jit
def greedy_partials(vocab: int, rows: int = 1, dynamic_rows: bool = False):
    if type(vocab) is not int or vocab < 1 or type(rows) is not int or rows < 1:
        raise ValueError("Positive vocabulary required")
    blocks = (vocab + 1023) // 1024
    rows = row_count(rows, dynamic_rows)

    @T.prim_func
    def main(
        Logits: T.Tensor((rows, vocab), T.float32),
        Values: T.Tensor((rows * blocks,), T.float32),
        Indices: T.Tensor((rows * blocks,), T.int32),
        Invalid: T.Tensor((rows * blocks,), T.int32),
    ):
        with T.Kernel(blocks, rows, threads=256) as (block, row):
            values = T.alloc_fragment((1024,), T.float32)
            indices = T.alloc_fragment((1024,), T.int32)
            invalid = T.alloc_fragment((1024,), T.int32)
            maximum = T.alloc_fragment((1,), T.float32)
            minimum = T.alloc_fragment((1,), T.int32)
            bad = T.alloc_fragment((1,), T.int32)
            T.annotate_layout(
                {
                    values: tilelang.Fragment(
                        (1024,),
                        forward_thread_fn=lambda i: i % 256,
                        forward_index_fn=lambda i: i // 256,
                    ),
                    indices: tilelang.Fragment(
                        (1024,),
                        forward_thread_fn=lambda i: i % 256,
                        forward_index_fn=lambda i: i // 256,
                    ),
                    invalid: tilelang.Fragment(
                        (1024,),
                        forward_thread_fn=lambda i: i % 256,
                        forward_index_fn=lambda i: i // 256,
                    ),
                    maximum: tilelang.Fragment(
                        (1,), forward_thread_fn=lambda i, rep: rep, replicate=256
                    ),
                    minimum: tilelang.Fragment(
                        (1,), forward_thread_fn=lambda i, rep: rep, replicate=256
                    ),
                    bad: tilelang.Fragment(
                        (1,), forward_thread_fn=lambda i, rep: rep, replicate=256
                    ),
                }
            )
            for i in T.Parallel(1024):
                values[i] = -T.infinity(T.float32)
                invalid[i] = 0
                if block * 1024 + i < vocab:
                    values[i] = Logits[row, block * 1024 + i]
                    invalid[i] = T.cast(
                        not T.call_pure_extern("bool", "isfinite", values[i]), T.int32
                    )
            T.reduce_max(values, maximum, dim=0)
            T.reduce_max(invalid, bad, dim=0)
            for i in T.Parallel(1024):
                indices[i] = 2147483647
                if block * 1024 + i < vocab and values[i] == maximum[0]:
                    indices[i] = block * 1024 + i
            T.reduce_min(indices, minimum, dim=0)
            if T.get_thread_binding() == 0:
                Values[row * blocks + block] = maximum[0]
                Indices[row * blocks + block] = minimum[0]
                Invalid[row * blocks + block] = bad[0]

    return main


@orin_jit
def greedy_merge(vocab: int, rows: int = 1, dynamic_rows: bool = False):
    if type(vocab) is not int or vocab < 1 or type(rows) is not int or rows < 1:
        raise ValueError("Positive vocabulary required")
    blocks = (vocab + 1023) // 1024
    width = max(32, 1 << (blocks - 1).bit_length())
    threads = min(256, width)
    rows = row_count(rows, dynamic_rows)

    @T.prim_func
    def main(
        Values: T.Tensor((rows * blocks,), T.float32),
        Indices: T.Tensor((rows * blocks,), T.int32),
        Invalid: T.Tensor((rows * blocks,), T.int32),
        Output: T.Tensor((rows * 2,), T.int32),
    ):
        with T.Kernel(rows, threads=threads) as row:
            values = T.alloc_fragment((width,), T.float32)
            indices = T.alloc_fragment((width,), T.int32)
            invalid = T.alloc_fragment((width,), T.int32)
            maximum = T.alloc_fragment((1,), T.float32)
            minimum = T.alloc_fragment((1,), T.int32)
            bad = T.alloc_fragment((1,), T.int32)
            T.annotate_layout(
                {
                    values: tilelang.Fragment(
                        (width,),
                        forward_thread_fn=lambda i: i % threads,
                        forward_index_fn=lambda i: i // threads,
                    ),
                    indices: tilelang.Fragment(
                        (width,),
                        forward_thread_fn=lambda i: i % threads,
                        forward_index_fn=lambda i: i // threads,
                    ),
                    invalid: tilelang.Fragment(
                        (width,),
                        forward_thread_fn=lambda i: i % threads,
                        forward_index_fn=lambda i: i // threads,
                    ),
                    maximum: tilelang.Fragment(
                        (1,), forward_thread_fn=lambda i, rep: rep, replicate=threads
                    ),
                    minimum: tilelang.Fragment(
                        (1,), forward_thread_fn=lambda i, rep: rep, replicate=threads
                    ),
                    bad: tilelang.Fragment(
                        (1,), forward_thread_fn=lambda i, rep: rep, replicate=threads
                    ),
                }
            )
            for i in T.Parallel(width):
                values[i] = -T.infinity(T.float32)
                invalid[i] = 0
                if i < blocks:
                    values[i] = Values[row * blocks + i]
                    invalid[i] = Invalid[row * blocks + i]
            T.reduce_max(values, maximum, dim=0)
            T.reduce_max(invalid, bad, dim=0)
            for i in T.Parallel(width):
                indices[i] = 2147483647
                if i < blocks and values[i] == maximum[0]:
                    indices[i] = Indices[row * blocks + i]
            T.reduce_min(indices, minimum, dim=0)
            if T.get_thread_binding() == 0:
                Output[row * 2] = minimum[0]
                Output[row * 2 + 1] = bad[0]

    return main
