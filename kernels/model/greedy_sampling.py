"""Exact f64 history processors and total-order greedy selection on SM87.

F64 scalars are stored as U64 bits in the package ABI. Rounded CUDA intrinsics
prevent fast-math contraction from changing the Rust sampler's arithmetic.
Scratch is request-private and rebuilt from the authoritative token history.
"""

import tilelang
import tilelang.language as T
from tools.operators.common import orin_jit


@orin_jit
def history_counts(vocab, context):
    @T.prim_func
    def kernel(
        History: T.Tensor((context,), T.int32),
        Length: T.Tensor((1,), T.int32),
        Counts: T.Tensor((vocab,), T.int32),
    ):
        with T.Kernel(256, threads=256) as block:
            tx = T.get_thread_binding()
            for i in T.serial(T.ceildiv(Length[0], 65536)):
                j = i * 65536 + block * 256 + tx
                if j < Length[0]:
                    token = History[j]
                    if token >= 0 and token < vocab:
                        T.atomic_add(Counts[token], 1)

    return kernel


def layout(size, threads):
    return tilelang.Fragment(
        (size,), forward_thread_fn=lambda j: j % threads, forward_index_fn=lambda j: j // threads
    )


@orin_jit
def penalized_partials(vocab, chunk=4096, threads=256):
    blocks = (vocab + chunk - 1) // chunk

    @T.prim_func
    def kernel(
        X: T.Tensor((vocab,), T.float32),
        Counts: T.Tensor((vocab,), T.int32),
        Parameters: T.Tensor((3,), T.uint64),
        Keys: T.Tensor((blocks,), T.uint64),
        IDs: T.Tensor((blocks,), T.int32),
        Bad: T.Tensor((blocks,), T.int32),
    ):
        with T.Kernel(blocks, threads=threads) as block:
            scores = T.alloc_fragment((chunk,), T.float64)
            keys = T.alloc_fragment((chunk,), T.uint64)
            indices = T.alloc_fragment((chunk,), T.int32)
            invalid = T.alloc_fragment((chunk,), T.int32)
            maximum = T.alloc_fragment((1,), T.uint64)
            minimum = T.alloc_fragment((1,), T.int32)
            bad = T.alloc_fragment((1,), T.int32)
            T.annotate_layout(
                {
                    scores: layout(chunk, threads),
                    keys: layout(chunk, threads),
                    indices: layout(chunk, threads),
                    invalid: layout(chunk, threads),
                    maximum: tilelang.Fragment(
                        (1,), forward_thread_fn=lambda j, rep: rep, replicate=threads
                    ),
                    minimum: tilelang.Fragment(
                        (1,), forward_thread_fn=lambda j, rep: rep, replicate=threads
                    ),
                    bad: tilelang.Fragment(
                        (1,), forward_thread_fn=lambda j, rep: rep, replicate=threads
                    ),
                }
            )
            for j in T.Parallel(chunk):
                col = block * chunk + j
                keys[j] = T.uint64(0)
                invalid[j] = 0
                if col < vocab:
                    scores[j] = T.cast(X[col], T.float64)
                    count = Counts[col]
                    if count > 0:
                        repeat = T.reinterpret(T.float64, Parameters[0])
                        if scores[j] < 0.0:
                            scores[j] = T.call_pure_extern(
                                "float64", "__dmul_rn", scores[j], repeat
                            )
                        else:
                            scores[j] = T.call_pure_extern(
                                "float64", "__ddiv_rn", scores[j], repeat
                            )
                        scores[j] = T.call_pure_extern(
                            "float64",
                            "__dsub_rn",
                            scores[j],
                            T.reinterpret(T.float64, Parameters[1]),
                        )
                    frequency = T.call_pure_extern(
                        "float64",
                        "__dmul_rn",
                        T.reinterpret(T.float64, Parameters[2]),
                        T.cast(count, T.float64),
                    )
                    scores[j] = T.call_pure_extern("float64", "__dsub_rn", scores[j], frequency)
                    invalid[j] = T.if_then_else(T.call_extern("bool", "isfinite", scores[j]), 0, 1)
                    bits = T.reinterpret(T.uint64, scores[j])
                    # IEEE total order, including -0 < +0. Finite scores have
                    # nonzero keys, leaving zero as the padded/invalid sentinel.
                    keys[j] = T.if_then_else(
                        invalid[j] != 0,
                        T.uint64(0),
                        T.if_then_else(
                            (bits >> 63) != 0, ~bits, bits ^ T.uint64(9223372036854775808)
                        ),
                    )
            T.reduce_max(keys, maximum, dim=0)
            T.reduce_sum(invalid, bad, dim=0)
            for j in T.Parallel(chunk):
                indices[j] = T.if_then_else(
                    block * chunk + j < vocab and keys[j] == maximum[0],
                    block * chunk + j,
                    2147483647,
                )
            T.reduce_min(indices, minimum, dim=0)
            if T.get_thread_binding() == 0:
                Keys[block] = maximum[0]
                IDs[block] = minimum[0]
                Bad[block] = bad[0]

    return kernel


@orin_jit
def penalized_merge(vocab, chunk=4096, threads=128):
    blocks = (vocab + chunk - 1) // chunk
    tile = ((blocks + threads - 1) // threads) * threads

    @T.prim_func
    def kernel(
        Keys: T.Tensor((blocks,), T.uint64),
        IDs: T.Tensor((blocks,), T.int32),
        Bad: T.Tensor((blocks,), T.int32),
        Token: T.Tensor((1,), T.int32),
        Status: T.Tensor((1,), T.int32),
    ):
        with T.Kernel(1, threads=threads):
            keys = T.alloc_fragment((tile,), T.uint64)
            indices = T.alloc_fragment((tile,), T.int32)
            invalid = T.alloc_fragment((tile,), T.int32)
            maximum = T.alloc_fragment((1,), T.uint64)
            minimum = T.alloc_fragment((1,), T.int32)
            bad = T.alloc_fragment((1,), T.int32)
            T.annotate_layout(
                {
                    keys: layout(tile, threads),
                    indices: layout(tile, threads),
                    invalid: layout(tile, threads),
                    maximum: tilelang.Fragment(
                        (1,), forward_thread_fn=lambda j, rep: rep, replicate=threads
                    ),
                    minimum: tilelang.Fragment(
                        (1,), forward_thread_fn=lambda j, rep: rep, replicate=threads
                    ),
                    bad: tilelang.Fragment(
                        (1,), forward_thread_fn=lambda j, rep: rep, replicate=threads
                    ),
                }
            )
            for j in T.Parallel(tile):
                keys[j] = T.if_then_else(j < blocks, Keys[j], T.uint64(0))
                invalid[j] = T.if_then_else(j < blocks, Bad[j], 0)
            T.reduce_max(keys, maximum, dim=0)
            T.reduce_sum(invalid, bad, dim=0)
            for j in T.Parallel(tile):
                indices[j] = 2147483647
                if j < blocks and keys[j] == maximum[0]:
                    indices[j] = IDs[j]
            T.reduce_min(indices, minimum, dim=0)
            if T.get_thread_binding() == 0:
                Status[0] = T.if_then_else(bad[0] != 0, 1, 0)
                Token[0] = T.if_then_else(bad[0] != 0, -1, minimum[0])

    return kernel
