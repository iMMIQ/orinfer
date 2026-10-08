"""Full-vocabulary exact top-k and categorical CDF; caller-owned stable buffers.

Only offline compilation imports common.orin_jit. Production math is TileLang.
Counter uniforms are a CPU/Rust ABI contract, not batch-index seeded GPU RNG.
"""

import math
import tilelang
import tilelang.language as T
from tools.operators.common import orin_jit

VOCAB = 248320
FP32_MAX = 3.4028234663852886e38
SEED = 20261002
MASK64 = (1 << 64) - 1


def splitmix64(x):
    """Wrapping u64 SplitMix permutation; identical operations required in Rust."""
    x = (x + 0x9E3779B97F4A7C15) & MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK64
    return (x ^ (x >> 31)) & MASK64


def counter_uniform(request_id, absolute_step, seed=SEED):
    """u64 IDs/absolute committed token step -> IEEE F64 uniform in [0,1).

    u = (splitmix64(splitmix64(seed ^ request_id) ^ absolute_step) >> 11)*2^-53.
    Branches preserving request identity and step reuse that draw; a deliberately
    independent stream receives a different stable request_id. Never use queue,
    resident row, retries, draft count, or graph replay count as the counter.
    """
    for value in (request_id, absolute_step, seed):
        if type(value) is not int or not 0 <= value <= MASK64:
            raise ValueError("seed/request_id/absolute_step must be u64")
    return (splitmix64(splitmix64(seed ^ request_id) ^ absolute_step) >> 11) * 2.0**-53


@orin_jit
def topk_partials(dtype="float16", vocab=VOCAB, k=3, chunk=4096, threads=256):
    assert dtype in ("float16", "bfloat16", "float32")
    assert 1 <= k <= min(16, vocab) and chunk % threads == 0
    rows = T.dynamic("rows")
    storage_dtype = "uint16" if dtype == "bfloat16" else dtype
    blocks = (vocab + chunk - 1) // chunk

    @T.prim_func
    def kernel(
        X: T.Tensor((rows, vocab), storage_dtype),
        PV: T.Tensor((rows, blocks, k), T.float32),
        PI: T.Tensor((rows, blocks, k), T.int32),
        Bad: T.Tensor((rows, blocks), T.int32),
    ):
        with T.Kernel(blocks, rows, threads=threads) as (block, row):
            values = T.alloc_fragment((chunk,), T.float32)
            indices = T.alloc_fragment((chunk,), T.int32)
            invalid = T.alloc_fragment((chunk,), T.int32)
            maximum = T.alloc_fragment((1,), T.float32)
            minimum = T.alloc_fragment((1,), T.int32)
            count = T.alloc_fragment((1,), T.int32)
            T.annotate_layout(
                {
                    values: tilelang.Fragment(
                        (chunk,),
                        forward_thread_fn=lambda j: j % threads,
                        forward_index_fn=lambda j: j // threads,
                    ),
                    indices: tilelang.Fragment(
                        (chunk,),
                        forward_thread_fn=lambda j: j % threads,
                        forward_index_fn=lambda j: j // threads,
                    ),
                    invalid: tilelang.Fragment(
                        (chunk,),
                        forward_thread_fn=lambda j: j % threads,
                        forward_index_fn=lambda j: j // threads,
                    ),
                    maximum: tilelang.Fragment(
                        (1,), forward_thread_fn=lambda j, rep: rep, replicate=threads
                    ),
                    minimum: tilelang.Fragment(
                        (1,), forward_thread_fn=lambda j, rep: rep, replicate=threads
                    ),
                    count: tilelang.Fragment(
                        (1,), forward_thread_fn=lambda j, rep: rep, replicate=threads
                    ),
                }
            )
            for j in T.Parallel(chunk):
                col = block * chunk + j
                values[j] = -T.infinity(T.float32)
                invalid[j] = 0
                if col < vocab:
                    if dtype == "bfloat16":
                        values[j] = T.reinterpret(T.float32, T.cast(X[row, col], T.uint32) << 16)
                    else:
                        values[j] = T.cast(X[row, col], T.float32)
                    invalid[j] = T.if_then_else(T.call_extern("bool", "isfinite", values[j]), 0, 1)
                    values[j] = T.if_then_else(invalid[j] > 0, -T.infinity(T.float32), values[j])
            T.reduce_sum(invalid, count, dim=0)
            if T.get_thread_binding() == 0:
                Bad[row, block] = count[0]
            for rank in T.serial(k):
                T.reduce_max(values, maximum, dim=0)
                for j in T.Parallel(chunk):
                    indices[j] = 2147483647
                    if block * chunk + j < vocab and values[j] == maximum[0]:
                        indices[j] = block * chunk + j
                T.reduce_min(indices, minimum, dim=0)
                if T.get_thread_binding() == 0:
                    PV[row, block, rank] = maximum[0]
                    PI[row, block, rank] = minimum[0]
                for j in T.Parallel(chunk):
                    if block * chunk + j == minimum[0]:
                        values[j] = -T.infinity(T.float32)

    return kernel


@orin_jit
def topk_merge(vocab=VOCAB, k=3, chunk=4096, threads=128):
    assert 1 <= k <= min(16, vocab)
    rows = T.dynamic("rows")
    blocks = (vocab + chunk - 1) // chunk
    tile = ((blocks * k + threads - 1) // threads) * threads

    @T.prim_func
    def kernel(
        PV: T.Tensor((rows, blocks, k), T.float32),
        PI: T.Tensor((rows, blocks, k), T.int32),
        Bad: T.Tensor((rows, blocks), T.int32),
        Values: T.Tensor((rows, k), T.float32),
        IDs: T.Tensor((rows, k), T.int32),
        Token: T.Tensor((rows,), T.int32),
        Status: T.Tensor((rows,), T.int32),
    ):
        with T.Kernel(rows, threads=threads) as row:
            values = T.alloc_fragment((tile,), T.float32)
            indices = T.alloc_fragment((tile,), T.int32)
            invalid = T.alloc_fragment((tile,), T.int32)
            maximum = T.alloc_fragment((1,), T.float32)
            minimum = T.alloc_fragment((1,), T.int32)
            count = T.alloc_fragment((1,), T.int32)
            T.annotate_layout(
                {
                    values: tilelang.Fragment(
                        (tile,),
                        forward_thread_fn=lambda j: j % threads,
                        forward_index_fn=lambda j: j // threads,
                    ),
                    indices: tilelang.Fragment(
                        (tile,),
                        forward_thread_fn=lambda j: j % threads,
                        forward_index_fn=lambda j: j // threads,
                    ),
                    invalid: tilelang.Fragment(
                        (tile,),
                        forward_thread_fn=lambda j: j % threads,
                        forward_index_fn=lambda j: j // threads,
                    ),
                    maximum: tilelang.Fragment(
                        (1,), forward_thread_fn=lambda j, rep: rep, replicate=threads
                    ),
                    minimum: tilelang.Fragment(
                        (1,), forward_thread_fn=lambda j, rep: rep, replicate=threads
                    ),
                    count: tilelang.Fragment(
                        (1,), forward_thread_fn=lambda j, rep: rep, replicate=threads
                    ),
                }
            )
            for j in T.Parallel(tile):
                values[j] = -T.infinity(T.float32)
                invalid[j] = 0
                if j < blocks * k:
                    values[j] = PV[row, j // k, j % k]
                if j < blocks:
                    invalid[j] = Bad[row, j]
            T.reduce_sum(invalid, count, dim=0)
            if T.get_thread_binding() == 0:
                Status[row] = T.if_then_else(count[0] > 0, 1, 0)
            for rank in T.serial(k):
                T.reduce_max(values, maximum, dim=0)
                for j in T.Parallel(tile):
                    indices[j] = 2147483647
                    if j < blocks * k and values[j] == maximum[0]:
                        indices[j] = PI[row, j // k, j % k]
                T.reduce_min(indices, minimum, dim=0)
                if T.get_thread_binding() == 0:
                    Values[row, rank] = T.if_then_else(
                        count[0] > 0, -T.infinity(T.float32), maximum[0]
                    )
                    IDs[row, rank] = T.if_then_else(count[0] > 0, -1, minimum[0])
                    if rank == 0:
                        Token[row] = T.if_then_else(count[0] > 0, -1, minimum[0])
                for j in T.Parallel(tile):
                    if j < blocks * k:
                        if PI[row, j // k, j % k] == minimum[0]:
                            values[j] = -T.infinity(T.float32)

    return kernel


@orin_jit
def sampling_mass(dtype="float16", vocab=VOCAB, k=3, temperature=1.0, chunk=4096, threads=256):
    assert dtype in ("float16", "bfloat16", "float32")
    assert math.isfinite(temperature) and temperature > 0
    rows = T.dynamic("rows")
    blocks = (vocab + chunk - 1) // chunk
    storage_dtype = "uint16" if dtype == "bfloat16" else dtype

    @T.prim_func
    def kernel(
        X: T.Tensor((rows, vocab), storage_dtype),
        Values: T.Tensor((rows, k), T.float32),
        Mass: T.Tensor((rows, blocks), T.float64),
    ):
        with T.Kernel(blocks, rows, threads=threads) as (block, row):
            weights = T.alloc_fragment((chunk,), T.float64)
            total = T.alloc_fragment((1,), T.float64)
            T.annotate_layout(
                {
                    weights: tilelang.Fragment(
                        (chunk,),
                        forward_thread_fn=lambda j: j % threads,
                        forward_index_fn=lambda j: j // threads,
                    ),
                    total: tilelang.Fragment(
                        (1,), forward_thread_fn=lambda j, rep: rep, replicate=threads
                    ),
                }
            )
            for j in T.Parallel(chunk):
                weights[j] = 0.0
                if block * chunk + j < vocab and T.call_extern("bool", "isfinite", Values[row, 0]):
                    # Double transform/exponential prevents finite FP32 range
                    # subtraction overflow and premature probability underflow.
                    if dtype == "bfloat16":
                        x = T.cast(
                            T.reinterpret(
                                T.float32, T.cast(X[row, block * chunk + j], T.uint32) << 16
                            ),
                            T.float64,
                        )
                    else:
                        x = T.cast(X[row, block * chunk + j], T.float64)
                    weights[j] = T.exp(
                        (x - T.cast(Values[row, 0], T.float64)) / T.float64(temperature)
                    )
            T.reduce_sum(weights, total, dim=0)
            if T.get_thread_binding() == 0:
                Mass[row, block] = total[0]

    return kernel


@orin_jit
def sampling_prefix(vocab=VOCAB, chunk=4096):
    rows = T.dynamic("rows")
    blocks = (vocab + chunk - 1) // chunk

    @T.prim_func
    def kernel(
        Mass: T.Tensor((rows, blocks), T.float64),
        Uniform: T.Tensor((rows,), T.float64),
        Meta: T.Tensor((rows, 2), T.float64),
        Status: T.Tensor((rows,), T.int32),
    ):
        with T.Kernel(rows, threads=32) as row:
            total = T.alloc_local((1,), T.float64)
            target = T.alloc_local((1,), T.float64)
            found = T.alloc_local((1,), T.int32)
            last = T.alloc_local((1,), T.int32)
            if T.get_thread_binding() == 0:
                Meta[row, 0] = -1.0
                Meta[row, 1] = 0.0
                Status[row] = Status[row] + T.if_then_else(
                    T.call_extern("bool", "isfinite", Uniform[row])
                    and Uniform[row] >= 0.0
                    and Uniform[row] < 1.0,
                    0,
                    2,
                )
                if Status[row] == 0:
                    total[0] = 0.0
                    for b in T.serial(blocks):
                        total[0] = total[0] + Mass[row, b]
                    target[0] = Uniform[row] * total[0]
                    found[0] = 0
                    last[0] = 0
                    for b in T.serial(blocks):
                        if Mass[row, b] > 0.0:
                            last[0] = b
                        if found[0] == 0:
                            if target[0] < Mass[row, b]:
                                Meta[row, 0] = T.cast(b, T.float64)
                                Meta[row, 1] = target[0]
                                found[0] = 1
                            else:
                                target[0] = target[0] - Mass[row, b]
                    if found[0] == 0:
                        Meta[row, 0] = T.cast(last[0], T.float64)
                        Meta[row, 1] = Mass[row, last[0]]

    return kernel


@orin_jit
def sampling_select(dtype="float16", vocab=VOCAB, k=3, temperature=1.0, chunk=4096, threads=256):
    assert math.isfinite(temperature) and temperature > 0
    rows = T.dynamic("rows")
    storage_dtype = "uint16" if dtype == "bfloat16" else dtype

    @T.prim_func
    def kernel(
        X: T.Tensor((rows, vocab), storage_dtype),
        Values: T.Tensor((rows, k), T.float32),
        Meta: T.Tensor((rows, 2), T.float64),
        Token: T.Tensor((rows,), T.int32),
        Status: T.Tensor((rows,), T.int32),
    ):
        with T.Kernel(rows, threads=threads) as row:
            weights = T.alloc_shared((chunk,), T.float64)
            target = T.alloc_local((1,), T.float64)
            found = T.alloc_local((1,), T.int32)
            last = T.alloc_local((1,), T.int32)
            for j in T.Parallel(chunk):
                weights[j] = 0.0
                col = T.cast(Meta[row, 0], T.int32) * chunk + j
                if Status[row] == 0 and col >= 0 and col < vocab:
                    if dtype == "bfloat16":
                        x = T.cast(
                            T.reinterpret(T.float32, T.cast(X[row, col], T.uint32) << 16), T.float64
                        )
                    else:
                        x = T.cast(X[row, col], T.float64)
                    weights[j] = T.exp(
                        (x - T.cast(Values[row, 0], T.float64)) / T.float64(temperature)
                    )
            T.sync_threads()
            if T.get_thread_binding() == 0:
                Token[row] = -1
                if Status[row] == 0:
                    target[0] = Meta[row, 1]
                    found[0] = 0
                    last[0] = -1
                    for j in T.serial(chunk):
                        if weights[j] > 0.0:
                            last[0] = T.cast(Meta[row, 0], T.int32) * chunk + j
                        if found[0] == 0:
                            if target[0] < weights[j]:
                                Token[row] = T.cast(Meta[row, 0], T.int32) * chunk + j
                                found[0] = 1
                            else:
                                target[0] = target[0] - weights[j]
                    # Endpoint rounding fallback: last positive-mass token.
                    if found[0] == 0:
                        Token[row] = last[0]

    return kernel


def launch(
    partials,
    merge,
    logits,
    partial_values,
    partial_ids,
    bad,
    values,
    ids,
    token,
    status,
    *,
    stream,
    sampling=None,
    uniform=None,
    mass=None,
    meta=None,
):
    """Two topk launches; optionally three additional full-CDF launches.

    Temperature zero uses sampling=None and ignores uniform. No penalties and
    top_p=1. All contiguous buffers disjoint and caller-owned; graph shape fixed.
    Every consumer must inspect status (bit1 nonfinite logits, bit2 bad uniform).
    """
    partials(logits, partial_values, partial_ids, bad, stream=stream)
    merge(partial_values, partial_ids, bad, values, ids, token, status, stream=stream)
    if sampling is not None:
        mass_kernel, prefix_kernel, select_kernel = sampling
        mass_kernel(logits, values, mass, stream=stream)
        prefix_kernel(mass, uniform, meta, status, stream=stream)
        select_kernel(logits, values, meta, token, status, stream=stream)
