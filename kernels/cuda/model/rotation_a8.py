"""Signed block128 rotation and row A8 with exact FP16 materialization boundaries."""

import tilelang
import tilelang.language as T
from orinfer_build.compiler.cuda.common import orin_jit
from kernels.shared.rows import row_count


REGISTER_ROTATION_SOURCE = """
__device__ __forceinline__ void orinfer_fht128(float* values, int size) {
    #pragma unroll
    for (int stage = 0; stage < 5; ++stage) {
        #pragma unroll
        for (int i = 0; i < size; ++i) {
            values[i] = __shfl_xor_sync(0xffffffff, values[i], 1 << stage)
                      + values[i] * (1 - 2 * ((int(threadIdx.x) >> stage) & 1));
        }
    }
    #pragma unroll
    for (int i = 0; i < size; i += 4) {
        float x0 = values[i], x1 = values[i+1], x2 = values[i+2], x3 = values[i+3];
        float a = x1 + x0, b = x0 - x1, c = x3 + x2, d = x2 - x3;
        values[i] = c + a;
        values[i+1] = d + b;
        values[i+2] = a - c;
        values[i+3] = b - d;
    }
}
__device__ __forceinline__ float orinfer_rotation_max(const float* values, int size) {
    float result = 0;
    #pragma unroll
    for (int i = 0; i < size; ++i) result = fmaxf(result, fabsf(values[i]));
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        result = fmaxf(result, __shfl_xor_sync(0xffffffff, result, mask));
    return result;
}
"""


def _register_block_rotation(width, inputs, threads, swiglu, plane_rows=0):
    """Share the register butterflies and FP16 boundaries across row and tile CTAs."""
    local_width = width // threads

    @T.macro
    def transform(X, Signs, row, offset, values, maxima, maximum):
        T.import_source(REGISTER_ROTATION_SOURCE)
        lane = T.get_thread_binding()
        for i in T.unroll(local_width):
            k = offset + (i // 4) * (threads * 4) + (lane // 32) * 128 + (i % 4) * 32 + lane % 32
            values[i] = 0
            if k < inputs:
                if swiglu:
                    gate = T.cast(X[row, k], T.float32)
                    e = T.exp(-T.abs(gate))
                    sigmoid = T.if_then_else(gate >= 0, 1 / (1 + e), e / (1 + e))
                    value = T.cast(
                        T.cast(
                            (gate * sigmoid)
                            * T.cast(
                                X[row + plane_rows, k] if plane_rows else X[row, k + inputs],
                                T.float32,
                            ),
                            T.float16,
                        ),
                        T.float32,
                    )
                    values[i] = value * T.cast(Signs[k], T.float32)
                else:
                    values[i] = T.cast(X[row, k], T.float32) * T.cast(Signs[k], T.float32)
        T.evaluate(
            T.call_extern(
                "void",
                "orinfer_fht128",
                T.access_ptr(values[0], "rw", local_width),
                local_width,
            )
        )
        for i in T.unroll(local_width):
            values[i] = T.cast(T.cast(values[i] * 0.08838834764831845, T.float16), T.float32)
        maximum[0] = T.call_pure_extern(
            "float32",
            "orinfer_rotation_max",
            T.access_ptr(values[0], "r", local_width),
            local_width,
        )
        if lane % 32 == 0:
            maxima[lane // 32] = maximum[0]
        T.sync_threads()
        maximum[0] = maxima[0]
        for i in T.unroll(1, threads // 32):
            maximum[0] = T.max(maximum[0], maxima[i])

    return transform


def _register_rotation_a8(capacity, inputs, swiglu, reciprocal, dynamic_rows):
    """Keep complete block128 butterflies in each warp's registers.

    Coordinates 0/32/64/96 of each butterfly belong to one thread. The first
    five stages use shuffles; the last two use its four local values. Padding
    completes the last warp's block without changing the row-wide A8 scale.
    """
    m = row_count(capacity, dynamic_rows)
    threads = 128
    width = ((inputs + threads * 4 - 1) // (threads * 4)) * threads * 4
    local_width = width // threads
    transform = _register_block_rotation(width, inputs, threads, swiglu)

    @T.prim_func
    def main(
        X: T.Tensor((m, inputs * (2 if swiglu else 1)), T.float16),
        Signs: T.Tensor((inputs,), T.int8),
        Q: T.Tensor((m, inputs), T.int8),
        S: T.Tensor((m, 1), T.float16),
    ):
        with T.Kernel(m, threads=threads) as row:
            values = T.alloc_local((local_width,), T.float32)
            maxima = T.alloc_shared((threads // 32,), T.float32)
            maximum = T.alloc_local((1,), T.float32)
            scale = T.alloc_local((1,), T.float16)
            inv = T.alloc_local((1,), T.float32)
            lane = T.get_thread_binding()
            transform(X, Signs, row, 0, values, maxima, maximum)
            scale[0] = _rotation_scale(maximum[0])
            if lane == 0:
                S[row, 0] = scale[0]
            if reciprocal:
                inv[0] = T.call_extern("float32", "__fdiv_rn", 1.0, T.cast(scale[0], T.float32))
            for i in T.unroll(local_width):
                k = (i // 4) * (threads * 4) + (lane // 32) * 128 + (i % 4) * 32 + lane % 32
                if k < inputs:
                    if reciprocal:
                        Q[row, k] = T.cast(_rotation_code(values[i], scale[0], inv[0]), T.int8)
                    else:
                        ratio = T.call_extern(
                            "float32", "__fdiv_rn", values[i], T.cast(scale[0], T.float32)
                        )
                        Q[row, k] = T.cast(T.max(-127.0, T.min(127.0, T.round(ratio))), T.int8)

    return main


def _rotation_scale(maximum):
    return T.if_then_else(
        maximum > 0,
        T.max(T.call_extern("float32", "__fdiv_rn", maximum, 127.0), 2**-24),
        1.0,
    )


@T.macro
def _rotation_code(value, scale, inv):
    # Materialize the ratio once in TIR. An ordinary Python expression helper
    # can duplicate the opaque rounded-division call when the guard is inlined.
    operand = T.cast(value, T.float32)
    divisor = T.cast(scale, T.float32)
    ratio = operand * inv
    adjusted = T.if_then_else(
        T.abs(ratio - T.round(ratio)) >= 0.5 - 2**-14,
        T.call_extern("float32", "__fdiv_rn", operand, divisor),
        ratio,
    )
    return T.max(-127.0, T.min(127.0, T.round(adjusted)))


def block_hadamard(width):
    """Unscaled block128 butterflies, also used by inverse weight gathers."""

    @T.macro
    def transform(values, current):
        for stage in T.unroll(5):
            for k in T.Parallel(width):
                other = T.call_pure_extern(
                    "float32", "__shfl_xor_sync", T.uint32(0xFFFFFFFF), values[k], 1 << stage, 32
                )
                values[k] = other + values[k] * (1 - 2 * ((k >> stage) & 1))
        for stage in T.unroll(5, 7):
            T.copy(values, current)
            T.sync_threads()
            for k in T.Parallel(width):
                values[k] = current[k ^ (1 << stage)] + values[k] * (1 - 2 * ((k >> stage) & 1))
            T.sync_threads()

    return transform


def _block_rotation(width, inputs, swiglu, plane_rows=0):
    """Shared block128 butterflies; each warp owns consecutive coordinates."""

    butterfly = block_hadamard(width)

    @T.macro
    def transform(X, Signs, row, offset, values, current):
        for k in T.Parallel(width, coalesced_width=T.int32(1)):
            values[k] = 0.0
            if offset + k < inputs:
                if swiglu:
                    gate = T.cast(X[row, offset + k], T.float32)
                    e = T.exp(-T.abs(gate))
                    sigmoid = T.if_then_else(gate >= 0, 1 / (1 + e), e / (1 + e))
                    value = T.cast(
                        T.cast(
                            (gate * sigmoid)
                            * T.cast(
                                X[row + plane_rows, offset + k]
                                if plane_rows
                                else X[row, offset + k + inputs],
                                T.float32,
                            ),
                            T.float16,
                        ),
                        T.float32,
                    )
                    values[k] = value * T.cast(Signs[offset + k], T.float32)
                else:
                    values[k] = T.cast(X[row, offset + k], T.float32) * T.cast(
                        Signs[offset + k], T.float32
                    )
        butterfly(values, current)
        for k in T.Parallel(width):
            values[k] = T.cast(T.cast(values[k] * 0.08838834764831845, T.float16), T.float32)

    return transform


def _register_rotation_tiles(M, K, block_k, swiglu, dynamic_rows, plane_rows, threads):
    if threads not in (64, 128, 256, 512) or block_k % (threads * 4):
        raise ValueError("Register tiles must contain complete warp butterflies")
    m = row_count(M, dynamic_rows)
    blocks = T.ceildiv(K, block_k)
    local_width = block_k // threads
    transform = _register_block_rotation(block_k, K, threads, swiglu, plane_rows)

    @T.prim_func
    def main(
        X: T.Tensor(
            (m + plane_rows, K) if plane_rows else (m, K * (2 if swiglu else 1)), T.float16
        ),
        Signs: T.Tensor((K,), T.int8),
        Rotated: T.Tensor((m, K), T.float16),
        Maxima: T.Tensor((m, blocks), T.float32),
    ):
        with T.Kernel(m, blocks, threads=threads) as (row, block):
            values = T.alloc_local((local_width,), T.float32)
            maxima = T.alloc_shared((threads // 32,), T.float32)
            maximum = T.alloc_local((1,), T.float32)
            transform(X, Signs, row, block * block_k, values, maxima, maximum)
            lane = T.get_thread_binding()
            if lane == 0:
                Maxima[row, block] = maximum[0]
            for i in T.unroll(local_width):
                k = (
                    block * block_k
                    + (i // 4) * (threads * 4)
                    + (lane // 32) * 128
                    + (i % 4) * 32
                    + lane % 32
                )
                if k < K:
                    Rotated[row, k] = values[i]

    return main


@orin_jit
def rotate_activation_tiles(
    M,
    K,
    block_k=512,
    swiglu=False,
    dynamic_rows=False,
    *,
    plane_rows=0,
    registers=False,
    threads=128,
):
    """Parallel transforms plus partial maxima; scale remains one per row."""
    if min(M, K) <= 0 or K % 128 or block_k not in (128, 256, 512, 1024, 2048):
        raise ValueError("Invalid tiled block128 rotation")
    if (
        type(plane_rows) is not int
        or plane_rows < 0
        or (plane_rows and (not swiglu or plane_rows < M))
    ):
        raise ValueError("Invalid SwiGLU plane capacity")
    if type(registers) is not bool or (not registers and threads != 128):
        raise ValueError("Invalid tiled rotation implementation")
    if registers:
        return _register_rotation_tiles(M, K, block_k, swiglu, dynamic_rows, plane_rows, threads)
    m = row_count(M, dynamic_rows)
    blocks = T.ceildiv(K, block_k)
    transform = _block_rotation(block_k, K, swiglu, plane_rows)

    @T.prim_func
    def main(
        X: T.Tensor(
            (m + plane_rows, K) if plane_rows else (m, K * (2 if swiglu else 1)), T.float16
        ),
        Signs: T.Tensor((K,), T.int8),
        Rotated: T.Tensor((m, K), T.float16),
        Maxima: T.Tensor((m, blocks), T.float32),
    ):
        with T.Kernel(m, blocks, threads=128) as (row, block):
            current = T.alloc_shared((block_k,), T.float32)
            values = T.alloc_fragment((block_k,), T.float32)
            absolute = T.alloc_fragment((block_k,), T.float32)
            maximum = T.alloc_fragment((1,), T.float32)
            T.annotate_layout(
                {
                    values: tilelang.Fragment(
                        (block_k,),
                        forward_thread_fn=lambda k: k % 128,
                        forward_index_fn=lambda k: k // 128,
                    )
                }
            )
            transform(X, Signs, row, block * block_k, values, current)
            for k in T.Parallel(block_k):
                absolute[k] = T.abs(values[k])
                if block * block_k + k < K:
                    Rotated[row, block * block_k + k] = values[k]
            T.reduce_max(absolute, maximum, dim=0)
            Maxima[row, block] = maximum[0]

    return main


@orin_jit
def quantize_rotation_tiles(M, K, block_k=512, dynamic_rows=False):
    """Every CTA derives the same global row scale from the small max table."""
    if min(M, K) <= 0 or K % 128 or block_k not in (128, 256, 512, 1024, 2048):
        raise ValueError("Invalid tiled rotation quantization")
    m = row_count(M, dynamic_rows)
    blocks = T.ceildiv(K, block_k)
    reduction_width = 1 << (((K + block_k - 1) // block_k) - 1).bit_length()

    @T.prim_func
    def main(
        Rotated: T.Tensor((m, K), T.float16),
        Maxima: T.Tensor((m, blocks), T.float32),
        Q: T.Tensor((m, K), T.int8),
        S: T.Tensor((m, 1), T.float16),
    ):
        with T.Kernel(m, blocks, threads=128) as (row, block):
            maxima = T.alloc_fragment((reduction_width,), T.float32)
            maximum = T.alloc_fragment((1,), T.float32)
            scale = T.alloc_fragment((1,), T.float16)
            inv = T.alloc_fragment((1,), T.float32)
            for j in T.Parallel(reduction_width):
                maxima[j] = 0.0
                if j < blocks:
                    maxima[j] = Maxima[row, j]
            T.reduce_max(maxima, maximum, dim=0)
            scale[0] = _rotation_scale(maximum[0])
            if block == 0:
                S[row, 0] = scale[0]
            inv[0] = T.call_extern("float32", "__fdiv_rn", 1.0, T.cast(scale[0], T.float32))
            for k in T.Parallel(block_k):
                if block * block_k + k < K:
                    Q[row, block * block_k + k] = T.cast(
                        _rotation_code(Rotated[row, block * block_k + k], scale[0], inv[0]), T.int8
                    )

    return main


@orin_jit
def rotate_activation_a8(
    M: int,
    K: int,
    swiglu=False,
    reciprocal=True,
    dynamic_rows: bool = False,
    registers: bool = False,
):
    """Signed block128 rotation followed by one row-level A8 quantization."""
    _validate_rotation(M, K)
    if type(registers) is not bool:
        raise ValueError("Invalid register rotation selection")
    if registers:
        return _register_rotation_a8(M, K, swiglu, reciprocal, dynamic_rows)
    return _rotation_a8(M, K, swiglu, reciprocal, dynamic_rows)


@orin_jit
def rotate_expert_a8(
    M: int,
    K: int,
    swiglu=False,
    reciprocal=True,
    dynamic_rows: bool = False,
    *,
    routing=None,
    top_k=1,
    experts=1,
):
    """Fuse exact A8 planes and optional routing, with explicit expert geometry.

    No routing writes one plane row per input. Scatter writes routed assignment
    rows; shared keeps input planes and emits each assignment's scale and slot.
    """
    _validate_rotation(M, K)
    if routing not in (None, "shared", "scatter"):
        raise ValueError("Invalid expert activation routing")
    if type(top_k) is not int or type(experts) is not int or not 1 <= top_k <= experts:
        raise ValueError("Invalid expert activation geometry")
    return _rotation_a8(
        M,
        K,
        swiglu,
        reciprocal,
        dynamic_rows,
        exact=True,
        routing=routing,
        top_k=top_k,
        experts=experts,
    )


def _validate_rotation(M, K):
    if type(M) is not int or M <= 0 or type(K) is not int or K <= 0 or K % 128:
        raise ValueError("Invalid rotation dimensions")


def _rotation_a8(
    M, K, swiglu, reciprocal, dynamic_rows, *, exact=False, routing=None, top_k=1, experts=1
):
    scatter = routing == "scatter"
    dispatch = routing is not None
    width = 2 * K if swiglu else K
    M = row_count(M, dynamic_rows)
    slots = M * top_k if dispatch else M
    transform = _block_rotation(K, K, swiglu)

    @T.macro
    def body(X, Signs, Q, S, Low, High, Sum, IDs, RelativeSlot, RowOffsets, DS, SlotMap, row):
        current = T.alloc_shared((K,), T.float32)
        values = T.alloc_fragment((K,), T.float32)
        quant = T.alloc_fragment((K,), T.int32)
        low = T.alloc_fragment((K,), T.uint32)
        high = T.alloc_fragment((K,), T.uint32)
        total = T.alloc_fragment((1,), T.int32)
        absolute = T.alloc_fragment((K,), T.float32)
        maximum = T.alloc_fragment((1,), T.float32)
        scale = T.alloc_fragment((1,), T.float16)
        inv = T.alloc_fragment((1,), T.float32)
        T.annotate_layout(
            {
                v: tilelang.Fragment(
                    (K,),
                    forward_thread_fn=lambda k: k % 128,
                    forward_index_fn=lambda k: k // 128,
                )
                for v in (values, quant, low, high)
            }
        )
        transform(X, Signs, row, 0, values, current)
        for k in T.Parallel(K):
            absolute[k] = T.abs(values[k])
        T.reduce_max(absolute, maximum, dim=0)
        scale[0] = _rotation_scale(maximum[0])
        S[row, 0] = scale[0]
        if reciprocal:
            inv[0] = T.call_extern("float32", "__fdiv_rn", 1.0, T.cast(scale[0], T.float32))
        for k in T.Parallel(K):
            if reciprocal:
                quant[k] = T.cast(_rotation_code(values[k], scale[0], inv[0]), T.int32)
            else:
                ratio = T.call_extern(
                    "float32", "__fdiv_rn", values[k], T.cast(scale[0], T.float32)
                )
                quant[k] = T.cast(T.max(-127.0, T.min(127.0, T.round(ratio))), T.int32)
            Q[row, k] = quant[k]
            if exact:
                position = ((k & 1) << 1) | ((k & 2) >> 1) | (k & 4)
                low[k] = (T.cast(quant[k], T.uint32) & 15) << (position * 4)
                high[k] = ((T.cast(quant[k], T.uint32) >> 4) & 15) << (position * 4)
        if exact:
            for stage in T.unroll(3):
                for k in T.Parallel(K):
                    low[k] |= T.call_pure_extern(
                        "uint32", "__shfl_xor_sync", T.uint32(0xFFFFFFFF), low[k], 1 << stage, 32
                    )
                    high[k] |= T.call_pure_extern(
                        "uint32", "__shfl_xor_sync", T.uint32(0xFFFFFFFF), high[k], 1 << stage, 32
                    )
            T.reduce_sum(quant, total, dim=0)
            if scatter:
                T.sync_threads()
                for choice in T.serial(top_k):
                    slot = RowOffsets[IDs[row, choice]] + RelativeSlot[row, choice]
                    for k in T.Parallel(K):
                        if k % 8 == 0:
                            Low[slot, k // 8] = low[k]
                            High[slot, k // 8] = high[k]
                    if T.get_thread_binding() == 0:
                        Sum[slot] = total[0]
                        SlotMap[row, choice] = slot
                        DS[slot, 0] = S[row, 0]
            else:
                for k in T.Parallel(K):
                    if k % 8 == 0:
                        Low[row, k // 8] = low[k]
                        High[row, k // 8] = high[k]
                Sum[row] = total[0]

    if not exact:

        @T.prim_func
        def main(
            X: T.Tensor((M, width), T.float16),
            Signs: T.Tensor((K,), T.int8),
            Q: T.Tensor((M, K), T.int8),
            S: T.Tensor((M, 1), T.float16),
        ):
            with T.Kernel(M, threads=128) as row:
                body(X, Signs, Q, S, Q, Q, S, Q, Q, Q, S, Q, row)

        return main

    if not dispatch:

        @T.prim_func
        def rotate_split_a8(
            X: T.Tensor((M, width), T.float16),
            Signs: T.Tensor((K,), T.int8),
            Q: T.Tensor((M, K), T.int8),
            S: T.Tensor((M, 1), T.float16),
            Low: T.Tensor((slots, K // 8), T.uint32),
            High: T.Tensor((slots, K // 8), T.uint32),
            Sum: T.Tensor((slots,), T.int32),
        ):
            with T.Kernel(M, threads=128) as row:
                body(X, Signs, Q, S, Low, High, Sum, Q, Q, Q, S, Q, row)

        return (
            rotate_split_a8.with_attr("global_symbol", "rows_rotate_split")
            if dynamic_rows
            else rotate_split_a8
        )

    @T.prim_func
    def rotate_dispatch_a8(
        X: T.Tensor((M, width), T.float16),
        Signs: T.Tensor((K,), T.int8),
        Q: T.Tensor((M, K), T.int8),
        S: T.Tensor((M, 1), T.float16),
        Low: T.Tensor((slots, K // 8), T.uint32),
        High: T.Tensor((slots, K // 8), T.uint32),
        Sum: T.Tensor((slots,), T.int32),
        IDs: T.Tensor((M, top_k), T.int32),
        RelativeSlot: T.Tensor((M, top_k), T.int32),
        RowOffsets: T.Tensor((experts,), T.int32),
        DS: T.Tensor((slots, 1), T.float16),
        SlotMap: T.Tensor((M, top_k), T.int32),
    ):
        with T.Kernel(M, threads=128) as row:
            body(X, Signs, Q, S, Low, High, Sum, IDs, RelativeSlot, RowOffsets, DS, SlotMap, row)
            if not scatter:
                for rank in T.Parallel(top_k):
                    slot = RowOffsets[IDs[row, rank]] + RelativeSlot[row, rank]
                    SlotMap[row, rank] = slot
                    DS[slot, 0] = S[row, 0]

    if scatter:
        return rotate_dispatch_a8.with_attr("global_symbol", "rows_rotate_scatter")
    return rotate_dispatch_a8


@orin_jit
def residual_norm_rotate_a8(M, K=5120, *, dynamic_rows=False, zero_centered=True):
    """Residual RMSNorm, signed rotation and A8 with both FP16 boundaries intact."""
    from kernels.cuda.operators.op02_residual_norm import residual_norm_body

    if type(K) is not int or K <= 0 or K % 256:
        raise ValueError("Fused residual rotation requires complete 256-column tiles")
    m = row_count(M, dynamic_rows)
    threads = 256
    normalize = residual_norm_body(K, 1e-6, threads, staged=True, zero_centered=zero_centered)
    transform = _block_rotation(K, K, False)

    @T.prim_func
    def main(
        X: T.Tensor((m, K), T.float16),
        R: T.Tensor((m, K), T.float32),
        W: T.Tensor((K,), T.float16),
        Y: T.Tensor((m, K), T.float16),
        RO: T.Tensor((m, K), T.float32),
        Signs: T.Tensor((K,), T.int8),
        Q: T.Tensor((m, K), T.int8),
        S: T.Tensor((m,), T.float16),
    ):
        with T.Kernel(m, threads=threads) as row:
            normalized = T.alloc_shared((1, K), T.float16)
            current = T.alloc_shared((K,), T.float32)
            values = T.alloc_fragment((K,), T.float32)
            absolute = T.alloc_fragment((K,), T.float32)
            maximum = T.alloc_fragment((1,), T.float32)
            scale = T.alloc_fragment((1,), T.float16)
            inv = T.alloc_fragment((1,), T.float32)
            T.annotate_layout(
                {
                    values: tilelang.Fragment(
                        (K,),
                        forward_thread_fn=lambda k: k % threads,
                        forward_index_fn=lambda k: k // threads,
                    )
                }
            )
            normalize(X, R, W, Y, RO, normalized, row)
            T.sync_threads()
            transform(normalized, Signs, 0, 0, values, current)
            for k in T.Parallel(K):
                absolute[k] = T.abs(values[k])
            T.reduce_max(absolute, maximum, dim=0)
            scale[0] = _rotation_scale(maximum[0])
            S[row] = scale[0]
            inv[0] = T.call_extern("float32", "__fdiv_rn", 1.0, T.cast(scale[0], T.float32))
            for k in T.Parallel(K):
                Q[row, k] = T.cast(_rotation_code(values[k], scale[0], inv[0]), T.int8)

    return main
