"""Sparse QSA with shared INT8 KV tiles for twelve GQA heads on SM87.

FP16 tensor-core QK/PV use FP32 accumulators and softmax. Native prefill can
stage FP16 KV in reusable layer workspace; all other launches unpack each tile.
"""

import tilelang
import tilelang.language as T
from orinfer_build.compiler.cuda.common import orin_jit
from kernels.shared.rows import row_count
from kernels.cuda.model.qsa import geometry


PACKED_SOURCE = r"""
#include <cuda_fp16.h>
__device__ __forceinline__ int qsa_warp_token(const int *src, bool live) {
    int token = -1;
    if ((threadIdx.x & 31) == 0 && live) {
        token = *src;
    }
    return __shfl_sync(0xffffffffu, token, 0);
}

union QsaHalf2Bits { unsigned int bits; __half2 value; };
__device__ __forceinline__ unsigned int qsa_dequant_pair(unsigned int code, unsigned short scale_bits) {
    QsaHalf2Bits values{0}, bias{0}, scale{0}, result{0};
    values.bits = (((code & 0xffu) | ((code & 0xff00u) << 8)) ^ 0x00800080u) | 0x64006400u;
    bias.bits = 0x64806480u;
    scale.bits = static_cast<unsigned int>(scale_bits) | (static_cast<unsigned int>(scale_bits) << 16);
    result.value = __hmul2(__hsub2(values.value, bias.value), scale.value);
    return result.bits;
}

__device__ __forceinline__ unsigned int qsa_scale_pair(
    unsigned int packed, __half2 scale) {
    QsaHalf2Bits values{packed}, bias{0x64806480u};
    values.value = __hmul2(__hsub2(values.value, bias.value), scale);
    return values.bits;
}

__device__ __forceinline__ void qsa_dequant_store_eight(
    void* dst, unsigned int a, unsigned int b, unsigned short scale_bits) {
    // Map signed bytes to [0, 255], then place each byte in a half mantissa.
    // The resulting lanes are 1024 + unsigned_byte. Subtracting 1152
    // recovers the exact signed value before the original rounded half multiply.
    constexpr unsigned int exponent = 0x64646464u;
    a ^= 0x80808080u;
    b ^= 0x80808080u;
    QsaHalf2Bits scale{static_cast<unsigned int>(scale_bits) * 0x00010001u};
    *reinterpret_cast<uint4*>(dst) = make_uint4(
        qsa_scale_pair(__byte_perm(a, exponent, 0x5150), scale.value),
        qsa_scale_pair(__byte_perm(a, exponent, 0x5352), scale.value),
        qsa_scale_pair(__byte_perm(b, exponent, 0x5150), scale.value),
        qsa_scale_pair(__byte_perm(b, exponent, 0x5352), scale.value));
}
"""


@orin_jit
def sparse_attention(
    m: int,
    capacity: int,
    splits: int = 8,
    packed: bool = False,
    block_size: int = 32,
    threads: int = 128,
    single_buffer: bool = False,
    split_size: int | None = 256,
    dynamic_rows: bool = False,
    skip_empty: bool = False,
    async_scales: bool = False,
    vector_dequant: bool = False,
    explicit_barriers: bool = False,
    expanded_capacity: int = 0,
    partitions_per_cta: int = 1,
):
    geometry(m, capacity)

    m = row_count(m, dynamic_rows)
    if splits not in (1, 2, 4, 8):
        raise ValueError("Invalid QSA splits")
    # Selected contains at most 512 four-token blocks plus three pending
    # tokens. Fixed boundaries make each query's FP16 probability rounding
    # independent of prefill extent and request batching. The final slot
    # includes the pending tail; no split boundary follows a launch heuristic.
    # None retains balanced partitioning only for offline numerical references;
    # zero is the single-split reference using the same output ABI.
    if split_size is not None and (
        type(split_size) is not int
        or split_size not in (0, 256, 512, 1024, 2048)
        or split_size > 0
        and splits * split_size < 2048
    ):
        raise ValueError("Fixed QSA partitions must cover all selected blocks")
    if (
        block_size not in (32, 64)
        or threads not in (128, 256)
        or block_size == 32
        and threads == 256
    ):
        raise ValueError("Invalid sparse attention tile/warp geometry")
    if type(single_buffer) is not bool or single_buffer and not packed:
        raise ValueError("Single KV buffer requires packed attention")
    if type(async_scales) is not bool or async_scales and not packed:
        raise ValueError("Async scales require packed attention")
    if type(vector_dequant) is not bool or vector_dequant and not packed:
        raise ValueError("Vector dequantization requires packed attention")
    if (
        type(explicit_barriers) is not bool
        or explicit_barriers
        and not (packed and single_buffer and async_scales and vector_dequant)
    ):
        raise ValueError("Explicit QSA barriers require the single-buffer packed prefill path")
    if expanded_capacity and not (0 < expanded_capacity <= capacity and explicit_barriers):
        raise ValueError("QSA expansion requires bounded KV and explicit barriers")
    if type(partitions_per_cta) is not int or partitions_per_cta not in (1, 2):
        raise ValueError("Invalid QSA partitions per CTA")
    if partitions_per_cta == 2 and not (
        expanded_capacity
        and skip_empty
        and split_size == 256
        and block_size == 32
        and threads == 128
    ):
        raise ValueError("Paired QSA partitions require the bounded native prefill schedule")
    block = block_size
    shared_softmax = bool(expanded_capacity and block == 32 and threads == 128)
    kv_width = 64 if packed else 256
    kv_dtype = T.uint32 if packed else T.int8

    @T.macro
    def reduce_tile(scores, pairs, scratch, partial, result, maximum):
        # Each lane owns two adjacent scores. Publish their reduction once,
        # then combine the four warps in the original XOR-butterfly order.
        tile = T.reshape(scores, (16, 16, 2))
        if maximum:
            T.reduce_max(tile, pairs, dim=2)
        else:
            T.reduce_sum(tile, pairs, dim=2)
        for i, pair in T.Parallel(16, 16):
            scratch[pair // 4, i, pair % 4] = pairs[i, pair]
        T.sync_threads()
        for i, slot in T.Parallel(16, 4):
            a = scratch[0, i, slot]
            b = scratch[1, i, slot]
            c = scratch[2, i, slot]
            d = scratch[3, i, slot]
            partial[i, slot] = T.max(T.max(a, c), T.max(b, d)) if maximum else (a + c) + (b + d)
        if maximum:
            T.reduce_max(partial, result, dim=1, clear=False)
        else:
            T.reduce_sum(partial, result, dim=1)

    @T.macro
    def dequantize(raw, scales, output):
        if vector_dequant:
            # The MMA shared layout keeps each eight-half group contiguous
            # and aligned. Store the four rounded half2 values in one vector.
            for j, word_pair in T.Parallel(block, 32, coalesced_width=T.int32(1)):
                T.call_extern(
                    "void",
                    "qsa_dequant_store_eight",
                    T.access_ptr(output[j, word_pair * 8], "w", 8),
                    raw[j, word_pair * 2],
                    raw[j, word_pair * 2 + 1],
                    T.reinterpret(T.uint16, scales[j, word_pair // 8]),
                )
        else:
            for j, pair in T.Parallel(block, 128, coalesced_width=T.int32(1)):
                values = T.call_pure_extern(
                    "uint32",
                    "qsa_dequant_pair",
                    raw[j, pair // 2] >> ((pair % 2) * 16),
                    T.reinterpret(T.uint16, scales[j, pair // 32]),
                )
                output[j, pair * 2] = T.reinterpret(T.float16, T.cast(values & 65535, T.uint16))
                output[j, pair * 2 + 1] = T.reinterpret(T.float16, T.cast(values >> 16, T.uint16))

    @T.macro
    def copy_expanded(Rows, Tokens, target, cursor, kh):
        for j, group in T.Parallel(block, 32, coalesced_width=T.int32(1)):
            token = Tokens[j]
            T.ptx_cp_async(
                T.access_ptr(target[j, group * 8], "w", 8),
                T.access_ptr(Rows[T.max(token, 0), kh, group * 8], "r", 8),
                8,
                predicate=token >= 0 and token <= cursor,
            )

    @T.macro
    def copy_register_tokens(Rows, Tokens, Target, cursor, kh):
        tid = T.get_thread_binding()
        lane = tid % 32
        warp = tid // 32
        for group in T.unroll(block // (threads // 32)):
            j = warp + group * (threads // 32)
            token = Tokens[group]
            T.ptx_cp_async(
                T.access_ptr(Target[j, lane * 8], "w", 8),
                T.access_ptr(Rows[T.max(token, 0), kh, lane * 8], "r", 8),
                8,
                predicate=token >= 0 and token <= cursor,
            )

    @T.macro
    def attend(Query, K, V, KS, VS, Selected, Position, Max, Den, Out, ExpandedK, ExpandedV):
        with T.Kernel(m, 2, splits // partitions_per_cta, threads=threads) as (
            row,
            kh,
            group_split,
        ):
            if packed:
                T.import_source(PACKED_SOURCE)
            selected_tile = T.alloc_shared((block,), T.int32)
            key_scales = T.alloc_shared((block, 4), T.float16)
            value_scales = T.alloc_shared((block, 4), T.float16)
            q = T.alloc_shared((16, 256), T.float16)
            k = T.alloc_shared((block, 256), T.float16)
            if packed:
                raw_k = T.alloc_shared((block, 64), T.uint32)
                raw_v = raw_k if single_buffer else T.alloc_shared((block, 64), T.uint32)
            else:
                v = T.alloc_shared((block, 256), T.float16)
            p = T.alloc_shared((16, block), T.float16)
            scores = T.alloc_fragment((16, block), T.float32)
            acc = T.alloc_fragment((16, 256), T.float32)
            maximum = T.alloc_fragment((16,), T.float32)
            denom = T.alloc_fragment((16,), T.float32)
            previous = T.alloc_fragment((16,), T.float32)
            correction = T.alloc_fragment((16,), T.float32)
            rowsum = T.alloc_fragment((16,), T.float32)
            if shared_softmax:
                pairs = T.alloc_fragment((16, 16), T.float32)
                # Separate buffers let sum publication overlap earlier max
                # readers. The leading tile barrier protects their reuse.
                maximum_scratch = T.alloc_shared((4, 16, 4), T.float32)
                sum_scratch = T.alloc_shared((4, 16, 4), T.float32)
                partial = T.alloc_fragment((16, 4), T.float32)
                T.annotate_layout(
                    {
                        partial: tilelang.Fragment(
                            (16, 4),
                            forward_thread_fn=lambda i, j, replica: (i % 8) * 4 + j + replica * 32,
                            forward_index_fn=lambda i, j: i // 8,
                            replicate=4,
                        )
                    }
                )
            for i, d in T.Parallel(16, 256):
                if explicit_barriers:
                    q[i, d] = T.if_then_else(i < 12, Query[row, kh * 12 + i, d], T.float16(0))
                else:
                    q[i, d] = 0
                    if i < 12:
                        q[i, d] = Query[row, kh * 12 + i, d]
            length = T.min((Position[0] + row + 1) // 4, 512) * 4 + (Position[0] + row + 1) % 4
            count = (
                T.ceildiv(length, splits)
                if split_size is None
                else length
                if split_size == 0
                else split_size
            )
            expanded = Position[0] + m <= expanded_capacity if expanded_capacity else False
            # Q stays resident while the original fixed partitions are
            # computed, rounded and written to their own output slots.
            for partition in T.serial(partitions_per_cta):
                split = group_split * partitions_per_cta + partition
                if partitions_per_cta == 1 or split * count < length:
                    T.clear(acc)
                    T.clear(denom)
                    T.fill(maximum, -3.402823466e38)
                    start = split * count
                    end = (
                        T.if_then_else(split == splits - 1, length, T.min(start + count, length))
                        if split_size
                        else T.min(start + count, length)
                    )
                    for tile in T.serial(T.ceildiv(T.max(0, end - start), block)):
                        # Retire all readers before reusing metadata for another tile.
                        # Prefetch once instead of serializing each channel on the
                        # same global Selected/scale loads. The masked tail reads a
                        # valid allocated slot but stores -1, never an out-of-range ID.
                        T.sync_threads()
                        if partitions_per_cta == 2:
                            token_ids = T.alloc_local((block // (threads // 32),), T.int32)
                        if partitions_per_cta == 2 and expanded:
                            # Each warp loads a complete KV row and keeps its
                            # selected token for both K and V.
                            warp = T.get_thread_binding() // 32
                            for group in T.unroll(block // (threads // 32)):
                                slot = start + tile * block + warp + group * (threads // 32)
                                token_ids[group] = T.call_pure_extern(
                                    "int32",
                                    "qsa_warp_token",
                                    T.access_ptr(Selected[row, T.min(slot, 2050)], "r", 1),
                                    slot < end,
                                )
                        else:
                            for j in T.Parallel(block):
                                slot = start + tile * block + j
                                selected_tile[j] = T.if_then_else(
                                    slot < end, Selected[row, T.min(slot, 2050)], -1
                                )
                            T.sync_threads()
                        if expanded_capacity and expanded:
                            if partitions_per_cta == 2:
                                copy_register_tokens(ExpandedK, token_ids, k, Position[0] + row, kh)
                            else:
                                copy_expanded(ExpandedK, selected_tile, k, Position[0] + row, kh)
                            T.ptx_commit_group()
                            T.ptx_wait_group(0)
                            T.sync_threads()
                        else:
                            if not async_scales:
                                for j, g in T.Parallel(block, 4):
                                    token = selected_tile[j]
                                    key_scales[j, g] = 0
                                    value_scales[j, g] = 0
                                    if token >= 0 and token <= Position[0] + row:
                                        key_scales[j, g] = KS[token, kh, g]
                                        value_scales[j, g] = VS[token, kh, g]
                                T.sync_threads()
                            # Gather packed INT8 KV in 16-byte asynchronous loads. Keep
                            # packed V until softmax finishes so K and V can share one
                            # FP16 tensor-core tile without increasing shared memory.
                            if packed:
                                for j, segment in T.Parallel(block, 16):
                                    token = selected_tile[j]
                                    valid = token >= 0 and token <= Position[0] + row
                                    safe = T.max(token, 0)
                                    T.ptx_cp_async(
                                        T.tvm_access_ptr(
                                            T.type_annotation(T.uint32),
                                            raw_k.data,
                                            j * 64 + segment * 4,
                                            4,
                                            2,
                                        ),
                                        T.tvm_access_ptr(
                                            T.type_annotation(T.uint32),
                                            K.data,
                                            (safe * 2 + kh) * 64 + segment * 4,
                                            4,
                                            1,
                                        ),
                                        4,
                                        predicate=valid,
                                    )
                                    if not single_buffer:
                                        T.ptx_cp_async(
                                            T.tvm_access_ptr(
                                                T.type_annotation(T.uint32),
                                                raw_v.data,
                                                j * 64 + segment * 4,
                                                4,
                                                2,
                                            ),
                                            T.tvm_access_ptr(
                                                T.type_annotation(T.uint32),
                                                V.data,
                                                (safe * 2 + kh) * 64 + segment * 4,
                                                4,
                                                1,
                                            ),
                                            4,
                                            predicate=valid,
                                        )
                                if async_scales:
                                    # Each group-64 scale row is one aligned 8-byte copy.
                                    # Retire it with the packed K tile before dequantizing.
                                    for j in T.Parallel(block):
                                        token = selected_tile[j]
                                        valid = token >= 0 and token <= Position[0] + row
                                        safe = T.max(token, 0)
                                        T.ptx_cp_async(
                                            T.access_ptr(key_scales[j, 0], "w", 4),
                                            T.access_ptr(KS[safe, kh, 0], "r", 4),
                                            4,
                                            predicate=valid,
                                        )
                                        T.ptx_cp_async(
                                            T.access_ptr(value_scales[j, 0], "w", 4),
                                            T.access_ptr(VS[safe, kh, 0], "r", 4),
                                            4,
                                            predicate=valid,
                                        )
                                T.ptx_commit_group()
                                T.ptx_wait_group(0)
                                T.sync_threads()
                                dequantize(raw_k, key_scales, k)
                            else:
                                for j, d in T.Parallel(block, 256, coalesced_width=T.int32(1)):
                                    slot = start + tile * block + j
                                    k[j, d] = 0
                                    v[j, d] = 0
                                    if slot < end:
                                        token = selected_tile[j]
                                        if token >= 0 and token <= Position[0] + row:
                                            k[j, d] = T.cast(K[token, kh, d], T.float32) * T.cast(
                                                key_scales[j, d // 64], T.float32
                                            )
                                            v[j, d] = T.cast(V[token, kh, d], T.float32) * T.cast(
                                                value_scales[j, d // 64], T.float32
                                            )
                            if single_buffer:
                                # All K words have been decoded into the separate FP16
                                # tile. Retire their readers, then overlap packed V loads
                                # into the same raw storage with QK and softmax.
                                T.sync_threads()
                                for j, segment in T.Parallel(block, 16):
                                    token = selected_tile[j]
                                    valid = token >= 0 and token <= Position[0] + row
                                    safe = T.max(token, 0)
                                    T.ptx_cp_async(
                                        T.tvm_access_ptr(
                                            T.type_annotation(T.uint32),
                                            raw_v.data,
                                            j * 64 + segment * 4,
                                            4,
                                            2,
                                        ),
                                        T.tvm_access_ptr(
                                            T.type_annotation(T.uint32),
                                            V.data,
                                            (safe * 2 + kh) * 64 + segment * 4,
                                            4,
                                            1,
                                        ),
                                        4,
                                        predicate=valid,
                                    )
                                T.ptx_commit_group()
                        T.gemm(q, k, scores, transpose_B=True, clear_accum=True)
                        for i in T.Parallel(16):
                            previous[i] = maximum[i]
                        if expanded_capacity:
                            # Prefill uses the same softmax recurrence for complete and
                            # partial tiles. Only the final partial tile needs a mask.
                            # Padded query heads are independent and never written back.
                            if start + (tile + 1) * block <= end:
                                for i, j in T.Parallel(16, block):
                                    scores[i, j] *= 0.0625
                            else:
                                for i, j in T.Parallel(16, block):
                                    scores[i, j] = T.if_then_else(
                                        start + tile * block + j < end,
                                        scores[i, j] * 0.0625,
                                        -3.402823466e38,
                                    )
                            if shared_softmax:
                                reduce_tile(scores, pairs, maximum_scratch, partial, maximum, True)
                            else:
                                T.reduce_max(scores, maximum, dim=1, clear=False)
                            if start + (tile + 1) * block <= end:
                                for i, j in T.Parallel(16, block):
                                    scores[i, j] = T.exp(scores[i, j] - maximum[i])
                                    p[i, j] = scores[i, j]
                            else:
                                for i, j in T.Parallel(16, block):
                                    scores[i, j] = T.if_then_else(
                                        start + tile * block + j < end,
                                        T.exp(scores[i, j] - maximum[i]),
                                        0,
                                    )
                                    p[i, j] = scores[i, j]
                        else:
                            for i, j in T.Parallel(16, block):
                                if i < 12 and start + tile * block + j < end:
                                    scores[i, j] *= 0.0625
                                else:
                                    scores[i, j] = -3.402823466e38
                            T.reduce_max(scores, maximum, dim=1, clear=False)
                            for i, j in T.Parallel(16, block):
                                if i < 12 and start + tile * block + j < end:
                                    scores[i, j] = T.exp(scores[i, j] - maximum[i])
                                else:
                                    scores[i, j] = 0
                                p[i, j] = scores[i, j]
                        if shared_softmax:
                            reduce_tile(scores, pairs, sum_scratch, partial, rowsum, False)
                        else:
                            T.reduce_sum(scores, rowsum, dim=1)
                        for i in T.Parallel(16):
                            correction[i] = T.exp(previous[i] - maximum[i])
                            denom[i] = denom[i] * correction[i] + rowsum[i]
                        for i, d in T.Parallel(16, 256):
                            acc[i, d] *= correction[i]
                        if expanded_capacity and expanded:
                            T.sync_threads()
                            if partitions_per_cta == 2:
                                copy_register_tokens(ExpandedV, token_ids, k, Position[0] + row, kh)
                            else:
                                copy_expanded(ExpandedV, selected_tile, k, Position[0] + row, kh)
                            T.ptx_commit_group()
                            T.ptx_wait_group(0)
                            T.sync_threads()
                            T.gemm(p, k, acc)
                        else:
                            if packed:
                                if single_buffer:
                                    T.ptx_wait_group(0)
                                T.sync_threads()
                                dequantize(raw_v, value_scales, k)
                                if explicit_barriers:
                                    # All probability and dequantized value writes must
                                    # finish before any warp starts the shared PV tile.
                                    T.sync_threads()
                                T.gemm(p, k, acc)
                            else:
                                T.gemm(p, v, acc)
                    for i, d in T.Parallel(16, 256):
                        if i < 12:
                            Out[row, kh * 12 + i, split, d] = acc[i, d]
                    for i in T.Parallel(16):
                        if i < 12:
                            Max[row, kh * 12 + i, split] = maximum[i]
                            Den[row, kh * 12 + i, split] = denom[i]

    if expanded_capacity:

        @T.prim_func
        def main(
            Query: T.Tensor((m, 24, 256), T.float16),
            K: T.Tensor((capacity, 2, kv_width), kv_dtype),
            V: T.Tensor((capacity, 2, kv_width), kv_dtype),
            KS: T.Tensor((capacity, 2, 4), T.float16),
            VS: T.Tensor((capacity, 2, 4), T.float16),
            ExpandedK: T.Tensor((expanded_capacity, 2, 256), T.float16),
            ExpandedV: T.Tensor((expanded_capacity, 2, 256), T.float16),
            Selected: T.Tensor((m, 2051), T.int32),
            Position: T.Tensor((1,), T.int32),
            Max: T.Tensor((m, 24, splits), T.float32),
            Den: T.Tensor((m, 24, splits), T.float32),
            Out: T.Tensor((m, 24, splits, 256), T.float32),
        ):
            attend(Query, K, V, KS, VS, Selected, Position, Max, Den, Out, ExpandedK, ExpandedV)
    else:

        @T.prim_func
        def main(
            Query: T.Tensor((m, 24, 256), T.float16),
            K: T.Tensor((capacity, 2, kv_width), kv_dtype),
            V: T.Tensor((capacity, 2, kv_width), kv_dtype),
            KS: T.Tensor((capacity, 2, 4), T.float16),
            VS: T.Tensor((capacity, 2, 4), T.float16),
            Selected: T.Tensor((m, 2051), T.int32),
            Position: T.Tensor((1,), T.int32),
            Max: T.Tensor((m, 24, splits), T.float32),
            Den: T.Tensor((m, 24, splits), T.float32),
            Out: T.Tensor((m, 24, splits, 256), T.float32),
        ):
            attend(Query, K, V, KS, VS, Selected, Position, Max, Den, Out, K, V)

    if type(skip_empty) is not bool:
        raise ValueError("Invalid QSA empty partition policy")
    if skip_empty:
        if split_size is None or split_size <= 0:
            raise ValueError("Empty partition elision requires fixed QSA boundaries")
        from tvm import tirx as tir

        axes = {}

        def axis(node):
            if isinstance(node, tir.For) and node.thread_binding is not None:
                axes[node.thread_binding.thread_tag] = node.loop_var

        tir.stmt_functor.post_order_visit(main.body, axis)
        position = next(b for b in main.buffer_map.values() if b.name == "Position")
        cursor = tir.BufferLoad(position, [0]) + axes["blockIdx.x"] + 1
        length = tir.min(tir.floordiv(cursor, 4), 512) * 4 + tir.floormod(cursor, 4)

        def wrap(node):
            if isinstance(node, tir.SBlock) and node.name_hint == "tilelang_root":
                return tir.SBlock(
                    node.iter_vars,
                    node.reads,
                    node.writes,
                    node.name_hint,
                    tir.IfThenElse(
                        axes["blockIdx.z"] * (split_size * partitions_per_cta) < length,
                        node.body,
                        None,
                    ),
                    node.init,
                    node.alloc_buffers,
                    node.match_buffers,
                    node.annotations,
                )
            return None

        main = tir.PrimFunc(
            main.params,
            tir.stmt_functor.ir_transform(main.body, None, wrap),
            main.ret_type,
            main.buffer_map,
            main.attrs,
        )
    return main


def expand_prefill_kv(m, capacity, limit):
    """Dequantize the live prefill context into two reusable FP16 workspaces."""
    if not 0 < limit <= capacity:
        raise ValueError("Invalid QSA expansion capacity")
    m = row_count(m, True)

    @T.prim_func
    def main(
        K: T.Tensor((capacity, 2, 64), T.uint32),
        V: T.Tensor((capacity, 2, 64), T.uint32),
        KS: T.Tensor((capacity, 2, 4), T.float16),
        VS: T.Tensor((capacity, 2, 4), T.float16),
        Position: T.Tensor((1,), T.int32),
        ExpandedK: T.Tensor((limit, 2, 256), T.float16),
        ExpandedV: T.Tensor((limit, 2, 256), T.float16),
    ):
        with T.Kernel(T.ceildiv(limit, 4), 2, threads=128) as (group, h):
            T.import_source(PACKED_SOURCE)
            end = Position[0] + m
            if end <= limit:
                for row, lane in T.Parallel(4, 32, coalesced_width=T.int32(1)):
                    token = group * 4 + row
                    if token < end:
                        T.call_extern(
                            "void",
                            "qsa_dequant_store_eight",
                            T.access_ptr(ExpandedK[token, h, lane * 8], "w", 8),
                            K[token, h, lane * 2],
                            K[token, h, lane * 2 + 1],
                            T.reinterpret(T.uint16, KS[token, h, lane // 8]),
                        )
                        T.call_extern(
                            "void",
                            "qsa_dequant_store_eight",
                            T.access_ptr(ExpandedV[token, h, lane * 8], "w", 8),
                            V[token, h, lane * 2],
                            V[token, h, lane * 2 + 1],
                            T.reinterpret(T.uint16, VS[token, h, lane // 8]),
                        )

    return main
