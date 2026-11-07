"""Group-64 symmetric INT8 KV. Only storage is quantized; attention uses
FP16 tensor-core operands and FP32 accumulation. Scales are stored FP16.
"""

import tilelang.language as T
from orinfer_build.compiler.cuda.common import orin_jit


HALF2_SOURCE = r"""
#include <cuda_fp16.h>
union KvHalf2Bits { unsigned int bits; __half2 value; };
__device__ __forceinline__ unsigned int kv_dequant_pair(unsigned int code, unsigned short scale_bits) {
    KvHalf2Bits values{0}, bias{0}, scale{0}, result{0}, lower{0}, upper{0};
    // Encode unsigned bytes xor 128 as exact FP16 integers 1024..1279.
    values.bits = (((code & 0xffu) | ((code & 0xff00u) << 8)) ^ 0x00800080u) | 0x64006400u;
    bias.bits = 0x64806480u; // 1152 in both lanes
    scale.bits = static_cast<unsigned int>(scale_bits) | (static_cast<unsigned int>(scale_bits) << 16);
    result.value = __hmul2(__hsub2(values.value, bias.value), scale.value);
    lower.bits = 0xfbfffbffu; upper.bits = 0x7bff7bffu;
    result.value = __hmin2(__hmax2(result.value, lower.value), upper.value);
    return result.bits;
}
"""


def int8_kv_tile(block_n, dimension=256, threads=128):
    """Shared packed-byte loader for prefill, decode and verification tiles."""
    if threads <= 0 or (block_n * (dimension // 4)) % threads:
        raise ValueError("KV tile must contain complete thread stripes")

    @T.macro
    def load(K, V, KS, VS, k, v, batch, head, base, end):
        for stripe in T.unroll(block_n * (dimension // 4) // threads, unroll_factor=1):
            j = (stripe * threads + T.get_thread_binding()) // (dimension // 4)
            d = (stripe * threads + T.get_thread_binding()) % (dimension // 4)
            token = base + j
            if token < end:
                kw = K[batch, token, head * (dimension // 4) + d]
                vw = V[batch, token, head * (dimension // 4) + d]
                sk = T.reinterpret(T.uint16, KS[batch, token, head * (dimension // 64) + d // 16])
                sv = T.reinterpret(T.uint16, VS[batch, token, head * (dimension // 64) + d // 16])
                k0 = T.call_extern("uint32", "kv_dequant_pair", kw, sk)
                k1 = T.call_extern("uint32", "kv_dequant_pair", kw >> 16, sk)
                v0 = T.call_extern("uint32", "kv_dequant_pair", vw, sv)
                v1 = T.call_extern("uint32", "kv_dequant_pair", vw >> 16, sv)
                k[j, d * 4] = T.reinterpret(T.float16, T.cast(k0 & 65535, T.uint16))
                k[j, d * 4 + 1] = T.reinterpret(T.float16, T.cast(k0 >> 16, T.uint16))
                k[j, d * 4 + 2] = T.reinterpret(T.float16, T.cast(k1 & 65535, T.uint16))
                k[j, d * 4 + 3] = T.reinterpret(T.float16, T.cast(k1 >> 16, T.uint16))
                v[j, d * 4] = T.reinterpret(T.float16, T.cast(v0 & 65535, T.uint16))
                v[j, d * 4 + 1] = T.reinterpret(T.float16, T.cast(v0 >> 16, T.uint16))
                v[j, d * 4 + 2] = T.reinterpret(T.float16, T.cast(v1 & 65535, T.uint16))
                v[j, d * 4 + 3] = T.reinterpret(T.float16, T.cast(v1 >> 16, T.uint16))
            else:
                for lane in T.unroll(4):
                    k[j, d * 4 + lane] = 0.0
                    v[j, d * 4 + lane] = 0.0

    return load


@orin_jit
def full_prepare_mrope_int8(
    max_pages: int,
    context: int,
    section: tuple[int, int, int],
    block_size: int = 128,
    max_position: int = 8704,
    query_heads: int = 24,
    kv_heads: int = 4,
    dimension: int = 256,
    rotary_dim: int = 64,
    gated: bool = True,
    zero_centered: bool = True,
):
    if min(query_heads, kv_heads) <= 0 or query_heads % kv_heads:
        raise ValueError("Invalid full-attention heads")
    if dimension % 64 or not 0 < rotary_dim <= dimension or rotary_dim % 2:
        raise ValueError("Invalid Q/K normalization or rotary geometry")
    qwidth = query_heads * dimension * (2 if gated else 1)
    vstart = qwidth + kv_heads * dimension
    width = vstart + kv_heads * dimension
    heads = query_heads + kv_heads
    rows = T.dynamic("rows")

    @T.prim_func
    def kernel(
        X: T.Tensor((rows, width), T.float16),
        WQ: T.Tensor((dimension,), T.float16),
        WK: T.Tensor((dimension,), T.float16),
        Cache: T.Tensor((max_position, rotary_dim), T.float16),
        Req: T.Tensor((rows,), T.int32),
        Pos: T.Tensor((rows,), T.int32),
        Pages: T.Tensor((1, max_pages), T.int32),
        Status: T.Tensor((1,), T.int32),
        MRope: T.Tensor((context, 3), T.int32),
        Q: T.Tensor((rows, query_heads, dimension), T.float16),
        Gate: T.Tensor((rows, query_heads, dimension), T.float16),
        K: T.Tensor((max_pages, block_size, kv_heads, dimension), T.int8),
        V: T.Tensor((max_pages, block_size, kv_heads, dimension), T.int8),
        KS: T.Tensor((max_pages, block_size, kv_heads, dimension // 64), T.float16),
        VS: T.Tensor((max_pages, block_size, kv_heads, dimension // 64), T.float16),
    ):
        with T.Kernel(rows * heads, threads=128) as work:
            row = work // heads
            head = work % heads
            values = T.alloc_fragment((dimension,), T.float32)
            square = T.alloc_fragment((dimension,), T.float32)
            total = T.alloc_fragment((1,), T.float32)
            norm = T.alloc_shared((dimension,), T.float16)
            ka = T.alloc_fragment((dimension // 64, 64), T.float32)
            va = T.alloc_fragment((dimension // 64, 64), T.float32)
            km = T.alloc_fragment((dimension // 64,), T.float32)
            vm = T.alloc_fragment((dimension // 64,), T.float32)
            ks = T.alloc_shared((dimension // 64,), T.float16)
            vs = T.alloc_shared((dimension // 64,), T.float16)
            if Status[0] == 0:
                for j in T.Parallel(dimension):
                    if head < query_heads:
                        values[j] = T.cast(
                            X[row, head * dimension * (2 if gated else 1) + j], T.float32
                        )
                        Gate[row, head, j] = (
                            X[row, head * dimension * 2 + dimension + j] if gated else 0.0
                        )
                    else:
                        values[j] = T.cast(
                            X[row, qwidth + (head - query_heads) * dimension + j], T.float32
                        )
                    square[j] = values[j] * values[j]
                T.reduce_sum(square, total, dim=0)
                for j in T.Parallel(dimension):
                    weight = T.if_then_else(head < query_heads, WQ[j], WK[j])
                    norm[j] = (values[j] * T.rsqrt(total[0] / dimension + 1e-6)) * (
                        (1.0 if zero_centered else 0.0) + T.cast(weight, T.float32)
                    )
                T.sync_threads()
                for j in T.Parallel(dimension):
                    values[j] = T.cast(norm[j], T.float32)
                    if j < rotary_dim:
                        idx = j % (rotary_dim // 2)
                        partner = T.if_then_else(
                            j < rotary_dim // 2, j + rotary_dim // 2, j - rotary_dim // 2
                        )
                        axis = T.if_then_else(
                            idx % 3 == 1 and idx < section[1] * 3,
                            1,
                            T.if_then_else(idx % 3 == 2 and idx < section[2] * 3, 2, 0),
                        )
                        position = MRope[Pos[row], axis]
                        a = T.cast(
                            T.cast(norm[j], T.float32) * T.cast(Cache[position, idx], T.float32),
                            T.float16,
                        )
                        b = T.cast(
                            T.cast(norm[partner], T.float32)
                            * T.cast(Cache[position, idx + rotary_dim // 2], T.float32),
                            T.float16,
                        )
                        values[j] = T.if_then_else(
                            j < rotary_dim // 2,
                            T.cast(a, T.float32) - T.cast(b, T.float32),
                            T.cast(a, T.float32) + T.cast(b, T.float32),
                        )
                    if head < query_heads:
                        Q[row, head, j] = values[j]
                if head >= query_heads:
                    # Quantize the exact FP16 values the reference cache stores.
                    for g, d in T.Parallel(dimension // 64, 64):
                        ka[g, d] = T.abs(T.cast(T.cast(values[g * 64 + d], T.float16), T.float32))
                        va[g, d] = T.abs(
                            T.cast(
                                X[row, vstart + (head - query_heads) * dimension + g * 64 + d],
                                T.float32,
                            )
                        )
                    T.reduce_max(ka, km, dim=1)
                    T.reduce_max(va, vm, dim=1)
                    for g in T.Parallel(dimension // 64):
                        # FP16 scales, bounded above zero even for subnormal input.
                        ks[g] = T.max(km[g] / 127.0, 0.000000059604644775390625)
                        vs[g] = T.max(vm[g] / 127.0, 0.000000059604644775390625)
                    T.sync_threads()
                    page = Pages[Req[row], Pos[row] // block_size]
                    for g in T.Parallel(dimension // 64):
                        KS[page, Pos[row] % block_size, head - query_heads, g] = ks[g]
                        VS[page, Pos[row] % block_size, head - query_heads, g] = vs[g]
                    for j in T.Parallel(dimension):
                        kval = T.cast(T.cast(values[j], T.float16), T.float32)
                        vval = T.cast(
                            X[row, vstart + (head - query_heads) * dimension + j], T.float32
                        )
                        kr = T.call_extern(
                            "float32", "__fdiv_rn", kval, T.cast(ks[j // 64], T.float32)
                        )
                        vr = T.call_extern(
                            "float32", "__fdiv_rn", vval, T.cast(vs[j // 64], T.float32)
                        )
                        K[page, Pos[row] % block_size, head - query_heads, j] = T.cast(
                            T.max(-127.0, T.min(127.0, T.round(kr))), T.int8
                        )
                        V[page, Pos[row] % block_size, head - query_heads, j] = T.cast(
                            T.max(-127.0, T.min(127.0, T.round(vr))), T.int8
                        )

    return kernel


@orin_jit
def dequantize_attention_kv(capacity, context, kv_heads=2, dimension=256, block_n=32, programs=128):
    from kernels.shared.rows import row_count, explicit_rows

    if min(capacity, context, kv_heads, programs) <= 0 or dimension % 64:
        raise ValueError("Invalid attention staging geometry")
    # One temporary bank is reused across layers and owners; only committed
    # KV is read. A bounded grid avoids launching for unused context capacity.
    m = row_count(capacity, True)
    load = int8_kv_tile(block_n, dimension, 128)
    workers = T.min(programs, m)

    @T.prim_func
    def decode(
        K: T.Tensor((1, context, kv_heads * dimension // 4), T.uint32),
        V: T.Tensor((1, context, kv_heads * dimension // 4), T.uint32),
        KS: T.Tensor((1, context, kv_heads * dimension // 64), T.float16),
        VS: T.Tensor((1, context, kv_heads * dimension // 64), T.float16),
        Lengths: T.Tensor((1,), T.int32),
        DK: T.Tensor((1, context, kv_heads * dimension), T.float16),
        DV: T.Tensor((1, context, kv_heads * dimension), T.float16),
    ):
        with T.Kernel(workers, kv_heads, threads=128) as (b, h):
            T.import_source(HALF2_SOURCE)
            k = T.alloc_shared((block_n, dimension), T.float16)
            v = T.alloc_shared((block_n, dimension), T.float16)
            end = T.min(context, Lengths[0])
            for stripe in T.serial(T.ceildiv(end, block_n * workers)):
                base = (stripe * workers + b) * block_n
                if base < end:
                    load(K, V, KS, VS, k, v, 0, h, base, end)
                    T.sync_threads()
                    for i, d in T.Parallel(block_n, dimension):
                        if base + i < end:
                            DK[0, base + i, h * dimension + d] = k[i, d]
                            DV[0, base + i, h * dimension + d] = v[i, d]
                    T.sync_threads()

    return explicit_rows(decode)
