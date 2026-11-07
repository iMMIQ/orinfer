"""Short HC up/mix uses a lossless BF16 read cache with explicit synchronization."""

import tilelang
import tilelang.language as T
from orinfer_build.compiler.cuda.common import orin_jit
from kernels.shared.rows import row_count, explicit_scalar

from kernels.cuda.model.half_math import SOURCE as HALF_MATH, stream_sigmoid

source = r"""
#include <tl_templates/cuda/instruction/mma.h>
__device__ __forceinline__ bfloat16_t hc_bf_bits(unsigned bits) {
 unsigned short value=(unsigned short)bits;
 return *reinterpret_cast<bfloat16_t*>(&value);
}
__device__ __forceinline__ unsigned long long hc_expand13_four(unsigned m,unsigned q,unsigned e) {
 unsigned s0=(m&255u)|((m&0xff00u)<<8);
 unsigned s1=((m>>16)&255u)|((m&0xff000000u)>>8);
 unsigned q0=(q&31u)|((q&0x3e0u)<<11);
 unsigned q1=((q>>10)&31u)|((q&0xf8000u)<<1);
 unsigned base=e|(e<<16);
 unsigned v0=(s0&0x007f007fu)|((s0&0x00800080u)<<8)|((q0+base)<<7);
 unsigned v1=(s1&0x007f007fu)|((s1&0x00800080u)<<8)|((q1+base)<<7);
 return v0|(static_cast<unsigned long long>(v1)<<32);
}
__device__ __forceinline__ void hc_store13_four(void* p,unsigned long long v) {
 unsigned s=(unsigned)__cvta_generic_to_shared(p);
 unsigned lo=(unsigned)v,hi=(unsigned)(v>>32);
 asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s),"r"(lo),"r"(hi):"memory");
}
"""


@orin_jit
def hc_up13_mix(
    M: int,
    H: int,
    rank: int,
    streams: int = 4,
    dynamic_rows: bool = False,
    stage_count: int = 1,
    base_exp: int | None = 99,
    patch_count: int = 0,
    block_m: int = 8,
    weight_left: bool = True,
):
    """BF16 up projection and branch mix, preserving every FP16 boundary."""
    if (
        not 1 <= M <= 1024
        or H != 2560
        or rank not in (256, 320)
        or streams != 4
        or stage_count != 1
        or not 0 <= patch_count <= 3
        or block_m not in (8, 16, 32, 64)
        or type(weight_left) is not bool
        or (base_exp is not None and (type(base_exp) is not int or not 0 <= base_exp <= 224))
    ):
        raise ValueError("Unsupported exact BF16 HC read cache geometry")
    exponent = T.dynamic("base_exp") if base_exp is None else base_exp
    bm, bn = block_m, 32
    M = row_count(M, dynamic_rows)

    @T.prim_func
    def hc_up13_vector_project(
        A: T.Tensor((M, rank), T.float16),
        Lo: T.Tensor((H // bn, rank // 64, streams * bn, 16), T.uint32),
        Hi: T.Tensor((H // bn, rank // 64, streams * bn, 10), T.uint32),
        Patches: T.Tensor((max(1, patch_count),), T.uint64),
        Normed: T.Tensor((M, streams, H), T.float16),
        Out: T.Tensor((M, H), T.float16),
    ):
        with T.Kernel(T.ceildiv(M, bm), T.ceildiv(H, bn), threads=128) as (by, bx):
            T.import_source(HALF_MATH + source)
            a = T.alloc_shared((bm, 64), T.bfloat16)
            w = T.alloc_shared((streams * bn, 64), T.bfloat16)
            lo = T.alloc_shared((streams * bn, 16), T.uint32)
            hi = T.alloc_shared((streams * bn, 10), T.uint32)
            shape = (streams * bn, bm) if weight_left else (bm, streams * bn)
            acc = T.alloc_fragment(shape, T.float32)
            rounded_up = T.alloc_shared(shape, T.float16)
            total = T.alloc_fragment((bm, bn), T.float32)
            T.annotate_layout(
                {
                    total: tilelang.Fragment(
                        (bm, bn),
                        forward_thread_fn=lambda i, j: (i % 4) * 32 + j,
                        forward_index_fn=lambda i, j: i // 4,
                    )
                }
            )
            T.clear(acc)
            for kg in T.Pipelined(rank // 64, num_stages=stage_count):
                for i, j in T.Parallel(bm, 64):
                    a[i, j] = 0
                    if by * bm + i < M:
                        a[i, j] = A[by * bm + i, kg * 64 + j]
                T.copy(Lo[bx, kg, 0, 0], lo)
                T.copy(Hi[bx, kg, 0, 0], hi)
                for n, j in T.Parallel(streams * bn, 16):
                    pos = j * 20
                    q = (
                        T.call_pure_extern(
                            "uint32",
                            "__funnelshift_r",
                            hi[n, pos // 32],
                            hi[n, T.min(pos // 32 + 1, 9)],
                            pos % 32,
                        )
                        & 1048575
                    )
                    v = T.call_pure_extern(
                        "uint64", "hc_expand13_four", lo[n, j], q, T.uint32(exponent)
                    )
                    T.call_extern("void", "hc_store13_four", T.access_ptr(w[n, j * 4], "w", 4), v)
                if patch_count:
                    for patch_id in T.unroll(patch_count):
                        patch = Patches[patch_id]
                        index = T.cast(patch >> 16, T.int32)
                        if index // (streams * bn * 64) == bx * (rank // 64) + kg:
                            T.sync_threads()
                            if T.get_thread_binding() == 0:
                                w[(index // 64) % (streams * bn), index % 64] = T.call_pure_extern(
                                    "bfloat16", "hc_bf_bits", T.cast(patch & 65535, T.uint32)
                                )
                            T.sync_threads()
                T.sync_threads()
                # N x 8 MMA avoids the sixteen-row activation tile for small
                # decode/verification batches. Preserve the same K traversal.
                if weight_left:
                    T.gemm(w, a, acc, transpose_B=True)
                else:
                    T.gemm(a, w, acc, transpose_B=True)
            T.copy(acc, rounded_up)
            T.clear(total)
            for branch in T.serial(streams):
                for i, j in T.Parallel(bm, bn):
                    if by * bm + i < M:
                        up = (
                            rounded_up[branch * bn + j, i]
                            if weight_left
                            else rounded_up[i, branch * bn + j]
                        )
                        gate = stream_sigmoid(T.cast(up, T.float32), "float16")
                        product = T.cast(
                            T.cast(gate, T.float32)
                            * T.cast(Normed[by * bm + i, branch, bx * bn + j], T.float32),
                            T.float16,
                        )
                        total[i, j] += T.cast(product, T.float32)
            for i, j in T.Parallel(bm, bn):
                if by * bm + i < M:
                    Out[by * bm + i, bx * bn + j] = total[i, j] / streams

    return (
        explicit_scalar(hc_up13_vector_project, exponent)
        if base_exp is None
        else hc_up13_vector_project
    )
