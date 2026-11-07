"""Exact E8P arithmetic shared by row-major, K-major and cached-input schedules."""

import tilelang.language as T
from kernels.cuda.model.integer_core import rescale

SOURCE = r"""
// Parity toggles every nibble's low bit and the first coordinate's sign.
// Fold both into one final XOR; PRMT then uses a constant source and does
// not depend on the population count. Selector nibbles contain only 0..3.
__device__ __forceinline__ unsigned nib_fast(unsigned c,unsigned b){unsigned pp=__popc(c&255u)&1u;unsigned sel=(c&0x33u)|((c&0xccu)<<6),r;asm("prmt.b32 %0, %1, 0, %2;":"=r"(r):"r"(0xeee00e00u),"r"(sel));return b^r^(pp*0x1111111fu);}

__device__ __forceinline__ unsigned nib_pair(const unsigned short* p){return *reinterpret_cast<const unsigned*>(p);}
__device__ __forceinline__ void nib_store2(signed char* p,unsigned a,unsigned b){*reinterpret_cast<uint2*>(p)=make_uint2(a,b);}

__device__ __forceinline__ void nib_mma_low(int* d,unsigned a0,unsigned a1,unsigned a2,unsigned a3,unsigned b0,unsigned b1) {
 asm volatile("mma.sync.aligned.m16n8k64.row.col.s32.s4.u4.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" : "+r"(d[0]),"+r"(d[1]),"+r"(d[2]),"+r"(d[3]) : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}
__device__ __forceinline__ void nib_mma_high(int* d,unsigned a0,unsigned a1,unsigned a2,unsigned a3,unsigned b0,unsigned b1) {
 asm volatile("mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" : "+r"(d[0]),"+r"(d[1]),"+r"(d[2]),"+r"(d[3]) : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}
__device__ __forceinline__ void nib_ldm4(const signed char* p,unsigned* out) {unsigned a=__cvta_generic_to_shared(p);asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(out[0]),"=r"(out[1]),"=r"(out[2]),"=r"(out[3]) : "r"(a));}
__device__ __forceinline__ void nib_ldm2(const signed char* p,unsigned* out) {unsigned a=__cvta_generic_to_shared(p);asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(out[0]),"=r"(out[1]) : "r"(a));}
__device__ __forceinline__ void nib_store(signed char* p,unsigned x){*reinterpret_cast<unsigned*>(p)=x;}
__device__ __forceinline__ int nib_sum(unsigned a,unsigned b){return __dp4a((int)a,0x01010101,__dp4a((int)b,0x01010101,0));}

// Produce one K64 partial at a time. The caller folds it into the final
// INT32 accumulator before reusing these four output registers.
__device__ __forceinline__ void e8p_fold_low(int* d,unsigned a0,unsigned a1,unsigned a2,unsigned a3,unsigned b0,unsigned b1) {
 asm volatile("mma.sync.aligned.m16n8k64.row.col.s32.s4.u4.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};" : "=r"(d[0]),"=r"(d[1]),"=r"(d[2]),"=r"(d[3]) : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(0));
}
__device__ __forceinline__ void e8p_fold_high(int* d,unsigned a0,unsigned a1,unsigned a2,unsigned a3,unsigned b0,unsigned b1) {
 asm volatile("mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};" : "=r"(d[0]),"=r"(d[1]),"=r"(d[2]),"=r"(d[3]) : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(0));
}
"""


# Exact row sums and selector-to-nibble conversion for dense direct operands.
DENSE_SOURCE = r"""
__device__ __forceinline__ int direct_reduce4(int s) {
 s += __shfl_xor_sync(0xffffffff,s,1);
 return s + __shfl_xor_sync(0xffffffff,s,2);
}
__device__ __forceinline__ int direct_row_sum(int s,int row) {
 int v;
 asm volatile("shfl.sync.idx.b32 %0, %1, %2, 31, 0xffffffff;" : "=r"(v) : "r"(s), "r"(row*4));
 return v;
}
__device__ __forceinline__ unsigned direct_nibble_basis(unsigned s) {
 unsigned r=0;
 #pragma unroll
 for(unsigned i=0;i<8;i++) {
  unsigned j=(i&4)|((i&1)<<1)|((i&2)>>1);
  unsigned c=(s>>(4*j))&7;
  int v=2*(c&3)+1;
  if(c&4)v=-v;
  r|=(unsigned(v)&15)<<(4*i);
 }
 return r;
}
"""


def low_plane(x, y):
    return (
        (x & T.uint32(0x00000F0F))
        | ((x >> 12) & T.uint32(0x0000F0F0))
        | ((y << 16) & T.uint32(0x0F0F0000))
        | ((y << 4) & T.uint32(0xF0F00000))
    )


def high_plane(x, y):
    return (
        ((x >> 4) & T.uint32(0x00000F0F))
        | ((x >> 16) & T.uint32(0x0000F0F0))
        | ((y << 12) & T.uint32(0x0F0F0000))
        | (y & T.uint32(0xF0F00000))
    )


def mma_pair(bm):
    @T.macro
    def product(lo, hi, v, a, nt, stripe):
        T.evaluate(
            T.call_extern(
                "void",
                "nib_mma_low",
                T.access_ptr(lo[nt * (bm // 2) + stripe * 4], "rw", 4),
                v[0],
                v[1],
                v[2],
                v[3],
                a[stripe * 2],
                a[stripe * 2 + 1],
            )
        )
        T.evaluate(
            T.call_extern(
                "void",
                "nib_mma_high",
                T.access_ptr(hi[nt * (bm // 2) + stripe * 4], "rw", 4),
                v[0],
                v[1],
                v[2],
                v[3],
                a[bm // 4 + stripe * 2],
                a[bm // 4 + 1 + stripe * 2],
            )
        )

    return product


def folded_pair(bm):
    """Keep the full A8 product in one INT32 bank across E8P stages."""

    @T.macro
    def product(acc, v, a, nt, stripe, coefficient):
        partial = T.alloc_local((4,), T.int32)
        T.evaluate(
            T.call_extern(
                "void",
                "e8p_fold_low",
                T.access_ptr(partial[0], "w", 4),
                v[0],
                v[1],
                v[2],
                v[3],
                a[stripe * 2],
                a[stripe * 2 + 1],
            )
        )
        for j in T.unroll(4):
            acc[nt * (bm // 2) + stripe * 4 + j] += coefficient * partial[j]
        T.evaluate(
            T.call_extern(
                "void",
                "e8p_fold_high",
                T.access_ptr(partial[0], "w", 4),
                v[0],
                v[1],
                v[2],
                v[3],
                a[bm // 4 + stripe * 2],
                a[bm // 4 + stripe * 2 + 1],
            )
        )
        for j in T.unroll(4):
            acc[nt * (bm // 2) + stripe * 4 + j] += (coefficient * 16) * partial[j]

    return product


def store_folded_rows(bm, bn, n, coefficient_sum):
    @T.macro
    def store(acc, Sum, WS, AS, Out, bx, warp, lane, count, expert, offset):
        for nt in T.unroll(2):
            for stripe in T.unroll(bm // 8):
                for r in T.unroll(2):
                    col = bx * bn + warp * 32 + nt * 16 + lane // 4 + r * 8
                    for t in T.unroll(2):
                        row = stripe * 8 + (lane % 4) * 2 + t
                        if row < count and col < n:
                            j = nt * (bm // 2) + stripe * 4 + r * 2 + t
                            value = acc[j] + coefficient_sum * Sum[offset + row]
                            Out[offset + row, col] = rescale(
                                value, WS[expert, col], AS[offset + row]
                            )

    return store


def store_rows(bm, bn, n):
    @T.macro
    def store(lo, hi, Sum, WS, AS, Out, bx, warp, lane, count, expert, offset, input_offset):
        for nt in T.unroll(2):
            for stripe in T.unroll(bm // 8):
                for r in T.unroll(2):
                    col = bx * bn + warp * 32 + nt * 16 + lane // 4 + r * 8
                    for t in T.unroll(2):
                        row = stripe * 8 + (lane % 4) * 2 + t
                        if row < count and col < n:
                            j = nt * (bm // 2) + stripe * 4 + r * 2 + t
                            value = 2 * (lo[j] + 16 * hi[j]) + Sum[input_offset + row]
                            Out[offset + row, col] = rescale(
                                value, WS[expert, col], AS[input_offset + row]
                            )

    return store
