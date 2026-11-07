"""Small SM87 instruction helpers shared by exact BF16 decode kernels."""

SOURCE = r"""
#include <tl_templates/cuda/instruction/mma.h>
__device__ __forceinline__ unsigned hc_raw_pair(const bfloat16_t*p){return *reinterpret_cast<const unsigned*>(p);}
__device__ __forceinline__ void hc_mma_bf16(float*d,unsigned a0,unsigned a1,unsigned a2,unsigned a3,unsigned b0,unsigned b1){
 asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]) : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}
"""
