"""Four-entry row-W8 codebook helper; metadata bounds prevent byte carry."""
import tilelang.language as T
from kernels.operators.op03_ffn_gate_up import _orin_jit

LUT4_SOURCE=r'''
#include <cuda_fp16.h>
#include <tl_templates/cuda/instruction/mma.h>
__device__ __forceinline__ unsigned int orin_w4_i8_lut4(
    unsigned short x, unsigned int table, unsigned char step) {
    unsigned int base = __byte_perm(table, table, x & 0x3333u);
    unsigned int high = (x >> 2) & 0x3333u;
    unsigned int doubled = __byte_perm(high, high, 0x1100u);
    unsigned int coarse = (doubled & 0x00030003u) |
                          ((doubled & 0x30003000u) >> 4);
    return (base + coarse * unsigned(step)) ^ 0x80808080u;
}
'''


@_orin_jit
def check_lut4_quartets():
    count=T.dynamic('count')
    @T.prim_func
    def lut4_check(X:T.Tensor((count,),T.uint16),Table:T.Tensor((count,),T.uint32),
                   Step:T.Tensor((count,),T.uint8),O:T.Tensor((count,),T.uint32)):
        with T.Kernel(T.ceildiv(count,256),threads=256) as bx:
            T.import_source(LUT4_SOURCE)
            for j in T.Parallel(256):
                i=bx*256+j
                if i<count:
                    O[i]=T.call_pure_extern('uint32','orin_w4_i8_lut4',X[i],Table[i],Step[i])
    return lut4_check
