"""Sparse QSA with shared INT8 KV tiles for twelve GQA heads on SM87.

FP16 tensor-core QK/PV with FP32 accumulators and softmax. Dequantization is
local to each selected tile; no expanded KV cache persists between launches.
"""
import tilelang.language as T
from tools.operators.common import orin_jit
from kernels.model.qsa import geometry


PACKED_SOURCE = r'''
#include <cuda_fp16.h>
union QsaHalf2Bits { unsigned int bits; __half2 value; };
__device__ __forceinline__ unsigned int qsa_dequant_pair(unsigned int code, unsigned short scale_bits) {
    QsaHalf2Bits values{0}, bias{0}, scale{0}, result{0};
    values.bits = (((code & 0xffu) | ((code & 0xff00u) << 8)) ^ 0x00800080u) | 0x64006400u;
    bias.bits = 0x64806480u;
    scale.bits = static_cast<unsigned int>(scale_bits) | (static_cast<unsigned int>(scale_bits) << 16);
    result.value = __hmul2(__hsub2(values.value, bias.value), scale.value);
    return result.bits;
}
'''


@orin_jit
def sparse_attention(m: int,capacity: int,splits: int = 8,packed: bool = False):
    geometry(m,capacity)
    if splits not in (1,2,4,8):raise ValueError('Invalid QSA splits')
    block=32
    kv_width=64 if packed else 256
    kv_dtype=T.uint32 if packed else T.int8
    @T.prim_func
    def main(Query:T.Tensor((m,24,256),T.float16),K:T.Tensor((capacity,2,kv_width),kv_dtype),
             V:T.Tensor((capacity,2,kv_width),kv_dtype),KS:T.Tensor((capacity,2,4),T.float16),
             VS:T.Tensor((capacity,2,4),T.float16),Selected:T.Tensor((m,2051),T.int32),
             Position:T.Tensor((1,),T.int32),Max:T.Tensor((m,24,splits),T.float32),
             Den:T.Tensor((m,24,splits),T.float32),Out:T.Tensor((m,24,splits,256),T.float32)):
        with T.Kernel(m,2,splits,threads=128) as (row,kh,split):
            if packed:T.import_source(PACKED_SOURCE)
            q=T.alloc_shared((16,256),T.float16)
            k=T.alloc_shared((block,256),T.float16);v=T.alloc_shared((block,256),T.float16)
            p=T.alloc_shared((16,block),T.float16)
            scores=T.alloc_fragment((16,block),T.float32)
            acc=T.alloc_fragment((16,256),T.float32)
            maximum=T.alloc_fragment((16,),T.float32);denom=T.alloc_fragment((16,),T.float32)
            previous=T.alloc_fragment((16,),T.float32);correction=T.alloc_fragment((16,),T.float32)
            rowsum=T.alloc_fragment((16,),T.float32)
            T.clear(acc);T.clear(denom);T.fill(maximum,-3.402823466e38)
            for i,d in T.Parallel(16,256):
                q[i,d]=0
                if i<12:q[i,d]=Query[row,kh*12+i,d]
            length=T.min((Position[0]+row+1)//4,512)*4+(Position[0]+row+1)%4
            count=T.ceildiv(length,splits);start=split*count;end=T.min(start+count,length)
            for tile in T.serial(T.ceildiv(T.max(0,end-start),block)):
                # Scalar signed loads avoid the aarch64 vector-cast lowering
                # through plain char (which is unsigned on this platform).
                if packed:
                    for j,pair in T.Parallel(block,128,coalesced_width=T.int32(1)):
                        slot=start+tile*block+j
                        k[j,pair*2]=0;k[j,pair*2+1]=0
                        v[j,pair*2]=0;v[j,pair*2+1]=0
                        if slot<end:
                            token=Selected[row,slot]
                            if token>=0 and token<=Position[0]+row:
                                kw=T.call_pure_extern('uint32','qsa_dequant_pair',K[token,kh,pair//2] >> ((pair%2)*16),
                                    T.reinterpret(T.uint16,KS[token,kh,pair//32]))
                                vw=T.call_pure_extern('uint32','qsa_dequant_pair',V[token,kh,pair//2] >> ((pair%2)*16),
                                    T.reinterpret(T.uint16,VS[token,kh,pair//32]))
                                k[j,pair*2]=T.reinterpret(T.float16,T.cast(kw&65535,T.uint16))
                                k[j,pair*2+1]=T.reinterpret(T.float16,T.cast(kw>>16,T.uint16))
                                v[j,pair*2]=T.reinterpret(T.float16,T.cast(vw&65535,T.uint16))
                                v[j,pair*2+1]=T.reinterpret(T.float16,T.cast(vw>>16,T.uint16))
                else:
                    for j,d in T.Parallel(block,256,coalesced_width=T.int32(1)):
                        slot=start+tile*block+j
                        k[j,d]=0;v[j,d]=0
                        if slot<end:
                            token=Selected[row,slot]
                            if token>=0 and token<=Position[0]+row:
                                k[j,d]=T.cast(K[token,kh,d],T.float32)*T.cast(KS[token,kh,d//64],T.float32)
                                v[j,d]=T.cast(V[token,kh,d],T.float32)*T.cast(VS[token,kh,d//64],T.float32)
                T.gemm(q,k,scores,transpose_B=True,clear_accum=True)
                for i in T.Parallel(16):previous[i]=maximum[i]
                for i,j in T.Parallel(16,block):
                    if i<12 and start+tile*block+j<end:scores[i,j]*=.0625
                    else:scores[i,j]=-3.402823466e38
                T.reduce_max(scores,maximum,dim=1,clear=False)
                for i,j in T.Parallel(16,block):
                    if i<12 and start+tile*block+j<end:scores[i,j]=T.exp(scores[i,j]-maximum[i])
                    else:scores[i,j]=0
                    p[i,j]=scores[i,j]
                T.reduce_sum(scores,rowsum,dim=1)
                for i in T.Parallel(16):
                    correction[i]=T.exp(previous[i]-maximum[i])
                    denom[i]=denom[i]*correction[i]+rowsum[i]
                for i,d in T.Parallel(16,256):acc[i,d]*=correction[i]
                T.gemm(p,v,acc)
            for i,d in T.Parallel(16,256):
                if i<12:Out[row,kh*12+i,split,d]=acc[i,d]
            for i in T.Parallel(16):
                if i<12:
                    Max[row,kh*12+i,split]=maximum[i];Den[row,kh*12+i,split]=denom[i]
    return main
