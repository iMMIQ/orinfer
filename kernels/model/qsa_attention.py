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
def sparse_attention(m: int,capacity: int,splits: int = 8,packed: bool = False,
                     block_size: int = 32, threads: int = 128, single_buffer: bool = False):
    geometry(m,capacity)
    if splits not in (1,2,4,8):raise ValueError('Invalid QSA splits')
    if block_size not in (32,64) or threads not in (128,256) or block_size==32 and threads==256:
        raise ValueError('Invalid sparse attention tile/warp geometry')
    if type(single_buffer) is not bool or single_buffer and not packed:
        raise ValueError('Single KV buffer requires packed attention')
    block=block_size
    kv_width=64 if packed else 256
    kv_dtype=T.uint32 if packed else T.int8
    @T.prim_func
    def main(Query:T.Tensor((m,24,256),T.float16),K:T.Tensor((capacity,2,kv_width),kv_dtype),
             V:T.Tensor((capacity,2,kv_width),kv_dtype),KS:T.Tensor((capacity,2,4),T.float16),
             VS:T.Tensor((capacity,2,4),T.float16),Selected:T.Tensor((m,2051),T.int32),
             Position:T.Tensor((1,),T.int32),Max:T.Tensor((m,24,splits),T.float32),
             Den:T.Tensor((m,24,splits),T.float32),Out:T.Tensor((m,24,splits,256),T.float32)):
        with T.Kernel(m,2,splits,threads=threads) as (row,kh,split):
            if packed:T.import_source(PACKED_SOURCE)
            selected_tile=T.alloc_shared((block,),T.int32)
            key_scales=T.alloc_shared((block,4),T.float16)
            value_scales=T.alloc_shared((block,4),T.float16)
            q=T.alloc_shared((16,256),T.float16)
            k=T.alloc_shared((block,256),T.float16)
            if packed:
                raw_k=T.alloc_shared((block,64),T.uint32)
                raw_v=raw_k if single_buffer else T.alloc_shared((block,64),T.uint32)
            else:
                v=T.alloc_shared((block,256),T.float16)
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
                # Retire all readers before reusing metadata for another tile.
                # Prefetch once instead of serializing each channel on the
                # same global Selected/scale loads. The masked tail reads a
                # valid allocated slot but stores -1, never an out-of-range ID.
                T.sync_threads()
                for j in T.Parallel(block):
                    slot=start+tile*block+j
                    selected_tile[j]=T.if_then_else(slot<end,Selected[row,T.min(slot,2050)],-1)
                T.sync_threads()
                for j,g in T.Parallel(block,4):
                    token=selected_tile[j]
                    key_scales[j,g]=0;value_scales[j,g]=0
                    if token>=0 and token<=Position[0]+row:
                        key_scales[j,g]=KS[token,kh,g];value_scales[j,g]=VS[token,kh,g]
                T.sync_threads()
                # Gather packed INT8 KV in 16-byte asynchronous loads. Keep
                # packed V until softmax finishes so K and V can share one
                # FP16 tensor-core tile without increasing shared memory.
                if packed:
                    for j,segment in T.Parallel(block,16):
                        token=selected_tile[j]
                        valid=token>=0 and token<=Position[0]+row
                        safe=T.max(token,0)
                        T.ptx_cp_async(
                            T.tvm_access_ptr(T.type_annotation(T.uint32),raw_k.data,j*64+segment*4,4,2),
                            T.tvm_access_ptr(T.type_annotation(T.uint32),K.data,(safe*2+kh)*64+segment*4,4,1),
                            4,predicate=valid)
                        if not single_buffer:
                            T.ptx_cp_async(
                                T.tvm_access_ptr(T.type_annotation(T.uint32),raw_v.data,j*64+segment*4,4,2),
                                T.tvm_access_ptr(T.type_annotation(T.uint32),V.data,(safe*2+kh)*64+segment*4,4,1),
                                4,predicate=valid)
                    T.ptx_commit_group();T.ptx_wait_group(0);T.sync_threads()
                    for j,pair in T.Parallel(block,128,coalesced_width=T.int32(1)):
                        kw=T.call_pure_extern('uint32','qsa_dequant_pair',raw_k[j,pair//2] >> ((pair%2)*16),
                            T.reinterpret(T.uint16,key_scales[j,pair//32]))
                        k[j,pair*2]=T.reinterpret(T.float16,T.cast(kw&65535,T.uint16))
                        k[j,pair*2+1]=T.reinterpret(T.float16,T.cast(kw>>16,T.uint16))
                else:
                    for j,d in T.Parallel(block,256,coalesced_width=T.int32(1)):
                        slot=start+tile*block+j
                        k[j,d]=0;v[j,d]=0
                        if slot<end:
                            token=selected_tile[j]
                            if token>=0 and token<=Position[0]+row:
                                k[j,d]=T.cast(K[token,kh,d],T.float32)*T.cast(key_scales[j,d//64],T.float32)
                                v[j,d]=T.cast(V[token,kh,d],T.float32)*T.cast(value_scales[j,d//64],T.float32)
                if single_buffer:
                    # All K words have been decoded into the separate FP16
                    # tile. Retire their readers, then overlap packed V loads
                    # into the same raw storage with QK and softmax.
                    T.sync_threads()
                    for j,segment in T.Parallel(block,16):
                        token=selected_tile[j]
                        valid=token>=0 and token<=Position[0]+row
                        safe=T.max(token,0)
                        T.ptx_cp_async(
                            T.tvm_access_ptr(T.type_annotation(T.uint32),raw_v.data,j*64+segment*4,4,2),
                            T.tvm_access_ptr(T.type_annotation(T.uint32),V.data,(safe*2+kh)*64+segment*4,4,1),
                            4,predicate=valid)
                    T.ptx_commit_group()
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
                if packed:
                    if single_buffer:T.ptx_wait_group(0)
                    T.sync_threads()
                    for j,pair in T.Parallel(block,128,coalesced_width=T.int32(1)):
                        vw=T.call_pure_extern('uint32','qsa_dequant_pair',raw_v[j,pair//2] >> ((pair%2)*16),
                            T.reinterpret(T.uint16,value_scales[j,pair//32]))
                        k[j,pair*2]=T.reinterpret(T.float16,T.cast(vw&65535,T.uint16))
                        k[j,pair*2+1]=T.reinterpret(T.float16,T.cast(vw>>16,T.uint16))
                    T.gemm(p,k,acc)
                else:
                    T.gemm(p,v,acc)
            for i,d in T.Parallel(16,256):
                if i<12:Out[row,kh*12+i,split,d]=acc[i,d]
            for i in T.Parallel(16):
                if i<12:
                    Max[row,kh*12+i,split]=maximum[i];Den[row,kh*12+i,split]=denom[i]
    return main
