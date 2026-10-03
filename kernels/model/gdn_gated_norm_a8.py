"""Per-token GDN gated RMSNorm and A8, preserving the FP16 mixer boundary.

One CTA owns all 48 heads of a token. Ordinary W[128] is shared across heads.
No MixerIn global write; only row A8 codes and FP16 scale are materialized.
"""
import tilelang.language as T
from tools.operators.common import orin_jit
from kernels.operators.op17_gdn_gated_norm import gated_norm_epilogue
from kernels.operators.op30_activation_quantization import quantized_code


@orin_jit
def gdn_gated_norm_a8(threads=256):
    assert threads in (128,256,512)
    rows=T.dynamic('rows')

    @T.prim_func
    def gated_norm_a8(X:T.Tensor((rows,48,128),T.float16),
                      Z:T.Tensor((rows,48,128),T.float16),
                      W:T.Tensor((128,),T.float16),
                      Q:T.Tensor((rows,6144),T.int8),
                      S:T.Tensor((rows,),T.float16)):
        with T.Kernel(rows,threads=threads) as row:
            values=T.alloc_fragment((64,128),T.float32)
            square=T.alloc_fragment((64,128),T.float32)
            sums=T.alloc_fragment((64,),T.float32)
            normalized=T.alloc_fragment((64,128),T.float16)
            absolute=T.alloc_fragment((64,128),T.float32)
            head_maximum=T.alloc_fragment((64,),T.float32)
            maximum=T.alloc_fragment((1,),T.float32)
            scale=T.alloc_fragment((1,),T.float16)
            for h,j in T.Parallel(64,128):
                values[h,j]=0.0
                if h<48:
                    values[h,j]=T.cast(X[row,h,j],T.float32)
                square[h,j]=values[h,j]*values[h,j]
            T.reduce_sum(square,sums,dim=1)
            for h,j in T.Parallel(64,128):
                normalized[h,j]=0.0
                if h<48:
                    normalized[h,j]=gated_norm_epilogue(values[h,j],Z[row,h,j],W[j],
                                                       T.rsqrt(sums[h]/128+1e-6))
                absolute[h,j]=T.abs(T.cast(normalized[h,j],T.float32))
            T.reduce_max(absolute,head_maximum,dim=1)
            T.reduce_max(head_maximum,maximum,dim=0)
            scale[0]=T.if_then_else(maximum[0]>0,
                T.max(T.call_extern('float32','__fdiv_rn',maximum[0],127.0),2**-24),1.0)
            S[row]=scale[0]
            for h,j in T.Parallel(64,128):
                if h<48:
                    Q[row,h*128+j]=quantized_code(T.cast(normalized[h,j],T.float32),scale[0])
    return gated_norm_a8
