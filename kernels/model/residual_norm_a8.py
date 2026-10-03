"""FP32 residual + zero-centered norm with the original FP16 A8 boundary.

Default preserves op02 arithmetic/reduction and op30 scale/RNE division.
Explicit reciprocal candidate permits code differences at FP32 half ties.
Y is retained by default for the dense GDN a/b projection; write_normalized
may be disabled only when its output has no consumer. The local FP16 norm
boundary is still retained for A8. No input/output may alias.
"""
import tilelang.language as T
from tools.operators.common import orin_jit
from kernels.operators.op30_activation_quantization import quantized_code


@orin_jit
def residual_norm_a8(hidden=5120,epsilon=1e-6,threads=256,quant_mode='strict',write_normalized=True):
    assert hidden>0 and epsilon>0 and threads in (128,256,512)
    assert quant_mode in ('strict','reciprocal')
    rows=T.dynamic('rows');columns=T.ceildiv(hidden,threads)*threads
    @T.prim_func
    def kernel(X:T.Tensor((rows,hidden),T.float16),R:T.Tensor((rows,hidden),T.float32),
               W:T.Tensor((hidden,),T.float16),Y:T.Tensor((rows,hidden),T.float16),
               RO:T.Tensor((rows,hidden),T.float32),Q:T.Tensor((rows,hidden),T.int8),
               S:T.Tensor((rows,),T.float16)):
        with T.Kernel(rows,threads=threads) as row:
            value=T.alloc_fragment((columns,),T.float32)
            square=T.alloc_fragment((columns,),T.float32)
            total=T.alloc_fragment((1,),T.float32)
            normalized=T.alloc_fragment((columns,),T.float16)
            absolute=T.alloc_fragment((columns,),T.float32)
            maximum=T.alloc_fragment((1,),T.float32)
            scale=T.alloc_fragment((1,),T.float16)
            reciprocal=T.alloc_fragment((1,),T.float32)
            for j in T.Parallel(columns):
                value[j]=0.0
                if j<hidden:
                    value[j]=T.cast(X[row,j],T.float32)+R[row,j]
                    RO[row,j]=value[j]
                square[j]=value[j]*value[j]
            T.reduce_sum(square,total,dim=0)
            for j in T.Parallel(columns):
                normalized[j]=0.0
                if j<hidden:
                    normalized[j]=(value[j]*T.rsqrt(total[0]/hidden+epsilon))*(T.cast(W[j],T.float32)+1.0)
                    if write_normalized:
                        Y[row,j]=normalized[j]
                absolute[j]=T.abs(T.cast(normalized[j],T.float32))
            T.reduce_max(absolute,maximum,dim=0)
            scale[0]=T.if_then_else(maximum[0]>0,
                T.max(T.call_extern('float32','__fdiv_rn',maximum[0],127.0),2**-24),1.0)
            S[row]=scale[0]
            if quant_mode=='reciprocal':
                reciprocal[0]=T.call_extern('float32','__fdiv_rn',1.0,T.cast(scale[0],T.float32))
            for j in T.Parallel(columns):
                if j<hidden:
                    if quant_mode=='strict':
                        Q[row,j]=quantized_code(T.cast(normalized[j],T.float32),scale[0])
                    else:
                        # Candidate permits one-code differences at FP32 ties.
                        # FP16 norm/scale boundaries and FP32 residual stay intact.
                        ratio=T.cast(normalized[j],T.float32)*reciprocal[0]
                        Q[row,j]=T.cast(T.max(-127.0,T.min(127.0,T.round(ratio))),T.int8)
    return kernel
