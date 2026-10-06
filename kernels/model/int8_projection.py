"""Row-scaled dense INT8 Tensor Core projections for Flash Next.

Only the final INT32 accumulator is rescaled. Weights remain stored INT8,
including the output head; no persistent FP16 or expanded expert copy exists.
"""
import tilelang
import tilelang.language as T

from tools.operators.common import orin_jit


@orin_jit
def int8_projection(M: int, N: int, K: int, output_dtype: str = 'float16', block_m: int = 16):
    if any(type(v) is not int or v <= 0 for v in (M,N,K)) or K%64 or K*128*128 > 2**31-1:
        raise ValueError('Invalid INT8 projection geometry')
    if output_dtype not in ('float16','float32'):
        raise ValueError('Invalid INT8 projection output dtype')
    if block_m not in (16,32,64):raise ValueError('Invalid INT8 row tile')
    @T.prim_func
    def main(A:T.Tensor((M,K),T.int8),Weight:T.Tensor((N,K),T.int8),
             WeightScale:T.Tensor((N,),T.float16),TokenScale:T.Tensor((M,),T.float16),
             Output:T.Tensor((M,N),output_dtype)):
        with T.Kernel(T.ceildiv(M,block_m),T.ceildiv(N,64),threads=128) as (by,bx):
            a = T.alloc_shared((block_m,64),T.int8)
            w = T.alloc_shared((64,64),T.int8)
            acc = T.alloc_fragment((block_m,64),T.int32)
            T.clear(acc)
            for kg in T.Pipelined(K//64,num_stages=2):
                T.copy(A[by*block_m,kg*64],a)
                T.copy(Weight[bx*64,kg*64],w)
                T.gemm(a,w,acc,transpose_B=True)
            for i,j in T.Parallel(block_m,64):
                if by*block_m+i < M and bx*64+j < N:
                    Output[by*block_m+i,bx*64+j] = ((T.cast(acc[i,j],T.float32)*
                        T.cast(WeightScale[bx*64+j],T.float32))*T.cast(TokenScale[by*block_m+i],T.float32))
    return main


@orin_jit
def int8_gemv(N: int, K: int, output_dtype: str = 'float16', block_n: int = 16):
    """Single-token signed DP4A; packed views alias the original INT8 buffers."""
    if any(type(v) is not int or v <= 0 for v in (N,K)) or K%128 or K*128*128 > 2**31-1:
        raise ValueError('Invalid INT8 GEMV geometry')
    if output_dtype not in ('float16','float32') or block_n not in (4,8,16,32):
        raise ValueError('Invalid INT8 GEMV output/tile')
    @T.prim_func
    def main(A:T.Tensor((1,K//4),T.int32), Weight:T.Tensor((N,K//4),T.int32),
             WeightScale:T.Tensor((N,),T.float16), TokenScale:T.Tensor((1,),T.float16),
             Output:T.Tensor((1,N),output_dtype)):
        with T.Kernel(T.ceildiv(N,block_n),threads=128) as bx:
            T.import_source('\n__device__ __forceinline__ int orinfer_dp4a(int a, int b, int c) { return __dp4a(a,b,c); }\n')
            acc=T.alloc_fragment((block_n,32),T.int32)
            total=T.alloc_fragment((block_n,),T.int32)
            T.annotate_layout({acc:tilelang.Fragment((block_n,32),
                forward_thread_fn=lambda n,k:(n%4)*32+k,
                forward_index_fn=lambda n,k:n//4)})
            T.clear(acc)
            for kg in T.serial(K//128):
                for n,k in T.Parallel(block_n,32):
                    if bx*block_n+n<N:
                        acc[n,k]=T.call_pure_extern('int32','orinfer_dp4a',
                            A[0,kg*32+k],Weight[bx*block_n+n,kg*32+k],acc[n,k])
            T.reduce_sum(acc,total,dim=1)
            for n in T.Parallel(block_n):
                if bx*block_n+n<N:
                    Output[0,bx*block_n+n]=((T.cast(total[n],T.float32)*
                        T.cast(WeightScale[bx*block_n+n],T.float32))*T.cast(TokenScale[0],T.float32))
    return main
