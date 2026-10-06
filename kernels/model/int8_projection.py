"""Row-scaled dense INT8 Tensor Core projections for Flash Next.

Only the final INT32 accumulator is rescaled. Weights remain stored INT8,
including the output head; no persistent FP16 or expanded expert copy exists.
"""
import tilelang
import tilelang.language as T

from tools.operators.common import orin_jit


@orin_jit
def int8_projection(M: int, N: int, K: int, output_dtype: str = 'float16', block_m: int = 16,
                    block_n: int = 64, block_k: int = 64, group_m: int = 0):
    if any(type(v) is not int or v <= 0 for v in (M,N,K)) or K%64 or K*128*128 > 2**31-1:
        raise ValueError('Invalid INT8 projection geometry')
    if output_dtype not in ('float16','float32'):
        raise ValueError('Invalid INT8 projection output dtype')
    if block_m not in (16,32,64,128):raise ValueError('Invalid INT8 row tile')
    if block_n not in (64,128) or block_k not in (64,128) or K%block_k:
        raise ValueError('Invalid INT8 column/reduction tile')
    if type(group_m) is not int or group_m not in (0,1,2,4,8):
        raise ValueError('Invalid INT8 row grouping')
    mt,nt=(M+block_m-1)//block_m,(N+block_n-1)//block_n
    @T.prim_func
    def main(A:T.Tensor((M,K),T.int8),Weight:T.Tensor((N,K),T.int8),
             WeightScale:T.Tensor((N,),T.float16),TokenScale:T.Tensor((M,),T.float16),
             Output:T.Tensor((M,N),output_dtype)):
        with T.Kernel(mt*nt,threads=128) as pid:
            by=T.alloc_var(T.int32);bx=T.alloc_var(T.int32)
            if group_m:
                first=(pid//(group_m*nt))*group_m
                actual=T.min(mt-first,group_m)
                by=first+(pid%(group_m*nt))%actual
                bx=(pid%(group_m*nt))//actual
            else:
                by=pid%mt;bx=pid//mt
            a = T.alloc_shared((block_m,block_k),T.int8)
            w = T.alloc_shared((block_n,block_k),T.int8)
            acc = T.alloc_fragment((block_m,block_n),T.int32)
            T.clear(acc)
            for kg in T.Pipelined(K//block_k,num_stages=2):
                T.copy(A[by*block_m,kg*block_k],a)
                T.copy(Weight[bx*block_n,kg*block_k],w)
                T.gemm(a,w,acc,transpose_B=True)
            for i,j in T.Parallel(block_m,block_n):
                if by*block_m+i < M and bx*block_n+j < N:
                    Output[by*block_m+i,bx*block_n+j] = ((T.cast(acc[i,j],T.float32)*
                        T.cast(WeightScale[bx*block_n+j],T.float32))*T.cast(TokenScale[by*block_m+i],T.float32))
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
