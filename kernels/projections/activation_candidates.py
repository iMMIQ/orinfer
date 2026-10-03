"""SM87 A8 candidates. W4 stays resident; W8 and activation codes are temporary."""
import tilelang.language as T
from candidates import orin_jit


@orin_jit
def grouped_q8(M:int,K:int,G:int=512):
    assert K%G==0
    @T.prim_func
    def kernel(A:T.Tensor((M,K),T.float16),Q:T.Tensor((M,K),T.int8),S:T.Tensor((M,K//G),T.float16)):
        with T.Kernel(M,K//G,threads=128) as (row,group):
            x=T.alloc_fragment((G,),T.float32)
            absolute=T.alloc_fragment((G,),T.float32)
            maximum=T.alloc_fragment((1,),T.float32)
            scale=T.alloc_fragment((1,),T.float16)
            for j in T.Parallel(G):
                x[j]=A[row,group*G+j]
                absolute[j]=T.abs(x[j])
            T.reduce_max(absolute,maximum,dim=0)
            scale[0]=T.if_then_else(maximum[0]>0,T.max(maximum[0]/127,2**-24),1)
            S[row,group]=scale[0]
            for j in T.Parallel(G):
                Q[row,group*G+j]=T.cast(T.min(127.0,T.max(-127.0,T.round(x[j]/T.cast(scale[0],T.float32)))),T.int8)
    return kernel


@orin_jit
def grouped_gemm(M:int,N:int,K:int,G:int=512,BM:int=64,BN:int=128,BK:int=128):
    assert K%G==0 and G%BK==0 and N%BN==0
    @T.prim_func
    def kernel(A:T.Tensor((M,K),T.int8),B:T.Tensor((N,K),T.int8),AS:T.Tensor((M,K//G),T.float16),BS:T.Tensor((N,),T.float16),C:T.Tensor((M,N),T.float16)):
        with T.Kernel(N//BN,T.ceildiv(M,BM),threads=128) as (bx,by):
            a=T.alloc_shared((BM,BK),T.int8)
            b=T.alloc_shared((BN,BK),T.int8)
            partial=T.alloc_fragment((BM,BN),T.int32)
            accum=T.alloc_fragment((BM,BN),T.float32)
            T.clear(accum)
            for group in T.serial(K//G):
                T.clear(partial)
                for tile in T.Pipelined(G//BK,num_stages=2):
                    T.copy(A[by*BM,group*G+tile*BK],a)
                    T.copy(B[bx*BN,group*G+tile*BK],b)
                    T.gemm(a,b,partial,transpose_B=True)
                for i,j in T.Parallel(BM,BN):
                    if by*BM+i<M:
                        accum[i,j]+=T.cast(partial[i,j],T.float32)*T.cast(AS[by*BM+i,group],T.float32)
            for i,j in T.Parallel(BM,BN):
                if by*BM+i<M:
                    C[by*BM+i,bx*BN+j]=accum[i,j]*T.cast(BS[bx*BN+j],T.float32)
    return kernel


@orin_jit
def masked_q8(M:int,K:int):
    @T.prim_func
    def kernel(A:T.Tensor((M,K),T.float16),Mask:T.Tensor((K,),T.uint8),Q:T.Tensor((M,K),T.int8),S:T.Tensor((M,),T.float16)):
        with T.Kernel(M,threads=256) as row:
            x=T.alloc_fragment((K,),T.float32)
            absolute=T.alloc_fragment((K,),T.float32)
            maximum=T.alloc_fragment((1,),T.float32)
            scale=T.alloc_fragment((1,),T.float16)
            for j in T.Parallel(K):
                x[j]=T.if_then_else(Mask[j]==0,T.cast(A[row,j],T.float32),0)
                absolute[j]=T.abs(x[j])
            T.reduce_max(absolute,maximum,dim=0)
            scale[0]=T.if_then_else(maximum[0]>0,T.max(maximum[0]/127,2**-24),1)
            S[row]=scale[0]
            for j in T.Parallel(K):
                Q[row,j]=T.cast(T.min(127.0,T.max(-127.0,T.round(x[j]/T.cast(scale[0],T.float32)))),T.int8)
    return kernel


@orin_jit
def outlier_correct(M:int,N:int,K:int,O:int=32,BM:int=64,BN:int=128):
    assert O%16==0 and N%BN==0
    @T.prim_func
    def kernel(A:T.Tensor((M,K),T.float16),Indices:T.Tensor((O,),T.int32),Side:T.Tensor((N,O),T.float16),Base:T.Tensor((M,N),T.float16),C:T.Tensor((M,N),T.float16)):
        with T.Kernel(N//BN,T.ceildiv(M,BM),threads=128) as (bx,by):
            a=T.alloc_shared((BM,O),T.float16)
            b=T.alloc_shared((BN,O),T.float16)
            accum=T.alloc_fragment((BM,BN),T.float32)
            for i,j in T.Parallel(BM,O):
                a[i,j]=T.if_then_else(by*BM+i<M,A[by*BM+i,Indices[j]],0)
            T.copy(Side[bx*BN,0],b)
            T.clear(accum)
            T.gemm(a,b,accum,transpose_B=True)
            for i,j in T.Parallel(BM,BN):
                if by*BM+i<M:
                    C[by*BM+i,bx*BN+j]=accum[i,j]+T.cast(Base[by*BM+i,bx*BN+j],T.float32)
    return kernel
