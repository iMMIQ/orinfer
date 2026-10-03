"""Two independent 32x32 diagonal inverses + compensated cross-block products.

inv([[A,0],[B,D]]) = [[inv(A),0],[-inv(D) B inv(A),inv(D)]].
Diagonal solves use the FP32 elimination recurrence. Cross products use three
FP16 products, FP32 accumulation. Explicit precision candidate, no state alias.
Only strict-lower L is read; implicit diagonal one and upper garbage ignored.
"""
import tilelang.language as T
from tools.operators.common import orin_jit


@T.macro
def product(ahi,alo,bhi,blo,result):
    T.clear(result)
    T.gemm(alo,bhi,result)
    T.gemm(ahi,blo,result)
    T.gemm(ahi,bhi,result)


@orin_jit
def gdn_chunk_solve_blocked():
    batch,chunks=T.dynamic('batch'),T.dynamic('chunks')
    @T.prim_func
    def kernel(L:T.Tensor((batch,48,chunks,64,64),T.float32),
               A:T.Tensor((batch,48,chunks,64,64),T.float32)):
        with T.Kernel(batch*48*chunks,threads=128) as matrix:
            ls=T.alloc_shared((64,64),T.float32)
            inv=T.alloc_shared((64,64),T.float32)
            reg=T.alloc_fragment((64,64),T.float32)
            T.annotate_layout({reg:T.Fragment((64,64),
                forward_thread_fn=lambda i,j:(i*64+j)%128,
                forward_index_fn=lambda i,j:(i*64+j)//128)})
            ah=T.alloc_shared((32,32),T.float16);al=T.alloc_shared((32,32),T.float16)
            bh=T.alloc_shared((32,32),T.float16);bl=T.alloc_shared((32,32),T.float16)
            middle=T.alloc_fragment((32,32),T.float32)
            result=T.alloc_fragment((32,32),T.float32)
            b,h,c=matrix//(48*chunks),(matrix//chunks)%48,matrix%chunks
            for i,j in T.Parallel(64,64):
                ls[i,j]=0.0
                if j<i:ls[i,j]=L[b,h,c,i,j]
                inv[i,j]=T.if_then_else(i==j,1.0,0.0)
                if j<i and i//32==j//32:inv[i,j]=-ls[i,j]
                reg[i,j]=inv[i,j]
            T.sync_threads()
            for pivot in T.serial(1,31):
                for i,j in T.Parallel(64,64):
                    if i//32==j//32 and i%32>pivot and j%32<pivot:
                        reg[i,j]=reg[i,j]-ls[i,(i//32)*32+pivot]*inv[(i//32)*32+pivot,j]
                    if i%32==pivot+1:inv[i,j]=reg[i,j]
                T.sync_threads()
            T.copy(reg,inv)
            T.sync_threads()
            for i,j in T.Parallel(32,32):
                va=inv[32+i,32+j];vb=ls[32+i,j]
                ah[i,j]=va;al[i,j]=va-T.cast(ah[i,j],T.float32)
                bh[i,j]=vb;bl[i,j]=vb-T.cast(bh[i,j],T.float32)
            product(ah,al,bh,bl,middle)
            for i,j in T.Parallel(32,32):
                va=middle[i,j];vb=inv[i,j]
                ah[i,j]=va;al[i,j]=va-T.cast(ah[i,j],T.float32)
                bh[i,j]=vb;bl[i,j]=vb-T.cast(bh[i,j],T.float32)
            product(ah,al,bh,bl,result)
            for i,j in T.Parallel(64,64):
                if i<32 or j>=32:A[b,h,c,i,j]=reg[i,j]
            for i,j in T.Parallel(32,32):A[b,h,c,32+i,j]=-result[i,j]
    return kernel
