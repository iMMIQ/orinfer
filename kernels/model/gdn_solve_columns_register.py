"""Independent inverse columns in registers: immutable shared L, no pivot sync.

Each lane owns a complete 64-element column. Pivot row data come from that
lane's registers, so two warps need no cross-warp pivot publication. Elimination
order stays FP32 and matches op13. Compile-time unrolling avoids dynamic
register-array indices; inspect the generated code/occupancy as well as timing.
"""
import tilelang.language as T
from tools.operators.common import orin_jit


@orin_jit
def gdn_chunk_solve_columns_register():
    batch,chunks=T.dynamic('batch'),T.dynamic('chunks')
    @T.prim_func
    def kernel(L:T.Tensor((batch,48,chunks,64,64),T.float32),
               A:T.Tensor((batch,48,chunks,64,64),T.float32)):
        with T.Kernel(batch*48*chunks,threads=64) as matrix:
            ls=T.alloc_shared((64,64),T.float32)
            reg=T.alloc_fragment((64,64),T.float32)
            pivot_row=T.alloc_fragment((64,),T.float32)
            T.annotate_layout({pivot_row:T.Fragment((64,),
                forward_thread_fn=lambda j:j,forward_index_fn=lambda j:0)})
            T.annotate_layout({reg:T.Fragment((64,64),
                forward_thread_fn=lambda i,j:j,
                forward_index_fn=lambda i,j:i)})
            b,h,c=matrix//(48*chunks),(matrix//chunks)%48,matrix%chunks
            for i,j in T.Parallel(64,64):
                ls[i,j]=T.if_then_else(j<i,L[b,h,c,i,j],0.0)
                reg[i,j]=T.if_then_else(i==j,1.0,-ls[i,j])
            T.sync_threads()
            for pivot in T.unroll(1,63):
                for j in T.Parallel(64):pivot_row[j]=reg[pivot,j]
                for i in T.unroll(64):
                    if i>pivot:
                        for j in T.Parallel(64):
                            if j<pivot:reg[i,j]=reg[i,j]-ls[i,pivot]*pivot_row[j]
            for i,j in T.Parallel(64,64):A[b,h,c,i,j]=reg[i,j]
    return kernel
