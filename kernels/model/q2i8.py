"""Q2 indices -> exact local INT8 codebooks -> SM87 INT8 Tensor Cores.

Weights remain packed. P[E,K/G,N,G/4] U8 and Book[E,K/G,N] U32 are
group-major; each Book contains four little-endian signed bytes. WS[E,N]
is FP16. Activations A[assignments,K] I8 use one AS per row. Accumulation
stays INT32 through the complete K dimension, with one FP32 epilogue.
No global W8 workspace or floating inner-K scales are used.
"""
import tilelang.language as T

from tools.operators.common import orin_jit


SOURCE = r'''
__device__ __forceinline__ unsigned int q2i8_expand(unsigned int packed, unsigned int table) {
    unsigned int select = (packed & 3u) | ((packed & 12u) << 2) |
                          ((packed & 48u) << 4) | ((packed & 192u) << 6);
    return __byte_perm(table, table, select);
}
'''


def _dimensions(rows, experts, n, k, group):
    if any(type(x) is not int or x <= 0 for x in (rows,experts,n,k)):
        raise ValueError('Positive Q2I8 dimensions required')
    if group not in (64,128) or k % group:
        raise ValueError('Q2I8 K must be a multiple of group64 or group128')
    # Signed bytes include -128; bound every partial sum without cancellation.
    if k*128*128 > 2**31-1:
        raise ValueError('Q2I8 INT32 accumulation could overflow')


@orin_jit
def _compile(rows, experts, N, K, group, grouped, tiles, direct_m, block_m, block_n):
    _dimensions(rows,experts,N,K,group)
    if block_m not in (16,32,64) or block_n not in (64,128):
        raise ValueError('Unsupported Q2I8 tile')
    if grouped and (block_m != 16 or type(tiles) is not int or tiles <= 0):
        raise ValueError('Grouped Q2I8 requires positive tile capacity and BM16')
    BM,BN,G = block_m,block_n,group

    @T.macro
    def project(A,P,Book,WS,AS,C,expert,offset,count,bx):
        T.import_source(SOURCE)
        a = T.alloc_shared((BM,G),T.int8)
        b = T.alloc_shared((BN,G),T.int8)
        packed = T.alloc_shared((BN,G//4),T.uint8)
        palette = T.alloc_shared((BN,),T.uint32)
        acc = T.alloc_fragment((BM,BN),T.int32)
        T.clear(acc)
        for kg in T.Pipelined(K//G,num_stages=2):
            for m,k in T.Parallel(BM,G):
                a[m,k] = 0
                if m < count:
                    a[m,k] = A[offset+m,kg*G+k]
            T.copy(P[expert,kg,bx*BN,0],packed)
            T.copy(Book[expert,kg,bx*BN],palette)
            for n,k in T.Parallel(BN,G):
                word = T.call_pure_extern('uint32','q2i8_expand',T.cast(packed[n,k//4],T.uint32),palette[n])
                b[n,k] = T.cast((word >> ((k%4)*8)) & 255,T.int8)
            T.gemm(a,b,acc,transpose_B=True)
        for m,n in T.Parallel(BM,BN):
            if m < count and bx*BN+n < N:
                C[offset+m,bx*BN+n] = ((T.cast(acc[m,n],T.float32)*T.cast(WS[expert,bx*BN+n],T.float32))
                                         *T.cast(AS[offset+m],T.float32))

    @T.prim_func
    def direct(A:T.Tensor((rows,K),T.int8),
               P:T.Tensor((experts,K//G,N,G//4),T.uint8),
               Book:T.Tensor((experts,K//G,N),T.uint32),
               WS:T.Tensor((experts,N),T.float16),
               AS:T.Tensor((rows,),T.float16),
               C:T.Tensor((rows,N),T.float16)):
        with T.Kernel(T.ceildiv(direct_m,BM),T.ceildiv(N,BN),experts,threads=128) as (by,bx,e):
            project(A,P,Book,WS,AS,C,e,e*direct_m+by*BM,T.min(BM,direct_m-by*BM),bx)

    @T.prim_func
    def routed(A:T.Tensor((rows,K),T.int8),
               P:T.Tensor((experts,K//G,N,G//4),T.uint8),
               Book:T.Tensor((experts,K//G,N),T.uint32),
               WS:T.Tensor((experts,N),T.float16),
               AS:T.Tensor((rows,),T.float16),
               Counts:T.Tensor((experts,),T.int32),
               RowOffsets:T.Tensor((experts,),T.int32),
               TileExpert:T.Tensor((tiles,),T.int32),
               TileRow:T.Tensor((tiles,),T.int32),
               TileCount:T.Tensor((1,),T.int32),
               C:T.Tensor((rows,N),T.float16)):
        with T.Kernel(tiles,T.ceildiv(N,BN),threads=128) as (tile,bx):
            meta = T.alloc_local((4,),T.int32)
            if tile < TileCount[0]:
                meta[0] = TileExpert[tile]
                meta[1] = TileRow[tile]
                meta[2] = RowOffsets[meta[0]]+meta[1]
                meta[3] = T.min(BM,Counts[meta[0]]-meta[1])
                project(A,P,Book,WS,AS,C,meta[0],meta[2],meta[3],bx)
    return routed if grouped else direct


def q2i8(E,M,N,K,*,group_size=128,block_m=16,block_n=64):
    """Pre-dispatched ABI (A,P,Book,WS,AS,C), flattened expert rows.

    Inputs are disjoint contiguous device buffers. All scales finite; no
    allocation/quantization/state changes occur inside this kernel.
    """
    return _compile(E*M,E,N,K,group_size,False,1,M,block_m,block_n)


def q2i8_grouped(assignments,E,tiles,N,K,*,group_size=128,block_n=64):
    """ABI (A,P,Book,WS,AS,Counts,RowOffsets,TileExpert,TileRow,TileCount,C).

    Routing inputs follow moe.expert_* contracts. Launch reads only live tile
    metadata and skips inactive experts; callers own bounds validation.
    """
    return _compile(assignments,E,N,K,group_size,True,tiles,1,16,block_n)
