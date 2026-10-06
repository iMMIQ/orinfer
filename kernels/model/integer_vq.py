"""Integer VQ4/E8P tiles expanded in shared memory for SM87 INT8 MMA.

Experimental ABI: group-major indices, integer table, output-row scale, A8
activation and token scale. Full K is accumulated in INT32. Tables can be
shared across experts; no persistent expanded W8 or floating inner-K scales.
"""
import tilelang
import tilelang.language as T

from tools.operators.common import orin_jit


SOURCE = r'''
__device__ __forceinline__ unsigned int integer_e8p_quad(unsigned int code, unsigned int basis, unsigned int half) {
    unsigned int parity = __popc(code & 255u) & 1u;
    unsigned int s = ((code & 255u) ^ parity) >> (half * 2u);
    unsigned int flags = (s & 1u) | ((s & 16u) << 4u) | ((s & 2u) << 15u) | ((s & 32u) << 19u);
    unsigned int mask = flags * 255u;
    // Ordinary word arithmetic with each byte's carry confined to its own
    // sign bit. CUDA's vadd4/vsub4 intrinsics expand into costly emulation on
    // SM87; these expressions preserve even zero-valued diagnostic tables.
    unsigned int negative = ((~basis & 0x7f7f7f7fu) + 0x01010101u) ^ (~basis & 0x80808080u);
    unsigned int oriented = basis ^ ((basis ^ negative) & mask);
    unsigned int plus = oriented | 0x01010101u; // every E8P basis byte is even
    unsigned int minus = (((oriented & 0x7f7f7f7fu) | 0x80808080u) - 0x01010101u) ^ (~oriented & 0x80808080u);
    return parity ? minus : plus;
}
'''


@orin_jit
def _compile(rows,E,N,K,kind,shared_table,grouped,tiles,M,BN,patched):
    if kind not in ('vq4','e8p') or any(type(x) is not int or x <= 0 for x in (rows,E,N,K,tiles)) or K%128 or K*128*128 > 2**31-1:
        raise ValueError('Invalid integer VQ geometry')
    if BN not in (64,128):raise ValueError('Invalid VQ output tile')
    D = 4 if kind == 'vq4' else 8
    PD = T.uint8 if D == 4 else T.uint16
    TE = 1 if shared_table else E
    BM,G = 16,128

    @T.macro
    def project(A,P,Book,Patch,WS,AS,C,expert,offset,count,bx):
        T.import_source(SOURCE)
        a = T.alloc_shared((BM,G),T.int8)
        b = T.alloc_shared((BN,G),T.int8)
        packed = T.alloc_shared((BN,G//D),PD)
        table = T.alloc_shared((256,D//4),T.uint32)
        patch = T.alloc_shared((BN,),T.uint16)
        decoded = T.alloc_fragment((BN,G//4),T.uint32)
        acc = T.alloc_fragment((BM,BN),T.int32)
        T.copy(Book[0 if shared_table else expert,0,0],table)
        T.clear(acc)
        for kg in T.Pipelined(K//G,num_stages=2):
            for m,k in T.Parallel(BM,G):
                a[m,k] = 0
                if m < count:a[m,k] = A[offset+m,kg*G+k]
            T.copy(P[expert,kg,bx*BN,0],packed)
            if patched:
                # Scalar metadata reads avoid mixing two asynchronous producer
                # stages in TileLang's software-pipeline dependency graph.
                for n in T.Parallel(BN):
                    patch[n] = 0
                    if bx*BN+n < N:patch[n] = Patch[expert,kg,bx*BN+n]
            # Decode each four-byte word once in registers. The separate,
            # layout-aware shared write lets TileLang vectorize four bytes;
            # a scalar alloc_var in the byte loop prevents that on SM87.
            for n,q in T.Parallel(BN,G//4):
                if D == 4:
                    decoded[n,q] = table[T.cast(packed[n,q],T.int32),0]
                else:
                    code = T.cast(packed[n,q//2],T.uint32)
                    word = table[T.cast(code >> 8,T.int32),q%2]
                    decoded[n,q] = T.call_pure_extern('uint32','integer_e8p_quad',code,word,T.cast(q%2,T.uint32))
                if patched:
                    shift = T.cast((patch[n]&3)*8,T.uint32)
                    replacement = (decoded[n,q] & ~(T.uint32(255) << shift)) | (T.cast((patch[n] >> 7)&255,T.uint32) << shift)
                    decoded[n,q] = T.if_then_else((patch[n]&32768) != 0 and (patch[n]&127)//4 == q,replacement,decoded[n,q])
            for n,k in T.Parallel(BN,G):
                b[n,k] = T.cast((decoded[n,k//4] >> ((k%4)*8)) & 255,T.int8)
            T.gemm(a,b,acc,transpose_B=True)
        for m,n in T.Parallel(BM,BN):
            if m < count and bx*BN+n < N:
                C[offset+m,bx*BN+n] = ((T.cast(acc[m,n],T.float32)*T.cast(WS[expert,bx*BN+n],T.float32))*T.cast(AS[offset+m],T.float32))

    @T.prim_func
    def direct(A:T.Tensor((rows,K),T.int8),P:T.Tensor((E,K//G,N,G//D),PD),
               Book:T.Tensor((TE,256,D//4),T.uint32),Patch:T.Tensor((E,K//G,N) if patched else (1,),T.uint16),WS:T.Tensor((E,N),T.float16),
               AS:T.Tensor((rows,),T.float16),C:T.Tensor((rows,N),T.float16)):
        with T.Kernel(T.ceildiv(M,BM),T.ceildiv(N,BN),E,threads=128) as (by,bx,e):
            project(A,P,Book,Patch,WS,AS,C,e,e*M+by*BM,T.min(BM,M-by*BM),bx)

    @T.prim_func
    def routed(A:T.Tensor((rows,K),T.int8),P:T.Tensor((E,K//G,N,G//D),PD),
               Book:T.Tensor((TE,256,D//4),T.uint32),Patch:T.Tensor((E,K//G,N) if patched else (1,),T.uint16),WS:T.Tensor((E,N),T.float16),AS:T.Tensor((rows,),T.float16),
               Counts:T.Tensor((E,),T.int32),Offsets:T.Tensor((E,),T.int32),
               TileExpert:T.Tensor((tiles,),T.int32),TileRow:T.Tensor((tiles,),T.int32),TileCount:T.Tensor((1,),T.int32),
               C:T.Tensor((rows,N),T.float16)):
        with T.Kernel(tiles,T.ceildiv(N,BN),threads=128) as (tile,bx):
            meta = T.alloc_local((4,),T.int32)
            if tile < TileCount[0]:
                meta[0] = TileExpert[tile];meta[1] = TileRow[tile]
                meta[2] = Offsets[meta[0]]+meta[1];meta[3] = T.min(BM,Counts[meta[0]]-meta[1])
                project(A,P,Book,Patch,WS,AS,C,meta[0],meta[2],meta[3],bx)
    return routed if grouped else direct


def integer_vq(E,M,N,K,*,kind,shared_table=False,block_n=64,patched=False):
    return _compile(E*M,E,N,K,kind,shared_table,False,1,M,block_n,patched)


def integer_vq_grouped(rows,E,tiles,N,K,*,kind,shared_table=False,block_n=64,patched=False):
    return _compile(rows,E,N,K,kind,shared_table,True,tiles,1,block_n,patched)


@orin_jit
def integer_e8p_gemv(E: int, routes: int, N: int, K: int, *, shared_input: bool, block_n: int = 8):
    """Single-token routes directly index packed banks, using exact INT8 DP4A."""
    assert E>0 and routes>0 and N>0 and K%128==0 and block_n in (4,8,16)
    assert K*128*128<2**31
    rows=1 if shared_input else routes
    @T.prim_func
    def main(A:T.Tensor((rows,K//4),T.int32), P:T.Tensor((E,K//128,N,16),T.uint16),
             Book:T.Tensor((1,256,2),T.uint32), WS:T.Tensor((E,N),T.float16),
             AS:T.Tensor((rows,),T.float16), IDs:T.Tensor((1,routes),T.int32),
             C:T.Tensor((routes,N),T.float16)):
        with T.Kernel(T.ceildiv(N,block_n),routes,threads=128) as (bx,route):
            T.import_source(SOURCE+'\n__device__ __forceinline__ int flash_dp4a(int a, int b, int c) { return __dp4a(a,b,c); }\n')
            table=T.alloc_shared((256,2),T.uint32)
            acc=T.alloc_fragment((block_n,32),T.int32)
            total=T.alloc_fragment((block_n,),T.int32)
            T.annotate_layout({acc:tilelang.Fragment((block_n,32),
                forward_thread_fn=lambda n,k:(n%4)*32+k,
                forward_index_fn=lambda n,k:n//4)})
            T.copy(Book[0,0,0],table);T.clear(acc)
            expert=IDs[0,route]
            row=0 if shared_input else route
            for kg in T.serial(K//128):
                for n,k in T.Parallel(block_n,32):
                    if bx*block_n+n<N:
                        code=T.cast(P[expert,kg,bx*block_n+n,k//2],T.uint32)
                        word=T.call_pure_extern('uint32','integer_e8p_quad',code,table[code>>8,k%2],T.cast(k%2,T.uint32))
                        acc[n,k]=T.call_pure_extern('int32','flash_dp4a',A[row,kg*32+k],T.cast(word,T.int32),acc[n,k])
            T.reduce_sum(acc,total,dim=1)
            for n in T.Parallel(block_n):
                if bx*block_n+n<N:
                    C[route,bx*block_n+n]=(T.cast(total[n],T.float32)*T.cast(WS[expert,bx*block_n+n],T.float32))*T.cast(AS[row],T.float32)
    return main


@orin_jit
def rotate_activation(M,K,swiglu=False):
    if type(M) is not int or M <= 0 or type(K) is not int or K <= 0 or K%128:
        raise ValueError('Invalid block128 activation rotation')
    width = K*2 if swiglu else K
    @T.prim_func
    def main(X:T.Tensor((M,width),T.float16),Signs:T.Tensor((K,),T.int8),Y:T.Tensor((M,K),T.float16)):
        with T.Kernel(M,threads=128) as row:
            current = T.alloc_shared((K,),T.float32)
            temporary = T.alloc_shared((K,),T.float32)
            # Scalar signed-byte conversion avoids packed int8 -> float4
            # lowering through plain char, whose signedness is target-dependent.
            for k in T.Parallel(K,coalesced_width=T.int32(1)):
                if swiglu:
                    g = T.cast(X[row,k],T.float32)
                    e = T.exp(-T.abs(g))
                    sigmoid = T.if_then_else(g >= 0,1/(1+e),e/(1+e))
                    value = T.cast(T.cast((g*sigmoid)*T.cast(X[row,k+K],T.float32),T.float16),T.float32)
                    current[k] = value*T.cast(Signs[k],T.float32)
                else:
                    current[k] = T.cast(X[row,k],T.float32)*T.cast(Signs[k],T.float32)
            T.sync_threads()
            for stage in T.unroll(7):
                for k in T.Parallel(K):
                    temporary[k] = current[k^(1 << stage)]+current[k]*(1-2*((k >> stage)&1))
                T.sync_threads()
                T.copy(temporary,current)
                T.sync_threads()
            for k in T.Parallel(K):Y[row,k] = current[k]*0.08838834764831845
    return main
