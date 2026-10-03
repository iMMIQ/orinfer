"""Paged GQA with vector tile copy and guarded crossing/tail fallback.

K/V expose the exact same physical token-major [pages,128,4,256] storage
as [pages,128,1024] to copy one selected head's rows. No KV conversion.
Same FP16 QK/PV operands and FP32 online statistics as op22 GQA.
"""
import tilelang.language as T
from tools.operators.common import orin_jit
from kernels.operators.op22_attention_decode import _check


@orin_jit
def _compile_staged(max_pages:int,num_pages:int,block_size:int,block_n:int,
                 nsplits:int,partials:bool,queries:int|None=None):
    """Explicit KV reuse candidate: six Q heads share each staged KV tile.

    QK and PV accumulate FP32 on FP16 tensorcores. Softmax/max/denominator are
    FP32, but PV probability operands round FP16, as in FlashAttention. This
    extra rounding is separate from the strict FP32 SIMT baseline.
    """
    # A sequence verification graph has several queries sharing one context.
    # Their absolute QueryPos still applies a separate causal limit per row.
    batch=T.dynamic('batch') if queries is None else queries
    contexts=batch if queries is None else 1

    @T.macro
    def online(Q,K,V,Pages,SeqLen,QueryPos,b,kh,split,out,maximum,denom):
        q=T.alloc_shared((16,256),T.float16)
        k=T.alloc_shared((block_n,256),T.float16)
        v=T.alloc_shared((block_n,256),T.float16)
        p=T.alloc_shared((16,block_n),T.float16)
        scores=T.alloc_fragment((16,block_n),T.float32)
        previous=T.alloc_fragment((16,),T.float32)
        correction=T.alloc_fragment((16,),T.float32)
        rowsum=T.alloc_fragment((16,),T.float32)
        T.clear(out);T.clear(denom);T.fill(maximum,-T.infinity(T.float32))
        for i,d in T.Parallel(16,256):
            q[i,d]=0.0
            if i<6:q[i,d]=Q[b,kh*6+i,d]
        context=b if queries is None else 0
        valid=T.max(0,T.min(T.min(SeqLen[context],QueryPos[b]+1),max_pages*block_size))
        width=T.ceildiv(valid,nsplits);start=split*width;end=T.min(start+width,valid)
        for tile in T.serial(T.ceildiv(T.max(0,end-start),block_n)):
            base=start+tile*block_n
            first=Pages[context,base//block_size]
            second=Pages[context,(T.min(base+block_n,end)-1)//block_size]
            if base+block_n<=end and base%block_size+block_n<=block_size and first>=0 and first<num_pages:
                # The branch predicate is CTA-uniform but opaque to TileLang's
                # async-copy synchronization analysis. Use synchronous vector
                # loads here and a barrier after both staging paths.
                T.copy(K[first,base%block_size,kh*256],k,prefer_instruction="sync")
                T.copy(V[first,base%block_size,kh*256],v,prefer_instruction="sync")
            else:
                for j,d in T.Parallel(block_n,256):
                    token=base+j
                    stage_page=T.if_then_else(base%block_size+j<block_size,first,second)
                    if token<end and stage_page>=0 and stage_page<num_pages:
                        k[j,d]=K[stage_page,token%block_size,kh*256+d]
                        v[j,d]=V[stage_page,token%block_size,kh*256+d]
                    else:
                        k[j,d]=0.0;v[j,d]=0.0
            T.sync_threads()
            T.clear(scores)
            T.gemm(q,k,scores,transpose_B=True)
            for i in T.Parallel(16):previous[i]=maximum[i]
            for i,j in T.Parallel(16,block_n):
                token=start+tile*block_n+j
                if i<6 and token<end:
                    page=T.if_then_else(base%block_size+j<block_size,first,second)
                    if page>=0 and page<num_pages:scores[i,j]*=.0625
                    else:scores[i,j]=-1e30
                else:scores[i,j]=-1e30
            T.reduce_max(scores,maximum,dim=1,clear=False)
            for i,j in T.Parallel(16,block_n):
                token=start+tile*block_n+j
                if i<6 and token<end:
                    page=T.if_then_else(base%block_size+j<block_size,first,second)
                    if page>=0 and page<num_pages:scores[i,j]=T.exp(scores[i,j]-maximum[i])
                    else:scores[i,j]=0.0
                else:scores[i,j]=0.0
                p[i,j]=scores[i,j]
            T.reduce_sum(scores,rowsum,dim=1)
            for i in T.Parallel(16):
                correction[i]=T.exp(previous[i]-maximum[i])
                denom[i]=denom[i]*correction[i]+rowsum[i]
            for i,d in T.Parallel(16,256):out[i,d]*=correction[i]
            T.gemm(p,v,out)

    if partials:
        @T.prim_func
        def kernel(Q:T.Tensor((batch,24,256),T.float16),
                   K:T.Tensor((num_pages,block_size,1024),T.float16),
                   V:T.Tensor((num_pages,block_size,1024),T.float16),
                   Pages:T.Tensor((contexts,max_pages),T.int32),
                   SeqLen:T.Tensor((contexts,),T.int32),QueryPos:T.Tensor((batch,),T.int32),
                   M:T.Tensor((batch,24,nsplits),T.float32),L:T.Tensor((batch,24,nsplits),T.float32),
                   O:T.Tensor((batch,24,nsplits,256),T.float32)):
            with T.Kernel(4,nsplits,batch,threads=128) as (kh,s,b):
                out=T.alloc_fragment((16,256),T.float32)
                maximum=T.alloc_fragment((16,),T.float32)
                denom=T.alloc_fragment((16,),T.float32)
                online(Q,K,V,Pages,SeqLen,QueryPos,b,kh,s,out,maximum,denom)
                for i,d in T.Parallel(16,256):
                    if i<6:O[b,kh*6+i,s,d]=out[i,d]
                for i in T.Parallel(16):
                    if i<6:
                        M[b,kh*6+i,s]=T.if_then_else(denom[i]>0,maximum[i],-T.infinity(T.float32))
                        L[b,kh*6+i,s]=denom[i]
    else:
        @T.prim_func
        def kernel(Q:T.Tensor((batch,24,256),T.float16),
                   K:T.Tensor((num_pages,block_size,1024),T.float16),
                   V:T.Tensor((num_pages,block_size,1024),T.float16),
                   Pages:T.Tensor((contexts,max_pages),T.int32),
                   SeqLen:T.Tensor((contexts,),T.int32),QueryPos:T.Tensor((batch,),T.int32),
                   RawGate:T.Tensor((batch,24,256),T.float16),Y:T.Tensor((batch,24,256),T.float16)):
            with T.Kernel(4,batch,threads=128) as (kh,b):
                out=T.alloc_fragment((16,256),T.float32)
                maximum=T.alloc_fragment((16,),T.float32)
                denom=T.alloc_fragment((16,),T.float32)
                online(Q,K,V,Pages,SeqLen,QueryPos,b,kh,0,out,maximum,denom)
                for i,d in T.Parallel(16,256):
                    if i<6:
                        Y[b,kh*6+i,d]=0.0
                        if denom[i]>0:
                            attn=T.cast(out[i,d]/denom[i],T.float16)
                            gate=T.cast(1/(1+T.exp(-T.cast(RawGate[b,kh*6+i,d],T.float32))),T.float16)
                            Y[b,kh*6+i,d]=T.cast(attn,T.float32)*T.cast(gate,T.float32)
    return kernel



def paged_attention_partials_gqa_staged(max_pages,num_pages,nsplits=8,block_size=128,block_n=64,
                                        queries=None):
    _check(max_pages,num_pages,block_size,32,nsplits)
    assert block_n in (32,64) and block_size==128
    assert queries is None or 1 <= queries <= 16
    return _compile_staged(max_pages,num_pages,block_size,block_n,nsplits,True,queries)
