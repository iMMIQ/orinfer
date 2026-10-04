"""Group-64 symmetric INT8 KV. Only storage is quantized; attention uses
FP16 tensor-core operands and FP32 accumulation. Scales are stored FP16.
"""
import tilelang.language as T
from tools.operators.common import orin_jit
from kernels.operators.op22_attention_decode import _check


HALF2_SOURCE = r"""
#include <cuda_fp16.h>
union KvHalf2Bits { unsigned int bits; __half2 value; };
__device__ __forceinline__ unsigned int kv_dequant_pair(unsigned int code, unsigned short scale_bits) {
    KvHalf2Bits values{0}, bias{0}, scale{0}, result{0}, lower{0}, upper{0};
    // Encode unsigned bytes xor 128 as exact FP16 integers 1024..1279.
    values.bits = (((code & 0xffu) | ((code & 0xff00u) << 8)) ^ 0x00800080u) | 0x64006400u;
    bias.bits = 0x64806480u; // 1152 in both lanes
    scale.bits = static_cast<unsigned int>(scale_bits) | (static_cast<unsigned int>(scale_bits) << 16);
    result.value = __hmul2(__hsub2(values.value, bias.value), scale.value);
    lower.bits = 0xfbfffbffu; upper.bits = 0x7bff7bffu;
    result.value = __hmin2(__hmax2(result.value, lower.value), upper.value);
    return result.bits;
}
"""


@orin_jit
def dequant_pairs_probe(rows: int):
    """Exhaustive byte/scale correctness probe for the packed converter."""
    @T.prim_func
    def kernel(C: T.Tensor((rows,),T.uint32), S: T.Tensor((rows,),T.float16),
               Y: T.Tensor((rows,2),T.float16)):
        with T.Kernel(T.ceildiv(rows,128),threads=128) as b:
            T.import_source(HALF2_SOURCE)
            for i in T.Parallel(128):
                row=b*128+i
                if row<rows:
                    bits=T.call_extern('uint32','kv_dequant_pair',C[row],T.reinterpret(T.uint16,S[row]))
                    Y[row,0]=T.reinterpret(T.float16,T.cast(bits&65535,T.uint16))
                    Y[row,1]=T.reinterpret(T.float16,T.cast(bits>>16,T.uint16))
    return kernel


@orin_jit
def full_prepare_mrope_int8(max_pages: int, context: int, section: tuple[int,int,int],
                       block_size: int = 128, max_position: int = 8704):
    rows=T.dynamic('rows')
    @T.prim_func
    def kernel(X:T.Tensor((rows,14336),T.float16),WQ:T.Tensor((256,),T.float16),WK:T.Tensor((256,),T.float16),
               Cache:T.Tensor((max_position,64),T.float16),Req:T.Tensor((rows,),T.int32),
               Pos:T.Tensor((rows,),T.int32),Pages:T.Tensor((1,max_pages),T.int32),Status:T.Tensor((1,),T.int32),
               MRope:T.Tensor((context,3),T.int32),Q:T.Tensor((rows,24,256),T.float16),
               Gate:T.Tensor((rows,24,256),T.float16),K:T.Tensor((max_pages,block_size,4,256),T.int8),
               V:T.Tensor((max_pages,block_size,4,256),T.int8),
               KS:T.Tensor((max_pages,block_size,4,4),T.float16),
               VS:T.Tensor((max_pages,block_size,4,4),T.float16)):
        with T.Kernel(rows*28,threads=128) as work:
            row=work//28;head=work%28
            values=T.alloc_fragment((256,),T.float32)
            square=T.alloc_fragment((256,),T.float32)
            total=T.alloc_fragment((1,),T.float32)
            norm=T.alloc_shared((256,),T.float16)
            ka=T.alloc_fragment((4,64),T.float32)
            va=T.alloc_fragment((4,64),T.float32)
            km=T.alloc_fragment((4,),T.float32)
            vm=T.alloc_fragment((4,),T.float32)
            ks=T.alloc_shared((4,),T.float16)
            vs=T.alloc_shared((4,),T.float16)
            if Status[0]==0:
                for j in T.Parallel(256):
                    if head<24:
                        values[j]=T.cast(X[row,head*512+j],T.float32)
                        Gate[row,head,j]=X[row,head*512+256+j]
                    else:values[j]=T.cast(X[row,12288+(head-24)*256+j],T.float32)
                    square[j]=values[j]*values[j]
                T.reduce_sum(square,total,dim=0)
                for j in T.Parallel(256):
                    weight=T.if_then_else(head<24,WQ[j],WK[j])
                    norm[j]=(values[j]*T.rsqrt(total[0]/256+1e-6))*(1+T.cast(weight,T.float32))
                T.sync_threads()
                for j in T.Parallel(256):
                    values[j]=T.cast(norm[j],T.float32)
                    if j<64:
                        idx=j%32;partner=T.if_then_else(j<32,j+32,j-32)
                        axis=T.if_then_else(idx%3==1 and idx<section[1]*3,1,
                             T.if_then_else(idx%3==2 and idx<section[2]*3,2,0))
                        position=MRope[Pos[row],axis]
                        a=T.cast(T.cast(norm[j],T.float32)*T.cast(Cache[position,idx],T.float32),T.float16)
                        b=T.cast(T.cast(norm[partner],T.float32)*T.cast(Cache[position,idx+32],T.float32),T.float16)
                        values[j]=T.if_then_else(j<32,T.cast(a,T.float32)-T.cast(b,T.float32),T.cast(a,T.float32)+T.cast(b,T.float32))
                    if head<24:Q[row,head,j]=values[j]
                if head>=24:
                    # Quantize the exact FP16 values the reference cache stores.
                    for g,d in T.Parallel(4,64):
                        ka[g,d]=T.abs(T.cast(T.cast(values[g*64+d],T.float16),T.float32))
                        va[g,d]=T.abs(T.cast(X[row,13312+(head-24)*256+g*64+d],T.float32))
                    T.reduce_max(ka,km,dim=1)
                    T.reduce_max(va,vm,dim=1)
                    for g in T.Parallel(4):
                        # FP16 scales, bounded above zero even for subnormal input.
                        ks[g]=T.max(km[g]/127.0,0.000000059604644775390625)
                        vs[g]=T.max(vm[g]/127.0,0.000000059604644775390625)
                    T.sync_threads()
                    page=Pages[Req[row],Pos[row]//block_size]
                    for g in T.Parallel(4):
                        KS[page,Pos[row]%block_size,head-24,g]=ks[g]
                        VS[page,Pos[row]%block_size,head-24,g]=vs[g]
                    for j in T.Parallel(256):
                        kval=T.cast(T.cast(values[j],T.float16),T.float32)
                        vval=T.cast(X[row,13312+(head-24)*256+j],T.float32)
                        kr=T.call_extern('float32','__fdiv_rn',kval,T.cast(ks[j//64],T.float32))
                        vr=T.call_extern('float32','__fdiv_rn',vval,T.cast(vs[j//64],T.float32))
                        K[page,Pos[row]%block_size,head-24,j]=T.cast(T.max(-127.0,T.min(127.0,T.round(kr))),T.int8)
                        V[page,Pos[row]%block_size,head-24,j]=T.cast(T.max(-127.0,T.min(127.0,T.round(vr))),T.int8)
    return kernel


@orin_jit
def attention_prefill_int8(batch: int, tq: int, tkv: int,
                      input_layout: str = "token_major",
                      output_layout: str = "token_major",
                      kv_layout: str = "token_major", gate_mode: str = "native_fp16",
                      block_m: int = 32, block_n: int = 32, threads: int = 128,
                      exp_mode: str = "precise"):
    """Build (Q,K,V,RawGate,Positions,Lengths,Y), all explicit tensors.

    Q/G/Y token_major [B,Tq,24,256] or head_major [B,24,Tq,256].
    K/V token_major INT8 [B,Tkv,4,256], FP16 group-64 scales.
    Metadata int32 [B,Tq], [B]. FP16 compute, scale=1/16, qh//6 GQA.
    native_fp16 rounds sigmoid and normalized attention before multiplication.
    fp32_fused is a separately identified candidate with only output rounding.
    """
    assert (batch is None or batch > 0) and tq > 0 and tkv > 0
    batch = T.dynamic('batch') if batch is None else batch
    assert input_layout in ("token_major", "head_major")
    assert output_layout in ("token_major", "head_major")
    assert kv_layout == "token_major"
    assert gate_mode in ("native_fp16", "fp32_fused")
    assert block_m in (16, 32, 64, 128) and block_n in (32, 64)
    assert exp_mode in ('precise','fast')
    @T.macro
    def probability_exp(value):
        return T.call_extern('float32','__expf',value) if exp_mode=='fast' else T.exp(value)
    qs = (batch,tq,6144) if input_layout == "token_major" else (batch,24,tq,256)
    ys = (batch,tq,24,256) if output_layout == "token_major" else (batch,24,tq,256)
    ks = (batch,tkv,256) if kv_layout == "token_major" else (batch,4,tkv,256)
    @T.prim_func
    def kernel(Q: T.Tensor(qs,T.float16), K: T.Tensor(ks,T.uint32),
               V: T.Tensor(ks,T.uint32), Gate: T.Tensor(qs,T.float16),
               Positions: T.Tensor((batch,tq),T.int32),
               Lengths: T.Tensor((batch,),T.int32), Y: T.Tensor(ys,T.float16),
               KS: T.Tensor((batch,tkv,16),T.float16),
               VS: T.Tensor((batch,tkv,16),T.float16)):
        with T.Kernel(T.ceildiv(tq,block_m),24,batch,threads=threads) as (qb,h,b):
            T.import_source(HALF2_SOURCE)
            q = T.alloc_shared((block_m,256),T.float16)
            k = T.alloc_shared((block_n,256),T.float16)
            v = T.alloc_shared((block_n,256),T.float16)
            p = T.alloc_shared((block_m,block_n),T.float16)
            score = T.alloc_fragment((block_m,block_n),T.float32)
            out = T.alloc_fragment((block_m,256),T.float32)
            maximum = T.alloc_fragment((block_m,),T.float32)
            previous = T.alloc_fragment((block_m,),T.float32)
            correction = T.alloc_fragment((block_m,),T.float32)
            denom = T.alloc_fragment((block_m,),T.float32)
            rowsum = T.alloc_fragment((block_m,),T.float32)
            pos = T.alloc_fragment((block_m,),T.int32)
            maxpos = T.alloc_fragment((1,),T.int32)
            T.fill(maximum,-1e30)
            T.clear(denom)
            T.clear(out)
            for i in T.Parallel(block_m):
                pos[i] = T.if_then_else(qb*block_m+i < tq,Positions[b,qb*block_m+i],-1)
            T.reduce_max(pos,maxpos,dim=0)
            if input_layout == "token_major":
                T.copy(Q[b,qb*block_m,h*256],q,prefer_instruction="sync")
            else:
                T.copy(Q[b,h,qb*block_m,0],q,prefer_instruction="sync")
            for kb in T.serial(T.ceildiv(T.max(0,T.min(Lengths[b],maxpos[0]+1)),block_n)):
                for j,d in T.Parallel(block_n,64):
                    token=kb*block_n+j
                    if token < Lengths[b] and token < tkv:
                        kw=K[b,token,(h//6)*64+d]; vw=V[b,token,(h//6)*64+d]
                        sk=T.reinterpret(T.uint16,KS[b,token,(h//6)*4+d//16])
                        sv=T.reinterpret(T.uint16,VS[b,token,(h//6)*4+d//16])
                        k0=T.call_extern('uint32','kv_dequant_pair',kw,sk)
                        k1=T.call_extern('uint32','kv_dequant_pair',kw>>16,sk)
                        v0=T.call_extern('uint32','kv_dequant_pair',vw,sv)
                        v1=T.call_extern('uint32','kv_dequant_pair',vw>>16,sv)
                        k[j,d*4]=T.reinterpret(T.float16,T.cast(k0&65535,T.uint16))
                        k[j,d*4+1]=T.reinterpret(T.float16,T.cast(k0>>16,T.uint16))
                        k[j,d*4+2]=T.reinterpret(T.float16,T.cast(k1&65535,T.uint16))
                        k[j,d*4+3]=T.reinterpret(T.float16,T.cast(k1>>16,T.uint16))
                        v[j,d*4]=T.reinterpret(T.float16,T.cast(v0&65535,T.uint16))
                        v[j,d*4+1]=T.reinterpret(T.float16,T.cast(v0>>16,T.uint16))
                        v[j,d*4+2]=T.reinterpret(T.float16,T.cast(v1&65535,T.uint16))
                        v[j,d*4+3]=T.reinterpret(T.float16,T.cast(v1>>16,T.uint16))
                    else:
                        for lane in T.unroll(4): k[j,d*4+lane]=0.0;v[j,d*4+lane]=0.0
                T.sync_threads()
                T.clear(score)
                T.gemm(q,k,score,transpose_B=True)
                for i in T.Parallel(block_m):
                    previous[i] = maximum[i]
                for i,j in T.Parallel(block_m,block_n):
                    score[i,j] = T.if_then_else(kb*block_n+j < Lengths[b] and kb*block_n+j <= pos[i],score[i,j]*0.0625,-1e30)
                T.reduce_max(score,maximum,dim=1,clear=False)
                for i,j in T.Parallel(block_m,block_n):
                    score[i,j] = T.if_then_else(kb*block_n+j < Lengths[b] and kb*block_n+j <= pos[i],probability_exp(score[i,j]-maximum[i]),0)
                    p[i,j] = score[i,j]
                T.reduce_sum(score,rowsum,dim=1)
                for i in T.Parallel(block_m):
                    correction[i] = probability_exp(previous[i]-maximum[i])
                    denom[i] = denom[i]*correction[i]+rowsum[i]
                for i,d in T.Parallel(block_m,256):
                    out[i,d] *= correction[i]
                T.gemm(p,v,out)
            for i,d in T.Parallel(block_m,256):
                if qb*block_m+i < tq:
                    if input_layout == "token_major":
                        raw = T.cast(Gate[b,qb*block_m+i,h*256+d],T.float32)
                    else:
                        raw = T.cast(Gate[b,h,qb*block_m+i,d],T.float32)
                    sig = 1/(1+T.exp(-raw))
                    attn = T.if_then_else(denom[i] > 0,out[i,d]/T.max(denom[i],1e-30),0)
                    if gate_mode == "native_fp16":
                        result = T.cast(T.cast(attn,T.float16),T.float32)*T.cast(T.cast(sig,T.float16),T.float32)
                    else:
                        result = attn*sig
                    if output_layout == "token_major":
                        Y[b,qb*block_m+i,h,d] = result
                    else:
                        Y[b,h,qb*block_m+i,d] = result
    return kernel


@orin_jit
def _compile_int8(max_pages:int,num_pages:int,block_size:int,block_n:int,
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
    def online(Q,K,V,KS,VS,Pages,SeqLen,QueryPos,b,kh,split,out,maximum,denom):
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
            for j,d in T.Parallel(block_n,64):
                token=base+j
                stage_page=T.if_then_else(base%block_size+j<block_size,first,second)
                if token<end and stage_page>=0 and stage_page<num_pages:
                    kw=K[stage_page,token%block_size,kh*64+d];vw=V[stage_page,token%block_size,kh*64+d]
                    sk=T.reinterpret(T.uint16,KS[stage_page,token%block_size,kh*4+d//16])
                    sv=T.reinterpret(T.uint16,VS[stage_page,token%block_size,kh*4+d//16])
                    k0=T.call_extern('uint32','kv_dequant_pair',kw,sk)
                    k1=T.call_extern('uint32','kv_dequant_pair',kw>>16,sk)
                    v0=T.call_extern('uint32','kv_dequant_pair',vw,sv)
                    v1=T.call_extern('uint32','kv_dequant_pair',vw>>16,sv)
                    k[j,d*4]=T.reinterpret(T.float16,T.cast(k0&65535,T.uint16))
                    k[j,d*4+1]=T.reinterpret(T.float16,T.cast(k0>>16,T.uint16))
                    k[j,d*4+2]=T.reinterpret(T.float16,T.cast(k1&65535,T.uint16))
                    k[j,d*4+3]=T.reinterpret(T.float16,T.cast(k1>>16,T.uint16))
                    v[j,d*4]=T.reinterpret(T.float16,T.cast(v0&65535,T.uint16))
                    v[j,d*4+1]=T.reinterpret(T.float16,T.cast(v0>>16,T.uint16))
                    v[j,d*4+2]=T.reinterpret(T.float16,T.cast(v1&65535,T.uint16))
                    v[j,d*4+3]=T.reinterpret(T.float16,T.cast(v1>>16,T.uint16))
                else:
                    for lane in T.unroll(4): k[j,d*4+lane]=0.0;v[j,d*4+lane]=0.0
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
                   K:T.Tensor((num_pages,block_size,256),T.uint32),
                   V:T.Tensor((num_pages,block_size,256),T.uint32),
                   Pages:T.Tensor((contexts,max_pages),T.int32),
                   SeqLen:T.Tensor((contexts,),T.int32),QueryPos:T.Tensor((batch,),T.int32),
                   M:T.Tensor((batch,24,nsplits),T.float32),L:T.Tensor((batch,24,nsplits),T.float32),
                   O:T.Tensor((batch,24,nsplits,256),T.float32),
                   KS:T.Tensor((num_pages,block_size,16),T.float16),
                   VS:T.Tensor((num_pages,block_size,16),T.float16)):
            with T.Kernel(4,nsplits,batch,threads=128) as (kh,s,b):
                T.import_source(HALF2_SOURCE)
                out=T.alloc_fragment((16,256),T.float32)
                maximum=T.alloc_fragment((16,),T.float32)
                denom=T.alloc_fragment((16,),T.float32)
                online(Q,K,V,KS,VS,Pages,SeqLen,QueryPos,b,kh,s,out,maximum,denom)
                for i,d in T.Parallel(16,256):
                    if i<6:O[b,kh*6+i,s,d]=out[i,d]
                for i in T.Parallel(16):
                    if i<6:
                        M[b,kh*6+i,s]=T.if_then_else(denom[i]>0,maximum[i],-T.infinity(T.float32))
                        L[b,kh*6+i,s]=denom[i]
    else:
        @T.prim_func
        def kernel(Q:T.Tensor((batch,24,256),T.float16),
                   K:T.Tensor((num_pages,block_size,256),T.uint32),
                   V:T.Tensor((num_pages,block_size,256),T.uint32),
                   Pages:T.Tensor((contexts,max_pages),T.int32),
                   SeqLen:T.Tensor((contexts,),T.int32),QueryPos:T.Tensor((batch,),T.int32),
                   RawGate:T.Tensor((batch,24,256),T.float16),Y:T.Tensor((batch,24,256),T.float16),
                   KS:T.Tensor((num_pages,block_size,16),T.float16),
                   VS:T.Tensor((num_pages,block_size,16),T.float16)):
            with T.Kernel(4,batch,threads=128) as (kh,b):
                T.import_source(HALF2_SOURCE)
                out=T.alloc_fragment((16,256),T.float32)
                maximum=T.alloc_fragment((16,),T.float32)
                denom=T.alloc_fragment((16,),T.float32)
                online(Q,K,V,KS,VS,Pages,SeqLen,QueryPos,b,kh,0,out,maximum,denom)
                for i,d in T.Parallel(16,256):
                    if i<6:
                        Y[b,kh*6+i,d]=0.0
                        if denom[i]>0:
                            attn=T.cast(out[i,d]/denom[i],T.float16)
                            gate=T.cast(1/(1+T.exp(-T.cast(RawGate[b,kh*6+i,d],T.float32))),T.float16)
                            Y[b,kh*6+i,d]=T.cast(attn,T.float32)*T.cast(gate,T.float32)
    return kernel



def paged_attention_partials_int8(max_pages,num_pages,nsplits=8,block_size=128,block_n=64,
                                        queries=None):
    _check(max_pages,num_pages,block_size,32,nsplits)
    assert block_n in (32,64) and block_size==128
    assert queries is None or 1 <= queries <= 16
    return _compile_int8(max_pages,num_pages,block_size,block_n,nsplits,True,queries)
