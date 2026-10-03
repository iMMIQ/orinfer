"""Causal online-softmax prefill, FP16 tensorcores / FP32 reductions, SM87.

No scores workspace. Production inputs and outputs have stable caller-owned
addresses. token_major Q/G/Y directly consume op20 [B*Tq,24,256]. Contiguous
KV is explicit; this entry does not interpret op20 paged KV/page tables.
"""
import tilelang.language as T
from tools.operators.common import orin_jit


def validate_metadata(positions, lengths, tkv):
    """CPU scheduler contract: empty rows are legal and produce zero.

    Positions may be unsorted. -1 denotes a padded/empty query. KV positions
    are contiguous absolute positions 0..length-1, not chunk-relative indices.
    """
    if len(positions) != len(lengths) or not positions:
        raise ValueError("metadata batch mismatch")
    width = len(positions[0])
    if width < 1 or any(len(row) != width for row in positions):
        raise ValueError("positions must be nonempty rectangular")
    for row, length in zip(positions, lengths):
        if type(length) is not int or not 0 <= length <= tkv:
            raise ValueError("length outside contiguous KV allocation")
        if any(type(p) is not int or p < -1 or p > 2147483646 for p in row):
            raise ValueError("invalid absolute query position")


@orin_jit
def attention_prefill(batch: int, tq: int, tkv: int,
                      input_layout: str = "token_major",
                      output_layout: str = "token_major",
                      kv_layout: str = "head_major", gate_mode: str = "native_fp16",
                      block_m: int = 32, block_n: int = 32, threads: int = 128):
    """Build (Q,K,V,RawGate,Positions,Lengths,Y), all explicit tensors.

    Q/G/Y token_major [B,Tq,24,256] or head_major [B,24,Tq,256].
    K/V head_major [B,4,Tkv,256] or token_major [B,Tkv,4,256].
    Metadata int32 [B,Tq], [B]. FP16 data, scale=1/16, qh//6 GQA.
    native_fp16 rounds sigmoid and normalized attention before multiplication.
    fp32_fused is a separately identified candidate with only output rounding.
    """
    assert (batch is None or batch > 0) and tq > 0 and tkv > 0
    batch = T.dynamic('batch') if batch is None else batch
    assert input_layout in ("token_major", "head_major")
    assert output_layout in ("token_major", "head_major")
    assert kv_layout in ("token_major", "head_major")
    assert gate_mode in ("native_fp16", "fp32_fused")
    assert block_m in (16, 32, 64) and block_n in (32, 64)
    qs = (batch,tq,24,256) if input_layout == "token_major" else (batch,24,tq,256)
    ys = (batch,tq,24,256) if output_layout == "token_major" else (batch,24,tq,256)
    ks = (batch,tkv,4,256) if kv_layout == "token_major" else (batch,4,tkv,256)
    @T.prim_func
    def kernel(Q: T.Tensor(qs,T.float16), K: T.Tensor(ks,T.float16),
               V: T.Tensor(ks,T.float16), Gate: T.Tensor(qs,T.float16),
               Positions: T.Tensor((batch,tq),T.int32),
               Lengths: T.Tensor((batch,),T.int32), Y: T.Tensor(ys,T.float16)):
        with T.Kernel(T.ceildiv(tq,block_m),24,batch,threads=threads) as (qb,h,b):
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
            for i,d in T.Parallel(block_m,256):
                if input_layout == "token_major":
                    q[i,d] = T.if_then_else(qb*block_m+i < tq,Q[b,qb*block_m+i,h,d],0)
                else:
                    q[i,d] = T.if_then_else(qb*block_m+i < tq,Q[b,h,qb*block_m+i,d],0)
            for kb in T.serial(T.ceildiv(T.max(0,T.min(Lengths[b],maxpos[0]+1)),block_n)):
                for j,d in T.Parallel(block_n,256):
                    if kv_layout == "head_major":
                        k[j,d] = T.if_then_else(kb*block_n+j < Lengths[b],K[b,h//6,kb*block_n+j,d],0)
                        v[j,d] = T.if_then_else(kb*block_n+j < Lengths[b],V[b,h//6,kb*block_n+j,d],0)
                    else:
                        k[j,d] = T.if_then_else(kb*block_n+j < Lengths[b],K[b,kb*block_n+j,h//6,d],0)
                        v[j,d] = T.if_then_else(kb*block_n+j < Lengths[b],V[b,kb*block_n+j,h//6,d],0)
                T.clear(score)
                T.gemm(q,k,score,transpose_B=True)
                for i in T.Parallel(block_m):
                    previous[i] = maximum[i]
                for i,j in T.Parallel(block_m,block_n):
                    score[i,j] = T.if_then_else(kb*block_n+j < Lengths[b] and kb*block_n+j <= pos[i],score[i,j]*0.0625,-1e30)
                T.reduce_max(score,maximum,dim=1,clear=False)
                for i,j in T.Parallel(block_m,block_n):
                    score[i,j] = T.if_then_else(kb*block_n+j < Lengths[b] and kb*block_n+j <= pos[i],T.exp(score[i,j]-maximum[i]),0)
                    p[i,j] = score[i,j]
                T.reduce_sum(score,rowsum,dim=1)
                for i in T.Parallel(block_m):
                    correction[i] = T.exp(previous[i]-maximum[i])
                    denom[i] = denom[i]*correction[i]+rowsum[i]
                for i,d in T.Parallel(block_m,256):
                    out[i,d] *= correction[i]
                T.gemm(p,v,out)
            for i,d in T.Parallel(block_m,256):
                if qb*block_m+i < tq:
                    if input_layout == "token_major":
                        raw = T.cast(Gate[b,qb*block_m+i,h,d],T.float32)
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
