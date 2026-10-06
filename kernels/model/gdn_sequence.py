"""Causal small-sequence GDN with exact prefix snapshots for speculation.

Each CTA keeps its initial FP32 state in registers while processing a chunk.
Verification saves either full per-position states or compact rank-one updates;
the caller commits only the accepted prefix, including the target token.
In-place prefill writes only the final state and supports chunks through 512.
Convolution history, position, and attention KV require separate handling.
"""
import tilelang.language as T
from tools.operators.common import orin_jit

HK, HV, DK, DV = 16, 48, 128, 128


@orin_jit
def gdn_sequence(tokens: int, q_scale: float = 128 ** -.5,
                 value_tile: int = 32, threads: int = 128, in_place: bool = False,
                 compact: bool = False):
    assert 1 <= tokens <= (512 if in_place else 16) and q_scale > 0
    assert value_tile in (16, 32, 64, 128) and threads in (64, 128, 256)
    assert not (compact and in_place)

    @T.prim_func
    def main(Q: T.Tensor((HK, tokens, DK), T.float16),
             K: T.Tensor((HK, tokens, DK), T.float16),
             V: T.Tensor((HV, tokens, DV), T.float16),
             G: T.Tensor((tokens, HV), T.float32),
             Beta: T.Tensor((tokens, HV), T.float32),
             State: T.Tensor((HV, DK, DV), T.float32),
             Prefix: T.Tensor((tokens, HV, DV) if compact else (1 if in_place else tokens, HV, DK, DV), T.float32),
             Out: T.Tensor((tokens, HV * DV), T.float16)):
        with T.Kernel(DV // value_tile, HV, threads=threads) as (bv, h):
            kh = h // 3
            state = T.alloc_fragment((DK, value_tile), T.float32)
            product = T.alloc_fragment((DK, value_tile), T.float32)
            predicted = T.alloc_fragment((value_tile,), T.float32)
            result = T.alloc_fragment((value_tile,), T.float32)
            update = T.alloc_fragment((value_tile,), T.float32)
            for i, j in T.Parallel(DK, value_tile):
                state[i, j] = State[h, i, bv * value_tile + j]
            for t in T.serial(tokens):
                for i, j in T.Parallel(DK, value_tile):
                    state[i, j] = state[i, j] * T.exp(G[t, h])
                    product[i, j] = state[i, j] * T.cast(K[kh, t, i], T.float32)
                T.reduce_sum(product, predicted, dim=0)
                if compact:
                    for j in T.Parallel(value_tile):
                        update[j]=Beta[t,h]*(T.cast(V[h,t,bv*value_tile+j],T.float32)-predicted[j])
                        Prefix[t,h,bv*value_tile+j]=update[j]
                for i, j in T.Parallel(DK, value_tile):
                    delta = Beta[t, h] * (T.cast(V[h, t, bv * value_tile + j], T.float32) - predicted[j])
                    if compact:delta=update[j]
                    state[i, j] = state[i, j] + T.cast(K[kh, t, i], T.float32) * delta
                    product[i, j] = state[i, j] * (T.cast(Q[kh, t, i], T.float32) * q_scale)
                T.reduce_sum(product, result, dim=0)
                for i, j in T.Parallel(DK, value_tile):
                    if in_place:
                        if t == tokens - 1:
                            State[h, i, bv * value_tile + j] = state[i, j]
                    elif not compact:
                        Prefix[t, h, i, bv * value_tile + j] = state[i, j]
                for j in T.Parallel(value_tile):
                    Out[t, h * DV + bv * value_tile + j] = result[j]
    return main


@orin_jit
def gdn_commit(tokens: int, value_tile: int = 32):
    """Replay saved FP32 rank-one updates in their original rounding order."""
    assert 1<=tokens<=8 and value_tile in (16,32,64,128)
    @T.prim_func
    def main(K:T.Tensor((HK,tokens,DK),T.float16), G:T.Tensor((tokens,HV),T.float32),
             Update:T.Tensor((tokens,HV,DV),T.float32), Accepted:T.Tensor((1,),T.int32),
             State:T.Tensor((HV,DK,DV),T.float32)):
        with T.Kernel(DV//value_tile,HV,threads=128) as (bv,h):
            state=T.alloc_fragment((DK,value_tile),T.float32)
            for i,j in T.Parallel(DK,value_tile):state[i,j]=State[h,i,bv*value_tile+j]
            for t in T.serial(tokens):
                if t<Accepted[0]:
                    for i,j in T.Parallel(DK,value_tile):
                        state[i,j]=state[i,j]*T.exp(G[t,h])
                        state[i,j]=state[i,j]+T.cast(K[h//3,t,i],T.float32)*Update[t,h,bv*value_tile+j]
            for i,j in T.Parallel(DK,value_tile):State[h,i,bv*value_tile+j]=state[i,j]
    return main
