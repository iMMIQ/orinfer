"""Causal small-sequence GDN with exact prefix snapshots for speculation.

Each CTA owns all key rows for distinct value columns and reads its initial
state once. Tokens update that state in sequence. Every resulting prefix is
saved in FP32; the caller commits only the accepted input prefix, including
the mandatory target token. This kernel does not mutate the initial state.
Convolution history, position, and attention KV require separate handling.
"""
import tilelang.language as T
from tools.operators.common import orin_jit

HK, HV, DK, DV = 16, 48, 128, 128


@orin_jit
def gdn_sequence(tokens: int, q_scale: float = 128 ** -.5,
                 value_tile: int = 32, threads: int = 128, in_place: bool = False):
    assert 1 <= tokens <= (128 if in_place else 16) and q_scale > 0
    assert value_tile in (16, 32, 64, 128) and threads in (64, 128, 256)

    @T.prim_func
    def main(Q: T.Tensor((HK, tokens, DK), T.float16),
             K: T.Tensor((HK, tokens, DK), T.float16),
             V: T.Tensor((HV, tokens, DV), T.float16),
             G: T.Tensor((tokens, HV), T.float32),
             Beta: T.Tensor((tokens, HV), T.float32),
             State: T.Tensor((HV, DK, DV), T.float32),
             Prefix: T.Tensor((tokens, HV, DK, DV), T.float32),
             Out: T.Tensor((tokens, HV * DV), T.float16)):
        with T.Kernel(DV // value_tile, HV, threads=threads) as (bv, h):
            kh = h // 3
            state = T.alloc_fragment((DK, value_tile), T.float32)
            product = T.alloc_fragment((DK, value_tile), T.float32)
            predicted = T.alloc_fragment((value_tile,), T.float32)
            result = T.alloc_fragment((value_tile,), T.float32)
            for i, j in T.Parallel(DK, value_tile):
                state[i, j] = State[h, i, bv * value_tile + j]
            for t in T.serial(tokens):
                for i, j in T.Parallel(DK, value_tile):
                    state[i, j] = state[i, j] * T.exp(G[t, h])
                    product[i, j] = state[i, j] * T.cast(K[kh, t, i], T.float32)
                T.reduce_sum(product, predicted, dim=0)
                for i, j in T.Parallel(DK, value_tile):
                    delta = Beta[t, h] * (T.cast(V[h, t, bv * value_tile + j], T.float32) - predicted[j])
                    state[i, j] = state[i, j] + T.cast(K[kh, t, i], T.float32) * delta
                    product[i, j] = state[i, j] * (T.cast(Q[kh, t, i], T.float32) * q_scale)
                T.reduce_sum(product, result, dim=0)
                for i, j in T.Parallel(DK, value_tile):
                    if in_place:
                        if t == tokens - 1:
                            State[h, i, bv * value_tile + j] = state[i, j]
                    else:
                        Prefix[t, h, i, bv * value_tile + j] = state[i, j]
                for j in T.Parallel(value_tile):
                    Out[t, h * DV + bv * value_tile + j] = result[j]
    return main
