"""Width-four causal depthwise conv, SiLU, L2 and private history (SM87).

Q scaling is an explicit build parameter; default Q is unscaled native prefill.
All buffers are caller owned, contiguous and mutually non-aliasing. No Torch
mathematics is used in this production kernel.
"""
import tilelang.language as T
from tools.operators.common import orin_jit

CHANNELS, KEY_HEADS, VALUE_HEADS, DIM = 10240, 16, 48, 128
Q_SCALE = 1.0


@orin_jit
def gdn_conv_prep(B: int | None = None, tokens: int | None = None,
                  tile_tokens: int = 16, normalize_round_fp16: bool = True,
                  q_scale: float = 1.0, qk_output_dtype: str = "float16",
                  conv_product_round_fp16: bool = True,
                  weight_dtype: str = "float16"):
    """Build (X,W,HI,lengths,positions,Q,K,V,HO,positions_out).

    X[B,T,10240] and W[10240,4] FP16; chronological raw HI/HO[B,3,10240]
    FP16. lengths[B]/positions[B] int32; positions is the logical first token
    offset of each independent request. 0 <= lengths <= T. Q/K[B,16,T,128],
    Q/K dtype is qk_output_dtype, V[B,48,T,128] FP16; padded output rows are zero. Empty chunks copy HI and
    positions exactly. Nonempty start position 0 ignores prior history.

    Native FP16 tap products are rounded before FP32 accumulation; False
    selects an explicit FP32-product candidate. FP32 conv+SiLU is rounded
    to FP16 before normalization. Optional half
    rounding of normalized Q matches staged native prefill; False matches
    the native fused-decode normalization. Q's scale is then applied in FP32
    with q_scale and the exported Q cast to qk_output_dtype. Default unscaled
    FP16 output follows native staged prefill; fused decode uses False,
    qk_output_dtype='float32', q_scale=128**-.5. Downstream must apply scaling
    only for q_scale=1.0 and must preserve this mode's rounding contract.
    """
    assert tile_tokens in (1, 2, 4, 8, 16)
    assert qk_output_dtype in ("float16", "float32") and q_scale > 0
    assert weight_dtype in ("float16", "float32")
    batch = T.dynamic("batch") if B is None else B
    time = T.dynamic("tokens") if tokens is None else tokens

    @T.prim_func
    def kernel(X: T.Tensor((batch, time, CHANNELS), T.float16),
               W: T.Tensor((CHANNELS, 4), weight_dtype),
               HI: T.Tensor((batch, 3, CHANNELS), T.float16),
               lengths: T.Tensor((batch,), T.int32),
               positions: T.Tensor((batch,), T.int32),
               Q: T.Tensor((batch, KEY_HEADS, time, DIM), qk_output_dtype),
               K: T.Tensor((batch, KEY_HEADS, time, DIM), qk_output_dtype),
               V: T.Tensor((batch, VALUE_HEADS, time, DIM), T.float16),
               HO: T.Tensor((batch, 3, CHANNELS), T.float16),
               positions_out: T.Tensor((batch,), T.int32)):
        with T.Kernel(T.ceildiv(time, tile_tokens), 80, batch, threads=128) as (bt, head, b):
            acc = T.alloc_fragment((tile_tokens, DIM), T.float32)
            raw = T.alloc_fragment((tile_tokens, DIM), T.float32)
            activated = T.alloc_fragment((tile_tokens, DIM), T.float16)
            value = T.alloc_fragment((tile_tokens, DIM), T.float32)
            square = T.alloc_fragment((tile_tokens, DIM), T.float32)
            total = T.alloc_fragment((tile_tokens,), T.float32)
            normalized = T.alloc_fragment((tile_tokens, DIM), T.float16)
            product = T.alloc_fragment((tile_tokens, DIM), T.float16)
            T.clear(acc)
            for tap in T.unroll(4):
                for ti, d in T.Parallel(tile_tokens, DIM):
                    raw[ti, d] = 0.0
                    if bt * tile_tokens + ti < lengths[b]:
                        if bt * tile_tokens + ti + tap >= 3:
                            raw[ti, d] = T.cast(X[b, bt * tile_tokens + ti + tap - 3, head * DIM + d], T.float32)
                        elif positions[b] + bt * tile_tokens + ti + tap >= 3:
                            raw[ti, d] = T.cast(HI[b, bt * tile_tokens + ti + tap, head * DIM + d], T.float32)
                    if conv_product_round_fp16:
                        product[ti, d] = raw[ti, d] * T.cast(W[head * DIM + d, tap], T.float32)
                        acc[ti, d] += T.cast(product[ti, d], T.float32)
                    else:
                        acc[ti, d] += raw[ti, d] * T.cast(W[head * DIM + d, tap], T.float32)
            for ti, d in T.Parallel(tile_tokens, DIM):
                activated[ti, d] = acc[ti, d] / (1.0 + T.exp(-acc[ti, d]))
                value[ti, d] = T.cast(activated[ti, d], T.float32)
                square[ti, d] = value[ti, d] * value[ti, d]
            if head < 32:
                T.reduce_sum(square, total, dim=1)
                for ti, d in T.Parallel(tile_tokens, DIM):
                    if normalize_round_fp16:
                        normalized[ti, d] = value[ti, d] * T.rsqrt(total[ti] + 1e-6)
                        value[ti, d] = T.cast(normalized[ti, d], T.float32)
                    else:
                        value[ti, d] *= T.rsqrt(total[ti] + 1e-6)
                    if bt * tile_tokens + ti < time:
                        if head < 16:
                            Q[b, head, bt * tile_tokens + ti, d] = value[ti, d] * q_scale
                        else:
                            K[b, head - 16, bt * tile_tokens + ti, d] = value[ti, d]
            else:
                for ti, d in T.Parallel(tile_tokens, DIM):
                    if bt * tile_tokens + ti < time:
                        V[b, head - 32, bt * tile_tokens + ti, d] = activated[ti, d]
            # Only the first time-block writes each channel's final raw history.
            # Q/K are represented once, so shared value heads never race here.
            if bt == 0:
                for i, d in T.Parallel(3, DIM):
                    if lengths[b] == 0:
                        HO[b, i, head * DIM + d] = HI[b, i, head * DIM + d]
                    elif lengths[b] + i >= 3:
                        HO[b, i, head * DIM + d] = X[b, lengths[b] + i - 3, head * DIM + d]
                    elif positions[b] + lengths[b] + i >= 3:
                        HO[b, i, head * DIM + d] = HI[b, lengths[b] + i, head * DIM + d]
                    else:
                        HO[b, i, head * DIM + d] = 0.0
                if head == 0:
                    positions_out[b] = positions[b] + lengths[b]
    return kernel


def launch(kernel, X, W, HI, lengths, positions, Q, K, V, HO, positions_out,
           stream=None):
    """Explicit-buffer launch. Resolve the current capture stream at every call."""
    return kernel(X, W, HI, lengths, positions, Q, K, V, HO, positions_out,
                  stream=stream)


def gdn_conv_decode(B: int | None = None, **rounding_and_scale):
    """One token per request, one-time-row tile, same explicit tensor API."""
    return gdn_conv_prep(B=B, tile_tokens=1, **rounding_and_scale)
