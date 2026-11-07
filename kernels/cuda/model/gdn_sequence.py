"""Causal small-sequence GDN with exact prefix snapshots for speculation.

Each CTA keeps its initial FP32 state in registers while processing a chunk.
Verification saves either full per-position states or compact rank-one updates;
the caller commits only the accepted prefix, including the target token.
In-place prefill writes only the final state and supports chunks through 4096.
Convolution history, position, and attention KV require separate handling.
"""

import tilelang.language as T
from orinfer_build.compiler.cuda.common import orin_jit
from kernels.shared.rows import row_count
from kernels.cuda.model.gdn_core import state_layout, predict, update_readout, rank_one

HK, HV, DK, DV = 16, 48, 128, 128


def _gdn_sequence(
    tokens: int,
    q_scale: float = 128**-0.5,
    value_tile: int = 32,
    threads: int = 128,
    in_place: bool = False,
    compact: bool = False,
    prefetch_qk: bool = False,
    dynamic_rows: bool = False,
    token_unroll: int = 1,
    key_heads: int = 16,
    value_heads: int = 48,
    shared_state: bool = False,
    initial_state=None,
):
    HK, HV = key_heads, value_heads
    if min(HK, HV) <= 0 or HV % HK:
        raise ValueError("Invalid GDN head geometry")
    assert 1 <= tokens <= (4096 if in_place else 16) and q_scale > 0
    assert value_tile in (16, 32, 64, 128) and threads in (32, 64, 128, 256)
    assert not (compact and in_place)
    if type(shared_state) is not bool or (shared_state and not in_place):
        raise ValueError("State staging requires an in-place GDN recurrence")
    if type(token_unroll) is not int or token_unroll not in (1, 2, 4, 8):
        raise ValueError("Invalid GDN token unrolling")
    if token_unroll != 1 and not prefetch_qk:
        raise ValueError("GDN token unrolling requires the prefetch recurrence")
    prediction = predict(value_tile, prefetched=prefetch_qk)
    advance = update_readout(value_tile, q_scale, prefetched=prefetch_qk, saved_delta=compact)
    tokens = count = row_count(tokens, dynamic_rows)
    if (
        type(prefetch_qk) is not bool
        or prefetch_qk
        and (not in_place or compact or threads != 32 or value_tile != 16)
    ):
        raise ValueError("Q/K prefetch requires a warp-owned in-place GDN tile")

    @T.prim_func
    def main(
        Q: T.Tensor((HK, tokens, DK), T.float16),
        K: T.Tensor((HK, tokens, DK), T.float16),
        V: T.Tensor((HV, tokens, DV), T.float16),
        G: T.Tensor((tokens, HV), T.float32),
        Beta: T.Tensor((tokens, HV), T.float32),
        State: T.Tensor((HV, DK, DV), T.float32),
        Prefix: T.Tensor(
            (tokens, HV, DV) if compact else (1 if in_place else tokens, HV, DK, DV), T.float32
        ),
        Out: T.Tensor((tokens, HV * DV), T.float16),
    ):
        with T.Kernel(DV // value_tile, HV, threads=threads) as (bv, h):
            kh = h // (HV // HK)
            if prefetch_qk:
                tid = T.get_thread_binding()
                qs = T.alloc_shared((2, DK), T.float16)
                ks = T.alloc_shared((2, DK), T.float16)
                vs = T.alloc_shared((2, value_tile), T.float16)
                gs = T.alloc_shared((2,), T.float32)
                bs = T.alloc_shared((2,), T.float32)
            state = T.alloc_fragment((DK, value_tile), T.float32)
            product = T.alloc_fragment((DK, value_tile), T.float32)
            if shared_state:
                state_tile = T.alloc_shared((DK, value_tile), T.float32)
            T.annotate_layout(
                {buffer: state_layout(value_tile, threads) for buffer in (state, product)}
            )
            predicted = T.alloc_fragment((value_tile,), T.float32)
            result = T.alloc_fragment((value_tile,), T.float32)
            update = T.alloc_fragment((value_tile,), T.float32)
            if initial_state is not None:
                initial_state(state, State, h, bv)
            elif shared_state:
                T.copy(State[h, 0:DK, bv * value_tile : (bv + 1) * value_tile], state_tile)
                T.copy(state_tile, state)
            else:
                for i, j in T.Parallel(DK, value_tile):
                    state[i, j] = State[h, i, bv * value_tile + j]
            if prefetch_qk:
                T.ptx_cp_async(
                    T.access_ptr(qs[0, tid * 4], "w", 4),
                    T.access_ptr(Q[kh, 0, tid * 4], "r", 4),
                    4,
                )
                T.ptx_cp_async(
                    T.access_ptr(ks[0, tid * 4], "w", 4),
                    T.access_ptr(K[kh, 0, tid * 4], "r", 4),
                    4,
                )
                if tid < value_tile // 4:
                    T.ptx_cp_async(
                        T.access_ptr(vs[0, tid * 4], "w", 4),
                        T.access_ptr(V[h, 0, bv * value_tile + tid * 4], "r", 4),
                        4,
                    )
                if tid == 0:
                    T.ptx_cp_async(T.access_ptr(gs[0], "w", 1), T.access_ptr(G[0, h], "r", 1), 1)
                    T.ptx_cp_async(T.access_ptr(bs[0], "w", 1), T.access_ptr(Beta[0, h], "r", 1), 1)
                T.ptx_commit_group()
            # Static token positions remove repeated double-buffer indexing
            # while retaining the original sequential FP32 recurrence.
            for group in T.serial(T.ceildiv(count, token_unroll)):
                for sub in T.unroll(token_unroll):
                    t = group * token_unroll + sub
                    if t < count:
                        if prefetch_qk:
                            T.ptx_wait_group(0)
                            T.sync_warp()
                            if t + 1 < count:
                                T.ptx_cp_async(
                                    T.access_ptr(qs[(t + 1) % 2, tid * 4], "w", 4),
                                    T.access_ptr(Q[kh, t + 1, tid * 4], "r", 4),
                                    4,
                                )
                                T.ptx_cp_async(
                                    T.access_ptr(ks[(t + 1) % 2, tid * 4], "w", 4),
                                    T.access_ptr(K[kh, t + 1, tid * 4], "r", 4),
                                    4,
                                )
                                if tid < value_tile // 4:
                                    T.ptx_cp_async(
                                        T.access_ptr(vs[(t + 1) % 2, tid * 4], "w", 4),
                                        T.access_ptr(
                                            V[h, t + 1, bv * value_tile + tid * 4], "r", 4
                                        ),
                                        4,
                                    )
                                if tid == 0:
                                    T.ptx_cp_async(
                                        T.access_ptr(gs[(t + 1) % 2], "w", 1),
                                        T.access_ptr(G[t + 1, h], "r", 1),
                                        1,
                                    )
                                    T.ptx_cp_async(
                                        T.access_ptr(bs[(t + 1) % 2], "w", 1),
                                        T.access_ptr(Beta[t + 1, h], "r", 1),
                                        1,
                                    )
                                T.ptx_commit_group()
                        prediction(
                            state,
                            product,
                            predicted,
                            ks if prefetch_qk else K,
                            gs if prefetch_qk else G,
                            kh,
                            t,
                            h,
                        )
                        if compact:
                            for j in T.Parallel(value_tile):
                                update[j] = Beta[t, h] * (
                                    T.cast(V[h, t, bv * value_tile + j], T.float32) - predicted[j]
                                )
                                Prefix[t, h, bv * value_tile + j] = update[j]
                        advance(
                            state,
                            product,
                            result,
                            qs if prefetch_qk else Q,
                            ks if prefetch_qk else K,
                            vs if prefetch_qk else V,
                            bs if prefetch_qk else Beta,
                            predicted,
                            update,
                            kh,
                            t,
                            h,
                            bv,
                        )
                        for i, j in T.Parallel(DK, value_tile):
                            if in_place:
                                if t == count - 1:
                                    if shared_state:
                                        state_tile[i, j] = state[i, j]
                                    else:
                                        State[h, i, bv * value_tile + j] = state[i, j]
                            elif not compact:
                                Prefix[t, h, i, bv * value_tile + j] = state[i, j]
                        for j in T.Parallel(value_tile):
                            Out[t, h * DV + bv * value_tile + j] = result[j]
            if shared_state:
                T.copy(state_tile, State[h, 0:DK, bv * value_tile : (bv + 1) * value_tile])

    return main


_compile_sequence = orin_jit(_gdn_sequence)
_compile_prefetch = orin_jit(
    _gdn_sequence,
    pass_configs={
        "tl.disable_thread_storage_sync": True,
        "tl.disable_shared_memory_reuse": True,
    },
)


def gdn_sequence(
    tokens: int,
    q_scale: float = 128**-0.5,
    value_tile: int = 32,
    threads: int = 128,
    in_place: bool = False,
    compact: bool = False,
    prefetch_qk: bool = False,
    dynamic_rows: bool = False,
    token_unroll: int = 1,
    key_heads: int = 16,
    value_heads: int = 48,
):
    """Preserve the recurrence and reductions while overlapping input reads.

    The optional prefill path has one warp per CTA and two private shared
    slots. Its explicit wait and warp barrier protect every slot reuse.
    """
    compiler = _compile_prefetch if prefetch_qk else _compile_sequence
    return compiler(
        tokens,
        q_scale,
        value_tile,
        threads,
        in_place,
        compact,
        prefetch_qk,
        dynamic_rows,
        token_unroll,
        key_heads,
        value_heads,
    )


@orin_jit
def gdn_commit(
    tokens: int,
    value_tile: int = 32,
    dynamic_rows: bool = False,
    key_heads: int = 16,
    value_heads: int = 48,
):
    """Replay saved FP32 rank-one updates in their original rounding order."""
    HK, HV = key_heads, value_heads
    if min(HK, HV) <= 0 or HV % HK:
        raise ValueError("Invalid GDN head geometry")
    assert 1 <= tokens <= 8 and value_tile in (16, 32, 64, 128)

    replay = rank_one(value_tile)
    tokens = row_count(tokens, dynamic_rows)

    @T.prim_func
    def main(
        K: T.Tensor((HK, tokens, DK), T.float16),
        G: T.Tensor((tokens, HV), T.float32),
        Update: T.Tensor((tokens, HV, DV), T.float32),
        Accepted: T.Tensor((1,), T.int32),
        State: T.Tensor((HV, DK, DV), T.float32),
    ):
        with T.Kernel(DV // value_tile, HV, threads=128) as (bv, h):
            state = T.alloc_fragment((DK, value_tile), T.float32)
            for i, j in T.Parallel(DK, value_tile):
                state[i, j] = State[h, i, bv * value_tile + j]
            for t in T.serial(tokens):
                if t < Accepted[0]:
                    replay(state, K, Update, T.exp(G[t, h]), h // (HV // HK), t, h, bv)
            for i, j in T.Parallel(DK, value_tile):
                State[h, i, bv * value_tile + j] = state[i, j]

    return main


@orin_jit
def gdn_commit_controlled(tokens=8, value_tile=16, key_heads=16, value_heads=48):
    """Same FP32 commit with a device-owned compact key stride per request.

    The fixed tensor extent is allocation capacity. Length supplies the actual
    saved key stride, allowing the independent-owner wrapper to join different
    verification widths without specializing their computation.
    """
    HK, HV = key_heads, value_heads
    if not 1 <= tokens <= 8 or HK <= 0 or HV % HK:
        raise ValueError("Invalid controlled GDN commit geometry")
    replay = rank_one(value_tile, staged=True)

    @T.prim_func
    def main(
        K: T.Tensor((HK * tokens * DK,), T.float16),
        G: T.Tensor((tokens, HV), T.float32),
        Update: T.Tensor((tokens, HV, DV), T.float32),
        Accepted: T.Tensor((1,), T.int32),
        Length: T.Tensor((1,), T.int32),
        State: T.Tensor((HV, DK, DV), T.float32),
    ):
        with T.Kernel(DV // value_tile, HV, threads=128) as (bv, h):
            keys = T.alloc_shared((tokens, DK), T.float16)
            for t, i in T.Parallel(tokens, DK):
                if t < Length[0]:
                    keys[t, i] = K[(h // (HV // HK) * Length[0] + t) * DK + i]
            T.sync_threads()
            state = T.alloc_fragment((DK, value_tile), T.float32)
            delta = T.alloc_fragment((value_tile,), T.float32)
            for i, j in T.Parallel(DK, value_tile):
                state[i, j] = State[h, i, bv * value_tile + j]
            for t in T.serial(tokens):
                if t < Accepted[0]:
                    for j in T.Parallel(value_tile):
                        delta[j] = Update[t, h, bv * value_tile + j]
                    replay(state, keys, delta, T.exp(G[t, h]), 0, t, h, bv)
            for i, j in T.Parallel(DK, value_tile):
                State[h, i, bv * value_tile + j] = state[i, j]

    return main
