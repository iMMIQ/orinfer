"""Bounded GDN logs with the original sequential FP32 recurrence.

Only dense writeback is deferred. Every step replays pending FP32 rank-one
updates in order, including the original rounding and reduction boundaries.
Request snapshots must preserve the base, keys, updates, gates and counters.
"""

import tilelang.language as T
from kernels.cuda.model.gdn_core import state_layout, predict, update_readout, rank_one


def validate_geometry(capacity, hk, hv, vt, threads):
    if (
        capacity not in (2, 4, 8, 16, 32)
        or min(hk, hv) <= 0
        or hv % hk
        or vt not in (16, 32, 64)
        or threads not in (64, 128, 256)
        or vt * 16 % threads
    ):
        raise ValueError("Invalid deferred GDN geometry")


def pending_updates(capacity, hk, hv, vt, threads):
    update = rank_one(vt, staged=True)

    @T.macro
    def apply(state, LogK, LogD, LogG, phase, h, bv):
        replay_k = T.alloc_shared((capacity, 128), T.float16)
        replay_d = T.alloc_shared((capacity, vt), T.float32)
        replay_g = T.alloc_shared((capacity,), T.float32)
        previous_delta = T.alloc_fragment((vt,), T.float32)
        decay = T.alloc_local((1,), T.float32)
        # Only the committed prefix is replayed. Initializing unused slots
        # adds shared-memory barriers without making any read valid.
        for s, i in T.Parallel(capacity, 128):
            if s < phase:
                replay_k[s, i] = LogK[s, h // (hv // hk), i]
        for s, j in T.Parallel(capacity, vt):
            if s < phase:
                replay_d[s, j] = LogD[s, h, bv * vt + j]
        for s in T.Parallel(capacity):
            if s < phase:
                replay_g[s] = LogG[s, h]
        T.sync_threads()
        for s in T.serial(phase):
            decay[0] = T.exp(replay_g[s])
            for j in T.Parallel(vt):
                previous_delta[j] = replay_d[s, j]
            update(state, replay_k, previous_delta, decay[0], 0, s, h, bv)

    return apply


def recurrent(capacity, *, key_heads=16, value_heads=48, value_tile=16, threads=64):
    HK, HV, DK, DV = key_heads, value_heads, 128, 128
    vt = value_tile
    validate_geometry(capacity, HK, HV, vt, threads)
    replay = pending_updates(capacity, HK, HV, vt, threads)
    prediction = predict(vt)
    advance = update_readout(vt, 128**-0.5, saved_delta=True)

    @T.prim_func
    def main(
        Q: T.Tensor((HK, 1, DK), T.float16),
        K: T.Tensor((HK, 1, DK), T.float16),
        V: T.Tensor((HV, 1, DV), T.float16),
        G: T.Tensor((1, HV), T.float32),
        Beta: T.Tensor((1, HV), T.float32),
        State: T.Tensor((HV, DK, DV), T.float32),
        LogK: T.Tensor((capacity, HK, DK), T.float16),
        LogD: T.Tensor((capacity, HV, DV), T.float32),
        LogG: T.Tensor((capacity, HV), T.float32),
        Count: T.Tensor((HV, DV), T.int32),
        Out: T.Tensor((1, HV * DV), T.float16),
    ):
        with T.Kernel(DV // vt, HV, threads=threads) as (bv, h):
            kh = h // (HV // HK)
            phase = T.alloc_local((1,), T.int32)
            phase[0] = Count[h, bv * vt]
            state = T.alloc_fragment((DK, vt), T.float32)
            product = T.alloc_fragment((DK, vt), T.float32)
            T.annotate_layout({buf: state_layout(vt, threads) for buf in (state, product)})
            predicted = T.alloc_fragment((vt,), T.float32)
            result = T.alloc_fragment((vt,), T.float32)
            delta = T.alloc_fragment((vt,), T.float32)
            # Preserve the reduction layout while loading directly into its
            # registers; staging the full state adds a copy and a CTA barrier.
            T.copy(State[h, 0:DK, bv * vt : (bv + 1) * vt], state)
            replay(state, LogK, LogD, LogG, phase[0], h, bv)
            prediction(state, product, predicted, K, G, kh, 0, h)
            for j in T.Parallel(vt):
                delta[j] = Beta[0, h] * (T.cast(V[h, 0, bv * vt + j], T.float32) - predicted[j])
                LogD[phase[0], h, bv * vt + j] = delta[j]
            advance(state, product, result, Q, K, V, Beta, predicted, delta, kh, 0, h, bv)
            for j in T.Parallel(vt):
                Out[0, h * DV + bv * vt + j] = result[j]
                Count[h, bv * vt + j] = T.if_then_else(phase[0] + 1 == capacity, 0, phase[0] + 1)
            if phase[0] + 1 == capacity:
                T.copy(state, State[h, 0:DK, bv * vt : (bv + 1) * vt])
            if bv == 0:
                if T.get_thread_binding() == 0:
                    LogG[phase[0], h] = G[0, h]
                if h % (HV // HK) == 0:
                    for i in T.Parallel(DK):
                        LogK[phase[0], kh, i] = K[kh, 0, i]

    return main


def materialize(capacity, *, key_heads=16, value_heads=48, value_tile=16, threads=64):
    """Canonicalize a request before dense prefill or speculative verification.

    Each CTA owns its counters. No other CTA reads these counters, so resetting
    them cannot race a neighbouring value tile's replay.
    """
    HK, HV, vt = key_heads, value_heads, value_tile
    validate_geometry(capacity, HK, HV, vt, threads)

    replay = pending_updates(capacity, HK, HV, vt, threads)

    @T.prim_func
    def main(
        State: T.Tensor((HV, 128, 128), T.float32),
        LogK: T.Tensor((capacity, HK, 128), T.float16),
        LogD: T.Tensor((capacity, HV, 128), T.float32),
        LogG: T.Tensor((capacity, HV), T.float32),
        Count: T.Tensor((HV, 128), T.int32),
    ):
        with T.Kernel(128 // vt, HV, threads=threads) as (bv, h):
            phase = T.alloc_local((1,), T.int32)
            phase[0] = Count[h, bv * vt]
            if phase[0] != 0:
                state = T.alloc_fragment((128, vt), T.float32)
                tile = T.alloc_shared((128, vt), T.float32)
                T.annotate_layout({state: state_layout(vt, threads)})
                T.copy(State[h, 0:128, bv * vt : (bv + 1) * vt], tile)
                T.copy(tile, state)
                replay(state, LogK, LogD, LogG, phase[0], h, bv)
                T.copy(state, tile)
                T.copy(tile, State[h, 0:128, bv * vt : (bv + 1) * vt])
                for j in T.Parallel(vt):
                    Count[h, bv * vt + j] = 0

    return main


def verification(capacity, *, key_heads=16, value_heads=48, tokens=8, value_tile=16, threads=32):
    """Verify from base plus pending updates without an intermediate writeback."""
    from tvm import tirx as tir
    from kernels.cuda.model.gdn_sequence import _gdn_sequence
    import inspect

    hk, hv, vt = key_heads, value_heads, value_tile
    logs = [
        tir.decl_buffer(shape, dtype, name=name)
        for name, shape, dtype in [
            ("LogK", (capacity, hk, 128), "float16"),
            ("LogD", (capacity, hv, 128), "float32"),
            ("LogG", (capacity, hv), "float32"),
            ("Count", (hv, 128), "int32"),
        ]
    ]
    logk, logd, logg, count = logs
    replay = pending_updates(capacity, hk, hv, vt, threads)

    @T.macro
    def initialize(state, Base, h, bv):
        for i, j in T.Parallel(128, vt):
            state[i, j] = Base[h, i, bv * vt + j]
        replay(state, logk, logd, logg, count[h, bv * vt], h, bv)

    fn = inspect.unwrap(_gdn_sequence)(
        tokens,
        value_tile=vt,
        threads=threads,
        compact=True,
        dynamic_rows=True,
        key_heads=hk,
        value_heads=hv,
        initial_state=initialize,
    )
    params = list(fn.params) + [b.data for b in logs]
    return tir.PrimFunc(
        params, fn.body, fn.ret_type, dict(fn.buffer_map) | {b.data: b for b in logs}, fn.attrs
    )


def commit(capacity, *, key_heads=16, value_heads=48, tokens=8, value_tile=16, threads=128):
    """Append an exact accepted prefix; canonicalize only when its log fills."""
    hk, hv, vt = key_heads, value_heads, value_tile
    validate_geometry(capacity, hk, hv, vt, threads)
    pending = pending_updates(capacity, hk, hv, vt, threads)
    advance = rank_one(vt, staged=True)

    @T.prim_func
    def main(
        K: T.Tensor((hk * tokens * 128,), T.float16),
        G: T.Tensor((tokens, hv), T.float32),
        Update: T.Tensor((tokens, hv, 128), T.float32),
        Accepted: T.Tensor((1,), T.int32),
        Length: T.Tensor((1,), T.int32),
        State: T.Tensor((hv, 128, 128), T.float32),
        LogK: T.Tensor((capacity, hk, 128), T.float16),
        LogD: T.Tensor((capacity, hv, 128), T.float32),
        LogG: T.Tensor((capacity, hv), T.float32),
        Count: T.Tensor((hv, 128), T.int32),
    ):
        with T.Kernel(128 // vt, hv, threads=threads) as (bv, h):
            phase = T.alloc_local((1,), T.int32)
            phase[0] = Count[h, bv * vt]
            keys = T.alloc_shared((tokens, 128), T.float16)
            for t, i in T.Parallel(tokens, 128):
                if t < Accepted[0]:
                    keys[t, i] = K[(h // (hv // hk) * Length[0] + t) * 128 + i]
            T.sync_threads()
            if phase[0] + Accepted[0] >= capacity:
                state = T.alloc_fragment((128, vt), T.float32)
                delta = T.alloc_fragment((vt,), T.float32)
                T.annotate_layout({state: state_layout(vt, threads)})
                for i, j in T.Parallel(128, vt):
                    state[i, j] = State[h, i, bv * vt + j]
                pending(state, LogK, LogD, LogG, phase[0], h, bv)
                for t in T.serial(tokens):
                    if t < Accepted[0]:
                        for j in T.Parallel(vt):
                            delta[j] = Update[t, h, bv * vt + j]
                        advance(state, keys, delta, T.exp(G[t, h]), 0, t, h, bv)
                for i, j in T.Parallel(128, vt):
                    State[h, i, bv * vt + j] = state[i, j]
                for j in T.Parallel(vt):
                    Count[h, bv * vt + j] = 0
            else:
                for t, j in T.Parallel(tokens, vt):
                    if t < Accepted[0]:
                        LogD[phase[0] + t, h, bv * vt + j] = Update[t, h, bv * vt + j]
                for j in T.Parallel(vt):
                    Count[h, bv * vt + j] = phase[0] + Accepted[0]
                if bv == 0:
                    for t in T.Parallel(tokens):
                        if t < Accepted[0]:
                            LogG[phase[0] + t, h] = G[t, h]
                    if h % (hv // hk) == 0:
                        for t, i in T.Parallel(tokens, 128):
                            if t < Accepted[0]:
                                LogK[phase[0] + t, h // (hv // hk), i] = keys[t, i]

    return main
