"""Compile-time GDN arithmetic shared by sequences, deferred decode and commit.

Schedulers own input staging, snapshots and dense writeback. These components
retain FP32 state, separate decay/update rounding and the sixteen-lane sum tree.
"""

import tilelang
import tilelang.language as T


def state_layout(value_tile, threads):
    values = value_tile * 16 // threads
    return tilelang.Fragment(
        (128, value_tile),
        forward_thread_fn=lambda i, j: (j // (2 * values)) * 32 + (i % 16) * 2 + (j // values) % 2,
        forward_index_fn=lambda i, j: (i // 16) * values + j % values,
    )


def predict(value_tile, *, prefetched=False):
    @T.macro
    def apply(state, product, predicted, Key, Gate, kh, t, h):
        for i, j in T.Parallel(128, value_tile):
            state[i, j] = state[i, j] * T.exp(Gate[t % 2] if prefetched else Gate[t, h])
            product[i, j] = state[i, j] * T.cast(
                Key[t % 2, i] if prefetched else Key[kh, t, i], T.float32
            )
        T.reduce_sum(product, predicted, dim=0)

    return apply


def update_readout(value_tile, q_scale, *, prefetched=False, saved_delta=False):
    @T.macro
    def apply(state, product, result, Query, Key, Value, Beta, predicted, update, kh, t, h, bv):
        for i, j in T.Parallel(128, value_tile):
            delta = (Beta[t % 2] if prefetched else Beta[t, h]) * (
                T.cast(
                    Value[t % 2, j] if prefetched else Value[h, t, bv * value_tile + j], T.float32
                )
                - predicted[j]
            )
            if saved_delta:
                delta = update[j]
            state[i, j] = (
                state[i, j]
                + T.cast(Key[t % 2, i] if prefetched else Key[kh, t, i], T.float32) * delta
            )
            product[i, j] = state[i, j] * (
                T.cast(Query[t % 2, i] if prefetched else Query[kh, t, i], T.float32) * q_scale
            )
        T.reduce_sum(product, result, dim=0)

    return apply


def rank_one(value_tile, *, staged=False):
    """Replay a saved update without changing its original FP32 association."""

    @T.macro
    def apply(state, Key, Delta, decay, kh, t, h, bv):
        for i, j in T.Parallel(128, value_tile):
            state[i, j] = state[i, j] * decay
            state[i, j] = state[i, j] + T.cast(
                Key[t, i] if staged else Key[kh, t, i], T.float32
            ) * (Delta[j] if staged else Delta[t, h, bv * value_tile + j])

    return apply
