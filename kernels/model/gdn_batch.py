"""M1 mixers over private request arenas, with a stable address-table ABI.

Pointers[128,3] contains FP32 state, FP16 chronological history and int32
position addresses. A null state marks padding. The caller retains each arena
and uploads the table before replay on the same ordered execution stream.
"""

import tilelang.language as T
from tools.operators.common import orin_jit

HK, HV, DK, DV, CHANNELS = 16, 48, 128, 128, 10240
POINTER_SOURCE = r"""
__device__ __forceinline__ void* orin_arena_pointer(unsigned long long address) {
    return reinterpret_cast<void*>(address);
}
"""


@orin_jit
def batch_gdn_recurrent(rows: int, value_tile: int = 32):
    assert 1 <= rows <= 128 and value_tile in (16, 32, 64, 128)

    @T.prim_func
    def kernel(
        Pointers: T.Tensor((128, 3), "uint64"),
        Q: T.Tensor((rows, HK, DK), "float16"),
        K: T.Tensor((rows, HK, DK), "float16"),
        V: T.Tensor((rows, HV, DV), "float16"),
        G: T.Tensor((rows, HV), "float32"),
        Beta: T.Tensor((rows, HV), "float32"),
        Out: T.Tensor((rows, HV, DV), "float16"),
    ):
        with T.Kernel(DV // value_tile, HV, rows, threads=128) as (bv, h, b):
            T.import_source(POINTER_SOURCE)
            if Pointers[b, 0] != 0:
                pointer = T.bind(
                    T.call_extern("handle", "orin_arena_pointer", Pointers[b, 0]),
                    var=T.ptr("float32"),
                )
                State = T.decl_buffer((HV, DK, DV), "float32", data=pointer)
                state = T.alloc_fragment((DK, value_tile), "float32")
                product = T.alloc_fragment((DK, value_tile), "float32")
                predicted = T.alloc_fragment((value_tile,), "float32")
                result = T.alloc_fragment((value_tile,), "float32")
                for i, j in T.Parallel(DK, value_tile):
                    state[i, j] = State[h, i, bv * value_tile + j] * T.exp(G[b, h])
                    product[i, j] = state[i, j] * T.cast(K[b, h // 3, i], "float32")
                T.reduce_sum(product, predicted, dim=0)
                for i, j in T.Parallel(DK, value_tile):
                    delta = Beta[b, h] * (
                        T.cast(V[b, h, bv * value_tile + j], "float32") - predicted[j]
                    )
                    state[i, j] += T.cast(K[b, h // 3, i], "float32") * delta
                    product[i, j] = state[i, j] * (T.cast(Q[b, h // 3, i], "float32") * DK**-0.5)
                T.reduce_sum(product, result, dim=0)
                for i, j in T.Parallel(DK, value_tile):
                    State[h, i, bv * value_tile + j] = state[i, j]
                for j in T.Parallel(value_tile):
                    Out[b, h, bv * value_tile + j] = T.cast(result[j], "float16")

    return kernel


@orin_jit
def batch_gdn_conv(rows: int):
    assert 1 <= rows <= 128

    @T.prim_func
    def kernel(
        Pointers: T.Tensor((128, 3), "uint64"),
        X: T.Tensor((rows, CHANNELS), "float16"),
        W: T.Tensor((CHANNELS, 4), "float16"),
        Q: T.Tensor((rows, HK, DK), "float16"),
        K: T.Tensor((rows, HK, DK), "float16"),
        V: T.Tensor((rows, HV, DV), "float16"),
    ):
        with T.Kernel(80, rows, threads=128) as (head, b):
            T.import_source(POINTER_SOURCE)
            if Pointers[b, 0] != 0:
                hp = T.bind(
                    T.call_extern("handle", "orin_arena_pointer", Pointers[b, 1]),
                    var=T.ptr("float16"),
                )
                sp = T.bind(
                    T.call_extern("handle", "orin_arena_pointer", Pointers[b, 2]),
                    var=T.ptr("int32"),
                )
                History = T.decl_buffer((3, CHANNELS), "float16", data=hp)
                Step = T.decl_buffer((1,), "int32", data=sp)
                # Each CTA owns one head's channels. Read all three taps before
                # writing chronological history; no other CTA reads this slice.
                h0 = T.alloc_fragment((DK,), "float16")
                h1 = T.alloc_fragment((DK,), "float16")
                h2 = T.alloc_fragment((DK,), "float16")
                acc = T.alloc_fragment((DK,), "float32")
                product = T.alloc_fragment((DK,), "float16")
                activated = T.alloc_fragment((DK,), "float16")
                values = T.alloc_fragment((DK,), "float32")
                squares = T.alloc_fragment((DK,), "float32")
                total = T.alloc_fragment((1,), "float32")
                for d in T.Parallel(DK):
                    h0[d] = T.if_then_else(Step[0] >= 3, History[0, head * DK + d], 0)
                    h1[d] = T.if_then_else(Step[0] >= 2, History[1, head * DK + d], 0)
                    h2[d] = T.if_then_else(Step[0] >= 1, History[2, head * DK + d], 0)
                T.clear(acc)
                for tap in T.unroll(4):
                    for d in T.Parallel(DK):
                        if tap == 0:
                            product[d] = T.cast(h0[d], "float32") * T.cast(
                                W[head * DK + d, tap], "float32"
                            )
                        elif tap == 1:
                            product[d] = T.cast(h1[d], "float32") * T.cast(
                                W[head * DK + d, tap], "float32"
                            )
                        elif tap == 2:
                            product[d] = T.cast(h2[d], "float32") * T.cast(
                                W[head * DK + d, tap], "float32"
                            )
                        else:
                            product[d] = T.cast(X[b, head * DK + d], "float32") * T.cast(
                                W[head * DK + d, tap], "float32"
                            )
                        acc[d] += T.cast(product[d], "float32")
                for d in T.Parallel(DK):
                    activated[d] = acc[d] / (1.0 + T.exp(-acc[d]))
                    values[d] = T.cast(activated[d], "float32")
                    squares[d] = values[d] * values[d]
                if head < 32:
                    T.reduce_sum(squares, total, dim=0)
                    for d in T.Parallel(DK):
                        if head < HK:
                            Q[b, head, d] = values[d] * T.rsqrt(total[0] + 1e-6)
                        else:
                            K[b, head - HK, d] = values[d] * T.rsqrt(total[0] + 1e-6)
                else:
                    for d in T.Parallel(DK):
                        V[b, head - 32, d] = activated[d]
                for d in T.Parallel(DK):
                    History[0, head * DK + d] = h1[d]
                for d in T.Parallel(DK):
                    History[1, head * DK + d] = h2[d]
                for d in T.Parallel(DK):
                    History[2, head * DK + d] = X[b, head * DK + d]

    return kernel
