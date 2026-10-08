"""Owned-state M1 recurrence, one pointer for FP32 state reads/writes.

Each CTA owns all K rows of distinct V columns in a single head. It loads
its complete owned slice before storing it; no other CTA reads that slice.
Other inputs/output must be disjoint from State. Stream/program ordering
separates steps and consumers. The immutable op10 API remains unchanged.
"""

import tilelang
import tilelang.language as T

HK, HV, DK, DV = 16, 48, 128, 128


@tilelang.jit(out_idx=[], execution_backend="nvrtc", target={"kind": "cuda", "arch": "sm_87"})
def _compile_inplace(
    q_scale: float,
    qk_dtype: str = "float16",
    output_dtype: str = "float16",
    value_tile: int = 16,
    threads: int = 128,
    beta_round_fp16: bool = False,
):
    assert qk_dtype in ("float16", "float32")
    assert output_dtype in ("float16", "float32")
    assert value_tile in (16, 32, 64, 128) and DV % value_tile == 0
    assert threads in (64, 128, 256)
    assert q_scale > 0
    batch = T.dynamic("batch")

    @T.prim_func
    def main(
        Q: T.Tensor((batch, HK, DK), qk_dtype),
        K: T.Tensor((batch, HK, DK), qk_dtype),
        V: T.Tensor((batch, HV, DV), "float16"),
        G: T.Tensor((batch, HV), "float32"),
        Beta: T.Tensor((batch, HV), "float32"),
        State: T.Tensor((batch, HV, DK, DV), "float32"),
        Out: T.Tensor((batch, HV, DV), output_dtype),
    ):
        with T.Kernel(DV // value_tile, batch * HV, threads=threads) as (bv, bh):
            b, h, kh = bh // HV, bh % HV, (bh % HV) // 3
            state = T.alloc_fragment((DK, value_tile), "float32")
            product = T.alloc_fragment((DK, value_tile), "float32")
            predicted = T.alloc_fragment((value_tile,), "float32")
            result = T.alloc_fragment((value_tile,), "float32")
            for i, j in T.Parallel(DK, value_tile):
                state[i, j] = State[b, h, i, bv * value_tile + j] * T.exp(G[b, h])
                product[i, j] = state[i, j] * T.cast(K[b, kh, i], "float32")
            T.reduce_sum(product, predicted, dim=0)
            for i, j in T.Parallel(DK, value_tile):
                if beta_round_fp16:
                    beta = T.cast(T.cast(Beta[b, h], "float16"), "float32")
                else:
                    beta = Beta[b, h]
                delta = beta * (T.cast(V[b, h, bv * value_tile + j], "float32") - predicted[j])
                state[i, j] = state[i, j] + T.cast(K[b, kh, i], "float32") * delta
                product[i, j] = state[i, j] * (T.cast(Q[b, kh, i], "float32") * q_scale)
            T.reduce_sum(product, result, dim=0)
            for i, j in T.Parallel(DK, value_tile):
                State[b, h, i, bv * value_tile + j] = state[i, j]
            for j in T.Parallel(value_tile):
                Out[b, h, bv * value_tile + j] = T.cast(result[j], output_dtype)

    return main


def gdn_recurrent_inplace(*, q_scale, value_tile=32, threads=128):
    kernel = _compile_inplace(q_scale, value_tile=value_tile, threads=threads)
    kernel.adapter.kernels = dict(kernel.adapter.kernels)
    return kernel
