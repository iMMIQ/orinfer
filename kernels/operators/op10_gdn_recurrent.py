"""SM87 GDN one-token recurrence, explicit immutable input and output state.

Q/K [B,16,128] normalized FP16 or FP32; V [B,48,128] FP16;
g/beta [B,48] FP32; state [B,48,128,128] FP32 in [K,V] order.
q_scale is mandatory: 1/sqrt(128) for unscaled Q, 1 for already-scaled Q.
All arithmetic is FP32, beta can explicitly round through native FP16.
Each CTA owns complete K and disjoint V columns. No cross-CTA norm fusion.
"""
import tilelang
import tilelang.language as T

HK, HV, DK, DV = 16, 48, 128, 128


@tilelang.jit(out_idx=[], execution_backend="nvrtc",
              target={"kind": "cuda", "arch": "sm_87"})
def _compile_recurrent(q_scale: float, qk_dtype: str = "float16",
                       output_dtype: str = "float16", value_tile: int = 16,
                       threads: int = 128, beta_round_fp16: bool = False):
    assert qk_dtype in ("float16", "float32")
    assert output_dtype in ("float16", "float32")
    assert value_tile in (16, 32, 64, 128) and DV % value_tile == 0
    assert threads in (64, 128, 256)
    assert q_scale > 0
    batch = T.dynamic("batch")

    @T.prim_func
    def main(Q: T.Tensor((batch, HK, DK), qk_dtype),
             K: T.Tensor((batch, HK, DK), qk_dtype),
             V: T.Tensor((batch, HV, DV), "float16"),
             G: T.Tensor((batch, HV), "float32"),
             Beta: T.Tensor((batch, HV), "float32"),
             StateIn: T.Tensor((batch, HV, DK, DV), "float32"),
             StateOut: T.Tensor((batch, HV, DK, DV), "float32"),
             Out: T.Tensor((batch, HV, DV), output_dtype)):
        with T.Kernel(DV // value_tile, batch * HV, threads=threads) as (bv, bh):
            b, h, kh = bh // HV, bh % HV, (bh % HV) // 3
            state = T.alloc_fragment((DK, value_tile), "float32")
            product = T.alloc_fragment((DK, value_tile), "float32")
            predicted = T.alloc_fragment((value_tile,), "float32")
            result = T.alloc_fragment((value_tile,), "float32")
            for i, j in T.Parallel(DK, value_tile):
                state[i, j] = StateIn[b, h, i, bv * value_tile + j] * T.exp(G[b, h])
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
                StateOut[b, h, i, bv * value_tile + j] = state[i, j]
            for j in T.Parallel(value_tile):
                Out[b, h, bv * value_tile + j] = T.cast(result[j], output_dtype)
    return main


def gdn_recurrent(*, q_scale, qk_dtype="float16", output_dtype="float16",
                  value_tile=16, threads=128, beta_round_fp16=False):
    """Compile a dynamic-B kernel; all buffers contiguous and preallocated.

    Sin/Sout must be distinct in the supported immutable API. Other buffers
    must not alias. Output state remains FP32, output values round once.
    """
    kernel = _compile_recurrent(q_scale, qk_dtype, output_dtype, value_tile,
                                threads, beta_round_fp16)
    kernel.adapter.kernels = dict(kernel.adapter.kernels)
    return kernel


def launch(kernel, q, k, v, g, beta, state_in, state_out, output, *, stream):
    """Launch on explicit current caller/capture stream, stable buffer addresses."""
    return kernel.adapter.func(q, k, v, g, beta, state_in, state_out, output, stream=stream)
