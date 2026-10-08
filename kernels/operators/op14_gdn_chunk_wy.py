"""FP32 SIMT GDN WY transform on SM87; no Torch or allocations.

K shares sixteen heads across 48 value heads via kh=vh//3. Each CTA owns
one [BT,32] output tile. Products, A and accumulators remain FP32: no TF32,
tensorcore operand casts, q_scale or materialized repeated K.
"""

import tilelang
import tilelang.language as T

HK, HV, D = 16, 48, 128


@tilelang.jit(out_idx=[], execution_backend="nvrtc", target={"kind": "cuda", "arch": "sm_87"})
def _compile(bt: int, k_dtype: str, value_tile: int):
    batch, chunks = T.dynamic("batch"), T.dynamic("chunks")

    @T.prim_func
    def main(
        A: T.Tensor((batch, HV, chunks, bt, bt), "float32"),
        K: T.Tensor((batch, HK, chunks, bt, D), k_dtype),
        V: T.Tensor((batch, HV, chunks, bt, D), "float16"),
        G: T.Tensor((batch, HV, chunks, bt), "float32"),
        Beta: T.Tensor((batch, HV, chunks, bt), "float32"),
        W: T.Tensor((batch, HV, chunks, bt, D), "float32"),
        U: T.Tensor((batch, HV, chunks, bt, D), "float32"),
    ):
        with T.Kernel(D // value_tile, HV * chunks, batch, threads=128) as (tile, hc, b):
            h, c = hc // chunks, hc % chunks
            a = T.alloc_shared((bt, bt), "float32")
            bk = T.alloc_shared((bt, value_tile), "float32")
            bv = T.alloc_shared((bt, value_tile), "float32")
            w = T.alloc_fragment((bt, value_tile), "float32")
            u = T.alloc_fragment((bt, value_tile), "float32")
            for i, j in T.Parallel(bt, bt):
                if j <= i:
                    a[i, j] = A[b, h, c, i, j]
                else:
                    a[i, j] = 0.0
            for i, d in T.Parallel(bt, value_tile):
                if Beta[b, h, c, i] != 0.0:
                    bk[i, d] = (
                        Beta[b, h, c, i]
                        * T.cast(K[b, h // 3, c, i, tile * value_tile + d], "float32")
                    ) * T.exp(G[b, h, c, i])
                    bv[i, d] = Beta[b, h, c, i] * T.cast(
                        V[b, h, c, i, tile * value_tile + d], "float32"
                    )
                else:
                    # Zero beta suppresses reads of padded K/V/G, including NaN.
                    bk[i, d] = 0.0
                    bv[i, d] = 0.0
            T.sync_threads()
            T.clear(w)
            T.clear(u)
            for j in T.serial(bt):
                for i, d in T.Parallel(bt, value_tile):
                    if j <= i:
                        w[i, d] += a[i, j] * bk[j, d]
                        u[i, d] += a[i, j] * bv[j, d]
            for i, d in T.Parallel(bt, value_tile):
                W[b, h, c, i, tile * value_tile + d] = w[i, d]
                U[b, h, c, i, tile * value_tile + d] = u[i, d]

    return main


def gdn_chunk_wy(bt=64, k_dtype="float16", value_tile=32):
    """Build dynamic-B/C W=A@(beta*K*exp(G)), U=A@(beta*V).

    Contiguous row-major, disjoint caller-owned buffers; A is FP32 unit-lower
    inverse, G cumulative FP32, beta FP32, V FP16, outputs FP32. Upper A is
    ignored. Finite lower A is required, including identity padded rows.
    Exact beta=0 skips K/V/G reads; otherwise they must be finite. Tail beta=0
    with identity A padding gives exact zero W/U. No workspace or q scaling.
    """
    if type(bt) is not int or bt not in (16, 32, 64):
        raise ValueError("BT must be 16, 32 or 64")
    if k_dtype not in ("float16", "float32"):
        raise ValueError("K dtype must be float16 or float32")
    if type(value_tile) is not int or value_tile not in (16, 32, 64):
        raise ValueError("value_tile must be 16, 32 or 64")
    kernel = _compile(bt, k_dtype, value_tile)
    kernel.adapter.kernels = dict(kernel.adapter.kernels)
    return kernel


def launch(kernel, a, k, v, cumulative_g, beta, w, u, *, stream):
    """Explicit caller/capture stream and preallocated stable output addresses."""
    return kernel.adapter.func(a, k, v, cumulative_g, beta, w, u, stream=stream)
