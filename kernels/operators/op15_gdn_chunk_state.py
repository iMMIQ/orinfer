"""SM87 FP32 cross-chunk GDN scan; logical state order is [K,V].

One CTA owns a complete K dimension and disjoint V columns for one B/H.
Chunks are scanned sequentially inside that CTA. No mutable input, workspace,
head expansion, half state, tensor cores, TF32, or inter-CTA synchronization.
"""

import tilelang
import tilelang.language as T

HK, HV, DK, DV = 16, 48, 128, 128


@tilelang.jit(out_idx=[], execution_backend="nvrtc", target={"kind": "cuda", "arch": "sm_87"})
def _compile(bt: int = 64, value_tile: int = 16, threads: int = 128):
    assert bt in (16, 32, 64)
    assert value_tile in (8, 16, 32) and DV % value_tile == 0
    assert threads in (128, 256)
    batch, chunks = T.dynamic("batch"), T.dynamic("chunks")

    @T.prim_func
    def main(
        K: T.Tensor((batch, HK, chunks, bt, DK), "float16"),
        G: T.Tensor((batch, HV, chunks, bt), "float32"),
        W: T.Tensor((batch, HV, chunks, bt, DK), "float32"),
        U: T.Tensor((batch, HV, chunks, bt, DV), "float32"),
        Sin: T.Tensor((batch, HV, DK, DV), "float32"),
        Senter: T.Tensor((batch, HV, chunks, DK, DV), "float32"),
        R: T.Tensor((batch, HV, chunks, bt, DV), "float32"),
        Sfinal: T.Tensor((batch, HV, DK, DV), "float32"),
    ):
        with T.Kernel(DV // value_tile, batch * HV, threads=threads) as (bv, bh):
            b, h, kh = bh // HV, bh % HV, (bh % HV) // 3
            state = T.alloc_shared((DK, value_tile), "float32")
            residual = T.alloc_shared((bt, value_tile), "float32")
            decay = T.alloc_shared((bt,), "float32")
            last_decay = T.alloc_shared((1,), "float32")
            accum = T.alloc_local((1,), "float32")
            for i, j in T.Parallel(DK, value_tile):
                state[i, j] = Sin[b, h, i, bv * value_tile + j]
            T.sync_threads()
            for c in T.serial(chunks):
                for i, j in T.Parallel(DK, value_tile):
                    Senter[b, h, c, i, bv * value_tile + j] = state[i, j]
                for t in T.Parallel(bt):
                    decay[t] = T.exp(G[b, h, c, bt - 1] - G[b, h, c, t])
                if T.get_thread_binding() == 0:
                    last_decay[0] = T.exp(G[b, h, c, bt - 1])
                for t, j in T.Parallel(bt, value_tile):
                    accum[0] = 0.0
                    for i in T.serial(DK):
                        accum[0] = accum[0] + W[b, h, c, t, i] * state[i, j]
                    residual[t, j] = U[b, h, c, t, bv * value_tile + j] - accum[0]
                    R[b, h, c, t, bv * value_tile + j] = residual[t, j]
                T.sync_threads()
                for i, j in T.Parallel(DK, value_tile):
                    accum[0] = 0.0
                    for t in T.serial(bt):
                        accum[0] = (
                            accum[0]
                            + (T.cast(K[b, kh, c, t, i], "float32") * decay[t]) * residual[t, j]
                        )
                    state[i, j] = last_decay[0] * state[i, j] + accum[0]
                T.sync_threads()
            for i, j in T.Parallel(DK, value_tile):
                Sfinal[b, h, i, bv * value_tile + j] = state[i, j]

    return main


def gdn_chunk_state(bt=64, value_tile=16, threads=128):
    """Compile dynamic positive B/C, contiguous inputs and independent outputs.

    Tail K/W/U rows must be exact zero; cumulative G repeats its last valid
    value. Sin/Senter/Sfinal use [K,V], requiring explicit transpose when
    importing native [V,K] state. Outputs and all inputs must not alias.
    """
    kernel = _compile(bt, value_tile, threads)
    kernel.adapter.kernels = dict(kernel.adapter.kernels)
    return kernel


def launch(kernel, k, g, w, u, initial_state, entering_states, residuals, final_state, *, stream):
    """Stable caller buffers and explicit caller/capture CUDA stream."""
    return kernel.adapter.func(
        k, g, w, u, initial_state, entering_states, residuals, final_state, stream=stream
    )
