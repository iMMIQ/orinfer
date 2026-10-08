"""Experimental GDN: factor token scalars onto the FP32 operand.

WY uses (A * beta * exp(G)) @ K and (A * beta) @ V, where K/V remain
exact FP16 operands. State update uses K.T @ (decay * R). Output computes
(Q @ S) * q_scale * exp(G) before adding QK @ R. Split FP32 operands into
FP16 hi/lo; retain FP32 accumulation and persistent state. Reassociation
changes FP32 rounding, and residual underflow is possible; explicit numerical
and state/graph validation is required. Defaults retain hi/lo compensation;
explicit policy flags omit selected low products while persistent state and
MMA accumulation stay FP32. AOT model binding selects the measured policy.
"""

import tilelang.language as T
from kernels.operators.op03_ffn_gate_up import _orin_jit


@T.macro
def gemm3(ahi, alo, bhi, blo, output):
    T.gemm(alo, bhi, output)
    T.gemm(ahi, blo, output)
    T.gemm(ahi, bhi, output)


@_orin_jit
def gdn_chunk_state_factored(
    bt=64, value_tile=32, operand_dtype="float16", compensate_residual=True, compensate_update=True
):
    assert bt in (16, 32, 64) and value_tile in (16, 32)
    assert operand_dtype == "float16"
    batch, chunks = T.dynamic("batch"), T.dynamic("chunks")

    @T.prim_func
    def main(
        K: T.Tensor((batch, 16, chunks, bt, 128), T.float16),
        G: T.Tensor((batch, 48, chunks, bt), T.float32),
        W: T.Tensor((batch, 48, chunks, bt, 128), T.float32),
        U: T.Tensor((batch, 48, chunks, bt, 128), T.float32),
        Sin: T.Tensor((batch, 48, 128, 128), T.float32),
        Senter: T.Tensor((batch, 48, chunks, 128, 128), T.float32),
        R: T.Tensor((batch, 48, chunks, bt, 128), T.float32),
        Sfinal: T.Tensor((batch, 48, 128, 128), T.float32),
    ):
        with T.Kernel(128 // value_tile, batch * 48, threads=128) as (bv, bh):
            b, h = bh // 48, bh % 48
            state = T.alloc_shared((128, value_tile), T.float32)
            shi = T.alloc_shared((128, value_tile), operand_dtype)
            slo = T.alloc_shared((128, value_tile), operand_dtype)
            whi = T.alloc_shared((bt, 128), operand_dtype)
            wlo = T.alloc_shared((bt, 128), operand_dtype)
            rhi = T.alloc_shared((bt, value_tile), operand_dtype)
            rlo = T.alloc_shared((bt, value_tile), operand_dtype)
            residual = T.alloc_fragment((bt, value_tile), T.float32)
            update = T.alloc_fragment((128, value_tile), T.float32)
            decay = T.alloc_shared((bt,), T.float32)
            for i, j in T.Parallel(128, value_tile):
                state[i, j] = Sin[b, h, i, bv * value_tile + j]
            for c in T.serial(chunks):
                for i, j in T.Parallel(128, value_tile):
                    value = state[i, j]
                    high = T.cast(value, operand_dtype)
                    shi[i, j] = high
                    if compensate_residual:
                        slo[i, j] = value - T.cast(high, T.float32)
                    Senter[b, h, c, i, bv * value_tile + j] = value
                for t, i in T.Parallel(bt, 128):
                    value = W[b, h, c, t, i]
                    high = T.cast(value, operand_dtype)
                    whi[t, i] = high
                    if compensate_residual:
                        wlo[t, i] = value - T.cast(high, T.float32)
                T.clear(residual)
                if compensate_residual:
                    gemm3(whi, wlo, shi, slo, residual)
                else:
                    T.gemm(whi, shi, residual)
                for t, j in T.Parallel(bt, value_tile):
                    value = U[b, h, c, t, bv * value_tile + j] - residual[t, j]
                    residual[t, j] = value
                    R[b, h, c, t, bv * value_tile + j] = value
                for t in T.Parallel(bt):
                    decay[t] = T.exp(G[b, h, c, bt - 1] - G[b, h, c, t])
                for t, j in T.Parallel(bt, value_tile):
                    value = residual[t, j] * decay[t]
                    high = T.cast(value, operand_dtype)
                    rhi[t, j] = high
                    if compensate_update:
                        rlo[t, j] = value - T.cast(high, T.float32)
                for t, i in T.Parallel(bt, 128):
                    whi[t, i] = K[b, h // 3, c, t, i]
                T.clear(update)
                if compensate_update:
                    T.gemm(whi, rlo, update, transpose_A=True)
                T.gemm(whi, rhi, update, transpose_A=True)
                for i, j in T.Parallel(128, value_tile):
                    state[i, j] = T.exp(G[b, h, c, bt - 1]) * state[i, j] + update[i, j]
            for i, j in T.Parallel(128, value_tile):
                Sfinal[b, h, i, bv * value_tile + j] = state[i, j]

    return main


@_orin_jit
def gdn_chunk_output_factored(
    q_scale=128**-0.5,
    bt=64,
    value_tile=32,
    operand_dtype="float16",
    compensate_cross=True,
    compensate_local=True,
):
    assert bt in (16, 32, 64) and value_tile in (16, 32)
    assert operand_dtype == "float16"
    batch, chunks = T.dynamic("batch"), T.dynamic("chunks")

    @T.prim_func
    def main(
        Q: T.Tensor((batch, 16, chunks, bt, 128), T.float16),
        G: T.Tensor((batch, 48, chunks, bt), T.float32),
        QK: T.Tensor((batch, 48, chunks, bt, bt), T.float32),
        Senter: T.Tensor((batch, 48, chunks, 128, 128), T.float32),
        R: T.Tensor((batch, 48, chunks, bt, 128), T.float32),
        Y: T.Tensor((batch, chunks * bt, 48, 128), T.float16),
    ):
        with T.Kernel(128 // value_tile, 48 * chunks, batch, threads=128) as (bv, hc, b):
            h, c = hc // chunks, hc % chunks
            qhi = T.alloc_shared((bt, 128), operand_dtype)

            shi = T.alloc_shared((128, value_tile), operand_dtype)
            slo = T.alloc_shared((128, value_tile), operand_dtype)
            khi = T.alloc_shared((bt, bt), operand_dtype)
            klo = T.alloc_shared((bt, bt), operand_dtype)
            rhi = T.alloc_shared((bt, value_tile), operand_dtype)
            rlo = T.alloc_shared((bt, value_tile), operand_dtype)
            output = T.alloc_fragment((bt, value_tile), T.float32)
            for t, i in T.Parallel(bt, 128):
                qhi[t, i] = Q[b, h // 3, c, t, i]
            for i, j in T.Parallel(128, value_tile):
                value = Senter[b, h, c, i, bv * value_tile + j]
                high = T.cast(value, operand_dtype)
                shi[i, j] = high
                if compensate_cross:
                    slo[i, j] = value - T.cast(high, T.float32)
            for i, j in T.Parallel(bt, bt):
                value = QK[b, h, c, i, j]
                high = T.cast(value, operand_dtype)
                khi[i, j] = high
                if compensate_local:
                    klo[i, j] = value - T.cast(high, T.float32)
            for t, j in T.Parallel(bt, value_tile):
                value = R[b, h, c, t, bv * value_tile + j]
                high = T.cast(value, operand_dtype)
                rhi[t, j] = high
                if compensate_local:
                    rlo[t, j] = value - T.cast(high, T.float32)
            T.clear(output)
            if compensate_cross:
                T.gemm(qhi, slo, output)
            T.gemm(qhi, shi, output)
            for t, j in T.Parallel(bt, value_tile):
                output[t, j] = (output[t, j] * q_scale) * T.exp(G[b, h, c, t])
            if compensate_local:
                gemm3(khi, klo, rhi, rlo, output)
            else:
                T.gemm(khi, rhi, output)
            for t, j in T.Parallel(bt, value_tile):
                Y[b, c * bt + t, h, bv * value_tile + j] = output[t, j]

    return main
