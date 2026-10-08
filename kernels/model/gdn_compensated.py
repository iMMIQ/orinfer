"""GDN matrix candidates using three products and FP32 state/accumulation.

X = X_hi + X_lo, hi=operand(X), lo=operand(X-FP32(hi)). Products retain hi*hi
and both cross terms; lo*lo is omitted. This is an explicit numerical change,
not bit-exact FP32. FP16 operands require finite representable values; small
residuals may underflow. The optional BF16 candidate is not launch-validated.
All persistent and working state remains FP32; global scratch is unchanged.
Original FP32 SIMT operators remain available.
"""

import tilelang.language as T
from kernels.operators.op03_ffn_gate_up import _orin_jit


@_orin_jit
def gdn_chunk_wy_compensated(
    bt=64, value_tile=64, compensate_transform=True, compensate_inputs=True, cached_scalars=False
):
    """Candidate FP32 WY outputs using three-product FP16 compensation.

    Preserves beta*K then exp(G) ordering and suppressed-lane/upper-triangle
    semantics. Omits lo*lo and may underflow operand residuals; not bit-exact.
    """
    assert bt in (16, 32, 64) and value_tile in (32, 64)
    batch, chunks = T.dynamic("batch"), T.dynamic("chunks")

    @T.prim_func
    def main(
        A: T.Tensor((batch, 48, chunks, bt, bt), T.float32),
        K: T.Tensor((batch, 16, chunks, bt, 128), T.float16),
        V: T.Tensor((batch, 48, chunks, bt, 128), T.float16),
        G: T.Tensor((batch, 48, chunks, bt), T.float32),
        Beta: T.Tensor((batch, 48, chunks, bt), T.float32),
        W: T.Tensor((batch, 48, chunks, bt, 128), T.float32),
        U: T.Tensor((batch, 48, chunks, bt, 128), T.float32),
    ):
        with T.Kernel(128 // value_tile, 48 * chunks, batch, threads=128) as (tile, hc, b):
            h, c = hc // chunks, hc % chunks
            ahi = T.alloc_shared((bt, bt), T.float16)
            alo = T.alloc_shared((bt, bt), T.float16)
            khi = T.alloc_shared((bt, value_tile), T.float16)
            klo = T.alloc_shared((bt, value_tile), T.float16)
            vhi = T.alloc_shared((bt, value_tile), T.float16)
            vlo = T.alloc_shared((bt, value_tile), T.float16)
            w = T.alloc_fragment((bt, value_tile), T.float32)
            u = T.alloc_fragment((bt, value_tile), T.float32)
            if cached_scalars:
                beta_shared = T.alloc_shared((bt,), T.float32)
                exp_shared = T.alloc_shared((bt,), T.float32)
                for i in T.Parallel(bt):
                    bval = T.alloc_var(T.float32)
                    eval = T.alloc_var(T.float32)
                    bval = Beta[b, h, c, i]
                    eval = 0.0
                    if bval != 0.0:
                        eval = T.exp(G[b, h, c, i])
                    beta_shared[i] = bval
                    exp_shared[i] = eval
            for i, j in T.Parallel(bt, bt):
                aval = T.alloc_var(T.float32)
                aval = 0.0
                if j <= i:
                    aval = A[b, h, c, i, j]
                ahi[i, j] = aval
                if compensate_transform:
                    alo[i, j] = aval - T.cast(ahi[i, j], T.float32)
            for i, d in T.Parallel(bt, value_tile):
                kval = T.alloc_var(T.float32)
                vval = T.alloc_var(T.float32)
                kval = 0.0
                vval = 0.0
                if cached_scalars:
                    if beta_shared[i] != 0.0:
                        kval = (
                            beta_shared[i]
                            * T.cast(K[b, h // 3, c, i, tile * value_tile + d], T.float32)
                        ) * exp_shared[i]
                        vval = beta_shared[i] * T.cast(
                            V[b, h, c, i, tile * value_tile + d], T.float32
                        )
                else:
                    if Beta[b, h, c, i] != 0.0:
                        kval = (
                            Beta[b, h, c, i]
                            * T.cast(K[b, h // 3, c, i, tile * value_tile + d], T.float32)
                        ) * T.exp(G[b, h, c, i])
                        vval = Beta[b, h, c, i] * T.cast(
                            V[b, h, c, i, tile * value_tile + d], T.float32
                        )
                khi[i, d] = kval
                if compensate_inputs:
                    klo[i, d] = kval - T.cast(khi[i, d], T.float32)
                vhi[i, d] = vval
                if compensate_inputs:
                    vlo[i, d] = vval - T.cast(vhi[i, d], T.float32)
            T.clear(w)
            T.clear(u)
            if compensate_transform:
                T.gemm(alo, khi, w)
            if compensate_inputs:
                T.gemm(ahi, klo, w)
            T.gemm(ahi, khi, w)
            if compensate_transform:
                T.gemm(alo, vhi, u)
            if compensate_inputs:
                T.gemm(ahi, vlo, u)
            T.gemm(ahi, vhi, u)
            T.copy(w, W[b, h, c, 0, tile * value_tile])
            T.copy(u, U[b, h, c, 0, tile * value_tile])

    return main


@T.macro
def gemm3(ahi, alo, bhi, blo, output):
    T.gemm(alo, bhi, output)
    T.gemm(ahi, blo, output)
    T.gemm(ahi, bhi, output)


@_orin_jit
def gdn_chunk_state_compensated(bt=64, value_tile=32, operand_dtype="float16"):
    assert bt in (16, 32, 64) and value_tile in (16, 32)
    assert operand_dtype in ("bfloat16", "float16")
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
                    slo[i, j] = value - T.cast(high, T.float32)
                    Senter[b, h, c, i, bv * value_tile + j] = value
                for t, i in T.Parallel(bt, 128):
                    value = W[b, h, c, t, i]
                    high = T.cast(value, operand_dtype)
                    whi[t, i] = high
                    wlo[t, i] = value - T.cast(high, T.float32)
                T.clear(residual)
                gemm3(whi, wlo, shi, slo, residual)
                for t, j in T.Parallel(bt, value_tile):
                    value = U[b, h, c, t, bv * value_tile + j] - residual[t, j]
                    high = T.cast(value, operand_dtype)
                    rhi[t, j] = high
                    rlo[t, j] = value - T.cast(high, T.float32)
                    R[b, h, c, t, bv * value_tile + j] = value
                for t in T.Parallel(bt):
                    decay[t] = T.exp(G[b, h, c, bt - 1] - G[b, h, c, t])
                for t, i in T.Parallel(bt, 128):
                    value = T.cast(K[b, h // 3, c, t, i], T.float32) * decay[t]
                    high = T.cast(value, operand_dtype)
                    whi[t, i] = high
                    wlo[t, i] = value - T.cast(high, T.float32)
                T.clear(update)
                T.gemm(wlo, rhi, update, transpose_A=True)
                T.gemm(whi, rlo, update, transpose_A=True)
                T.gemm(whi, rhi, update, transpose_A=True)
                for i, j in T.Parallel(128, value_tile):
                    state[i, j] = T.exp(G[b, h, c, bt - 1]) * state[i, j] + update[i, j]
            for i, j in T.Parallel(128, value_tile):
                Sfinal[b, h, i, bv * value_tile + j] = state[i, j]

    return main


@_orin_jit
def gdn_chunk_output_compensated(q_scale=128**-0.5, bt=64, value_tile=32, operand_dtype="float16"):
    assert bt in (16, 32, 64) and value_tile in (16, 32)
    assert operand_dtype in ("bfloat16", "float16")
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
            qlo = T.alloc_shared((bt, 128), operand_dtype)
            shi = T.alloc_shared((128, value_tile), operand_dtype)
            slo = T.alloc_shared((128, value_tile), operand_dtype)
            khi = T.alloc_shared((bt, bt), operand_dtype)
            klo = T.alloc_shared((bt, bt), operand_dtype)
            rhi = T.alloc_shared((bt, value_tile), operand_dtype)
            rlo = T.alloc_shared((bt, value_tile), operand_dtype)
            output = T.alloc_fragment((bt, value_tile), T.float32)
            for t, i in T.Parallel(bt, 128):
                value = (T.cast(Q[b, h // 3, c, t, i], T.float32) * q_scale) * T.exp(G[b, h, c, t])
                high = T.cast(value, operand_dtype)
                qhi[t, i] = high
                qlo[t, i] = value - T.cast(high, T.float32)
            for i, j in T.Parallel(128, value_tile):
                value = Senter[b, h, c, i, bv * value_tile + j]
                high = T.cast(value, operand_dtype)
                shi[i, j] = high
                slo[i, j] = value - T.cast(high, T.float32)
            for i, j in T.Parallel(bt, bt):
                value = QK[b, h, c, i, j]
                high = T.cast(value, operand_dtype)
                khi[i, j] = high
                klo[i, j] = value - T.cast(high, T.float32)
            for t, j in T.Parallel(bt, value_tile):
                value = R[b, h, c, t, bv * value_tile + j]
                high = T.cast(value, operand_dtype)
                rhi[t, j] = high
                rlo[t, j] = value - T.cast(high, T.float32)
            T.clear(output)
            gemm3(qhi, qlo, shi, slo, output)
            gemm3(khi, klo, rhi, rlo, output)
            for t, j in T.Parallel(bt, value_tile):
                Y[b, c * bt + t, h, bv * value_tile + j] = output[t, j]

    return main
