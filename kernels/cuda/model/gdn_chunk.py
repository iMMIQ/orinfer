"""Chunkwise GDN using FP16 MMA operands and a persistent FP32 state.

The triangular WY solve stays FP32. Intermediate operands round to FP16;
this is numerically close to the sequential recurrence, not bit-identical.
Both kernels take the live row count, including a masked partial final chunk.
"""

import tilelang.language as T
from kernels.shared.rows import row_count


def prepare(M, HK=16, HV=32, C=16, dynamic_rows=False):
    if not 1 <= M <= 4096 or min(HK, HV) <= 0 or HV % HK or C not in (16, 32):
        raise ValueError("Invalid chunkwise GDN geometry")
    m = row_count(M, dynamic_rows)
    nc = T.ceildiv(m, C)

    @T.prim_func
    def main(
        Q: T.Tensor((HK, m, 128), T.float16),
        K: T.Tensor((HK, m, 128), T.float16),
        V: T.Tensor((HV, m, 128), T.float16),
        G: T.Tensor((m, HV), T.float32),
        Beta: T.Tensor((m, HV), T.float32),
        W: T.Tensor((HV, nc, C, 128), T.float16),
        U: T.Tensor((HV, nc, C, 128), T.float16),
        D: T.Tensor((HV, nc, C, C), T.float16),
        GP: T.Tensor((HV, nc, C, 2), T.float32),
    ):
        with T.Kernel(nc, HV, threads=64 if C == 16 else 128) as (chunk, h):
            kh = h // (HV // HK)
            start = chunk * C
            qs = T.alloc_shared((C, 128), T.float16)
            ks = T.alloc_shared((C, 128), T.float16)
            gates = T.alloc_fragment((C,), T.float32)
            g = T.alloc_shared((C,), T.float32)
            b = T.alloc_shared((C,), T.float32)
            dot = T.alloc_fragment((C, C), T.float32)
            lower = T.alloc_shared((C, C), T.float32)
            ih = T.alloc_shared((C, C), T.float16)
            rhs = T.alloc_shared((C, 256), T.float16)
            wy = T.alloc_fragment((C, 256), T.float32)
            for i, j in T.Parallel(C, 128):
                qs[i, j] = T.if_then_else(start + i < m, Q[kh, start + i, j], T.float16(0))
                ks[i, j] = T.if_then_else(start + i < m, K[kh, start + i, j], T.float16(0))
            for i in T.Parallel(C):
                gates[i] = T.if_then_else(start + i < m, G[start + i, h], T.float32(0))
                b[i] = T.if_then_else(start + i < m, Beta[start + i, h], T.float32(0))
            T.cumsum(gates, dim=0)
            for i in T.Parallel(C):
                g[i] = gates[i]
            for i in T.Parallel(C):
                GP[h, chunk, i, 0] = T.exp(g[i])
                GP[h, chunk, i, 1] = T.exp(g[C - 1] - g[i])
            T.clear(dot)
            T.gemm(ks, ks, dot, transpose_B=True)
            for i, j in T.Parallel(C, C):
                lower[i, j] = T.if_then_else(
                    i > j, b[i] * dot[i, j] * T.exp(g[i] - g[j]), T.float32(0)
                )
            T.sync_threads()
            # Columns are independent. Unroll both dimensions so the FP32
            # forward solve stays in registers and needs no per-row barrier.
            column = T.get_thread_binding() % C
            solved = T.alloc_local((C,), T.float32)
            scalar = T.alloc_local((1,), T.float32)
            if T.get_thread_binding() < C:
                for i in T.unroll(C):
                    scalar[0] = T.if_then_else(i == column, T.float32(1), T.float32(0))
                    for k in T.unroll(i):
                        scalar[0] -= lower[i, k] * solved[k]
                    solved[i] = scalar[0]
                for i in T.unroll(C):
                    ih[i, column] = solved[i]
            T.sync_threads()
            for i, j in T.Parallel(C, 256):
                if j < 128:
                    rhs[i, j] = b[i] * T.cast(ks[i, j], T.float32) * T.exp(g[i])
                else:
                    rhs[i, j] = T.if_then_else(
                        start + i < m,
                        b[i] * T.cast(V[h, start + i, j - 128], T.float32),
                        T.float32(0),
                    )
            T.clear(wy)
            T.gemm(ih, rhs, wy)
            for i, j in T.Parallel(C, 256):
                if j < 128:
                    W[h, chunk, i, j] = wy[i, j]
                else:
                    U[h, chunk, i, j - 128] = wy[i, j]
            T.clear(dot)
            T.gemm(qs, ks, dot, transpose_B=True)
            for i, j in T.Parallel(C, C):
                D[h, chunk, i, j] = T.if_then_else(
                    j <= i, dot[i, j] * T.exp(g[i] - g[j]) * T.float32(128**-0.5), T.float32(0)
                )

    return main


def recurrent(M, HK=16, HV=32, C=16, BV=32, dynamic_rows=False, threads=None):
    if (
        not 1 <= M <= 4096
        or min(HK, HV) <= 0
        or HV % HK
        or C not in (16, 32)
        or BV not in (16, 32, 64, 128)
    ):
        raise ValueError("Invalid chunkwise GDN geometry")
    if threads is None:
        threads = 64 if BV == 16 and C == 16 else 128
    if threads not in (64, 128, 256, 512) or threads // 32 > (C // 16) * (BV // 8):
        raise ValueError("Invalid chunkwise GDN warp geometry")
    m = row_count(M, dynamic_rows)
    nc = T.ceildiv(m, C)

    @T.prim_func
    def main(
        Q: T.Tensor((HK, m, 128), T.float16),
        K: T.Tensor((HK, m, 128), T.float16),
        W: T.Tensor((HV, nc, C, 128), T.float16),
        U: T.Tensor((HV, nc, C, 128), T.float16),
        D: T.Tensor((HV, nc, C, C), T.float16),
        GP: T.Tensor((HV, nc, C, 2), T.float32),
        State: T.Tensor((HV, 128, 128), T.float32),
        Out: T.Tensor((m, HV * 128), T.float16),
    ):
        with T.Kernel(128 // BV, HV, threads=threads) as (bv, h):
            kh = h // (HV // HK)
            state = T.alloc_fragment((128, BV), T.float32)
            sh = T.alloc_shared((128, BV), T.float16)
            w = T.alloc_shared((C, 128), T.float16)
            q = T.alloc_shared((C, 128), T.float16)
            k = T.alloc_shared((C, 128), T.float16)
            delta = T.alloc_fragment((C, BV), T.float32)
            ds = T.alloc_shared((C, BV), T.float16)
            d = T.alloc_shared((C, C), T.float16)
            out = T.alloc_fragment((C, BV), T.float32)
            update = T.alloc_fragment((128, BV), T.float32)
            g = T.alloc_shared((C, 2), T.float32)
            for i, j in T.Parallel(128, BV):
                state[i, j] = State[h, i, bv * BV + j]
            for chunk in T.serial(nc):
                for i, j in T.Parallel(128, BV):
                    sh[i, j] = state[i, j]
                T.copy(W[h, chunk, :, :], w)
                T.copy(D[h, chunk, :, :], d)
                T.copy(GP[h, chunk, :, :], g)
                for i, j in T.Parallel(C, 128):
                    q[i, j] = T.if_then_else(
                        chunk * C + i < m,
                        T.cast(Q[kh, chunk * C + i, j], T.float32) * g[i, 0] * T.float32(128**-0.5),
                        T.float32(0),
                    )
                    k[i, j] = T.if_then_else(
                        chunk * C + i < m,
                        T.cast(K[kh, chunk * C + i, j], T.float32) * g[i, 1],
                        T.float32(0),
                    )
                T.clear(delta)
                T.gemm(w, sh, delta)
                for i, j in T.Parallel(C, BV):
                    ds[i, j] = T.cast(U[h, chunk, i, bv * BV + j], T.float32) - delta[i, j]
                T.clear(out)
                T.gemm(q, sh, out)
                T.gemm(d, ds, out)
                for i, j in T.Parallel(C, BV):
                    if chunk * C + i < m:
                        Out[chunk * C + i, h * 128 + bv * BV + j] = out[i, j]
                T.clear(update)
                T.gemm(k, ds, update, transpose_A=True)
                for i, j in T.Parallel(128, BV):
                    state[i, j] = state[i, j] * g[C - 1, 0] + update[i, j]
            for i, j in T.Parallel(128, BV):
                State[h, i, bv * BV + j] = state[i, j]

    return main
