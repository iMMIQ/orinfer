"""Portable E8P Q2A8 GEMM kernels for non-Orin discrete CUDA GPUs (SM80/86/89/90/100+).

Reads Orinfer's exact physical compressed E8P U16 weight banks (`packed_k=128` or
`packed_k=64`) directly from HBM without persistent FP16 expansion, decodes each
K128 tile on-chip from the 2 KiB `integer-e8p-spread29-v1` INT8 basis table, and
executes exact INT8 x INT8 -> INT32 Tensor Core MMA via portable `T.gemm`.

Zero SM87-specific PTX (`mma.s4`, `prmt`, manual `ldmatrix` / bank swizzles) is
emitted, so TileLang lowers cleanly to optimal native Tensor Cores across Ampere,
Ada, Hopper, and Blackwell architectures.
"""

import tilelang.language as T

from kernels.shared.rows import row_count
from orinfer_build.compiler.cuda.common import cuda_jit


@T.macro
def _decode_e8p_tile_k128(P_s, Book_s, W_s, bn):
    """Portable pure-TileLang E8P U16 -> INT8 codeword decode for one [BN, 128] tile."""
    for n, g in T.Parallel(bn, 16):
        code = T.cast(P_s[n, g], T.uint32)
        idx = code >> 8
        low = code & T.uint32(255)
        p1 = low ^ (low >> 4)
        p2 = p1 ^ (p1 >> 2)
        parity = (p2 ^ (p2 >> 1)) & T.uint32(1)
        mask = low ^ parity
        shift = 1 - 2 * T.cast(parity, T.int32)
        for d in T.unroll(8):
            bval = T.cast(Book_s[idx, d], T.int32)
            bit_pos = T.cast(((d & 1) << 2) | (d >> 1), T.uint32)
            flip = T.cast((mask >> bit_pos) & T.uint32(1), T.int32)
            W_s[n, g * 8 + d] = T.cast(bval * (1 - 2 * flip) + shift, T.int8)


@cuda_jit
def e8p_grouped_gemm(
    rows,
    E,
    tiles,
    N,
    K,
    *,
    block_m=32,
    block_n=64,
    num_stages=2,
    dynamic_rows=False,
    packed_k=128,
):
    """Portable grouped MoE E8P Q2A8 GEMM over compressed Orinfer U16 expert banks.

    Matches Orinfer's exact integer dot-product and FP32 rescaling order
    `(float32(acc) * float32(WS[expert, n])) * float32(AS[row])`.
    Supports both `packed_k=128` (`[E, K//128, N, 16]`) and `packed_k=64`
    (`[E, K//64, N, 8]`) physical banks.
    """
    if (
        any(type(v) is not int or v <= 0 for v in (rows, E, tiles, N, K))
        or K % 128
        or K * 128 * 128 > 2**31 - 1
        or block_m not in (16, 32, 64)
        or block_n not in (64, 128)
        or num_stages not in (1, 2)
        or packed_k not in (64, 128)
        or type(dynamic_rows) is not bool
    ):
        raise ValueError("Invalid portable E8P grouped GEMM geometry")

    BM, BN, BK = block_m, block_n, 128
    K128 = K // 128
    rows_sym = row_count(rows, dynamic_rows)
    launch_tiles = (
        T.min(tiles, T.ceildiv(rows_sym, BM) + T.min(E, rows_sym)) if dynamic_rows else tiles
    )

    @T.prim_func
    def main(
        A: T.Tensor((rows_sym, K), T.int8),
        P: T.Tensor((E, K // packed_k, N, packed_k // 8), T.uint16),
        Book: T.Tensor((256, 8), T.int8),
        WS: T.Tensor((E, N), T.float16),
        AS: T.Tensor((rows_sym,), T.float16),
        Counts: T.Tensor((E,), T.int32),
        Offsets: T.Tensor((E,), T.int32),
        TileExpert: T.Tensor((tiles,), T.int32),
        TileRow: T.Tensor((tiles,), T.int32),
        TileCount: T.Tensor((1,), T.int32),
        Out: T.Tensor((rows_sym, N), T.float16),
    ):
        with T.Kernel(T.ceildiv(N, BN), launch_tiles, threads=128) as (bx, tile):
            Book_s = T.alloc_shared((256, 8), T.int8)
            A_s = T.alloc_shared((BM, BK), T.int8)
            P_s = T.alloc_shared((BN, 16), T.uint16)
            W_s = T.alloc_shared((BN, BK), T.int8)
            C_f = T.alloc_fragment((BM, BN), T.int32)
            C_s = T.alloc_shared((BM, BN), T.int32)
            AS_s = T.alloc_shared((BM,), T.float32)
            WS_s = T.alloc_shared((BN,), T.float32)

            if tile < TileCount[0]:
                expert = TileExpert[tile]
                first = TileRow[tile]
                offset = Offsets[expert] + first
                count = T.min(BM, Counts[expert] - first)

                T.copy(Book[0, 0], Book_s)
                T.clear(C_f)

                for kb in T.Pipelined(K128, num_stages=num_stages):
                    for m, k in T.Parallel(BM, BK):
                        A_s[m, k] = T.if_then_else(m < count, A[offset + m, kb * BK + k], 0)
                    if packed_k == 64:
                        for half in T.unroll(2):
                            T.copy(
                                P[expert, kb * 2 + half, bx * BN : bx * BN + BN, 0:8],
                                P_s[:, half * 8 : (half + 1) * 8],
                            )
                    else:
                        T.copy(P[expert, kb, bx * BN, 0], P_s)

                    _decode_e8p_tile_k128(P_s, Book_s, W_s, BN)
                    T.gemm(A_s, W_s, C_f, transpose_B=True)

                T.copy(C_f, C_s)
                for i in T.Parallel(BM):
                    AS_s[i] = T.if_then_else(i < count, T.cast(AS[offset + i], T.float32), 0.0)
                for j in T.Parallel(BN):
                    col = bx * BN + j
                    WS_s[j] = T.if_then_else(col < N, T.cast(WS[expert, col], T.float32), 0.0)
                for i, j in T.Parallel(BM, BN):
                    if i < count and bx * BN + j < N:
                        Out[offset + i, bx * BN + j] = T.cast(
                            (T.cast(C_s[i, j], T.float32) * WS_s[j]) * AS_s[i], T.float16
                        )

    return main


@cuda_jit
def e8p_dense_gemm(
    rows,
    N,
    K,
    *,
    block_m=64,
    block_n=64,
    num_stages=2,
    dynamic_rows=False,
    packed_k=128,
):
    """Portable dense (non-grouped) E8P Q2A8 GEMM over compressed Orinfer U16 weights."""
    if (
        any(type(v) is not int or v <= 0 for v in (rows, N, K))
        or K % 128
        or K * 128 * 128 > 2**31 - 1
        or block_m not in (16, 32, 64)
        or block_n not in (64, 128)
        or num_stages not in (1, 2)
        or packed_k not in (64, 128)
        or type(dynamic_rows) is not bool
    ):
        raise ValueError("Invalid portable E8P dense GEMM geometry")

    BM, BN, BK = block_m, block_n, 128
    K128 = K // 128
    rows_sym = row_count(rows, dynamic_rows)

    @T.prim_func
    def main(
        A: T.Tensor((rows_sym, K), T.int8),
        P: T.Tensor((K // packed_k, N, packed_k // 8), T.uint16),
        Book: T.Tensor((256, 8), T.int8),
        WS: T.Tensor((N,), T.float16),
        AS: T.Tensor((rows_sym,), T.float16),
        Out: T.Tensor((rows_sym, N), T.float16),
    ):
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(rows_sym, BM), threads=128) as (bx, by):
            Book_s = T.alloc_shared((256, 8), T.int8)
            A_s = T.alloc_shared((BM, BK), T.int8)
            P_s = T.alloc_shared((BN, 16), T.uint16)
            W_s = T.alloc_shared((BN, BK), T.int8)
            C_f = T.alloc_fragment((BM, BN), T.int32)
            C_s = T.alloc_shared((BM, BN), T.int32)
            AS_s = T.alloc_shared((BM,), T.float32)
            WS_s = T.alloc_shared((BN,), T.float32)

            T.copy(Book[0, 0], Book_s)
            T.clear(C_f)

            for kb in T.Pipelined(K128, num_stages=num_stages):
                T.copy(A[by * BM, kb * BK], A_s)
                if packed_k == 64:
                    for half in T.unroll(2):
                        T.copy(
                            P[kb * 2 + half, bx * BN : bx * BN + BN, 0:8],
                            P_s[:, half * 8 : (half + 1) * 8],
                        )
                else:
                    T.copy(P[kb, bx * BN, 0], P_s)

                _decode_e8p_tile_k128(P_s, Book_s, W_s, BN)
                T.gemm(A_s, W_s, C_f, transpose_B=True)

            T.copy(C_f, C_s)
            for i in T.Parallel(BM):
                r = by * BM + i
                AS_s[i] = T.if_then_else(r < rows_sym, T.cast(AS[r], T.float32), 0.0)
            for j in T.Parallel(BN):
                col = bx * BN + j
                WS_s[j] = T.if_then_else(col < N, T.cast(WS[col], T.float32), 0.0)
            for i, j in T.Parallel(BM, BN):
                if by * BM + i < rows_sym and bx * BN + j < N:
                    Out[by * BM + i, bx * BN + j] = T.cast(
                        (T.cast(C_s[i, j], T.float32) * WS_s[j]) * AS_s[i], T.float16
                    )

    return main
