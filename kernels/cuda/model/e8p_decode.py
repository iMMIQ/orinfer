"""Warp-owned E8P decode tiles; original compressed bank and integer arithmetic.

TileLang owns staging, routing, split-K and synchronization. Imported helpers
only expose PRMT and one SM87 INT8 MMA instruction, with no kernel or loops.
"""

import tilelang.language as T

from orinfer_build.compiler.cuda.common import orin_jit


SOURCE = r"""
__device__ __forceinline__ unsigned e8p_prmt(unsigned positive, unsigned negative, unsigned selector) {
    unsigned result;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(result) : "r"(positive), "r"(negative), "r"(selector));
    return result;
}
__device__ __forceinline__ void e8p_mma(int* d, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
"""


@orin_jit(
    pass_configs={
        "tl.disable_thread_storage_sync": True,
        # Warp-owned stages may finish at different times. Their LUT cannot alias
        # split-K partials before the final CTA barrier.
        "tl.disable_shared_memory_reuse": True,
        "tl.if_stmt_binding_inline_replayable_binds": False,
    }
)
def e8p_decode_grouped(
    rows, E, tiles, N, K, *, block_n=64, split_k=1, parity_xor=False, packed_k=128
):
    if (
        any(type(v) is not int or v <= 0 for v in (rows, E, tiles, N, K))
        or type(parity_xor) is not bool
        or type(packed_k) is not int
        or packed_k not in (64, 128)
        or K % 128
        or K * 128 * 128 > 2**31 - 1
        or type(block_n) is not int
        or type(split_k) is not int
        or block_n not in (32, 64, 128)
        or split_k not in (1, 2, 4)
        or block_n * 2 * split_k > 512
    ):
        raise ValueError("Invalid E8P decode geometry")
    # XOR parity requires a nonzero spread29 basis, checked by offline builders.
    BN, SK, G = block_n, split_k, 128
    padding = 16 if parity_xor else 24
    threads = BN * 2 * SK
    iterations = (K // G + SK - 1) // SK

    @T.prim_func
    def main(
        A: T.Tensor((rows, K), T.int8),
        P: T.Tensor((E, K // packed_k, N, packed_k // 8), T.uint16),
        Book: T.Tensor((1, 256, 1), T.uint32),
        Patch: T.Tensor((1,), T.uint16),  # Offline VQ API slot; omitted from exported CUDA ABI.
        WS: T.Tensor((E, N), T.float16),
        AS: T.Tensor((rows,), T.float16),
        Counts: T.Tensor((E,), T.int32),
        Offsets: T.Tensor((E,), T.int32),
        TileExpert: T.Tensor((tiles,), T.int32),
        TileRow: T.Tensor((tiles,), T.int32),
        TileCount: T.Tensor((1,), T.int32),
        C: T.Tensor((rows, N), T.float16),
    ):
        with T.Kernel(tiles, T.ceildiv(N, BN), threads=threads) as (tile, bx):
            T.import_source(SOURCE)
            tid = T.get_thread_binding()
            lane, warp, sk = tid % 32, (tid // 32) % (BN // 16), tid // (BN * 2)
            group, tc = lane // 4, lane % 4
            book = T.alloc_shared((512,), T.uint32)
            # Pad only shared staging: global compressed weights stay unchanged.
            packed = T.alloc_shared((2, SK, BN, padding), T.uint16)
            partial = T.alloc_shared((SK, BN // 16, 32, 4), T.int32)
            acc = T.alloc_local((4,), T.int32)
            decoded = T.alloc_local((4,), T.uint32)
            inputs = T.alloc_local((2,), T.uint32)
            if tile < TileCount[0]:
                expert, first = TileExpert[tile], TileRow[tile]
                offset = Offsets[expert] + first
                count = T.min(8, Counts[expert] - first)
                if count > 0:
                    for i in T.serial(T.ceildiv(256, threads)):
                        index = i * threads + tid
                        if index < 256:
                            book[index] = Book[0, index, 0]
                            parity = (
                                T.call_pure_extern("uint32", "__popc", T.cast(index, T.uint32)) & 1
                            )
                            signs = T.cast(index, T.uint32) ^ parity
                            low = ((signs & 0x11) << 2) | ((signs & 0x22) << 9)
                            high = (((signs >> 2) & 0x11) << 2) | (((signs >> 2) & 0x22) << 9)
                            book[index + 256] = (
                                (parity * T.uint32(0x02020202) if parity_xor else parity)
                                | low
                                | (high << 16)
                            )
                    T.sync_threads()
                    for j in T.unroll(4):
                        acc[j] = 0
                    row = bx * BN + warp * 16 + lane // 2
                    T.ptx_cp_async(
                        T.access_ptr(
                            packed[
                                0,
                                sk,
                                warp * 16 + lane // 2,
                                ((lane % 2) * 8)
                                ^ ((((lane // 2 >> 2) & 1) * 8) if parity_xor else 0),
                            ],
                            "w",
                            8,
                        ),
                        T.access_ptr(
                            P[
                                expert,
                                T.min(sk, K // G - 1) * (G // packed_k)
                                + ((lane % 2) if packed_k == 64 else 0),
                                T.min(row, N - 1),
                                0 if packed_k == 64 else (lane % 2) * 8,
                            ],
                            "r",
                            8,
                        ),
                        8,
                        predicate=(row < N) & (sk < K // G),
                    )
                    T.ptx_commit_group()
                    for iteration in T.unroll(iterations, unroll_factor=1):
                        kg, slot = iteration * SK + sk, iteration % 2
                        T.ptx_wait_group(0)
                        T.sync_warp()
                        if iteration + 1 < iterations:
                            next_kg = (iteration + 1) * SK + sk
                            T.ptx_cp_async(
                                T.access_ptr(
                                    packed[
                                        (iteration + 1) % 2,
                                        sk,
                                        warp * 16 + lane // 2,
                                        ((lane % 2) * 8)
                                        ^ ((((lane // 2 >> 2) & 1) * 8) if parity_xor else 0),
                                    ],
                                    "w",
                                    8,
                                ),
                                T.access_ptr(
                                    P[
                                        expert,
                                        T.min(next_kg, K // G - 1) * (G // packed_k)
                                        + ((lane % 2) if packed_k == 64 else 0),
                                        T.min(row, N - 1),
                                        0 if packed_k == 64 else (lane % 2) * 8,
                                    ],
                                    "r",
                                    8,
                                ),
                                8,
                                predicate=(row < N) & (next_kg < K // G),
                            )
                            T.ptx_commit_group()
                        for kk in T.unroll(4):
                            for j in T.unroll(4):
                                code = T.cast(
                                    packed[
                                        slot,
                                        sk,
                                        warp * 16 + group + (j % 2) * 8,
                                        (kk * 4 + tc // 2 + (j // 2) * 2)
                                        ^ ((((group >> 2) & 1) * 8) if parity_xor else 0),
                                    ],
                                    T.uint32,
                                )
                                flags = book[256 + (code & 255)]
                                if parity_xor:
                                    selector = (
                                        book[code >> 8] ^ (flags & T.uint32(0x44444444))
                                    ) >> ((tc % 2) * 16)
                                    decoded[j] = T.call_pure_extern(
                                        "uint32",
                                        "e8p_prmt",
                                        T.uint32(0x010B0703),
                                        T.uint32(0x01F7FBFF),
                                        selector,
                                    ) ^ (flags & T.uint32(0x02020202))
                                else:
                                    selector = (book[code >> 8] >> ((tc % 2) * 16)) ^ (
                                        (flags >> ((tc % 2) * 16)) & 0x4444
                                    )
                                    shift = (flags & 1) * T.uint32(0x02020202)
                                    decoded[j] = T.call_pure_extern(
                                        "uint32",
                                        "e8p_prmt",
                                        T.uint32(0x010B0703) - shift,
                                        T.uint32(0x01F7FBFF) - shift,
                                        selector,
                                    )
                            for j in T.unroll(2):
                                inputs[j] = 0
                                if group < count and kg < K // G:
                                    inputs[j] = T.ldg32(
                                        A[offset + group, kg * G + kk * 32 + (tc + j * 4) * 4]
                                    )
                            T.evaluate(
                                T.call_extern(
                                    "void",
                                    "e8p_mma",
                                    acc.access_ptr("rw"),
                                    decoded[0],
                                    decoded[1],
                                    decoded[2],
                                    decoded[3],
                                    inputs[0],
                                    inputs[1],
                                )
                            )
                    if SK > 1:
                        for j in T.unroll(4):
                            partial[sk, warp, lane, j] = acc[j]
                        T.sync_threads()
                        if sk == 0:
                            for s in T.unroll(1, SK):
                                for j in T.unroll(4):
                                    acc[j] += partial[s, warp, lane, j]
                    if sk == 0:
                        for r in T.unroll(2):
                            output_row = bx * BN + warp * 16 + group + r * 8
                            if output_row < N:
                                for t in T.unroll(2):
                                    token = tc * 2 + t
                                    if token < count:
                                        C[offset + token, output_row] = (
                                            T.cast(acc[r * 2 + t], T.float32)
                                            * T.cast(WS[expert, output_row], T.float32)
                                        ) * T.cast(AS[offset + token], T.float32)

    return main
