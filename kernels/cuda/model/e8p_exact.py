"""Exact E8P/A8 products using two INT4 Tensor Core products on SM87.

For the nonzero spread29 basis, decoded w=2*y+1 with signed INT4 y.
A8 a=low_u4+16*high_s4, so dot(w,a)=2*(dot(y,low)+16*dot(y,high))+sum(a).
TileLang owns every kernel, loop, stage and barrier. Helpers expose instructions.
The K64 code layout and matching within-eight permutation are lossless.
"""

import tilelang.language as T
from orinfer_build.compiler.cuda.common import orin_jit
from kernels.shared.rows import row_count

from kernels.cuda.model.e8p_core import SOURCE, mma_pair, store_rows, low_plane, high_plane


@orin_jit
def split_a8(rows, K, dynamic_rows=False):
    if any(type(x) is not int or x <= 0 for x in (rows, K)) or K % 128:
        raise ValueError("Invalid exact A8 split geometry")
    rows = row_count(rows, dynamic_rows)
    W = 1 << ((K // 8 - 1).bit_length())

    @T.prim_func
    def pack_exact_a8(
        A: T.Tensor((rows, K), T.int8),
        Low: T.Tensor((rows, K // 8), T.uint32),
        High: T.Tensor((rows, K // 8), T.uint32),
        Sum: T.Tensor((rows,), T.int32),
    ):
        with T.Kernel(rows, threads=128) as row:
            T.import_source(SOURCE)
            sums = T.alloc_fragment((W,), T.int32)
            total = T.alloc_fragment((1,), T.int32)
            for j in T.Parallel(W):
                sums[j] = 0
                if j < K // 8:
                    x = T.ldg32(A[row, j * 8])
                    y = T.ldg32(A[row, j * 8 + 4])
                    Low[row, j] = low_plane(x, y)
                    High[row, j] = high_plane(x, y)
                    sums[j] = T.call_pure_extern("int32", "nib_sum", x, y)
            T.reduce_sum(sums, total, dim=0)
            Sum[row] = total[0]

    return pack_exact_a8


@orin_jit(
    pass_configs={
        "tl.disable_thread_storage_sync": True,
        "tl.disable_shared_memory_reuse": True,
        "tl.if_stmt_binding_inline_replayable_binds": False,
    }
)
def e8p_exact_grouped(rows, E, tiles, N, K, *, block_m=16, stages=2, dynamic_rows=False):
    if (
        any(type(x) is not int or x <= 0 for x in (rows, E, tiles, N, K))
        or K % 128
        or K * 128 * 128 > 2**31 - 1
        or type(block_m) is not int
        or block_m not in (8, 16)
        or type(stages) is not int
        or stages not in (2, 3, 4)
        or K // 64 < stages - 1
    ):
        raise ValueError("Invalid exact E8P geometry")
    BM, BN, BK = block_m, 128, 64
    # NCU identifies excess shared wavefronts for strided stores. Swizzle
    # sixteen-byte row halves so ldmatrix and writes both use distinct banks.
    # The small decode tile benefits from weight swizzling only.
    input_stride, input_mask = (32, 16) if BM == 16 else (48, 0)
    rows = row_count(rows, dynamic_rows)
    launch_tiles = T.min(tiles, T.ceildiv(rows, BM) + T.min(E, rows)) if dynamic_rows else tiles

    multiply, store = mma_pair(BM), store_rows(BM, BN, N)

    @T.prim_func
    def exact_e8p_int4(
        Low: T.Tensor((rows, K // 2), T.int8),
        High: T.Tensor((rows, K // 2), T.int8),
        Sum: T.Tensor((rows,), T.int32),
        P: T.Tensor((E, K // 64, N, 8), T.uint16),
        Book: T.Tensor((1, 256, 1), T.uint32),
        WS: T.Tensor((E, N), T.float16),
        AS: T.Tensor((rows,), T.float16),
        Counts: T.Tensor((E,), T.int32),
        Offsets: T.Tensor((E,), T.int32),
        TE: T.Tensor((tiles,), T.int32),
        TR: T.Tensor((tiles,), T.int32),
        TC: T.Tensor((1,), T.int32),
        Out: T.Tensor((rows, N), T.float16),
    ):
        with T.Kernel(T.ceildiv(N, BN), launch_tiles, threads=128) as (bx, tile):
            T.import_source(SOURCE)
            tid = T.get_thread_binding()
            lane = tid % 32
            warp = tid // 32
            packed = T.alloc_shared((stages, BN, 8), T.uint16)
            inputs = T.alloc_shared((stages, 2, BM, input_stride), T.int8)
            weights = T.alloc_shared((BN, 32), T.int8)
            book = T.alloc_shared((256,), T.uint32)
            lo = T.alloc_local((BM,), T.int32)
            hi = T.alloc_local((BM,), T.int32)
            v = T.alloc_local((4,), T.uint32)
            a = T.alloc_local((BM // 2,), T.uint32)
            if tile < TC[0]:
                expert, first = TE[tile], TR[tile]
                offset = Offsets[expert] + first
                count = T.min(BM, Counts[expert] - first)
                for i in T.unroll(2):
                    book[tid + i * 128] = Book[0, tid + i * 128, 0]
                for i in T.unroll(BM):
                    lo[i] = 0
                    hi[i] = 0
                T.sync_threads()
                for p in T.unroll(stages - 1):
                    T.ptx_cp_async(
                        T.access_ptr(packed[p, tid, 0], "w", 8),
                        T.access_ptr(P[expert, p, T.min(bx * BN + tid, N - 1), 0], "r", 8),
                        8,
                        predicate=bx * BN + tid < N,
                    )
                    if tid < BM * 2:
                        T.ptx_cp_async(
                            T.access_ptr(
                                inputs[
                                    p,
                                    0,
                                    tid // 2,
                                    ((tid % 2) * 16) ^ (((tid // 8) % 2) * input_mask),
                                ],
                                "w",
                                16,
                            ),
                            T.access_ptr(
                                Low[T.min(offset + tid // 2, rows - 1), p * 32 + (tid % 2) * 16],
                                "r",
                                16,
                            ),
                            16,
                            predicate=offset + tid // 2 < rows,
                        )
                        T.ptx_cp_async(
                            T.access_ptr(
                                inputs[
                                    p,
                                    1,
                                    tid // 2,
                                    ((tid % 2) * 16) ^ (((tid // 8) % 2) * input_mask),
                                ],
                                "w",
                                16,
                            ),
                            T.access_ptr(
                                High[T.min(offset + tid // 2, rows - 1), p * 32 + (tid % 2) * 16],
                                "r",
                                16,
                            ),
                            16,
                            predicate=offset + tid // 2 < rows,
                        )
                    T.ptx_commit_group()
                for step in T.serial(K // BK):
                    slot = step % stages
                    # The drain tail has fewer groups in flight than stages-1;
                    # wait_group must still force this step's group complete.
                    if stages == 2:
                        T.ptx_wait_group(0)
                    elif stages == 3:
                        if step + 1 < K // BK:
                            T.ptx_wait_group(1)
                        else:
                            T.ptx_wait_group(0)
                    else:
                        if step + 2 < K // BK:
                            T.ptx_wait_group(2)
                        elif step + 1 < K // BK:
                            T.ptx_wait_group(1)
                        else:
                            T.ptx_wait_group(0)
                    T.sync_threads()
                    if step + stages - 1 < K // BK:
                        nxt = (step + stages - 1) % stages
                        T.ptx_cp_async(
                            T.access_ptr(packed[nxt, tid, 0], "w", 8),
                            T.access_ptr(
                                P[
                                    expert,
                                    step + stages - 1,
                                    T.min(bx * BN + tid, N - 1),
                                    0,
                                ],
                                "r",
                                8,
                            ),
                            8,
                            predicate=bx * BN + tid < N,
                        )
                        if tid < BM * 2:
                            T.ptx_cp_async(
                                T.access_ptr(
                                    inputs[
                                        nxt,
                                        0,
                                        tid // 2,
                                        ((tid % 2) * 16) ^ (((tid // 8) % 2) * input_mask),
                                    ],
                                    "w",
                                    16,
                                ),
                                T.access_ptr(
                                    Low[
                                        T.min(offset + tid // 2, rows - 1),
                                        (step + stages - 1) * 32 + (tid % 2) * 16,
                                    ],
                                    "r",
                                    16,
                                ),
                                16,
                                predicate=offset + tid // 2 < rows,
                            )
                            T.ptx_cp_async(
                                T.access_ptr(
                                    inputs[
                                        nxt,
                                        1,
                                        tid // 2,
                                        ((tid % 2) * 16) ^ (((tid // 8) % 2) * input_mask),
                                    ],
                                    "w",
                                    16,
                                ),
                                T.access_ptr(
                                    High[
                                        T.min(offset + tid // 2, rows - 1),
                                        (step + stages - 1) * 32 + (tid % 2) * 16,
                                    ],
                                    "r",
                                    16,
                                ),
                                16,
                                predicate=offset + tid // 2 < rows,
                            )
                        T.ptx_commit_group()
                    for i in T.unroll(4):
                        n = tid // 4 + i * 32
                        both = T.call_pure_extern(
                            "uint32",
                            "nib_pair",
                            T.access_ptr(packed[slot, n, (tid % 4) * 2], "r", 2),
                        )
                        c0 = both & T.uint32(65535)
                        c1 = both >> 16
                        d0 = T.call_pure_extern("uint32", "nib_fast", c0, book[c0 >> 8])
                        d1 = T.call_pure_extern("uint32", "nib_fast", c1, book[c1 >> 8])
                        T.evaluate(
                            T.call_extern(
                                "void",
                                "nib_store2",
                                T.access_ptr(
                                    weights[n, ((tid % 4) * 8) ^ (((n // 4) % 2) * 16)], "w", 8
                                ),
                                d0,
                                d1,
                            )
                        )
                    T.sync_threads()
                    for stripe in T.unroll(BM // 8):
                        m = lane % 8 + stripe * 8
                        col = (lane // 8 % 2) * 16
                        T.evaluate(
                            T.call_extern(
                                "void",
                                "nib_ldm2",
                                T.access_ptr(
                                    inputs[slot, 0, m, col ^ (((m // 4) % 2) * input_mask)], "r", 16
                                ),
                                T.access_ptr(a[stripe * 2], "w", 2),
                            )
                        )
                        T.evaluate(
                            T.call_extern(
                                "void",
                                "nib_ldm2",
                                T.access_ptr(
                                    inputs[slot, 1, m, col ^ (((m // 4) % 2) * input_mask)], "r", 16
                                ),
                                T.access_ptr(a[BM // 4 + stripe * 2], "w", 2),
                            )
                        )
                    for nt in T.unroll(2):
                        n = warp * 32 + nt * 16 + lane % 16
                        col = (lane // 16) * 16
                        T.evaluate(
                            T.call_extern(
                                "void",
                                "nib_ldm4",
                                T.access_ptr(weights[n, col ^ (((n // 4) % 2) * 16)], "r", 16),
                                v.access_ptr("w"),
                            )
                        )
                        for stripe in T.unroll(BM // 8):
                            multiply(lo, hi, v, a, nt, stripe)
                    T.sync_threads()
                store(lo, hi, Sum, WS, AS, Out, bx, warp, lane, count, expert, offset, offset)

    return exact_e8p_int4
