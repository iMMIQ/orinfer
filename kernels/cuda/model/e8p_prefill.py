"""Exact grouped E8P products with K64-major activation planes.

All warps decode directly into MMA registers. Four activation stages accompany
four compressed K64 stages for Q2 or two compressed K128 stages for Q4.
Cooperative copies hide cp.async completion waits without a producer warp.
Integer products and rounding match the row-major exact expert recipe.
"""

import tilelang
import tilelang.language as T
from orinfer_build.compiler.cuda.common import orin_jit

from kernels.cuda.model.e8p_core import (
    SOURCE as EXACT_SOURCE,
    DENSE_SOURCE,
    folded_pair,
    store_folded_rows,
    mma_pair,
    store_rows,
    low_plane,
    high_plane,
)
from kernels.shared.rows import row_count

SOURCE = (
    EXACT_SOURCE
    + DENSE_SOURCE
    + r"""
// Move parity out of the selector: flipping its first bit changes the low
// nibble by 0xe, while parity changes every decoded nibble by 0x1.
__device__ __forceinline__ unsigned nib_prefill(unsigned c,unsigned b){unsigned pp=__popc(c&255u)&1u;unsigned sel=(c&0x33u)|((c&0xccu)<<6);return b^__byte_perm(0xeee00e00u,0u,sel)^(pp*0x1111111fu);}
"""
)


def _split_a8_block(K):
    width = max(32, 1 << ((K // 8 - 1).bit_length()))

    @T.macro
    def pack(A, Low, High, Sum, first, count):
        T.import_source(SOURCE)
        sums = T.alloc_fragment((4, width), T.int32)
        total = T.alloc_fragment((4,), T.int32)
        T.annotate_layout(
            {
                sums: tilelang.Fragment(
                    (4, width),
                    forward_thread_fn=lambda i, j: i * 32 + j % 32,
                    forward_index_fn=lambda i, j: j // 32,
                )
            }
        )
        for i, j in T.Parallel(4, width):
            sums[i, j] = 0
            if i < count and j < K // 8:
                x = T.ldg32(A[first + i, j * 8])
                y = T.ldg32(A[first + i, j * 8 + 4])
                Low[j // 8, first + i, j % 8] = low_plane(x, y)
                High[j // 8, first + i, j % 8] = high_plane(x, y)
                sums[i, j] = T.call_pure_extern("int32", "nib_sum", x, y)
        T.reduce_sum(sums, total, dim=1)
        for i in T.Parallel(4):
            if i < count:
                Sum[first + i] = total[i]

    return pack


@orin_jit
def split_prefill_a8(rows, K, dynamic_rows=False):
    if any(type(x) is not int or x <= 0 for x in (rows, K)) or K % 128:
        raise ValueError("Invalid K64-major A8 geometry")
    rows = row_count(rows, dynamic_rows)
    pack = _split_a8_block(K)

    @T.prim_func
    def pack_exact_a8(
        A: T.Tensor((rows, K), T.int8),
        Low: T.Tensor((K // 64, rows, 8), T.uint32),
        High: T.Tensor((K // 64, rows, 8), T.uint32),
        Sum: T.Tensor((rows,), T.int32),
    ):
        with T.Kernel(T.ceildiv(rows, 4), threads=128) as by:
            pack(A, Low, High, Sum, by * 4, T.min(4, rows - by * 4))

    return pack_exact_a8


@orin_jit
def split_routed_prefill_a8(rows, K, experts, block_m, dynamic_rows=False):
    if (
        any(type(x) is not int or x <= 0 for x in (rows, K, experts, block_m))
        or K % 128
        or block_m % 4
    ):
        raise ValueError("Invalid routed A8 plane geometry")
    capacity = (rows + block_m - 1) // block_m + min(experts, rows)
    rows = row_count(rows, dynamic_rows)
    pack = _split_a8_block(K)

    @T.prim_func
    def main(
        A: T.Tensor((rows, K), T.int8),
        Counts: T.Tensor((experts,), T.int32),
        Offsets: T.Tensor((experts,), T.int32),
        TE: T.Tensor((capacity,), T.int32),
        TR: T.Tensor((capacity,), T.int32),
        TC: T.Tensor((1,), T.int32),
        Low: T.Tensor((K // 64, rows, 8), T.uint32),
        High: T.Tensor((K // 64, rows, 8), T.uint32),
        Sum: T.Tensor((rows,), T.int32),
    ):
        with T.Kernel(T.ceildiv(rows, block_m) + T.min(experts, rows), threads=128) as tile:
            metadata = T.alloc_local((3,), T.int32)
            if tile < TC[0]:
                metadata[0] = TE[tile]
                metadata[1] = Offsets[metadata[0]] + TR[tile]
                metadata[2] = T.min(block_m, Counts[metadata[0]] - TR[tile])
                for group in T.serial(block_m // 4):
                    pack(
                        A,
                        Low,
                        High,
                        Sum,
                        metadata[1] + group * 4,
                        T.min(4, metadata[2] - group * 4),
                    )

    return main


@orin_jit(
    pass_configs={
        "tl.disable_thread_storage_sync": True,
        "tl.disable_shared_memory_reuse": True,
        "tl.if_stmt_binding_inline_replayable_binds": False,
    }
)
def e8p_prefill_grouped(
    rows,
    E,
    tiles,
    N,
    K,
    dynamic_rows=False,
    sparse_rows=False,
    row_major=False,
    codec="e8p_q2",
    minimum_blocks=1,
    block_m=16,
):
    """One routed-tile mainloop; optionally predicate empty MMA row stripes.

    Routing and computation share the selected row tile. Sparse candidates
    mask complete hardware M8 stripes inside this body. Offline tuning selects
    both row reuse and whether masking pays off.
    """
    if (
        any(type(x) is not int or x <= 0 for x in (rows, E, tiles, N, K))
        or type(sparse_rows) is not bool
        or type(row_major) is not bool
        or codec not in ("e8p_q2", "e8p_q4")
        or type(minimum_blocks) is not int
        or not 1 <= minimum_blocks <= 16
        or block_m not in (16, 32)
        or (codec == "e8p_q2" and block_m != 16)
        or K % 128
        or K * 128 * 128 > 2**31 - 1
    ):
        raise ValueError("Invalid exact E8P geometry")
    BM, BN, BK = block_m, 128, 64
    q4 = codec == "e8p_q4"
    packed_k, packed_dtype = (128, T.uint32) if q4 else (64, T.uint16)
    # Matching swizzle for sixteen-byte activation halves and ldmatrix reads.
    input_stride, input_mask = 32, 16
    rows = row_count(rows, dynamic_rows)
    # Group-major strides depend on live rows, rather than allocation capacity.
    group_stride, row_stride = (32, K // 2) if row_major else (rows * 32, 32)
    launch_tiles = T.min(tiles, T.ceildiv(rows, BM) + T.min(E, rows)) if dynamic_rows else tiles

    @T.macro
    def prefetch_weight(P, packed, slot, step, bx, expert, tid):
        if q4:
            # Four lanes fetch a complete physical K128 row. Pair slots retain
            # both K64 operands until their consumers pass the CTA barrier.
            for group in T.unroll(4):
                weight_row = tid // 4 + group * (BN // 4)
                col = (tid % 4) * 4
                T.ptx_cp_async(
                    T.access_ptr(
                        packed[(step // 2) % 2, weight_row, col ^ (((weight_row // 2) % 4) * 4)],
                        "w",
                        4,
                    ),
                    T.access_ptr(
                        P[expert, step // 2, T.min(bx * BN + weight_row, N - 1), col], "r", 4
                    ),
                    4,
                    predicate=bx * BN + weight_row < N,
                )
        else:
            T.ptx_cp_async(
                T.access_ptr(packed[slot, tid, 0], "w", 8),
                T.access_ptr(P[expert, step, T.min(bx * BN + tid, N - 1), 0], "r", 8),
                8,
                predicate=bx * BN + tid < N,
            )

    tile_rows = BM
    multiply = folded_pair(tile_rows) if q4 else mma_pair(tile_rows)
    store = store_folded_rows(tile_rows, BN, N, 5) if q4 else store_rows(tile_rows, BN, N)

    @T.macro
    def project(Low, High, Sum, P, WS, AS, Out, packed, inputs, book, bx, expert, offset, count):
        tid = T.get_thread_binding()
        lane, warp = tid % 32, tid // 32
        lo = T.alloc_local((tile_rows,), T.int32)
        if not q4:
            hi = T.alloc_local((tile_rows,), T.int32)
        v = T.alloc_local((4,), T.uint32)
        a = T.alloc_local((tile_rows // 2,), T.uint32)
        for i in T.unroll(tile_rows):
            lo[i] = 0
            if not q4:
                hi[i] = 0
        T.sync_threads()
        for initial in T.unroll(min(3, K // BK)):
            if not q4 or initial % 2 == 0:
                prefetch_weight(P, packed, initial, initial, bx, expert, tid)
            if tid < tile_rows * 2:
                T.ptx_cp_async(
                    T.access_ptr(
                        inputs[
                            initial,
                            0,
                            tid // 2,
                            ((tid % 2) * 16) ^ (((tid // 8) % 2) * input_mask),
                        ],
                        "w",
                        16,
                    ),
                    T.access_ptr(
                        Low[
                            (initial) * group_stride
                            + T.min(offset + tid // 2, rows - 1) * row_stride
                            + (tid % 2) * 16
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
                            initial,
                            1,
                            tid // 2,
                            ((tid % 2) * 16) ^ (((tid // 8) % 2) * input_mask),
                        ],
                        "w",
                        16,
                    ),
                    T.access_ptr(
                        High[
                            (initial) * group_stride
                            + T.min(offset + tid // 2, rows - 1) * row_stride
                            + (tid % 2) * 16
                        ],
                        "r",
                        16,
                    ),
                    16,
                    predicate=offset + tid // 2 < rows,
                )
            T.ptx_commit_group()
        # Four static input positions avoid repeated address calculations.
        # The leading barrier also protects previous input/weight readers
        # before their shared slots are prefetched again.
        for group in T.serial(T.ceildiv(K // BK, 4)):
            for sub in T.unroll(4):
                step = group * 4 + sub
                if step < K // BK:
                    slot = step % 4
                    if q4 and step >= 4 and step % 2 == 0:
                        # The first half shares the next group's K128 copy.
                        if step + 2 < K // BK:
                            T.ptx_wait_group(1)
                        else:
                            T.ptx_wait_group(0)
                    elif step + 2 < K // BK:
                        T.ptx_wait_group(2)
                    elif step + 1 < K // BK:
                        T.ptx_wait_group(1)
                    else:
                        T.ptx_wait_group(0)
                    T.sync_threads()
                    if step + 3 < K // BK:
                        nxt = (step + 3) % 4
                        # In steady state, refill only after both halves
                        # of the old pair have completed their reads.
                        if not q4 or (step >= 2 and step % 2 == 0):
                            prefetch_weight(P, packed, nxt, step + 3, bx, expert, tid)
                        if tid < tile_rows * 2:
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
                                        (step + 3) * group_stride
                                        + T.min(offset + tid // 2, rows - 1) * row_stride
                                        + (tid % 2) * 16
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
                                        (step + 3) * group_stride
                                        + T.min(offset + tid // 2, rows - 1) * row_stride
                                        + (tid % 2) * 16
                                    ],
                                    "r",
                                    16,
                                ),
                                16,
                                predicate=offset + tid // 2 < rows,
                            )
                        T.ptx_commit_group()
                    for stripe in T.unroll(tile_rows // 8):
                        if not sparse_rows or stripe * 8 < count:
                            r_index = lane % 8 + stripe * 8
                            col = (lane // 8 % 2) * 16
                            T.evaluate(
                                T.call_extern(
                                    "void",
                                    "nib_ldm2",
                                    T.access_ptr(
                                        inputs[
                                            slot,
                                            0,
                                            r_index,
                                            col ^ (((r_index // 4) % 2) * input_mask),
                                        ],
                                        "r",
                                        16,
                                    ),
                                    T.access_ptr(a[stripe * 2], "w", 2),
                                )
                            )
                            T.evaluate(
                                T.call_extern(
                                    "void",
                                    "nib_ldm2",
                                    T.access_ptr(
                                        inputs[
                                            slot,
                                            1,
                                            r_index,
                                            col ^ (((r_index // 4) % 2) * input_mask),
                                        ],
                                        "r",
                                        16,
                                    ),
                                    T.access_ptr(a[tile_rows // 4 + stripe * 2], "w", 2),
                                )
                            )
                    for nt in T.unroll(2):
                        for level in T.unroll(2 if q4 else 1):
                            for rr in T.unroll(4):
                                n = warp * 32 + nt * 16 + lane // 4 + (rr % 2) * 8
                                col = lane % 4 + (rr // 2) * 4
                                if q4:
                                    code = (
                                        packed[
                                            (step // 2) % 2,
                                            n,
                                            ((step % 2) * 8 + col) ^ (((n // 2) % 4) * 4),
                                        ]
                                        >> (level * 16)
                                    ) & T.uint32(65535)
                                    v[rr] = T.call_pure_extern(
                                        "uint32", "nib_fast", code, book[code >> 8]
                                    )
                                else:
                                    code = T.cast(packed[slot, n, col], T.uint32)
                                    v[rr] = T.call_pure_extern(
                                        "uint32", "nib_prefill", code, book[code >> 8]
                                    )
                            for stripe in T.unroll(tile_rows // 8):
                                if not sparse_rows or stripe * 8 < count:
                                    if q4:
                                        if level == 0:
                                            multiply(lo, v, a, nt, stripe, 8)
                                        else:
                                            multiply(lo, v, a, nt, stripe, 2)
                                    else:
                                        multiply(lo, hi, v, a, nt, stripe)
        if q4:
            store(lo, Sum, WS, AS, Out, bx, warp, lane, count, expert, offset)
        else:
            store(lo, hi, Sum, WS, AS, Out, bx, warp, lane, count, expert, offset, offset)

    @T.prim_func
    def exact_e8p_int4(
        Low: T.Tensor((rows * K // 2,), T.int8),
        High: T.Tensor((rows * K // 2,), T.int8),
        Sum: T.Tensor((rows,), T.int32),
        P: T.Tensor((E, K // packed_k, N, packed_k // 8), packed_dtype),
        Book: T.Tensor((256, 1) if q4 else (1, 256, 1), T.uint32),
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
            T.annotate_min_blocks_per_sm(minimum_blocks)
            tid = T.get_thread_binding()
            packed = T.alloc_shared((2, BN, 16) if q4 else (4, BN, 8), packed_dtype)
            inputs = T.alloc_shared((4, 2, BM, input_stride), T.int8)
            book = T.alloc_shared((256,), T.uint32)
            if tile < TC[0]:
                if q4:
                    metadata = T.alloc_local((3,), T.int32)
                    metadata[0] = TE[tile]
                    metadata[1] = Offsets[metadata[0]] + TR[tile]
                    metadata[2] = T.min(BM, Counts[metadata[0]] - TR[tile])
                    expert, offset, count = metadata[0], metadata[1], metadata[2]
                    for i in T.Parallel(256):
                        book[i] = T.call_pure_extern("uint32", "direct_nibble_basis", Book[i, 0])
                else:
                    expert, first = TE[tile], TR[tile]
                    offset = Offsets[expert] + first
                    count = T.min(BM, Counts[expert] - first)
                    for i in T.unroll(2):
                        book[tid + i * 128] = Book[0, tid + i * 128, 0]
                project(
                    Low,
                    High,
                    Sum,
                    P,
                    WS,
                    AS,
                    Out,
                    packed,
                    inputs,
                    book,
                    bx,
                    expert,
                    offset,
                    count,
                )

    return exact_e8p_int4


@orin_jit
def dispatch_prefill_a8(M, K=2560, E=512, top_k=10, dynamic_rows=False):
    if any(type(x) is not int or x <= 0 for x in (M, K, E, top_k)) or K % 128 or top_k > E:
        raise ValueError("Invalid direct E8P dispatch geometry")
    BM = 4
    W = max(32, 1 << ((K // 8 - 1).bit_length()))
    M = row_count(M, dynamic_rows)
    slots = M * top_k

    @T.prim_func
    def dispatch_planes(
        A: T.Tensor((M, K), T.int8),
        Scale: T.Tensor((M, 1), T.float16),
        IDs: T.Tensor((M, top_k), T.int32),
        RelativeSlot: T.Tensor((M, top_k), T.int32),
        RowOffsets: T.Tensor((E,), T.int32),
        Low: T.Tensor((K // 64, slots, 8), T.uint32),
        High: T.Tensor((K // 64, slots, 8), T.uint32),
        Sum: T.Tensor((slots,), T.int32),
        DS: T.Tensor((slots, 1), T.float16),
        SlotMap: T.Tensor((M, top_k), T.int32),
    ):
        with T.Kernel(T.ceildiv(slots, BM), threads=BM * 32) as by:
            T.import_source(SOURCE)
            sums = T.alloc_fragment((BM, W), T.int32)
            total = T.alloc_fragment((BM,), T.int32)
            T.annotate_layout(
                {
                    sums: tilelang.Fragment(
                        (BM, W),
                        forward_thread_fn=lambda i, j: i * 32 + j % 32,
                        forward_index_fn=lambda i, j: j // 32,
                    )
                }
            )
            for i, j in T.Parallel(BM, W):
                assignment = by * BM + i
                row = assignment // top_k
                rank = assignment % top_k
                sums[i, j] = 0
                if assignment < slots and j < K // 8:
                    slot = RowOffsets[IDs[row, rank]] + RelativeSlot[row, rank]
                    x = T.ldg32(A[row, j * 8])
                    y = T.ldg32(A[row, j * 8 + 4])
                    Low[j // 8, slot, j % 8] = low_plane(x, y)
                    High[j // 8, slot, j % 8] = high_plane(x, y)
                    sums[i, j] = T.call_pure_extern("int32", "nib_sum", x, y)
            T.reduce_sum(sums, total, dim=1)
            for i in T.Parallel(BM):
                assignment = by * BM + i
                row = assignment // top_k
                rank = assignment % top_k
                if assignment < slots:
                    slot = RowOffsets[IDs[row, rank]] + RelativeSlot[row, rank]
                    Sum[slot] = total[i]
                    DS[slot, 0] = Scale[row, 0]
                    SlotMap[row, rank] = slot

    return dispatch_planes
