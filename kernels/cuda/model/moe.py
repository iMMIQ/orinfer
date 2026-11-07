"""Explicit routing and deterministic mixture reduction for SM87 MoE.

Router logits and probabilities remain FP32. Top-k uses ascending expert IDs
to break exact ties, and can normalize over the selected experts or all experts.
Dispatch owns the output-slot map. Optional expert expansion uses a reusable
temporary workspace and retains the single packed weight bank.
"""

import tilelang.language as T

from orinfer_build.compiler.cuda.common import orin_jit
from kernels.shared.rows import row_count


@orin_jit
def router_projection(M, H, E, dynamic_rows=False, block_m=16, block_n=32, block_k=128, stages=2):
    """One floating projection for the router and the shared expert's gate."""
    from kernels.cuda.model.floating_core import mma_loop

    if (
        min(M, H, E) <= 0
        or block_m not in (16, 32, 64, 128)
        or block_n not in (32, 64, 128)
        or block_k not in (128, 256)
        or H % block_k
        or stages not in (1, 2)
    ):
        raise ValueError("Invalid floating router geometry")
    m = row_count(M, dynamic_rows)
    product = mma_loop(m, H, block_m, block_n, block_k, dtype="float16", stages=stages)

    @T.prim_func
    def main(
        X: T.Tensor((m, H), T.float16),
        W: T.Tensor((E + 1, H), T.float16),
        Logits: T.Tensor((m, E), T.float32),
        SharedGate: T.Tensor((m,), T.float16),
    ):
        with T.Kernel(T.ceildiv(m, block_m), T.ceildiv(E + 1, block_n), threads=128) as (by, bx):
            acc = T.alloc_fragment((block_m, block_n), T.float32)
            product(X, W, acc, by, bx, 0)
            for i, j in T.Parallel(block_m, block_n):
                if by * block_m + i < m:
                    if bx * block_n + j < E:
                        Logits[by * block_m + i, bx * block_n + j] = acc[i, j]
                    elif bx * block_n + j == E:
                        SharedGate[by * block_m + i] = acc[i, j]

    return main


@orin_jit
def integer_experts(
    M,
    N,
    K,
    E,
    top_k,
    *,
    codec="e8p_q4",
    block_m=16,
    block_n=64,
    block_k=128,
    stages=2,
    dynamic_rows=False,
    tail_m=0,
    minimum_blocks=1,
):
    """Routed bank/row views around the shared dense integer MMA computation."""
    from kernels.cuda.model.integer_core import tensor_core, tensor_core_tiles
    from kernels.cuda.model.e8p_q4 import decode_source

    if (
        min(M, N, K, E, top_k) <= 0
        or top_k > E
        or K % 128
        or block_m not in (8, 16, 32, 64, 128, 256)
        or block_n not in (32, 64, 128)
        or codec not in ("w8", "e8p_q4")
        or (codec == "w8" and (block_m == 8 or block_n == 32))
        or block_k not in (128, 256)
        or K % block_k
        or stages not in (1, 2)
        or type(minimum_blocks) is not int
        or minimum_blocks <= 0
    ):
        raise ValueError("Invalid grouped integer projection geometry")
    tiles = (M * top_k + block_m - 1) // block_m + min(E, M * top_k)
    m = row_count(M, dynamic_rows)
    rows = m * top_k
    packed = codec == "e8p_q4"
    threads = 64 if packed and block_n == 32 else 128
    operands = dict(
        stages=stages,
        codec=codec,
        source=decode_source((4, 1)) if packed else "",
        packed_k=128 if packed else 0,
        grouped=True,
    )
    if tail_m:
        _, product = tensor_core_tiles(
            rows, N, K, block_m, block_n, block_k, tail_m=tail_m, **operands
        )
    else:
        product = tensor_core(
            rows,
            N,
            K,
            block_m,
            block_n,
            block_k,
            **operands,
        )

    @T.prim_func
    def main(
        A: T.Tensor((rows, K), T.int8),
        W: T.Tensor((E, K // 128, N, 16) if packed else (E, N, K), T.uint32 if packed else T.int8),
        Book: T.Tensor((256, 1), T.uint32),
        WS: T.Tensor((E, N), T.float16),
        AS: T.Tensor((rows,), T.float16),
        Counts: T.Tensor((E,), T.int32),
        Offsets: T.Tensor((E,), T.int32),
        TileExpert: T.Tensor((tiles,), T.int32),
        TileRow: T.Tensor((tiles,), T.int32),
        TileCount: T.Tensor((1,), T.int32),
        Out: T.Tensor((rows, N), T.float16),
    ):
        with T.Kernel(
            T.ceildiv(N, block_n), T.ceildiv(rows, block_m) + T.min(E, rows), threads=threads
        ) as (bx, tile):
            T.annotate_min_blocks_per_sm(minimum_blocks)
            metadata = T.alloc_local((3,), T.int32)
            if tile < TileCount[0]:
                # Routing is immutable for this launch. Materialize these loads
                # before barriers and MMA so opaque operations cannot cause the
                # compiler to reload the address expressions inside the K loop.
                metadata[0] = TileExpert[tile]
                metadata[1] = Offsets[metadata[0]] + TileRow[tile]
                metadata[2] = T.min(block_m, Counts[metadata[0]] - TileRow[tile])
                product(A, W, Book, WS, WS, AS, Out, 0, bx, metadata[0], metadata[1], metadata[2])

    return main


@orin_jit
def expand_experts(N, K, E, block_n=64, block_k=256):
    """Decode only routed banks into a reusable, temporary INT8 workspace."""
    from kernels.cuda.model.e8p_q4 import decode_source
    from kernels.cuda.model.integer_core import prepare_e8p_book, weight_loader

    if min(N, K, E) <= 0 or N % block_n or K % block_k:
        raise ValueError("Invalid routed E8P expansion geometry")
    load = weight_loader(
        "e8p_q4", N, block_n, block_k, decode_source((4, 1)), packed_k=128, grouped=True
    )

    @T.prim_func
    def main(
        W: T.Tensor((E, K // 128, N, 16), T.uint32),
        Book: T.Tensor((256, 1), T.uint32),
        Counts: T.Tensor((E,), T.int32),
        Out: T.Tensor((E, N, K), T.int8),
    ):
        with T.Kernel(N // block_n, K // block_k, E, threads=128) as (bx, kg, expert):
            if Counts[expert] > 0:
                book = T.alloc_shared((512, 1), T.uint32)
                tile = T.alloc_shared((block_n, block_k), T.int8)
                prepare_e8p_book(Book, book)
                load(W, book, tile, bx, kg, expert)
                T.copy(tile, Out[expert, bx * block_n, kg * block_k])

    return main


@orin_jit
def expert_histogram(M: int, E: int = 512, top_k: int = 10, dynamic_rows: bool = False):
    """Build (IDs, Counts[E], RelativeSlot[M,top_k]), all INT32.

    Stable assignment order is token first, route rank second. No atomics or
    host synchronization. IDs must be in [0,E); each token's IDs are unique.
    """
    if any(type(x) is not int or x <= 0 for x in (M, E, top_k)) or top_k > E:
        raise ValueError("Invalid MoE dispatch dimensions")
    width = 1 << (M * top_k - 1).bit_length()

    M = row_count(M, dynamic_rows)

    @T.prim_func
    def main(
        IDs: T.Tensor((M, top_k), T.int32),
        Counts: T.Tensor((E,), T.int32),
        RelativeSlot: T.Tensor((M, top_k), T.int32),
    ):
        with T.Kernel(E, threads=128) as expert:
            matches = T.alloc_fragment((width,), T.int32)
            prefix = T.alloc_fragment((width,), T.int32)
            for j in T.Parallel(width):
                matches[j] = T.if_then_else(
                    j < M * top_k, T.cast(IDs[j // top_k, j % top_k] == expert, T.int32), 0
                )
                prefix[j] = matches[j]
            T.cumsum(prefix, dim=0)
            for j in T.Parallel(width):
                if j == width - 1:
                    Counts[expert] = prefix[j]
                if j < M * top_k and matches[j] != 0:
                    RelativeSlot[j // top_k, j % top_k] = prefix[j] - 1

    return main


@orin_jit
def expert_offsets(E: int = 512, block_m: int = 16):
    """Build (Counts, RowOffsets[E], TileOffsets[E], TileCount[1]), INT32."""
    if type(E) is not int or E <= 0 or block_m not in (8, 16, 32, 64, 128, 256):
        raise ValueError("Invalid MoE expert prefix dimensions")
    width = 1 << (E - 1).bit_length()

    @T.prim_func
    def main(
        Counts: T.Tensor((E,), T.int32),
        RowOffsets: T.Tensor((E,), T.int32),
        TileOffsets: T.Tensor((E,), T.int32),
        TileCount: T.Tensor((1,), T.int32),
    ):
        with T.Kernel(1, threads=128):
            counts = T.alloc_fragment((width,), T.int32)
            rows = T.alloc_fragment((width,), T.int32)
            tiles = T.alloc_fragment((width,), T.int32)
            for e in T.Parallel(width):
                counts[e] = T.if_then_else(e < E, Counts[e], 0)
                rows[e] = counts[e]
                tiles[e] = T.ceildiv(counts[e], block_m)
            T.cumsum(rows, dim=0)
            T.cumsum(tiles, dim=0)
            for e in T.Parallel(E):
                RowOffsets[e] = rows[e] - counts[e]
                TileOffsets[e] = tiles[e] - T.ceildiv(counts[e], block_m)
            for e in T.Parallel(width):
                if e == width - 1:
                    TileCount[0] = tiles[e]

    return main


@orin_jit
def expert_tiles(M: int, E: int = 512, top_k: int = 10, block_m: int = 16, dynamic_rows=False):
    """Build (Counts, TileOffsets, TileExpert, TileRow), all INT32.

    Output capacity ceil(M*top_k/block_m)+min(E,M*top_k) bounds all distributions.
    Only the prefix indicated by expert_offsets.TileCount is initialized/read.
    """
    if any(type(x) is not int or x <= 0 for x in (M, E, top_k)) or block_m not in (
        8,
        16,
        32,
        64,
        128,
        256,
    ):
        raise ValueError("Invalid MoE tile dimensions")
    assignments = M * top_k
    capacity = (assignments + block_m - 1) // block_m + min(E, assignments)
    M = row_count(M, dynamic_rows)
    expert_capacity = (M + block_m - 1) // block_m

    @T.prim_func
    def main(
        Counts: T.Tensor((E,), T.int32),
        TileOffsets: T.Tensor((E,), T.int32),
        TileExpert: T.Tensor((capacity,), T.int32),
        TileRow: T.Tensor((capacity,), T.int32),
    ):
        with T.Kernel(E, threads=128) as e:
            for tile in T.Parallel(expert_capacity):
                if tile * block_m < Counts[e]:
                    destination = TileOffsets[e] + tile
                    TileExpert[destination] = e
                    TileRow[destination] = tile * block_m

    return main


@orin_jit
def expert_dispatch(
    M: int,
    H: int,
    E: int = 512,
    top_k: int = 10,
    scale_group: int = 64,
    dynamic_rows: bool = False,
):
    """Build (A, Scale, IDs, RelativeSlot, RowOffsets, D, DS, SlotMap).

    A[M,H]/D[M*top_k,H] are INT8; Scale/DS are FP16. Others INT32.
    scale_group defaults to64; H uses one per-token scale for Q2I8.
    Dispatch copies quantized activations only. Source/destination must be disjoint.
    Inputs come from router_topk/expert_histogram/expert_offsets, without invalid
    or duplicate per-token IDs. The SlotMap is consumed by moe_combine.
    """
    if any(type(x) is not int or x <= 0 for x in (M, H, E, top_k)) or H % 64 or top_k > E:
        raise ValueError("Invalid MoE activation dispatch dimensions")
    if type(scale_group) is not int or scale_group <= 0 or H % scale_group:
        raise ValueError("Invalid activation scale group")
    groups = H // scale_group
    M = row_count(M, dynamic_rows)

    @T.prim_func
    def main(
        A: T.Tensor((M, H), T.int8),
        Scale: T.Tensor((M, groups), T.float16),
        IDs: T.Tensor((M, top_k), T.int32),
        RelativeSlot: T.Tensor((M, top_k), T.int32),
        RowOffsets: T.Tensor((E,), T.int32),
        D: T.Tensor((M * top_k, H), T.int8),
        DS: T.Tensor((M * top_k, groups), T.float16),
        SlotMap: T.Tensor((M, top_k), T.int32),
    ):
        with T.Kernel(M * top_k, threads=128) as assignment:
            row, rank = assignment // top_k, assignment % top_k
            slot = RowOffsets[IDs[row, rank]] + RelativeSlot[row, rank]
            SlotMap[row, rank] = slot
            for col in T.Parallel(H):
                D[slot, col] = A[row, col]
            for group in T.Parallel(groups):
                DS[slot, group] = Scale[row, group]

    return main


@orin_jit
def router_topk_warp(M: int, E: int, top_k: int, dynamic_rows=False, warps: int = 4):
    """One warp per row, register candidates and native SM80 integer reductions.

    Ordered FP32 keys retain exact scores and signed-zero equivalence. Ranking
    uses two 32-bit warp reductions instead of a cross-warp 64-bit reduction.
    The selected probability sum keeps the original FP32 tree and boundaries.
    """
    if min(M, E, top_k) <= 0 or E % 32 or top_k > 32 or top_k > E or warps not in (1, 2, 4):
        raise ValueError("Invalid warp router geometry")
    m = row_count(M, dynamic_rows)
    local = E // 32

    @T.prim_func
    def main(
        Logits: T.Tensor((m, E), T.float32),
        IDs: T.Tensor((m, top_k), T.int32),
        Prob: T.Tensor((m, top_k), T.float32),
    ):
        with T.Kernel(T.ceildiv(m, warps), threads=warps * 32) as block:
            tx = T.get_thread_binding()
            row = block * warps + tx // 32
            lane = tx % 32
            if row < m:
                keys = T.alloc_local((local,), T.uint32)
                best = T.alloc_var(T.uint32)
                best_id = T.alloc_var(T.int32)
                selected = T.alloc_var(T.float32, 0.0)
                maximum = T.alloc_var(T.float32, 0.0)
                for j in T.unroll(local):
                    value = Logits[row, j * 32 + lane]
                    bits = T.reinterpret(T.uint32, value)
                    keys[j] = T.if_then_else(
                        value == 0,
                        T.uint32(0x80000000),
                        T.if_then_else((bits >> 31) != 0, ~bits, bits ^ T.uint32(0x80000000)),
                    )
                for rank in T.serial(top_k):
                    best = 0
                    for j in T.unroll(local):
                        best = T.max(best, keys[j])
                    best = T.call_pure_extern(
                        "uint32", "__reduce_max_sync", T.uint32(0xFFFFFFFF), best
                    )
                    best_id = E
                    for j in T.unroll(local):
                        if keys[j] == best:
                            best_id = T.min(best_id, j * 32 + lane)
                    best_id = T.call_pure_extern(
                        "int32", "__reduce_min_sync", T.uint32(0xFFFFFFFF), best_id
                    )
                    bits = T.if_then_else((best >> 31) != 0, best ^ T.uint32(0x80000000), ~best)
                    score = T.reinterpret(T.float32, bits)
                    if rank == 0:
                        maximum = score
                    if lane == rank:
                        IDs[row, rank] = best_id
                        selected = score
                    for j in T.unroll(local):
                        if j * 32 + lane == best_id:
                            keys[j] = 0
                mass = T.alloc_var(T.float32)
                denominator = T.alloc_var(T.float32)
                mass = T.if_then_else(lane < top_k, T.exp(selected - maximum), 0.0)
                denominator = mass
                for stage in T.unroll(5):
                    denominator += T.call_pure_extern(
                        "float32",
                        "__shfl_xor_sync",
                        T.uint32(0xFFFFFFFF),
                        denominator,
                        1 << (4 - stage),
                        32,
                    )
                if lane < top_k:
                    Prob[row, lane] = mass / denominator

    return main


@orin_jit
def router_topk(
    M: int, E: int = 512, top_k: int = 10, renormalize: bool = True, dynamic_rows: bool = False
):
    """Build (Logits[M,E], IDs[M,top_k], Prob[M,top_k]), F32/I32/F32.

    Logits must be finite. The normalized route is equivalent to FP32 softmax,
    top-k and renormalization, while avoiding a full probability materialization.
    """
    if any(type(x) is not int or x <= 0 for x in (M, E, top_k)) or top_k > E:
        raise ValueError("Invalid MoE router dimensions")
    if type(renormalize) is not bool:
        raise ValueError("renormalize must be a boolean")
    width = 1 << (E - 1).bit_length()
    selected_width = 1 << (top_k - 1).bit_length()

    M = row_count(M, dynamic_rows)

    @T.prim_func
    def main(
        Logits: T.Tensor((M, E), T.float32),
        IDs: T.Tensor((M, top_k), T.int32),
        Prob: T.Tensor((M, top_k), T.float32),
    ):
        with T.Kernel(M, threads=128) as row:
            values = T.alloc_fragment((width,), T.float32)
            # A total-order key selects the score and its tie-breaking ID in
            # one reduction. Finite FP32 scores occupy the high word; smaller
            # expert IDs win equal scores. Canonicalize signed zero because
            # the original floating comparison treats both zeros as equal.
            keys = T.alloc_fragment((width,), T.uint64)
            maximum = T.alloc_fragment((1,), T.uint64)
            selected = T.alloc_shared((selected_width,), T.float32)
            numerator = T.alloc_fragment((selected_width,), T.float32)
            denominator = T.alloc_fragment((1,), T.float32)
            full_exp = T.alloc_fragment((width,), T.float32)
            global_max = T.alloc_fragment((1,), T.float32)
            for j in T.Parallel(width):
                values[j] = T.if_then_else(j < E, Logits[row, j], -T.infinity(T.float32))
                bits = T.reinterpret(values[j], T.uint32)
                ordered = T.if_then_else(
                    values[j] == 0,
                    T.uint32(0x80000000),
                    T.if_then_else(
                        (bits & T.uint32(0x80000000)) != 0,
                        ~bits,
                        bits ^ T.uint32(0x80000000),
                    ),
                )
                keys[j] = T.if_then_else(
                    j < E,
                    (T.cast(ordered, T.uint64) << 32) | T.cast(E - j, T.uint64),
                    T.uint64(0),
                )
            T.reduce_max(values, global_max, dim=0)
            if not renormalize:
                for j in T.Parallel(width):
                    full_exp[j] = T.exp(values[j] - global_max[0])
                T.reduce_sum(full_exp, denominator, dim=0)
            for rank in T.serial(top_k):
                T.reduce_max(keys, maximum, dim=0)
                selected_id = E - T.cast(maximum[0] & T.uint64(0xFFFFFFFF), T.int32)
                ordered = T.cast(maximum[0] >> 32, T.uint32)
                bits = T.if_then_else(
                    (ordered & T.uint32(0x80000000)) != 0,
                    ordered ^ T.uint32(0x80000000),
                    ~ordered,
                )
                for j in T.Parallel(width):
                    if j == selected_id:
                        keys[j] = 0
                IDs[row, rank] = selected_id
                selected[rank] = T.reinterpret(bits, T.float32)
            for rank in T.Parallel(selected_width):
                numerator[rank] = T.if_then_else(
                    rank < top_k, T.exp(selected[rank] - global_max[0]), 0.0
                )
            if renormalize:
                T.reduce_sum(numerator, denominator, dim=0)
            for rank in T.Parallel(top_k):
                Prob[row, rank] = numerator[rank] / denominator[0]

    return main


@orin_jit
def moe_combine(
    M: int,
    H: int,
    slots: int,
    top_k: int = 10,
    dynamic_rows: bool = False,
):
    """Build (Expert, SlotMap, Prob, Shared, SharedGate, Output).

    Expert[slots,H], Shared[M,H], SharedGate[M] and Output[M,H] are FP16.
    SlotMap[M,top_k] is INT32; Prob[M,top_k] is FP32. Invalid slots are ignored.
    SharedGate contains raw gate logits. Accumulation and sigmoid are FP32,
    with one final FP16 cast. Each output element has one owner; no atomics.
    """
    if any(type(x) is not int or x <= 0 for x in (M, H, slots, top_k)):
        raise ValueError("Invalid MoE mixture dimensions")
    width = 256

    M = row_count(M, dynamic_rows)
    count = M
    # Dynamic decode has exactly top_k assignments per live request. A
    # compile-time capacity bound would silently drop valid larger batches.
    if dynamic_rows:
        slots = M * top_k

    @T.prim_func
    def main(
        Expert: T.Tensor((slots, H), T.float16),
        SlotMap: T.Tensor((M, top_k), T.int32),
        Prob: T.Tensor((M, top_k), T.float32),
        Shared: T.Tensor((M, H), T.float16),
        SharedGate: T.Tensor((M,), T.float16),
        Output: T.Tensor((M, H), T.float16),
    ):
        with T.Kernel(M, T.ceildiv(H, width), threads=128) as (row, block):
            if row < count:
                acc = T.alloc_fragment((width,), T.float32)
                T.clear(acc)
                for rank in T.serial(top_k):
                    slot = SlotMap[row, rank]
                    if slot >= 0 and slot < slots:
                        for j in T.Parallel(width):
                            col = block * width + j
                            if col < H:
                                acc[j] += T.cast(Expert[slot, col], T.float32) * Prob[row, rank]
                gate = 1.0 / (1.0 + T.exp(-T.cast(SharedGate[row], T.float32)))
                for j in T.Parallel(width):
                    col = block * width + j
                    if col < H:
                        Output[row, col] = acc[j] + gate * T.cast(Shared[row, col], T.float32)
            else:
                for j in T.Parallel(width):
                    col = block * width + j
                    if col < H:
                        Output[row, col] = 0

    return main


@orin_jit
def expert_histogram_local(M, E=512, K=10, block=1024, dynamic_rows=False):
    if (
        any(type(x) is not int or x <= 0 for x in (M, E, K, block))
        or K > E
        or block not in (512, 1024, 2048)
    ):
        raise ValueError("Invalid segmented MoE histogram geometry")
    M = row_count(M, dynamic_rows)
    segments = (M * K + block - 1) // block
    count = M

    @T.prim_func
    def main(
        IDs: T.Tensor((M, K), T.int32),
        Partial: T.Tensor((E, segments), T.int32),
        Slots: T.Tensor((M, K), T.int32),
    ):
        with T.Kernel(segments, E, threads=128) as (segment, expert):
            matches = T.alloc_fragment((block,), T.int32)
            prefix = T.alloc_fragment((block,), T.int32)
            for j in T.Parallel(block):
                assignment = segment * block + j
                matches[j] = 0
                if assignment < count * K:
                    matches[j] = T.cast(IDs[assignment // K, assignment % K] == expert, T.int32)
                prefix[j] = matches[j]
            T.cumsum(prefix, dim=0)
            for j in T.Parallel(block):
                assignment = segment * block + j
                if j == block - 1:
                    Partial[expert, segment] = prefix[j]
                if assignment < count * K and matches[j] != 0:
                    Slots[assignment // K, assignment % K] = prefix[j] - 1

    return main


@orin_jit
def expert_histogram_offsets(M, E=512, K=10, block=1024, dynamic_rows=False):
    if (
        any(type(x) is not int or x <= 0 for x in (M, E, K, block))
        or K > E
        or block not in (512, 1024, 2048)
    ):
        raise ValueError("Invalid segmented MoE histogram geometry")
    segments = (M * K + block - 1) // block
    width = 1 << (segments - 1).bit_length()
    M = row_count(M, dynamic_rows)
    segments = (M * K + block - 1) // block

    @T.prim_func
    def main(
        Partial: T.Tensor((E, segments), T.int32),
        Offsets: T.Tensor((E, segments), T.int32),
        Counts: T.Tensor((E,), T.int32),
    ):
        with T.Kernel(E, threads=128) as expert:
            prefix = T.alloc_fragment((width,), T.int32)
            for j in T.Parallel(width):
                prefix[j] = 0
                if j < segments:
                    prefix[j] = Partial[expert, j]
            T.cumsum(prefix, dim=0)
            for j in T.Parallel(width):
                if j < segments:
                    Offsets[expert, j] = prefix[j] - Partial[expert, j]
                if j == width - 1:
                    Counts[expert] = prefix[j]

    return main


@orin_jit
def expert_histogram_finish(M, E=512, K=10, block=1024, dynamic_rows=False):
    if (
        any(type(x) is not int or x <= 0 for x in (M, E, K, block))
        or K > E
        or block not in (512, 1024, 2048)
    ):
        raise ValueError("Invalid segmented MoE histogram geometry")
    M = row_count(M, dynamic_rows)
    segments = (M * K + block - 1) // block
    count = M
    assignments = M * K

    @T.prim_func
    def main(
        IDs: T.Tensor((M, K), T.int32),
        Offsets: T.Tensor((E, segments), T.int32),
        Slots: T.Tensor((M, K), T.int32),
    ):
        with T.Kernel(T.ceildiv(assignments, 256), threads=128) as tile:
            for j in T.Parallel(256):
                assignment = tile * 256 + j
                if assignment < count * K:
                    row, rank = assignment // K, assignment % K
                    Slots[row, rank] += Offsets[IDs[row, rank], assignment // block]

    return main


@orin_jit
def single_route_metadata(E: int = 512, top_k: int = 10, capacity: int = 11):
    """Single-token dispatch in expert-ID order, with no atomics or host reads.

    IDs must be distinct and in [0, E), as guaranteed by router_topk. Each
    selected expert owns exactly one row/tile. Preserve the unused tile tail.
    """
    if (
        any(type(v) is not int or v <= 0 for v in (E, top_k, capacity))
        or not top_k <= E
        or capacity < top_k
    ):
        raise ValueError("Invalid single-token MoE routing geometry")

    @T.prim_func
    def single_route_metadata_kernel(
        IDs: T.Tensor((1, top_k), T.int32),
        Counts: T.Tensor((E,), T.int32),
        Relative: T.Tensor((1, top_k), T.int32),
        Offsets: T.Tensor((E,), T.int32),
        TileOffsets: T.Tensor((E,), T.int32),
        TileExpert: T.Tensor((capacity,), T.int32),
        TileRow: T.Tensor((capacity,), T.int32),
        TileCount: T.Tensor((1,), T.int32),
    ):
        with T.Kernel(1, threads=128):
            ids = T.alloc_shared((top_k,), T.int32)
            T.copy(IDs[0, 0], ids)
            counts = T.alloc_fragment((E,), T.int32)
            offsets = T.alloc_fragment((E,), T.int32)
            T.clear(counts)
            T.clear(offsets)
            for rank in T.unroll(top_k):
                for expert in T.Parallel(E):
                    counts[expert] += T.cast(ids[rank] == expert, T.int32)
                    offsets[expert] += T.cast(ids[rank] < expert, T.int32)
            T.copy(counts, Counts)
            T.copy(offsets, Offsets)
            T.copy(offsets, TileOffsets)
            for rank in T.Parallel(top_k):
                position = T.alloc_var(T.int32)
                position = 0
                for other in T.unroll(top_k):
                    position += T.cast(ids[other] < ids[rank], T.int32)
                Relative[0, rank] = 0
                TileExpert[position] = ids[rank]
                TileRow[position] = 0
            if T.get_thread_binding() == 0:
                TileCount[0] = top_k

    return single_route_metadata_kernel


@orin_jit
def route_metadata(M: int, E: int = 512, top_k: int = 10, block_m: int = 8, dynamic_rows=False):
    """One block emits stable counts, relative slots and matching expert tiles.

    Each expert owns its counter; assignments are visited in token/rank order.
    Only live assignments are scanned. Unique top-k IDs bound each expert's
    count by M, and each tile is written by its expert without atomics.
    """
    if not 1 <= M <= 1024 or E != 512 or top_k != 10 or block_m not in (8, 16):
        raise ValueError("Invalid decode route metadata geometry")
    capacity = (M * top_k + block_m - 1) // block_m + min(E, M * top_k)
    M = row_count(M, dynamic_rows)

    @T.prim_func
    def rows_route_metadata(
        IDs: T.Tensor((M, top_k), T.int32),
        Counts: T.Tensor((E,), T.int32),
        RelativeSlot: T.Tensor((M, top_k), T.int32),
        RowOffsets: T.Tensor((E,), T.int32),
        TileOffsets: T.Tensor((E,), T.int32),
        TileCount: T.Tensor((1,), T.int32),
        TileExpert: T.Tensor((capacity,), T.int32),
        TileRow: T.Tensor((capacity,), T.int32),
    ):
        with T.Kernel(1, threads=512):
            counts = T.alloc_fragment((E,), T.int32)
            rows = T.alloc_fragment((E,), T.int32)
            tiles = T.alloc_fragment((E,), T.int32)
            T.clear(counts)
            for assignment in T.serial(M * top_k):
                for expert in T.Parallel(E):
                    if IDs[assignment // top_k, assignment % top_k] == expert:
                        RelativeSlot[assignment // top_k, assignment % top_k] = counts[expert]
                        counts[expert] += 1
            for expert in T.Parallel(E):
                Counts[expert] = counts[expert]
                rows[expert] = counts[expert]
                tiles[expert] = T.ceildiv(counts[expert], block_m)
            T.cumsum(rows, dim=0)
            T.cumsum(tiles, dim=0)
            for expert in T.Parallel(E):
                RowOffsets[expert] = rows[expert] - counts[expert]
                TileOffsets[expert] = tiles[expert] - T.ceildiv(counts[expert], block_m)
                if expert == E - 1:
                    TileCount[0] = tiles[expert]
            for tile in T.serial(T.ceildiv(M, block_m)):
                for expert in T.Parallel(E):
                    if tile * block_m < counts[expert]:
                        index = tiles[expert] - T.ceildiv(counts[expert], block_m) + tile
                        TileExpert[index] = expert
                        TileRow[index] = tile * block_m

    return rows_route_metadata
