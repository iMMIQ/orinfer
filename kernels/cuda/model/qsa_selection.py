"""Exact partitioned QSA selection on SM87.

Each CTA retains its local top-512 using the original nonnegative FP32 bits.
A bounded union contains every global winner, including lower-ID ties. The
merge preserves greater-then-equal ascending IDs and the incomplete tail.
"""

import tilelang.language as T
from kernels.cuda.model.qsa import geometry
from kernels.shared.rows import row_count
from orinfer_build.compiler.cuda.common import orin_jit


def _direct_limit(blocks: int, query_major: bool):
    if type(blocks) is not int or blocks not in (0, 2048) or (blocks and not query_major):
        raise ValueError("Direct sparse QSA selection requires query-major scores")


@orin_jit
def select(
    m: int,
    capacity: int,
    dynamic_rows=False,
    tile: int = 512,
    threads: int = 128,
    parts: int = 2,
    *,
    query_major: bool = False,
    dense_scores: bool = False,
    direct_blocks: int = 0,
):
    blocks = geometry(m, capacity)
    if tile not in (256, 512, 1024) or threads not in (128, 256) or parts not in (2, 4, 8):
        raise ValueError("Invalid QSA selection geometry")
    if dense_scores and not query_major:
        raise ValueError("Direct dense QSA selection requires query-major scores")
    _direct_limit(direct_blocks, dense_scores)
    partitioned_after = max(512, direct_blocks)
    m = row_count(m, dynamic_rows)
    score_shape = (m * blocks,) if query_major else (blocks, m)

    def histogram(length):

        @T.macro
        def accumulate(Values, Hist, Control, count, digit):
            for i in T.Parallel(length):
                bucket = T.alloc_var(T.int32, 256)
                if i < count:
                    key = Values[i]
                    if digit == 0 or key >> (32 - digit * 8) == Control[1] >> (32 - digit * 8):
                        bucket = (key >> (24 - digit * 8)) & 255
                peers = T.call_pure_extern(
                    "uint32", "__match_any_sync", T.uint32(4294967295), bucket
                )
                if (
                    bucket < 256
                    and T.get_thread_binding() % 32
                    == T.call_pure_extern("int32", "__ffs", peers) - 1
                ):
                    T.atomic_add(Hist[bucket], T.call_pure_extern("int32", "__popc", peers))

        return accumulate

    tile_histogram = histogram(tile)

    @T.macro
    def choose(Hist, Control, digit):
        cumulative = T.alloc_fragment((256,), T.int32)
        T.sync_threads()
        for i in T.Parallel(256):
            cumulative[i] = Hist[255 - i]
        T.cumsum(cumulative, dim=0)
        for i in T.Parallel(256):
            before = cumulative[i] - Hist[255 - i]
            if before < Control[2] and cumulative[i] >= Control[2]:
                Control[1] = Control[1] | ((255 - i) << (24 - digit * 8))
                Control[6] = Control[2] - before
        T.sync_threads()
        if T.get_thread_binding() == 0:
            Control[2] = Control[6]
        T.sync_threads()

    @T.prim_func
    def main(
        Scores: T.Tensor(score_shape, T.float32),
        Position: T.Tensor((1,), T.int32),
        Selected: T.Tensor((m, parts * 1024), T.int32),
    ):
        with T.Kernel(m, parts, threads=threads) as (row, part):
            hist = T.alloc_shared((256,), T.int32)
            control = T.alloc_shared((7,), T.int32)
            equal_ids = T.alloc_shared((512,), T.int32)
            values = T.alloc_fragment((tile,), T.int32)
            flags = T.alloc_fragment((tile,), T.int32)
            ranks = T.alloc_fragment((tile,), T.int32)
            equals = T.alloc_fragment((tile,), T.int32)
            equal_ranks = T.alloc_fragment((tile,), T.int32)
            live_blocks = (Position[0] + m) // 4
            causal_full = (Position[0] + row + 1) // 4
            start = part * T.ceildiv(causal_full, parts)
            full = T.max(0, T.min(T.ceildiv(causal_full, parts), causal_full - start))
            tiles = T.ceildiv(full, tile)
            # Bounded rows read Scores in the merger. A local
            # partition that fits can retain its candidates in ID order.
            if not dense_scores or causal_full > partitioned_after:
                if dense_scores and full <= 512:
                    for i in T.Parallel(512):
                        key = T.alloc_var(T.int32, -1)
                        identifier = T.alloc_var(T.int32, -1)
                        if i < full:
                            key = T.reinterpret(T.int32, Scores[row * live_blocks + start + i])
                            identifier = start + i
                        Selected[row, part * 1024 + i] = key
                        Selected[row, part * 1024 + 512 + i] = identifier
                else:
                    T.clear(control)
                    if T.get_thread_binding() == 0:
                        control[3] = -1
                    T.sync_threads()
                    if full > 0:
                        if T.get_thread_binding() == 0:
                            control[2] = T.min(512, full)
                        T.sync_threads()
                        for digit in T.unroll(4):
                            T.clear(hist)
                            T.sync_threads()
                            for block in T.serial(tiles):
                                for i in T.Parallel(tile):
                                    values[i] = 0
                                    if block * tile + i < full:
                                        if query_major:
                                            values[i] = T.reinterpret(
                                                T.int32,
                                                Scores[
                                                    row * live_blocks + start + block * tile + i
                                                ],
                                            )
                                        else:
                                            values[i] = T.reinterpret(
                                                T.int32, Scores[start + block * tile + i, row]
                                            )
                                tile_histogram(
                                    values, hist, control, T.min(tile, full - block * tile), digit
                                )
                            choose(hist, control, digit)
                        if T.get_thread_binding() == 0:
                            control[3] = control[1]
                        T.sync_threads()
                    for i in T.Parallel(1024):
                        Selected[row, part * 1024 + i] = -1
                    for block in T.serial(tiles):
                        for i in T.Parallel(tile):
                            flags[i] = 0
                            equals[i] = 0
                            if block * tile + i < full:
                                key = T.alloc_var(T.int32)
                                if query_major:
                                    key = T.reinterpret(
                                        T.int32,
                                        Scores[row * live_blocks + start + block * tile + i],
                                    )
                                else:
                                    key = T.reinterpret(
                                        T.int32, Scores[start + block * tile + i, row]
                                    )
                                values[i] = key
                                flags[i] = T.cast(key > control[3], T.int32)
                                equals[i] = T.cast(key == control[3], T.int32)
                            ranks[i] = flags[i]
                            equal_ranks[i] = equals[i]
                        T.cumsum(ranks, dim=0)
                        T.cumsum(equal_ranks, dim=0)
                        for i in T.Parallel(tile):
                            if flags[i] != 0:
                                Selected[row, part * 1024 + control[4] + ranks[i] - 1] = values[i]
                                Selected[row, part * 1024 + 512 + control[4] + ranks[i] - 1] = (
                                    start + block * tile + i
                                )
                            if equals[i] != 0 and control[5] + equal_ranks[i] <= 512:
                                equal_ids[control[5] + equal_ranks[i] - 1] = (
                                    start + block * tile + i
                                )
                        T.sync_threads()
                        for i in T.Parallel(tile):
                            if i == tile - 1:
                                control[4] += ranks[i]
                                control[5] += equal_ranks[i]
                        T.sync_threads()
                    for i in T.Parallel(512):
                        if i < T.min(full, 512) - control[4]:
                            Selected[row, part * 1024 + control[4] + i] = control[3]
                            Selected[row, part * 1024 + 512 + control[4] + i] = equal_ids[i]

    symbol = "qsa_select_dense_partial" if dense_scores else "qsa_select_partial"
    return main.with_attr("global_symbol", symbol)


@orin_jit
def merge(
    m: int,
    parts: int,
    dynamic_rows=False,
    threads: int = 128,
    *,
    warp_sort=False,
    score_capacity: int = 0,
    direct_blocks: int = 0,
):
    """Merge bounded local winners, then sort IDs in the original emission order.

    No global score rescan or cluster synchronization is needed. Local top-512
    contains every possible global winner, including lower-ID score ties.
    """
    if parts not in (2, 4, 8) or threads not in (128, 256):
        raise ValueError("Invalid QSA candidate merge geometry")
    _direct_limit(direct_blocks, score_capacity > 0)
    blocks = geometry(m, score_capacity) if score_capacity else 0
    m = row_count(m, dynamic_rows)
    if score_capacity:
        from tvm import tirx as tir

        scores = tir.decl_buffer((m * blocks,), "float32", name="Scores")
    width = parts * 512

    source = (
        f"#define QSA_SORT_WIDTH {width}\n#define QSA_SORT_THREADS {threads}\n"
        + r"""
__device__ __forceinline__ void qsa_sort_keys(int* keys) {
    // A sub-warp XOR exchanges elements at the same register index.
    int current[QSA_SORT_WIDTH / QSA_SORT_THREADS];
    const int lane = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < QSA_SORT_WIDTH / QSA_SORT_THREADS; ++i)
        current[i] = keys[i * QSA_SORT_THREADS + lane];
    __syncthreads();
    #pragma unroll
    for (int width = 2; width <= QSA_SORT_WIDTH; width *= 2) {
        #pragma unroll
        for (int distance = width / 2; distance >= 1; distance /= 2) {
            if (distance >= 32) {
                #pragma unroll
                for (int i = 0; i < QSA_SORT_WIDTH / QSA_SORT_THREADS; ++i)
                    keys[i * QSA_SORT_THREADS + lane] = current[i];
                __syncthreads();
            }
            #pragma unroll
            for (int i = 0; i < QSA_SORT_WIDTH / QSA_SORT_THREADS; ++i) {
                const int index = i * QSA_SORT_THREADS + lane;
                const int value = current[i];
                const int peer = distance < 32
                    ? __shfl_xor_sync(0xffffffffu, value, distance)
                    : keys[index ^ distance];
                const bool smaller = ((index & width) == 0) == ((index & distance) == 0);
                current[i] = smaller ? min(value, peer) : max(value, peer);
            }
            if (distance >= 32)
                __syncthreads();
        }
    }
    #pragma unroll
    for (int i = 0; i < QSA_SORT_WIDTH / QSA_SORT_THREADS; ++i)
        keys[i * QSA_SORT_THREADS + lane] = current[i];
}
"""
    )

    def cutoff(length):
        @T.macro
        def search(values, hist, control, cumulative):
            for digit in T.unroll(4):
                T.clear(hist)
                T.sync_threads()
                for i in T.Parallel(length):
                    bucket = T.alloc_var(T.int32, 256)
                    key = values[i]
                    if key >= 0:
                        if digit == 0 or key >> 32 - digit * 8 == control[0] >> 32 - digit * 8:
                            bucket = key >> 24 - digit * 8 & 255
                    peers = T.call_pure_extern(
                        "uint32", "__match_any_sync", T.uint32(4294967295), bucket
                    )
                    if (
                        bucket < 256
                        and T.get_thread_binding() % 32
                        == T.call_pure_extern("int32", "__ffs", peers) - 1
                    ):
                        T.atomic_add(hist[bucket], T.call_pure_extern("int32", "__popc", peers))
                T.sync_threads()
                for i in T.Parallel(256):
                    cumulative[i] = hist[255 - i]
                T.cumsum(cumulative, dim=0)
                for i in T.Parallel(256):
                    before = cumulative[i] - hist[255 - i]
                    if before < control[1] and cumulative[i] >= control[1]:
                        control[0] = control[0] | 255 - i << 24 - digit * 8
                        control[2] = control[1] - before
                T.sync_threads()
                if T.get_thread_binding() == 0:
                    control[1] = control[2]
                T.sync_threads()

        return search

    merged_cutoff = cutoff(width)
    bounded_cutoff = cutoff(direct_blocks) if direct_blocks else None
    tie_stride = 4096

    @T.prim_func
    def main(
        Candidates: T.Tensor((m, parts * 1024), T.int32),
        Position: T.Tensor((1,), T.int32),
        Selected: T.Tensor((m, 2051), T.int32),
    ):
        with T.Kernel(m, threads=threads) as row:
            values = T.alloc_shared((width,), T.int32)
            keys = T.alloc_shared((width,), T.int32)
            hist = T.alloc_shared((256,), T.int32)
            control = T.alloc_shared((3,), T.int32)
            cumulative = T.alloc_fragment((256,), T.int32)
            next_keys = T.alloc_fragment((width,), T.int32)
            full = (Position[0] + row + 1) // 4
            tail = (Position[0] + row + 1) % 4
            if score_capacity and full <= 512:
                # All blocks survive. Preserve the original output order:
                # IDs above the cutoff first, then minimum-score ties by ID.
                score = T.alloc_fragment((512,), T.int32)
                minimum = T.alloc_fragment((1,), T.int32)
                flags = T.alloc_fragment((512,), T.int32)
                ranks = T.alloc_fragment((512,), T.int32)
                chosen = T.alloc_fragment((1,), T.int32)
                live_blocks = (Position[0] + m) // 4
                for i in T.Parallel(512):
                    score[i] = 2147483647
                    if i < full:
                        score[i] = T.reinterpret(T.int32, scores[row * live_blocks + i])
                T.reduce_min(score, minimum, dim=0)
                for i in T.Parallel(512):
                    flags[i] = T.cast(i < full and score[i] > minimum[0], T.int32)
                    ranks[i] = flags[i]
                T.cumsum(ranks, dim=0)
                T.reduce_sum(flags, chosen, dim=0)
                for i in T.Parallel(2051):
                    if i >= full * 4:
                        Selected[row, i] = T.if_then_else(i < full * 4 + tail, i, -1)
                for i, j in T.Parallel(512, 4):
                    if i < full:
                        order = ranks[i] - 1 if flags[i] != 0 else chosen[0] + i - ranks[i]
                        Selected[row, order * 4 + j] = i * 4 + j
            elif direct_blocks and full <= direct_blocks:
                score = T.alloc_fragment((direct_blocks,), T.int32)
                ranks = T.alloc_fragment((direct_blocks,), T.int32)
                live_blocks = (Position[0] + m) // 4
                for i in T.Parallel(direct_blocks):
                    score[i] = -1
                    if i < full:
                        score[i] = T.reinterpret(T.int32, scores[row * live_blocks + i])
                T.clear(control)
                if T.get_thread_binding() == 0:
                    control[1] = 512
                T.sync_threads()
                bounded_cutoff(score, hist, control, cumulative)
                # Both ranks fit in 12 bits. One inclusive scan preserves ID
                # order in each group without materializing local winners.
                for i in T.Parallel(direct_blocks):
                    ranks[i] = T.if_then_else(
                        score[i] > control[0],
                        1,
                        T.if_then_else(score[i] == control[0], tie_stride, 0),
                    )
                T.cumsum(ranks, dim=0)
                for i in T.Parallel(3):
                    Selected[row, 2048 + i] = T.if_then_else(i < tail, full * 4 + i, -1)
                for i, j in T.Parallel(direct_blocks, 4):
                    equal_rank = ranks[i] // tie_stride
                    if score[i] > control[0]:
                        output_block = ranks[i] % tie_stride - 1
                        Selected[row, output_block * 4 + j] = i * 4 + j
                    elif score[i] == control[0] and equal_rank <= control[1]:
                        output_block = 512 - control[1] + equal_rank - 1
                        Selected[row, output_block * 4 + j] = i * 4 + j
            else:
                for i in T.Parallel(width):
                    values[i] = Candidates[row, i // 512 * 1024 + i % 512]
                T.clear(control)
                if T.get_thread_binding() == 0:
                    control[1] = T.min(512, full)
                T.sync_threads()
                if full > 0:
                    merged_cutoff(values, hist, control, cumulative)
                for i in T.Parallel(width):
                    keys[i] = 2147483647
                    if values[i] >= control[0] and values[i] >= 0:
                        identifier = Candidates[row, i // 512 * 1024 + 512 + i % 512]
                        keys[i] = identifier | T.if_then_else(
                            values[i] == control[0], 1073741824, 0
                        )
                T.sync_threads()
                if warp_sort:
                    T.import_source(source)
                    T.evaluate(
                        T.call_extern("void", "qsa_sort_keys", T.access_ptr(keys[0], "rw", width))
                    )
                    T.sync_threads()
                else:
                    for level in T.unroll(width.bit_length() - 1):
                        for step in T.unroll(level + 1):
                            for i in T.Parallel(width):
                                other = i ^ 1 << level - step
                                smaller = (i & 1 << level + 1 == 0) == (i & 1 << level - step == 0)
                                next_keys[i] = T.if_then_else(
                                    smaller,
                                    T.min(keys[i], keys[other]),
                                    T.max(keys[i], keys[other]),
                                )
                            T.sync_threads()
                            T.copy(next_keys, keys)
                            T.sync_threads()
                for i in T.Parallel(2051):
                    Selected[row, i] = -1
                    if i < T.min(full, 512) * 4:
                        Selected[row, i] = (keys[i // 4] & 1073741823) * 4 + i % 4
                    elif i < T.min(full, 512) * 4 + tail:
                        Selected[row, i] = full * 4 + i - T.min(full, 512) * 4

    if score_capacity:
        pointer = tir.Var("Scores_handle", "handle")
        main = tir.PrimFunc(
            [*main.params, pointer],
            main.body,
            main.ret_type,
            {**main.buffer_map, pointer: scores},
            main.attrs,
        )
    symbol = "qsa_select_dense_merge" if score_capacity else "qsa_select_merge"
    return main.with_attr("global_symbol", symbol)
