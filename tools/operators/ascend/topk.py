"""Bounded, exact A2 sparse-index TopK using local selection and merge levels.

The first implementation preserves FP32 index scores. Every level retains only
K candidates; no stage sorts the complete long-context row in UB.
"""

import tilelang.language as T


@T.macro
def select_chunk(
    x,
    pairs,
    positions,
    mask,
    lanes,
    halves,
    floor_int,
    floor_float,
    offsets,
    gather_offsets,
    selected_indices,
    selected_values,
    selected_mask,
    valid,
    start,
    selected,
    block,
):
    if valid > 0:
        T.tile.createvecindex(positions, T.float32(0.0))
        T.tile.compare(mask, positions, T.cast(valid, "float32"), "LT")
        T.tile.select(x, mask, x, -T.infinity("float32"), "VSEL_TENSOR_SCALAR_MODE")
        if selected == 512:
            if valid <= 512:
                T.tile.topk(pairs, x, selected, 512)
            elif valid <= 1024:
                T.tile.topk(pairs, x, selected, 1024)
            elif valid <= 2048:
                T.tile.topk(pairs, x, selected, 2048)
            else:
                T.tile.topk(pairs, x, selected, block)
        else:
            T.tile.topk(pairs, x, selected, block)
    else:
        T.tile.fill(pairs, -T.infinity("float32"))
    # Add the chunk origin only to index lanes, using vector math.
    T.tile.createvecindex(lanes, T.float32(0.0))
    T.tile.mul(halves, lanes, 0.5)
    T.tile.cast(floor_int, halves, "CAST_FLOOR", 2 * selected)
    T.tile.cast(floor_float, floor_int, "CAST_NONE", 2 * selected)
    T.tile.mul(floor_float, floor_float, 2.0)
    T.tile.sub(offsets, lanes, floor_float)
    T.tile.mul(offsets, offsets, T.cast(start, "float32"))
    T.tile.add(pairs, pairs, offsets)
    # Reuse the vector work buffers to canonicalize masked lanes.
    # Every padding candidate is (-inf, -1), including short tails.
    T.reinterpretcast(selected_indices, halves, "float")
    T.reinterpretcast(selected_values, floor_float, "float")
    T.tile.gather_mask(selected_indices, pairs, "P1010")
    T.tile.gather_mask(selected_values, pairs, "P0101")
    T.tile.compare(selected_mask, selected_values, -T.infinity("float32"), "GT")
    T.tile.select(
        selected_indices,
        selected_mask,
        selected_indices,
        -1.0,
        "VSEL_TENSOR_SCALAR_MODE",
    )
    T.copy(selected_values, lanes[:selected])
    T.copy(selected_indices, lanes[selected:])
    # Every offset is an exactly representable FP32 integer.
    # Generate the interleave permutation with vector operations.
    T.tile.createvecindex(offsets, T.float32(0.0))
    T.tile.mul(halves, offsets, 0.5)
    T.tile.cast(floor_int, halves, "CAST_FLOOR", 2 * selected)
    T.tile.cast(floor_float, floor_int, "CAST_NONE", 2 * selected)
    T.tile.mul(halves, floor_float, 2.0)
    T.tile.sub(offsets, offsets, halves)
    T.tile.mul(offsets, offsets, float(selected))
    T.tile.add(offsets, offsets, floor_float)
    T.tile.mul(offsets, offsets, 4.0)
    T.tile.cast(floor_int, offsets, "CAST_RINT", 2 * selected)
    T.reinterpretcast(gather_offsets, floor_int, "uint32_t")
    T.tile.gather(pairs, lanes, gather_offsets, 0)


def local(rows, length, selected=512):
    block = 4096 if selected <= 512 else 2048
    if selected not in (512, 2048) or length > 1 << 24:
        raise ValueError("A2 select supports K=512/2048 and exact FP32 indices below 2^24")
    chunks = (length + block - 1) // block

    @T.prim_func
    def main(
        scores: T.Tensor((rows, length), "float32"),
        lengths: T.Tensor((rows,), "int32"),
        out: T.Tensor((rows, chunks, 2 * selected), "float32"),
    ):
        with T.Kernel(T.ceildiv(rows * chunks, 2), is_npu=True) as (cid, vid):
            x = T.alloc_ub((block,), "float32")
            positions = T.alloc_ub((block,), "float32")
            mask = T.alloc_ub((block // 8,), "uint8")
            pairs = T.alloc_ub((2 * selected,), "float32")
            lanes = T.alloc_ub((2 * selected,), "float32")
            halves = T.alloc_ub((2 * selected,), "float32")
            floor_int = T.alloc_ub((2 * selected,), "int32")
            floor_float = T.alloc_ub((2 * selected,), "float32")
            offsets = T.alloc_ub((2 * selected,), "float32")
            gather_offsets = T.alloc_ub((2 * selected,), "uint32")
            selected_indices = T.alloc_ub((selected,), "float32")
            selected_values = T.alloc_ub((selected,), "float32")
            selected_mask = T.alloc_ub((selected // 8,), "uint8")
            index = cid * 2 + vid
            row = index // chunks
            chunk = index % chunks
            if row < rows:
                start = chunk * block
                available = T.min(block, length - start)
                valid = T.min(available, T.max(0, lengths[row] - start))
                if valid > 0:
                    T.tile.fill(x, -T.infinity("float32"))
                    T.copy(scores[row, start : start + available], x[:available])
                select_chunk(
                    x,
                    pairs,
                    positions,
                    mask,
                    lanes,
                    halves,
                    floor_int,
                    floor_float,
                    offsets,
                    gather_offsets,
                    selected_indices,
                    selected_values,
                    selected_mask,
                    valid,
                    start,
                    selected,
                    block,
                )
                T.copy(pairs, out[row, chunk, :])

    return main


@T.macro
def merge_group(
    candidates,
    out,
    source_row,
    output_row,
    source_group,
    output_group,
    sources,
    selected,
    a,
    b,
    c,
    d,
    merged,
):
    """The same ordered four-way merge for GM and UB tree levels."""
    T.tile.fill(a, -T.infinity("float32"))
    T.tile.fill(b, -T.infinity("float32"))
    T.tile.fill(c, -T.infinity("float32"))
    T.tile.fill(d, -T.infinity("float32"))
    T.copy(candidates[source_row, source_group * 4, :], a)
    if source_group * 4 + 1 < sources:
        T.copy(candidates[source_row, source_group * 4 + 1, :], b)
    if source_group * 4 + 2 < sources:
        T.copy(candidates[source_row, source_group * 4 + 2, :], c)
    if source_group * 4 + 3 < sources:
        T.copy(candidates[source_row, source_group * 4 + 3, :], d)
    T.tile.merge_sort(merged, a, b, c, d)
    T.copy(merged[: 2 * selected], out[output_row, output_group, :])


def merge(rows, sources, selected=512):
    groups = (sources + 3) // 4

    @T.prim_func
    def main(
        candidates: T.Tensor((rows, sources, 2 * selected), "float32"),
        out: T.Tensor((rows, groups, 2 * selected), "float32"),
    ):
        with T.Kernel(T.ceildiv(rows * groups, 2), is_npu=True) as (cid, vid):
            a = T.alloc_ub((2 * selected,), "float32")
            b = T.alloc_ub((2 * selected,), "float32")
            c = T.alloc_ub((2 * selected,), "float32")
            d = T.alloc_ub((2 * selected,), "float32")
            merged = T.alloc_ub((8 * selected,), "float32")
            index = cid * 2 + vid
            row = index // groups
            group = index % groups
            if row < rows:
                merge_group(
                    candidates, out, row, row, group, group, sources, selected, a, b, c, d, merged
                )

    return main


def merge_tree(rows, sources, selected=512):
    """Preserve the four-way tree, bypassing canonical empty causal subtrees."""
    if sources not in (32, 64) or selected < 32 or selected > 512 or selected % 32:
        raise ValueError("Fused TopK tree requires32/64 sources and32-aligned selected<=512")
    second_groups = (sources + 15) // 16

    @T.prim_func
    def main(
        candidates: T.Tensor((rows, sources, 2 * selected), "float32"),
        lengths: T.Tensor((rows, 1), "int32"),
        out: T.Tensor((rows, 1, 2 * selected), "float32"),
    ):
        with T.Kernel(T.ceildiv(rows, 2), is_npu=True) as (cid, vid):
            a = T.alloc_ub((2 * selected,), "float32")
            b = T.alloc_ub((2 * selected,), "float32")
            c = T.alloc_ub((2 * selected,), "float32")
            d = T.alloc_ub((2 * selected,), "float32")
            merged = T.alloc_ub((8 * selected,), "float32")
            first_stage = T.alloc_ub((1, 4, 2 * selected), "float32")
            second_stage = T.alloc_ub((1, second_groups, 2 * selected), "float32")
            empty = T.alloc_ub((2 * selected,), "float32")
            row = cid * 2 + vid
            if row < rows:
                live = T.max(1, T.ceildiv(T.min(sources * 4096, T.max(0, lengths[row, 0])), 4096))
                if live < sources:
                    # The producer canonicalizes every inactive leaf to (-inf, -1).
                    # Merging four such leaves preserves this exact bit pattern.
                    T.copy(candidates[row, live, :], empty)
                for parent in T.serial(second_groups):
                    if parent * 16 < live:
                        for child in T.serial(4):
                            group = parent * 4 + child
                            if group * 4 < live:
                                merge_group(
                                    candidates,
                                    first_stage,
                                    row,
                                    0,
                                    group,
                                    child,
                                    sources,
                                    selected,
                                    a,
                                    b,
                                    c,
                                    d,
                                    merged,
                                )
                            else:
                                T.copy(empty, first_stage[0, child, :])
                        merge_group(
                            first_stage,
                            second_stage,
                            0,
                            0,
                            0,
                            parent,
                            4,
                            selected,
                            a,
                            b,
                            c,
                            d,
                            merged,
                        )
                    else:
                        T.copy(empty, second_stage[0, parent, :])
                merge_group(
                    second_stage, out, 0, row, 0, 0, second_groups, selected, a, b, c, d, merged
                )
                T.pipe_barrier("ALL")

    return main


def finalize(rows, selected=512, output_count=0):
    count = output_count or selected
    if not 1 <= count <= selected:
        raise ValueError("Finalized output count must be within the selected candidates")

    @T.prim_func
    def main(
        candidates: T.Tensor((rows, 2 * selected), "float32"),
        indices: T.Tensor((rows, count), "int32"),
        values: T.Tensor((rows, count), "float32"),
    ):
        with T.Kernel(T.ceildiv(rows, 2), is_npu=True) as (cid, vid):
            pairs = T.alloc_ub((2 * selected,), "float32")
            vi = T.alloc_ub((selected,), "float32")
            vv = T.alloc_ub((selected,), "float32")
            oi = T.alloc_ub((selected,), "int32")
            mask = T.alloc_ub((selected // 8,), "uint8")
            row = cid * 2 + vid
            if row < rows:
                T.copy(candidates[row, :], pairs)
                T.tile.gather_mask(vi, pairs, "P1010")
                T.tile.gather_mask(vv, pairs, "P0101")
                T.tile.compare(mask, vv, -T.infinity("float32"), "GT")
                T.tile.select(vi, mask, vi, -1.0, "VSEL_TENSOR_SCALAR_MODE")
                T.tile.cast(oi, vi, "CAST_RINT", selected)
                T.copy(oi[:count], indices[row, :])
                T.copy(vv[:count], values[row, :])

    return main
