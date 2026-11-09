"""Stream causal INT8 index scores into exact local TopK with bounded scratch."""

import tilelang.language as T


def causal_topk(
    rows,
    length,
    tile_rows=8,
    tile_s=128,
    candidate_blocks=0,
    candidate_block_size=8,
    chunk_size=4096,
    emit_block_scores=False,
):
    from kernels.ascend.candidate import load_keys, load_scales, mask_scores
    from kernels.ascend.indexer_cube import weighted_head_sum
    from kernels.ascend.topk import select_chunk
    from kernels.ascend.kv_int8 import _contiguous_view

    if length < 1 or length > 1 << 24 or tile_rows not in (1, 2, 4, 8) or tile_s not in (64, 128):
        raise ValueError("Invalid fused causal index geometry")
    if chunk_size not in (512, 1024, 2048, 4096):
        raise ValueError("Invalid fused index chunk size")
    heads, dimension, block, selected, slots = 32, 128, chunk_size, 512, 64
    if candidate_blocks and (tile_rows != 1 or candidate_block_size != 8):
        raise ValueError("Candidate fusion requires one query and eight-key blocks")
    if emit_block_scores and candidate_blocks:
        raise ValueError("Block score emission requires the full-history source")
    result_stride = 2 * selected + (block // candidate_block_size if emit_block_scores else 0)
    score_length = (
        min(length, candidate_blocks * candidate_block_size) if candidate_blocks else length
    )
    chunks, lanes = (score_length + block - 1) // block, tile_s // 2
    factor = dimension**-0.5 * heads**-0.5

    @T.prim_func
    def main(
        query: T.Tensor((rows * heads, dimension), "int8"),
        query_scale: T.Tensor((rows, heads), "float16"),
        keys: T.Tensor((length, dimension), "int8"),
        key_scale: T.Tensor((length,), "float16"),
        weights: T.Tensor((rows, heads), "bfloat16"),
        indices: T.Tensor((rows, candidate_blocks + 8 if candidate_blocks != 0 else 1), "int32"),
        candidates: T.Tensor((rows, chunks, result_stride), "float32"),
        cube_workspace: T.Tensor((slots, tile_rows * heads, tile_s), "int32"),
        score_workspace: T.Tensor((slots, tile_rows, block), "float32"),
    ):
        with T.Kernel(
            T.min(slots, T.ceildiv(rows, tile_rows) * chunks) if candidate_blocks != 0 else slots,
            is_npu=True,
        ) as (cid, vid):
            active_slots = (
                T.min(slots, T.ceildiv(rows, tile_rows) * chunks)
                if candidate_blocks != 0
                else slots
            )
            T.reads(
                query[:, :],
                query_scale[:, :],
                keys[:, :],
                key_scale[:],
                weights[:, :],
                indices[:, :],
                cube_workspace[:, :, :],
                score_workspace[:, :, :],
            )
            T.writes(candidates[:, :, :], cube_workspace[:, :, :], score_workspace[:, :, :])
            live_chunks = T.max(
                1, T.ceildiv(T.min(score_length, T.max(0, indices[rows - 1, 0])), block)
            )
            with T.Scope("C"):
                a = T.alloc_L1((tile_rows * heads, dimension), "int8")
                b = T.alloc_L1((tile_s, dimension), "int8")
                c = T.alloc_L0C((tile_rows * heads, tile_s), "int32")
                for iteration in T.serial(
                    T.ceildiv(T.ceildiv(rows, tile_rows) * live_chunks - cid, active_slots)
                ):
                    logical = cid + iteration * active_slots
                    first, chunk = logical // live_chunks * tile_rows, logical % live_chunks
                    valid_rows = T.min(tile_rows, rows - first)
                    start = chunk * block
                    valid_keys = T.min(
                        block,
                        T.max(0, T.min(score_length, indices[first + valid_rows - 1, 0]) - start),
                    )
                    if valid_keys > 0:
                        T.copy(
                            query[first * heads : (first + valid_rows) * heads, :],
                            a[: valid_rows * heads, :],
                        )
                    for tile in T.serial(T.ceildiv(valid_keys, tile_s)):
                        column = start + tile * tile_s
                        valid = T.min(tile_s, score_length - column)
                        if candidate_blocks != 0:
                            load_keys(
                                keys,
                                indices,
                                first,
                                column,
                                b,
                                tile_s,
                                dimension,
                                candidate_block_size,
                            )
                        else:
                            T.copy(keys[column : column + valid, :], b[:valid, :])
                        T.gemm_v0(a, b, c, transpose_B=True, init=True)
                        T.copy(
                            c[: valid_rows * heads, :], cube_workspace[cid, : valid_rows * heads, :]
                        )
                        T.set_cross_flag("FIX", 0)
                        T.wait_cross_flag(1)
                    T.set_cross_flag("FIX", 2)
                    T.wait_cross_flag(3)
            with T.Scope("V"):
                raw = T.alloc_ub((heads, lanes), "int32")
                dots = T.alloc_ub((heads, lanes), "float32")
                qs_half = T.alloc_ub((tile_rows * heads, 1), "float16")
                qs = T.alloc_ub((tile_rows * heads, 1), "float32")
                ks_half = T.alloc_ub((lanes,), "float16")
                ks = T.alloc_ub((1, lanes), "float32")
                w_bf = T.alloc_ub((tile_rows * heads, 1), "bfloat16")
                w_half = T.alloc_ub((tile_rows * heads, 1), "float16")
                w = T.alloc_ub((tile_rows * heads, 1), "float32")
                partial = T.alloc_ub((1, lanes), "float32")
                total = T.alloc_ub((1, lanes), "float32")
                expanded = T.alloc_ub((heads, lanes), "float32")
                x = T.alloc_ub((block,), "float32")
                positions = T.alloc_ub((block,), "float32")
                mask = T.alloc_ub((block // 8,), "uint8")
                pairs = T.alloc_ub((2 * selected,), "float32")
                pair_lanes = T.alloc_ub((2 * selected,), "float32")
                halves = T.alloc_ub((2 * selected,), "float32")
                floor_int = T.alloc_ub((2 * selected,), "int32")
                floor_float = T.alloc_ub((2 * selected,), "float32")
                offsets = T.alloc_ub((2 * selected,), "float32")
                gather_offsets = T.alloc_ub((2 * selected,), "uint32")
                selected_indices = T.alloc_ub((selected,), "float32")
                selected_values = T.alloc_ub((selected,), "float32")
                selected_mask = T.alloc_ub((selected // 8,), "uint8")
                block_maxima = T.alloc_ub((block // candidate_block_size,), "float32")
                block_view = _contiguous_view(
                    x, (block // candidate_block_size, candidate_block_size)
                )
                select_chunk(
                    x,
                    pairs,
                    positions,
                    mask,
                    pair_lanes,
                    halves,
                    floor_int,
                    floor_float,
                    offsets,
                    gather_offsets,
                    selected_indices,
                    selected_values,
                    selected_mask,
                    0,
                    0,
                    selected,
                    block,
                )
                for item in T.serial(
                    T.ceildiv(rows * (chunks - live_chunks) - cid * 2 - vid, active_slots * 2)
                ):
                    flat = cid * 2 + vid + item * active_slots * 2
                    row = flat // T.max(1, chunks - live_chunks)
                    chunk = live_chunks + flat % T.max(1, chunks - live_chunks)
                    T.copy(pairs, candidates[row, chunk, : 2 * selected])
                for iteration in T.serial(
                    T.ceildiv(T.ceildiv(rows, tile_rows) * live_chunks - cid, active_slots)
                ):
                    logical = cid + iteration * active_slots
                    first, chunk = logical // live_chunks * tile_rows, logical % live_chunks
                    valid_rows = T.min(tile_rows, rows - first)
                    start = chunk * block
                    valid_keys = T.min(
                        block,
                        T.max(0, T.min(score_length, indices[first + valid_rows - 1, 0]) - start),
                    )
                    for row_offset in T.serial(valid_rows):
                        T.copy(
                            query_scale[first + row_offset, :],
                            qs_half[row_offset * heads : (row_offset + 1) * heads, 0],
                        )
                        T.copy(
                            weights[first + row_offset, :],
                            w_bf[row_offset * heads : (row_offset + 1) * heads, 0],
                        )
                    T.tile.cast(qs, qs_half, "CAST_NONE", tile_rows * heads)
                    T.tile.cast(w, w_bf, "CAST_NONE", tile_rows * heads)
                    T.tile.mul(w, w, factor)
                    T.tile.cast(w_half, w, "CAST_RINT", tile_rows * heads)
                    T.tile.cast(w, w_half, "CAST_NONE", tile_rows * heads)
                    for tile in T.serial(T.ceildiv(valid_keys, tile_s)):
                        column = start + tile * tile_s + vid * lanes
                        T.wait_cross_flag(0)
                        for row_offset in T.serial(valid_rows):
                            valid = T.min(
                                lanes, T.min(score_length, indices[first + row_offset, 0]) - column
                            )
                            if valid > 0:
                                T.copy(
                                    cube_workspace[
                                        cid,
                                        row_offset * heads : (row_offset + 1) * heads,
                                        vid * lanes : (vid + 1) * lanes,
                                    ],
                                    raw,
                                )
                                T.tile.cast(dots, raw, "CAST_RINT", heads * lanes)
                                if candidate_blocks != 0:
                                    load_scales(
                                        key_scale,
                                        indices,
                                        first + row_offset,
                                        column,
                                        length,
                                        ks,
                                        lanes,
                                        candidate_block_size,
                                    )
                                else:
                                    T.copy(key_scale[column : column + valid], ks_half[:valid])
                                    T.tile.cast(ks, ks_half, "CAST_NONE", lanes)
                                weighted_head_sum(
                                    dots,
                                    total,
                                    partial,
                                    qs[row_offset * heads : (row_offset + 1) * heads, :],
                                    ks,
                                    w[row_offset * heads : (row_offset + 1) * heads, :],
                                    expanded,
                                    True,
                                    True,
                                    heads,
                                )
                                if candidate_blocks != 0:
                                    mask_scores(
                                        total,
                                        indices,
                                        first + row_offset,
                                        column,
                                        lanes,
                                        candidate_block_size,
                                    )
                                T.copy(
                                    total[0, :valid],
                                    score_workspace[
                                        cid,
                                        row_offset,
                                        tile * tile_s + vid * lanes : tile * tile_s
                                        + vid * lanes
                                        + valid,
                                    ],
                                )
                        T.set_cross_flag("MTE3", 1)
                    T.wait_cross_flag(2)
                    for group in T.serial(T.ceildiv(tile_rows, 2)):
                        row_offset = group * 2 + vid
                        if row_offset < valid_rows:
                            row = first + row_offset
                            valid = T.min(
                                block, T.max(0, T.min(score_length, indices[row, 0]) - start)
                            )
                            if valid > 0:
                                T.copy(score_workspace[cid, row_offset, :], x)
                            select_chunk(
                                x,
                                pairs,
                                positions,
                                mask,
                                pair_lanes,
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
                            T.copy(pairs, candidates[row, chunk, : 2 * selected])
                            if emit_block_scores:
                                if valid > 0:
                                    T.reduce_max(block_view, block_maxima, dim=1)
                                else:
                                    T.tile.fill(block_maxima, -T.infinity("float32"))
                                T.copy(block_maxima, candidates[row, chunk, 2 * selected :])
                    T.set_cross_flag("MTE3", 3)
                    T.pipe_barrier("ALL")

    return (
        main.with_attr("orinfer.coupled_resources", True)
        .with_attr("orinfer.explicit_cross_sync", True)
        .with_attr(
            "orinfer.component_sources",
            [
                "kernels/ascend/indexer_cube.py",
                "kernels/ascend/topk.py",
                "kernels/ascend/candidate.py",
                "kernels/ascend/kv_int8.py",
            ],
        )
    )
