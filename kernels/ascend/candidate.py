"""HF candidate-block selection and compact-to-absolute sparse index mapping."""

import tilelang.language as T

from kernels.ascend.kv_int8 import _contiguous_view


@T.macro
def load_keys(keys, indices, row, start, output, tile, dimension, block):
    """Gather whole INT8 key blocks into the same Cube input tile."""
    for part in T.serial(tile // block):
        key_start = indices[row, 8 + start // block + part] * block
        safe_key = T.max(0, key_start)
        T.copy(
            keys[safe_key : safe_key + block, :dimension],
            output[part * block : (part + 1) * block, :],
        )


@T.macro
def load_scales(scales, indices, row, start, length, output, lanes, block):
    """Load eight FP16 scales through aligned DMA, then compact in vector UB."""
    candidate_scale_padded_half = T.alloc_ub((lanes // block, 16), "float16")
    candidate_scale_padded_float = T.alloc_ub((1, lanes * 2), "float32")
    candidate_scale_positions = T.alloc_ub((1, lanes), "float32")
    candidate_scale_offsets_float = T.alloc_ub((1, lanes), "float32")
    candidate_scale_offsets_int = T.alloc_ub((lanes,), "int32")
    candidate_scale_offsets = T.alloc_ub((lanes,), "uint32")
    T.tile.fill(candidate_scale_padded_half, 0.0)
    for part in T.serial(lanes // block):
        key_start = indices[row, 8 + start // block + part] * block
        if key_start >= 0 and key_start < length:
            T.copy(scales[key_start : key_start + block], candidate_scale_padded_half[part, :block])
    T.tile.cast(candidate_scale_padded_float, candidate_scale_padded_half, "CAST_NONE", lanes * 2)
    T.tile.createvecindex(candidate_scale_positions, T.float32(0.0))
    T.tile.mul(candidate_scale_offsets_float, candidate_scale_positions, 1.0 / block)
    T.tile.cast(candidate_scale_offsets_int, candidate_scale_offsets_float, "CAST_FLOOR", lanes)
    T.tile.cast(candidate_scale_offsets_float, candidate_scale_offsets_int, "CAST_NONE", lanes)
    T.tile.mul(candidate_scale_offsets_float, candidate_scale_offsets_float, float(block))
    T.tile.add(candidate_scale_positions, candidate_scale_positions, candidate_scale_offsets_float)
    T.tile.mul(candidate_scale_positions, candidate_scale_positions, 4.0)
    T.tile.cast(candidate_scale_offsets_int, candidate_scale_positions, "CAST_RINT", lanes)
    T.reinterpretcast(candidate_scale_offsets, candidate_scale_offsets_int, "uint32_t")
    T.tile.gather(output, candidate_scale_padded_float, candidate_scale_offsets, 0)


@T.macro
def mask_scores(values, indices, row, start, lanes, block):
    """Preserve causal masking for invalid candidates and a partial last block."""
    for part in T.serial(lanes // block):
        key_start = indices[row, 8 + start // block + part] * block
        if key_start < 0 or key_start + block > indices[row, 0]:
            for lane in T.serial(block):
                if key_start < 0 or key_start + lane >= indices[row, 0]:
                    values[0, part * block + lane] = -T.infinity("float32")


def select_blocks(rows, length, selected=2048, block_size=8):
    """Select by each block's maximum score; short histories keep every block.

    Only valid source scores are read, including the partial causal final block.
    The first output column retains the original causal length for the common
    INT8 score kernel. All other columns are block IDs, padded with -1.
    """
    if selected != 2048 or block_size != 8 or length % block_size:
        raise ValueError("Candidate selection requires K2048, block8 and aligned capacity")
    span = selected * block_size

    @T.prim_func
    def main(
        scores: T.Tensor((rows, length), "float32"),
        lengths: T.Tensor((rows,), "int32"),
        out: T.Tensor((rows, selected + 8), "int32"),
    ):
        with T.Kernel(T.ceildiv(rows, 2), is_npu=True) as (cid, vid):
            flat = T.alloc_ub((span,), "float32")
            source = _contiguous_view(flat, (selected, block_size))
            block_scores = T.alloc_ub((selected,), "float32")
            current = T.alloc_ub((2 * selected,), "float32")
            pairs = T.alloc_ub((2 * selected,), "float32")
            empty = T.alloc_ub((2 * selected,), "float32")
            merged = T.alloc_ub((8 * selected,), "float32")
            positions = T.alloc_ub((2 * selected,), "float32")
            halves = T.alloc_ub((2 * selected,), "float32")
            integral = T.alloc_ub((2 * selected,), "int32")
            ids = T.alloc_ub((selected,), "float32")
            ids_i32 = T.alloc_ub((selected,), "int32")
            header = T.alloc_ub((8,), "int32")
            mask = T.alloc_ub((selected // 8,), "uint8")
            row = cid * 2 + vid
            if row < rows:
                live = T.min(length, T.max(0, lengths[row]))
                T.tile.fill(header, 0)
                header[0] = live
                T.copy(header, out[row, :8])
                if live <= span:
                    T.tile.createvecindex(ids, T.float32(0.0))
                else:
                    T.tile.fill(current, -T.infinity("float32"))
                    T.tile.fill(empty, -T.infinity("float32"))
                    for chunk in T.serial(T.ceildiv(live, span)):
                        start = chunk * span
                        valid = T.min(span, live - start)
                        T.tile.fill(flat, -T.infinity("float32"))
                        T.copy(scores[row, start : start + valid], flat[:valid])
                        for tail in T.serial(block_size):
                            if valid + tail < T.ceildiv(valid, block_size) * block_size:
                                flat[valid + tail] = -T.infinity("float32")
                        T.reduce_max(source, block_scores, dim=1)
                        # Bounded DMA zero-pads the complete UB destination.
                        # Mask whole padding blocks as well as the partial one.
                        T.tile.createvecindex(ids, T.float32(0.0))
                        T.tile.compare(
                            mask, ids, T.cast(T.ceildiv(valid, block_size), "float32"), "LT"
                        )
                        T.tile.select(
                            block_scores,
                            mask,
                            block_scores,
                            -T.infinity("float32"),
                            "VSEL_TENSOR_SCALAR_MODE",
                        )
                        T.tile.topk(pairs, block_scores, selected, selected)
                        # Offset only odd (index) lanes; preserve block scores.
                        T.tile.createvecindex(positions, T.float32(0.0))
                        T.tile.mul(halves, positions, 0.5)
                        T.tile.cast(integral, halves, "CAST_FLOOR", 2 * selected)
                        T.tile.cast(halves, integral, "CAST_NONE", 2 * selected)
                        T.tile.mul(halves, halves, 2.0)
                        T.tile.sub(positions, positions, halves)
                        T.tile.mul(positions, positions, T.cast(chunk * selected, "float32"))
                        T.tile.add(pairs, pairs, positions)
                        if chunk == 0:
                            T.copy(pairs, current)
                        else:
                            T.tile.merge_sort(merged, current, pairs, empty, empty)
                            T.copy(merged[: 2 * selected], current)
                    T.tile.gather_mask(ids, current, "P1010")
                    # The reference publishes candidate IDs in address order.
                    T.tile.mul(block_scores, ids, -1.0)
                    T.tile.topk(pairs, block_scores, selected, selected)
                    T.tile.gather_mask(ids, pairs, "P0101")
                    T.tile.mul(ids, ids, -1.0)
                T.tile.compare(mask, ids, T.cast(T.ceildiv(live, block_size), "float32"), "LT")
                T.tile.select(ids, mask, ids, -1.0, "VSEL_TENSOR_SCALAR_MODE")
                T.tile.compare(mask, ids, 0.0, "GE")
                T.tile.select(ids, mask, ids, -1.0, "VSEL_TENSOR_SCALAR_MODE")
                T.tile.cast(ids_i32, ids, "CAST_RINT", selected)
                T.copy(ids_i32, out[row, 8:])

    return main.with_attr("orinfer.component_sources", ["kernels/ascend/kv_int8.py"])


@T.macro
def absolute_indices(candidates, row, original, result, lanes, candidate_blocks, block_size):
    mapping_block_ids = T.alloc_ub((candidate_blocks,), "int32")
    mapping_block_values = T.alloc_ub((candidate_blocks,), "float32")
    mapping_compact = T.alloc_ub((lanes,), "float32")
    mapping_block = T.alloc_ub((lanes,), "float32")
    mapping_remainder = T.alloc_ub((lanes,), "float32")
    mapping_integer = T.alloc_ub((lanes,), "int32")
    mapping_offsets = T.alloc_ub((lanes,), "uint32")
    mapping_mask = T.alloc_ub((lanes // 8,), "uint8")
    T.copy(candidates[row, 8:], mapping_block_ids)
    T.tile.cast(mapping_block_values, mapping_block_ids, "CAST_NONE", candidate_blocks)
    # Inactive packet rows may contain non-finite partials. Bound addresses
    # before the gather; the original values still determine final validity.
    T.tile.compare(mapping_mask, original, 0.0, "GE")
    T.tile.select(mapping_compact, mapping_mask, original, 0.0, "VSEL_TENSOR_SCALAR_MODE")
    T.tile.compare(mapping_mask, mapping_compact, float(candidate_blocks * block_size), "LT")
    T.tile.select(mapping_compact, mapping_mask, mapping_compact, 0.0, "VSEL_TENSOR_SCALAR_MODE")
    T.tile.mul(mapping_block, mapping_compact, 1.0 / block_size)
    T.tile.cast(mapping_integer, mapping_block, "CAST_FLOOR", lanes)
    T.tile.cast(mapping_block, mapping_integer, "CAST_NONE", lanes)
    T.tile.mul(mapping_remainder, mapping_block, float(block_size))
    T.tile.sub(mapping_remainder, mapping_compact, mapping_remainder)
    T.tile.mul(mapping_block, mapping_block, 4.0)
    T.tile.cast(mapping_integer, mapping_block, "CAST_RINT", lanes)
    T.reinterpretcast(mapping_offsets, mapping_integer, "uint32_t")
    T.tile.gather(result, mapping_block_values, mapping_offsets, 0)
    T.tile.mul(result, result, float(block_size))
    T.tile.add(result, result, mapping_remainder)
    T.tile.compare(mapping_mask, original, 0.0, "GE")
    T.tile.select(result, mapping_mask, result, -1.0, "VSEL_TENSOR_SCALAR_MODE")
    T.tile.compare(mapping_mask, original, float(candidate_blocks * block_size), "LT")
    T.tile.select(result, mapping_mask, result, -1.0, "VSEL_TENSOR_SCALAR_MODE")
    T.tile.compare(mapping_mask, result, 0.0, "GE")
    T.tile.select(result, mapping_mask, result, -1.0, "VSEL_TENSOR_SCALAR_MODE")
    T.tile.compare(mapping_mask, result, T.cast(candidates[row, 0], "float32"), "LT")
    T.tile.select(result, mapping_mask, result, -1.0, "VSEL_TENSOR_SCALAR_MODE")


def restore(rows, selected=512, candidate_blocks=2048, block_size=8):
    """Restore compact TopK indices in place before the existing TP gather."""
    lanes = selected // 2

    @T.prim_func
    def main(
        candidates: T.Tensor((rows, candidate_blocks + 8), "int32"),
        indices: T.Tensor((rows, selected), "int32"),
    ):
        with T.Kernel(rows, is_npu=True) as (cid, vid):
            incoming = T.alloc_ub((lanes,), "int32")
            original = T.alloc_ub((lanes,), "float32")
            result = T.alloc_ub((lanes,), "float32")
            output = T.alloc_ub((lanes,), "int32")
            T.copy(indices[cid, vid * lanes : (vid + 1) * lanes], incoming)
            T.tile.cast(original, incoming, "CAST_NONE", lanes)
            absolute_indices(candidates, cid, original, result, lanes, candidate_blocks, block_size)
            T.tile.cast(output, result, "CAST_RINT", lanes)
            T.copy(output, indices[cid, vid * lanes : (vid + 1) * lanes])

    return main


def finalize(rows, selected=512, candidate_blocks=2048, block_size=8):
    """Decode selected pairs and restore absolute indices in one vector launch."""
    lanes = selected // 2

    @T.prim_func
    def main(
        candidates: T.Tensor((rows, candidate_blocks + 8), "int32"),
        pairs: T.Tensor((rows, 2 * selected), "float32"),
        indices: T.Tensor((rows, selected), "int32"),
        values: T.Tensor((rows, selected), "float32"),
    ):
        with T.Kernel(rows, is_npu=True) as (cid, vid):
            packed = T.alloc_ub((2 * lanes,), "float32")
            original = T.alloc_ub((lanes,), "float32")
            scores = T.alloc_ub((lanes,), "float32")
            result = T.alloc_ub((lanes,), "float32")
            output = T.alloc_ub((lanes,), "int32")
            mask = T.alloc_ub((lanes // 8,), "uint8")
            T.copy(pairs[cid, vid * 2 * lanes : (vid + 1) * 2 * lanes], packed)
            T.tile.gather_mask(original, packed, "P1010")
            T.tile.gather_mask(scores, packed, "P0101")
            T.tile.compare(mask, scores, -T.infinity("float32"), "GT")
            T.tile.select(original, mask, original, -1.0, "VSEL_TENSOR_SCALAR_MODE")
            absolute_indices(candidates, cid, original, result, lanes, candidate_blocks, block_size)
            T.tile.cast(output, result, "CAST_RINT", lanes)
            T.copy(output, indices[cid, vid * lanes : (vid + 1) * lanes])
            T.copy(scores, values[cid, vid * lanes : (vid + 1) * lanes])

    return main
