"""Group-64 INT8 attention cache with FP16 scales and BF16 compute boundaries."""

import tilelang.language as T

from tools.operators.ascend.vector_math import div_127_fp32, div_fp32_rows


@T.macro
def decode_values(q, half_values, values, half_scales, scale, broadcast, y, groups):
    T.tile.cast(half_values, q, "CAST_NONE", groups * 64)
    T.tile.cast(values, half_values, "CAST_NONE", groups * 64)
    T.tile.cast(scale, half_scales, "CAST_NONE", groups)
    T.tile.broadcast(broadcast, scale)
    T.tile.mul(values, values, broadcast)
    T.tile.cast(y, values, "CAST_RINT", groups * 64)


@T.macro
def decode_row(cache, scales, index, q, half_values, values, half_scales, scale, broadcast, y):
    T.copy(cache[index, :, :], q)
    T.copy(scales[index, :], half_scales[:, 0])
    decode_values(q, half_values, values, half_scales, scale, broadcast, y, 8)


def _scatter(rows, table_rows, reference_bf16=False, logical_capacity=0, staging_rows=0):
    """Round against the published FP16 scale before storing a unique cache row.

    ``-1`` skips a row; other invalid indices set the existing error flag. The
    reference variant retains the same quantized values in a BF16 cache solely
    for validating physical layouts and state transitions.
    """
    dtype = "bfloat16" if reference_bf16 else "int8"

    @T.macro
    def run(source, indices, cache, scales, errors, staged):
        with T.Kernel(T.ceildiv(rows, 2), is_npu=True) as (cid, vid):
            xb = T.alloc_ub((8, 64), "bfloat16")
            x = T.alloc_ub((8, 64), "float32")
            absolute = T.alloc_ub((8, 64), "float32")
            maximum = T.alloc_ub((8, 1), "float32")
            scale = T.alloc_ub((8, 1), "float32")
            divisor = T.alloc_ub((8, 1), "float32")
            scale_high = T.alloc_ub((8, 1), "float32")
            scale_low = T.alloc_ub((8, 1), "float32")
            half_scales = T.alloc_ub((8, 1), "float16")
            floor_half = T.alloc_ub((8, 1), "float16")
            floor = T.alloc_ub((8, 1), "float32")
            broadcast = T.alloc_ub((8, 64), "float32")
            zero = T.alloc_ub((8, 64), "float32")
            residual = T.alloc_ub((8, 64), "float32")
            high = T.alloc_ub((8, 64), "float32")
            low = T.alloc_ub((8, 64), "float32")
            product = T.alloc_ub((8, 64), "float32")
            rounded = T.alloc_ub((8, 64), "float16")
            q = T.alloc_ub((8, 64), "int8")
            y = T.alloc_ub((8, 64), "bfloat16")
            flag = T.alloc_ub((8,), "int32")
            row = cid * 2 + vid
            if row < rows:
                if staging_rows:
                    end = T.min(logical_capacity, T.max(0, indices[0]))
                    begin = T.max(0, end - 127)
                    for part in T.serial(T.ceildiv(end - begin, rows)):
                        history_index = begin + row + part * rows
                        if history_index < end:
                            physical = history_index % table_rows
                            decode_row(
                                cache,
                                scales,
                                physical,
                                q,
                                rounded,
                                x,
                                half_scales,
                                scale,
                                broadcast,
                                y,
                            )
                            T.copy(y, staged[physical, :, :])
                            T.pipe_barrier("ALL")
                source_index = indices[row]
                index = source_index % table_rows if logical_capacity else source_index
                if source_index >= 0 and source_index < (logical_capacity or table_rows):
                    T.copy(source[row, :, :], xb)
                    T.tile.cast(x, xb, "CAST_NONE", 512)
                    T.tile.min(x, x, 8319008.0)
                    T.tile.max(x, x, -8319008.0)
                    T.tile.abs(absolute, x)
                    T.reduce_max(absolute, maximum, dim=1)
                    T.tile.fill(divisor, 127.0)
                    div_127_fp32(maximum, divisor, scale, scale_high, scale_low)
                    T.tile.fill(floor_half, 2.0**-24)
                    T.tile.cast(floor, floor_half, "CAST_NONE", 8)
                    T.tile.max(scale, scale, floor)
                    T.tile.min(scale, scale, 65504.0)
                    T.tile.cast(half_scales, scale, "CAST_RINT", 8)
                    T.tile.cast(scale, half_scales, "CAST_NONE", 8)
                    T.tile.broadcast(broadcast, scale)
                    T.tile.fill(zero, 0.0)
                    # FP16 scales fit the scalar refiner's high component
                    # exactly; the low component is zero.
                    div_fp32_rows(x, broadcast, broadcast, zero, x, residual, high, low, product)
                    T.tile.min(x, x, 127.0)
                    T.tile.max(x, x, -127.0)
                    T.tile.round(x, x, 512)
                    T.tile.cast(rounded, x, "CAST_RINT", 512)
                    T.tile.cast(q, rounded, "CAST_RINT", 512)
                    if reference_bf16:
                        T.tile.cast(rounded, q, "CAST_NONE", 512)
                        T.tile.cast(x, rounded, "CAST_NONE", 512)
                        T.tile.mul(x, x, broadcast)
                        T.tile.cast(y, x, "CAST_RINT", 512)
                        T.copy(y, cache[index, :, :])
                    else:
                        T.copy(q, cache[index, :, :])
                    if staging_rows:
                        T.tile.cast(rounded, q, "CAST_NONE", 512)
                        T.tile.cast(x, rounded, "CAST_NONE", 512)
                        T.tile.mul(x, x, broadcast)
                        T.tile.cast(y, x, "CAST_RINT", 512)
                        T.copy(y, staged[index, :, :])
                    T.copy(half_scales[:, 0], scales[index, :])
                flag[0] = T.if_then_else(
                    source_index < -1 or source_index >= (logical_capacity or table_rows), 1, 0
                )
                T.copy(flag[:1], errors[row : row + 1])
                T.pipe_barrier("ALL")

    if staging_rows:
        if reference_bf16 or staging_rows != table_rows or not logical_capacity:
            raise ValueError("Staged scatter requires matching INT8 circular cache geometry")

        @T.prim_func
        def main(
            source: T.Tensor((rows, 8, 64), "bfloat16"),
            indices: T.Tensor((rows,), "int32"),
            cache: T.Tensor((table_rows, 8, 64), dtype),
            scales: T.Tensor((table_rows, 8), "float16"),
            errors: T.Tensor((rows,), "int32"),
            staged: T.Tensor((staging_rows, 8, 64), "bfloat16"),
        ):
            run(source, indices, cache, scales, errors, staged)
    else:

        @T.prim_func
        def main(
            source: T.Tensor((rows, 8, 64), "bfloat16"),
            indices: T.Tensor((rows,), "int32"),
            cache: T.Tensor((table_rows, 8, 64), dtype),
            scales: T.Tensor((table_rows, 8), "float16"),
            errors: T.Tensor((rows,), "int32"),
        ):
            run(source, indices, cache, scales, errors, cache)

    return main


def scatter(rows, table_rows, reference_bf16=False, logical_capacity=0):
    return _scatter(rows, table_rows, reference_bf16, logical_capacity)


def scatter_staged(rows, table_rows, logical_capacity):
    """Stage contiguous current rows and their prior 127 rows without new storage."""
    return _scatter(rows, table_rows, False, logical_capacity, table_rows)


def decode_prefix(rows, capacity, ratio=1, windowed=False):
    """Decode only the aligned prefix/window needed by a prefill packet.

    One shared BF16 workspace serves all SWA layers. A compressed source's
    decoded prefix is reused by its following consumer layers.
    """
    if ratio not in (1, 2) or capacity < 1:
        raise ValueError("Invalid INT8 KV prefix geometry")

    @T.prim_func
    def main(
        cache: T.Tensor((capacity, 8, 64), "int8"),
        scales: T.Tensor((capacity, 8), "float16"),
        positions: T.Tensor((rows,), "int32"),
        out: T.Tensor((capacity, 8, 64), "bfloat16"),
    ):
        with T.Kernel(T.ceildiv(capacity, 2), is_npu=True) as (cid, vid):
            q = T.alloc_ub((8, 64), "int8")
            half_values = T.alloc_ub((8, 64), "float16")
            values = T.alloc_ub((8, 64), "float32")
            half_scales = T.alloc_ub((8, 1), "float16")
            scale = T.alloc_ub((8, 1), "float32")
            broadcast = T.alloc_ub((8, 64), "float32")
            y = T.alloc_ub((8, 64), "bfloat16")
            index = cid * 2 + vid
            begin = T.max(0, (positions[0] - 127) // 128) * 128 if windowed else 0
            end = T.min(capacity, T.ceildiv((positions[rows - 1] + 1) // ratio, 128) * 128)
            if index >= begin and index < end:
                decode_row(
                    cache,
                    scales,
                    index,
                    q,
                    half_values,
                    values,
                    half_scales,
                    scale,
                    broadcast,
                    y,
                )
                T.copy(y, out[index, :, :])
                T.pipe_barrier("ALL")

    return main


@T.macro
def gather_tile(
    swa,
    swa_scales,
    long_kv,
    long_scales,
    swa_indices,
    long_indices,
    out,
    mask,
    q,
    zeros,
    half_values,
    values,
    half_scales,
    scale,
    broadcast,
    y,
    flag,
    row,
    first,
    capacity,
    compressed_capacity,
    compressed,
    storage,
    tile_slots,
    bulk_store,
    groups,
    staged_swa,
    staged,
    staged_rows,
):
    if tile_slots == 1 or not bulk_store or staged:
        T.tile.fill(y, 0.0)
    if tile_slots > 1 and (not staged or first >= 128):
        T.reinterpretcast(q, zeros, "int8_t")
        T.tile.fill(zeros, 0)
        T.tile.fill(half_scales, 0.0)
    for lane in T.serial(tile_slots):
        slot = first + lane
        flag[lane] = -1
        if slot < 128:
            source = swa_indices[row, slot]
            if source >= 0 and source < capacity:
                if staged:
                    T.copy(
                        staged_swa[source % storage, :, :],
                        staged_rows[lane * 8 : (lane + 1) * 8, :],
                    )
                else:
                    T.copy(swa[source % storage, :, :], q[lane * 8 : (lane + 1) * 8, :])
                    if tile_slots == 1:
                        T.copy(swa_scales[source % storage, :], half_scales[:, 0])
                    else:
                        # FP16 scale rows are 16 bytes; scalar packing
                        # avoids unaligned 32-byte UB DMA destinations.
                        for group in T.unroll(8):
                            half_scales[lane * 8 + group, 0] = swa_scales[source % storage, group]
                    if tile_slots == 1:
                        decode_values(
                            q, half_values, values, half_scales, scale, broadcast, y, groups
                        )
                flag[lane] = 0
        else:
            if compressed:
                source = long_indices[row, slot - 128]
                if source >= 0 and source < compressed_capacity:
                    T.copy(long_kv[source, :, :], q[lane * 8 : (lane + 1) * 8, :])
                    if tile_slots == 1:
                        T.copy(long_scales[source, :], half_scales[:, 0])
                    else:
                        for group in T.unroll(8):
                            half_scales[lane * 8 + group, 0] = long_scales[source, group]
                    if tile_slots == 1:
                        decode_values(
                            q,
                            half_values,
                            values,
                            half_scales,
                            scale,
                            broadcast,
                            y,
                            groups,
                        )
                    flag[lane] = 0
    if tile_slots > 1 and (not staged or first >= 128):
        decode_values(q, half_values, values, half_scales, scale, broadcast, y, groups)
    if bulk_store:
        T.copy(y, out[row, first : first + tile_slots, :])
    else:
        for lane in T.serial(tile_slots):
            T.copy(y[lane, :], out[row, first + lane, :])
    T.copy(flag[:tile_slots], mask[row, first : first + tile_slots])
    T.pipe_barrier("ALL")


def gather_attention(
    rows,
    capacity,
    compressed_capacity,
    compressed=True,
    swa_storage_rows=0,
    tile_slots=1,
    bulk_store=False,
    reuse_long=False,
):
    """Gather cache rows and decode groups using one shared quantization body.

    The selected-slot tile is an offline schedule choice. Larger tiles batch
    casts and scale broadcasts across slots; every group keeps the same FP16
    scale and BF16 rounding boundary as the one-slot schedule.
    ``reuse_long`` preserves the previously gathered long slots and mask. The
    offline plan must prove their inputs and owner layout have not changed
    since a complete gather in this packet; only the SWA slots are updated.
    """
    selected = 640 if compressed else 128
    if reuse_long and not compressed:
        raise ValueError("Long KV reuse requires compressed attention")
    storage = swa_storage_rows or capacity
    if storage < 128 or storage > capacity:
        raise ValueError("SWA storage must cover the complete sliding window")
    if tile_slots not in (1, 4, 8, 16):
        raise ValueError("KV slot tiles must fit the A2 UB and divide both selected widths")
    groups = tile_slots * 8
    tiles = (128 if reuse_long else selected) // tile_slots

    @T.prim_func
    def main(
        swa: T.Tensor((storage, 8, 64), "int8"),
        swa_scales: T.Tensor((storage, 8), "float16"),
        long_kv: T.Tensor((compressed_capacity, 8, 64), "int8"),
        long_scales: T.Tensor((compressed_capacity, 8), "float16"),
        swa_indices: T.Tensor((rows, 128), "int32"),
        long_indices: T.Tensor((rows, 512), "int32"),
        out: T.Tensor((rows, selected, 512), "bfloat16"),
        mask: T.Tensor((rows, selected), "int32"),
    ):
        with T.Kernel(T.ceildiv(rows * tiles, 2), is_npu=True) as (cid, vid):
            q = T.alloc_ub((groups, 64), "int8")
            zeros = T.alloc_ub((groups, 16), "int32")
            half_values = T.alloc_ub((groups, 64), "float16")
            values = T.alloc_ub((groups, 64), "float32")
            half_scales = T.alloc_ub((groups, 1), "float16")
            scale = T.alloc_ub((groups, 1), "float32")
            broadcast = T.alloc_ub((groups, 64), "float32")
            y = T.alloc_ub((tile_slots, 512), "bfloat16")
            staged_rows = T.alloc_ub((groups, 64), "bfloat16")
            T.reinterpretcast(staged_rows, y, "bfloat16_t")
            flag = T.alloc_ub((max(8, tile_slots),), "int32")
            flat = cid * 2 + vid
            row, first = flat // tiles, flat % tiles * tile_slots
            if row < rows:
                gather_tile(
                    swa,
                    swa_scales,
                    long_kv,
                    long_scales,
                    swa_indices,
                    long_indices,
                    out,
                    mask,
                    q,
                    zeros,
                    half_values,
                    values,
                    half_scales,
                    scale,
                    broadcast,
                    y,
                    flag,
                    row,
                    first,
                    capacity,
                    compressed_capacity,
                    compressed,
                    storage,
                    tile_slots,
                    bulk_store,
                    groups,
                    swa,
                    False,
                    staged_rows,
                )

    return main
