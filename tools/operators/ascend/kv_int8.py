"""Group-64 INT8 attention cache with FP16 scales and BF16 compute boundaries."""

import tilelang.language as T

from tools.operators.ascend.vector_math import div_127_fp32, div_fp32_rows


@T.macro
def decode_row(cache, scales, index, q, half_values, values, half_scales, scale, broadcast, y):
    T.copy(cache[index, :, :], q)
    T.copy(scales[index, :], half_scales[:, 0])
    T.tile.cast(half_values, q, "CAST_NONE", 512)
    T.tile.cast(values, half_values, "CAST_NONE", 512)
    T.tile.cast(scale, half_scales, "CAST_NONE", 8)
    T.tile.broadcast(broadcast, scale)
    T.tile.mul(values, values, broadcast)
    T.tile.cast(y, values, "CAST_RINT", 512)


def scatter(rows, table_rows, reference_bf16=False):
    """Round against the published FP16 scale before storing a unique cache row.

    ``-1`` skips a row; other invalid indices set the existing error flag. The
    reference variant retains the same quantized values in a BF16 cache solely
    for validating physical layouts and state transitions.
    """
    dtype = "bfloat16" if reference_bf16 else "int8"

    @T.prim_func
    def main(
        source: T.Tensor((rows, 8, 64), "bfloat16"),
        indices: T.Tensor((rows,), "int32"),
        cache: T.Tensor((table_rows, 8, 64), dtype),
        scales: T.Tensor((table_rows, 8), "float16"),
        errors: T.Tensor((rows,), "int32"),
    ):
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
                index = indices[row]
                if index >= 0 and index < table_rows:
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
                    T.copy(half_scales[:, 0], scales[index, :])
                flag[0] = T.if_then_else(index < -1 or index >= table_rows, 1, 0)
                T.copy(flag[:1], errors[row : row + 1])
                T.pipe_barrier("ALL")

    return main


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


def gather_attention(rows, capacity, compressed_capacity, compressed=True):
    """Read selected INT8 rows directly into the bounded decode compute tile."""
    selected = 640 if compressed else 128

    @T.prim_func
    def main(
        swa: T.Tensor((capacity, 8, 64), "int8"),
        swa_scales: T.Tensor((capacity, 8), "float16"),
        long_kv: T.Tensor((compressed_capacity, 8, 64), "int8"),
        long_scales: T.Tensor((compressed_capacity, 8), "float16"),
        swa_indices: T.Tensor((rows, 128), "int32"),
        long_indices: T.Tensor((rows, 512), "int32"),
        out: T.Tensor((rows, selected, 8, 64), "bfloat16"),
        mask: T.Tensor((rows, selected), "int32"),
    ):
        with T.Kernel(T.ceildiv(rows * selected, 2), is_npu=True) as (cid, vid):
            q = T.alloc_ub((8, 64), "int8")
            half_values = T.alloc_ub((8, 64), "float16")
            values = T.alloc_ub((8, 64), "float32")
            half_scales = T.alloc_ub((8, 1), "float16")
            scale = T.alloc_ub((8, 1), "float32")
            broadcast = T.alloc_ub((8, 64), "float32")
            y = T.alloc_ub((8, 64), "bfloat16")
            flag = T.alloc_ub((8,), "int32")
            flat = cid * 2 + vid
            row, slot = flat // selected, flat % selected
            if row < rows:
                T.tile.fill(y, 0.0)
                flag[0] = -1
                if slot < 128:
                    source = swa_indices[row, slot]
                    if source >= 0 and source < capacity:
                        decode_row(
                            swa,
                            swa_scales,
                            source,
                            q,
                            half_values,
                            values,
                            half_scales,
                            scale,
                            broadcast,
                            y,
                        )
                        flag[0] = 0
                else:
                    if compressed:
                        source = long_indices[row, slot - 128]
                        if source >= 0 and source < compressed_capacity:
                            decode_row(
                                long_kv,
                                long_scales,
                                source,
                                q,
                                half_values,
                                values,
                                half_scales,
                                scale,
                                broadcast,
                                y,
                            )
                            flag[0] = 0
                T.copy(y, out[row, slot, :, :])
                T.copy(flag[:1], mask[row, slot : slot + 1])
                T.pipe_barrier("ALL")

    return main
