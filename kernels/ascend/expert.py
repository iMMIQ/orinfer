"""Shared A8 bit decomposition and fused grouped-GMM SwiGLU/A8 boundary.

GMM applies channel scales before its FP16 partial boundary. The epilogue
combines those scaled planes before the existing BF16 projection/activation.
"""

import tilelang.language as T

from kernels.ascend.vector_math import quantize_bf16_row


@T.macro
def split_row(x, packed, row, reduction):
    xf = T.alloc_ub((reduction,), "float16")
    high = T.alloc_ub((reduction,), "float16")
    high4 = T.alloc_ub((reduction,), "int4")
    low4 = T.alloc_ub((reduction,), "int4")
    high_bytes = T.alloc_ub((reduction // 2,), "int8")
    low_bytes = T.alloc_ub((reduction // 2,), "int8")
    T.tile.cast(xf, x, "CAST_NONE", reduction)
    T.tile.mul(high, xf, 0.0625)
    T.tile.cast(high4, high, "CAST_FLOOR", reduction)
    T.tile.cast(high, high4, "CAST_NONE", reduction)
    T.tile.mul(high, high, 16.0)
    T.tile.sub(xf, xf, high)
    T.tile.add(xf, xf, -8.0)
    T.tile.cast(low4, xf, "CAST_NONE", reduction)
    T.set_flag("V", "MTE3", 7)
    T.wait_flag("V", "MTE3", 7)
    T.reinterpretcast(high_bytes, high4, "int8_t")
    T.reinterpretcast(low_bytes, low4, "int8_t")
    T.copy(high_bytes, packed[row * 2, :])
    T.copy(low_bytes, packed[row * 2 + 1, :])


def gather_split(rows, table_rows, width, experts):
    """Gather live rows directly into paired signed INT4 planes and counts."""
    if width % 64 or experts < 1 or experts % 4:
        raise ValueError("Split gather requires aligned K and expert counts")

    @T.prim_func
    def main(
        table: T.Tensor((table_rows, width), "int8"),
        scales: T.Tensor((table_rows,), "float32"),
        indices: T.Tensor((rows,), "int32"),
        groups: T.Tensor((experts,), "int64"),
        packed: T.Tensor((rows * 2, width // 2), "int8"),
        out_scales: T.Tensor((rows,), "float32"),
        errors: T.Tensor((rows,), "int32"),
        split_groups: T.Tensor((experts,), "int64"),
    ):
        with T.Kernel(T.ceildiv(rows, 2), is_npu=True) as (cid, vid):
            with T.Scope("V"):
                x = T.alloc_ub((width,), "int8")
                zeros = T.alloc_ub((width // 2,), "int16")
                scale = T.alloc_ub((8,), "float32")
                flag = T.alloc_ub((8,), "int32")
                counts64 = T.alloc_ub((experts,), "int64")
                counts32 = T.alloc_ub((experts * 2,), "int32")
                doubled64 = T.alloc_ub((experts,), "int64")
                T.reinterpretcast(x, zeros, "int8_t")
                T.reinterpretcast(counts32, counts64, "int32_t")
                T.reinterpretcast(doubled64, counts32, "int64_t")
                row = cid * 2 + vid
                if row == 0:
                    T.copy(groups, counts64)
                    T.set_flag("MTE2", "V", 7)
                    T.wait_flag("MTE2", "V", 7)
                    T.tile.mul(counts32, counts32, 2)
                    T.set_flag("V", "MTE3", 7)
                    T.wait_flag("V", "MTE3", 7)
                    T.copy(doubled64, split_groups)
                if row < rows:
                    index = indices[row]
                    T.tile.fill(zeros, 0)
                    T.tile.fill(scale, 0.0)
                    if index >= 0 and index < table_rows:
                        T.copy(table[index, :], x)
                        T.copy(scales[index : index + 1], scale[:1])
                    if index != -1:
                        split_row(x, packed, row, width)
                    T.copy(scale[:1], out_scales[row : row + 1])
                    flag[0] = T.if_then_else(index < -1 or index >= table_rows, 1, 0)
                    T.copy(flag[:1], errors[row : row + 1])

    return main.with_attr("orinfer.explicit_cross_sync", True)


def finish(rows, width, experts, tile_n=256, slots=24):
    if width % tile_n or tile_n % 32 or slots not in (24, 32, 48, 64):
        raise ValueError("Invalid grouped epilogue geometry")

    @T.prim_func
    def main(
        raw: T.Tensor((rows * 2, width * 2), "float16"),
        assist: T.Tensor((experts, width * 2), "float32"),
        token_scale: T.Tensor((rows,), "float32"),
        groups: T.Tensor((experts,), "int64"),
        out: T.Tensor((rows, width), "int8"),
        scales: T.Tensor((rows,), "float32"),
    ):
        with T.Kernel(slots, is_npu=True) as (cid, vid):
            with T.Scope("V"):
                start = T.alloc_var("int32", init=0)
                cursor = T.alloc_var("int32", init=0)
                ri = T.alloc_ub((2, width * 2), "float16")
                rf = T.alloc_ub((2, width * 2), "float32")
                high = T.alloc_ub((1, tile_n * 2), "float32")
                low = T.alloc_ub((1, tile_n * 2), "float32")
                bias = T.alloc_ub((1, tile_n * 2), "float32")
                projected = T.alloc_ub((1, tile_n * 2), "bfloat16")
                gb = T.alloc_ub((1, tile_n), "bfloat16")
                ub = T.alloc_ub((1, tile_n), "bfloat16")
                gate = T.alloc_ub((1, tile_n), "float32")
                up = T.alloc_ub((1, tile_n), "float32")
                exponential = T.alloc_ub((1, tile_n), "float32")
                boundary = T.alloc_ub((1, tile_n), "bfloat16")
                gated = T.alloc_ub((1, width), "bfloat16")
                for expert in T.serial(experts):
                    count = T.min(rows - start, T.max(0, T.cast(groups[expert], "int32")))
                    lane = cid * 2 + vid
                    first = T.if_then_else(lane >= cursor, lane - cursor, lane + slots * 2 - cursor)
                    for item in T.serial(T.ceildiv(T.max(0, count - first), slots * 2)):
                        row = start + first + item * slots * 2
                        T.copy(raw[row * 2 : row * 2 + 2, :], ri)
                        T.tile.cast(rf, ri, "CAST_NONE", width * 4)
                        for column in T.serial(width // tile_n):
                            T.copy(rf[0, column * tile_n : (column + 1) * tile_n], high[0, :tile_n])
                            T.copy(
                                rf[0, width + column * tile_n : width + (column + 1) * tile_n],
                                high[0, tile_n:],
                            )
                            T.copy(rf[1, column * tile_n : (column + 1) * tile_n], low[0, :tile_n])
                            T.copy(
                                rf[1, width + column * tile_n : width + (column + 1) * tile_n],
                                low[0, tile_n:],
                            )
                            T.tile.mul(high, high, 16.0)
                            T.tile.add(high, high, low)
                            T.copy(
                                assist[expert, column * tile_n : (column + 1) * tile_n],
                                bias[0, :tile_n],
                            )
                            T.copy(
                                assist[
                                    expert, width + column * tile_n : width + (column + 1) * tile_n
                                ],
                                bias[0, tile_n:],
                            )
                            T.tile.add(high, high, bias)
                            T.tile.mul(high, high, token_scale[row])
                            T.tile.cast(projected, high, "CAST_RINT", 2 * tile_n)
                            T.copy(projected[:, :tile_n], gb)
                            T.copy(projected[:, tile_n:], ub)
                            T.tile.cast(gate, gb, "CAST_NONE", tile_n)
                            T.tile.cast(up, ub, "CAST_NONE", tile_n)
                            T.tile.min(gate, gate, 10.0)
                            T.tile.min(up, up, 10.0)
                            T.tile.max(up, up, -10.0)
                            T.tile.mul(exponential, gate, -1.0)
                            T.tile.exp(exponential, exponential)
                            T.tile.add(exponential, exponential, 1.0)
                            T.tile.div(gate, gate, exponential)
                            T.tile.mul(up, up, gate)
                            T.tile.cast(boundary, up, "CAST_RINT", tile_n)
                            T.copy(boundary, gated[:, column * tile_n : (column + 1) * tile_n])
                        quantize_bf16_row(gated, out, scales, row, width)
                    start = start + count
                    cursor = (cursor + count) % (slots * 2)

    return main.with_attr("orinfer.explicit_cross_sync", True)
