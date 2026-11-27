"""Shared A8 bit decomposition and fused grouped-GMM SwiGLU/A8 boundary.

GMM applies channel scales before its FP16 partial boundary. The epilogue
combines those scaled planes before the existing BF16 projection/activation.
"""

import tilelang.language as T

from kernels.ascend.quantization import COMPONENT_SOURCES, quantize_bf16_row


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


@T.macro
def _dequant_swiglu_row(
    raw,
    bias,
    token_scale,
    row,
    width,
    ri0,
    ri1,
    high,
    low,
    projected,
    gb,
    ub,
    gate,
    up,
    exponential,
    gated,
):
    T.copy(raw[row * 2 : row * 2 + 1, :], ri0)
    T.copy(raw[row * 2 + 1 : row * 2 + 2, :], ri1)
    T.tile.cast(high, ri0, "CAST_NONE", width * 2)
    T.tile.cast(low, ri1, "CAST_NONE", width * 2)
    T.tile.mul(high, high, 16.0)
    T.tile.add(high, high, low)
    T.tile.add(high, high, bias)
    T.tile.mul(high, high, token_scale[row])
    T.tile.cast(projected, high, "CAST_RINT", width * 2)
    T.copy(projected[:, :width], gb)
    T.copy(projected[:, width:], ub)
    T.tile.cast(gate, gb, "CAST_NONE", width)
    T.tile.cast(up, ub, "CAST_NONE", width)
    T.tile.min(gate, gate, 10.0)
    T.tile.min(up, up, 10.0)
    T.tile.max(up, up, -10.0)
    T.tile.mul(exponential, gate, -1.0)
    T.tile.exp(exponential, exponential)
    T.tile.add(exponential, exponential, 1.0)
    T.tile.div(gate, gate, exponential)
    T.tile.mul(up, up, gate)
    T.tile.cast(gated, up, "CAST_RINT", width)


def gather_split(rows, table_rows, width, experts, table_stride=0, slots=20):
    """Gather live rows directly into paired signed INT4 planes and counts.

    Persistent 20-slot (40-AIV) 1-wave schedule loads 64 indices per block and
    skips dead (-1) rows without zeroing the 5 KB activation buffer or issuing
    per-row scalar GM writes.
    """
    stride = table_stride or width
    if width % 64 or stride < width or stride % 32 or experts < 1 or experts % 4:
        raise ValueError("Split gather requires aligned K, stride and expert counts")
    if slots not in (20, 24, 32, 48, 64):
        raise ValueError("Unsupported split gather slot count")
    block = 64

    @T.prim_func
    def main(
        table: T.Tensor((table_rows, stride), "int8"),
        scales: T.Tensor((table_rows,), "float32"),
        indices: T.Tensor((rows,), "int32"),
        groups: T.Tensor((experts,), "int64"),
        packed: T.Tensor((rows * 2, width // 2), "int8"),
        out_scales: T.Tensor((rows,), "float32"),
        errors: T.Tensor((rows,), "int32"),
        split_groups: T.Tensor((experts,), "int64"),
    ):
        with T.Kernel(slots, is_npu=True) as (cid, vid):
            with T.Scope("V"):
                x = T.alloc_ub((width,), "int8")
                zeros = T.alloc_ub((width // 2,), "int16")
                scale = T.alloc_ub((8,), "float32")
                idx_ub = T.alloc_ub((block,), "int32")
                scale_ub = T.alloc_ub((block,), "float32")
                err_ub = T.alloc_ub((block,), "int32")
                counts64 = T.alloc_ub((experts,), "int64")
                counts32 = T.alloc_ub((experts * 2,), "int32")
                doubled64 = T.alloc_ub((experts,), "int64")
                T.reinterpretcast(x, zeros, "int8_t")
                T.reinterpretcast(counts32, counts64, "int32_t")
                T.reinterpretcast(doubled64, counts32, "int64_t")
                lane = cid * 2 + vid
                lanes = slots * 2
                if lane == 0:
                    T.copy(groups, counts64)
                    T.set_flag("MTE2", "V", 7)
                    T.wait_flag("MTE2", "V", 7)
                    T.tile.mul(counts32, counts32, 2)
                    T.set_flag("V", "MTE3", 7)
                    T.wait_flag("V", "MTE3", 7)
                    T.copy(doubled64, split_groups)
                step_size = T.min(block, T.max(8, T.ceildiv(rows, lanes)))
                chunks = T.ceildiv(rows, step_size)
                for step in T.serial(T.ceildiv(T.max(0, chunks - lane), lanes)):
                    part = lane + step * lanes
                    base = part * step_size
                    valid = T.min(step_size, rows - base)
                    T.tile.fill(scale_ub, 0.0)
                    T.tile.fill(err_ub, 0)
                    T.copy(indices[base : base + valid], idx_ub[:valid])
                    T.pipe_barrier("ALL")
                    for j in T.serial(valid):
                        index = idx_ub[j]
                        if index != -1:
                            if index >= 0 and index < table_rows:
                                T.copy(table[index, :width], x)
                                T.copy(scales[index : index + 1], scale[:1])
                                T.pipe_barrier("ALL")
                                scale_ub[j] = scale[0]
                            else:
                                T.tile.fill(zeros, 0)
                                err_ub[j] = 1
                                T.pipe_barrier("ALL")
                            split_row(x, packed, base + j, width)
                    T.pipe_barrier("ALL")
                    T.copy(scale_ub[:valid], out_scales[base : base + valid])
                    T.copy(err_ub[:valid], errors[base : base + valid])
                    T.pipe_barrier("ALL")

    return main.with_attr("orinfer.explicit_cross_sync", True).with_attr(
        "orinfer.component_sources", COMPONENT_SOURCES
    )


def finish(rows, width, experts, tile_n=256, slots=20):
    if width % tile_n or tile_n % 32 or slots not in (20, 24, 32, 48, 64):
        raise ValueError("Invalid grouped epilogue geometry")
    # Default 24-slot callers from existing plans map cleanly to 20 slots (40 AIVs = 1 wave on 910B3).
    launch_slots = 20 if slots == 24 else slots

    @T.prim_func
    def main_full(
        raw: T.Tensor((rows * 2, width * 2), "float16"),
        assist: T.Tensor((experts, width * 2), "float32"),
        token_scale: T.Tensor((rows,), "float32"),
        groups: T.Tensor((experts,), "int64"),
        out: T.Tensor((rows, width), "int8"),
        scales: T.Tensor((rows,), "float32"),
    ):
        with T.Kernel(launch_slots, is_npu=True) as (cid, vid):
            with T.Scope("V"):
                prefix = T.alloc_ub((8,), "int32")
                prefix[0] = 0
                ri0 = T.alloc_ub((1, width * 2), "float16")
                ri1 = T.alloc_ub((1, width * 2), "float16")
                high = T.alloc_ub((1, width * 2), "float32")
                low = T.alloc_ub((1, width * 2), "float32")
                bias = T.alloc_ub((1, width * 2), "float32")
                projected = T.alloc_ub((1, width * 2), "bfloat16")
                gb = T.alloc_ub((1, width), "bfloat16")
                ub = T.alloc_ub((1, width), "bfloat16")
                gate = T.alloc_ub((1, width), "float32")
                up = T.alloc_ub((1, width), "float32")
                exponential = T.alloc_ub((1, width), "float32")
                gated = T.alloc_ub((1, width), "bfloat16")
                lane = cid * 2 + vid
                lanes = launch_slots * 2
                for expert in T.serial(experts):
                    start = prefix[0]
                    cursor = start % lanes
                    count = T.min(rows - start, T.max(0, T.cast(groups[expert], "int32")))
                    if lane >= cursor:
                        prefix[1] = lane - cursor
                    else:
                        prefix[1] = lane + lanes - cursor
                    first = prefix[1]
                    items = T.ceildiv(T.max(0, count - first), lanes)
                    if items > 0:
                        T.copy(assist[expert : expert + 1, :], bias)
                    for item in T.serial(items):
                        row = start + first + item * lanes
                        _dequant_swiglu_row(
                            raw,
                            bias,
                            token_scale,
                            row,
                            width,
                            ri0,
                            ri1,
                            high,
                            low,
                            projected,
                            gb,
                            ub,
                            gate,
                            up,
                            exponential,
                            gated,
                        )
                        quantize_bf16_row(gated, out, scales, row, width)
                    prefix[0] = start + count

    if tile_n == width:
        return main_full.with_attr("orinfer.explicit_cross_sync", True).with_attr(
            "orinfer.component_sources", COMPONENT_SOURCES
        )

    @T.prim_func
    def main(
        raw: T.Tensor((rows * 2, width * 2), "float16"),
        assist: T.Tensor((experts, width * 2), "float32"),
        token_scale: T.Tensor((rows,), "float32"),
        groups: T.Tensor((experts,), "int64"),
        out: T.Tensor((rows, width), "int8"),
        scales: T.Tensor((rows,), "float32"),
    ):
        with T.Kernel(launch_slots, is_npu=True) as (cid, vid):
            with T.Scope("V"):
                prefix = T.alloc_ub((8,), "int32")
                prefix[0] = 0
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
                    start = prefix[0]
                    cursor = start % (launch_slots * 2)
                    count = T.min(rows - start, T.max(0, T.cast(groups[expert], "int32")))
                    lane = cid * 2 + vid
                    if lane >= cursor:
                        prefix[1] = lane - cursor
                    else:
                        prefix[1] = lane + launch_slots * 2 - cursor
                    first = prefix[1]
                    for item in T.serial(T.ceildiv(T.max(0, count - first), launch_slots * 2)):
                        row = start + first + item * launch_slots * 2
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
                    prefix[0] = start + count

    return main.with_attr("orinfer.explicit_cross_sync", True).with_attr(
        "orinfer.component_sources", COMPONENT_SOURCES
    )


def finish_split(rows, width, experts, slots=20):
    """Fused up_proj W4A4 epilogue + SwiGLU + A8 quantization directly into paired INT4 planes."""
    from kernels.ascend.vector_math import div_127_fp32, div_fp32_scalar

    if width % 64 or slots not in (20, 24, 32, 48, 64):
        raise ValueError("Invalid split grouped epilogue geometry")
    launch_slots = 20 if slots == 24 else slots

    @T.prim_func
    def main(
        raw: T.Tensor((rows * 2, width * 2), "float16"),
        assist: T.Tensor((experts, width * 2), "float32"),
        token_scale: T.Tensor((rows,), "float32"),
        groups: T.Tensor((experts,), "int64"),
        packed: T.Tensor((rows * 2, width // 2), "int8"),
        scales: T.Tensor((rows,), "float32"),
    ):
        with T.Kernel(launch_slots, is_npu=True) as (cid, vid):
            with T.Scope("V"):
                prefix = T.alloc_ub((8,), "int32")
                prefix[0] = 0
                ri0 = T.alloc_ub((1, width * 2), "float16")
                ri1 = T.alloc_ub((1, width * 2), "float16")
                high = T.alloc_ub((1, width * 2), "float32")
                low = T.alloc_ub((1, width * 2), "float32")
                bias = T.alloc_ub((1, width * 2), "float32")
                projected = T.alloc_ub((1, width * 2), "bfloat16")
                gb = T.alloc_ub((1, width), "bfloat16")
                ub = T.alloc_ub((1, width), "bfloat16")
                gate = T.alloc_ub((1, width), "float32")
                up = T.alloc_ub((1, width), "float32")
                exponential = T.alloc_ub((1, width), "float32")
                gated = T.alloc_ub((1, width), "bfloat16")
                a = T.alloc_ub((1, width), "float32")
                absolute = T.alloc_ub((1, width), "float32")
                maximum = T.alloc_ub((1, 1), "float32")
                divisor = T.alloc_ub((1, 1), "float32")
                scale_residual = T.alloc_ub((1, 1), "float32")
                scale_negative = T.alloc_ub((1, 1), "float32")
                scale_high = T.alloc_ub((1, 1), "float32")
                scale_low = T.alloc_ub((1, 1), "float32")
                scale = T.alloc_ub((1, 1), "float32")
                scale_tile = T.alloc_ub((1, width), "float32")
                residual = T.alloc_ub((1, width), "float32")
                quotient_low = T.alloc_ub((1, width), "float32")
                product = T.alloc_ub((1, width), "float32")
                rounded = T.alloc_ub((width,), "float16")
                hi_f = T.alloc_ub((width,), "float16")
                hi4 = T.alloc_ub((width,), "int4")
                lo4 = T.alloc_ub((width,), "int4")
                hi_b = T.alloc_ub((width // 2,), "int8")
                lo_b = T.alloc_ub((width // 2,), "int8")
                lane = cid * 2 + vid
                lanes = launch_slots * 2
                for expert_idx in T.serial(experts):
                    start = prefix[0]
                    cursor = start % lanes
                    count = T.min(rows - start, T.max(0, T.cast(groups[expert_idx], "int32")))
                    if lane >= cursor:
                        prefix[1] = lane - cursor
                    else:
                        prefix[1] = lane + lanes - cursor
                    first = prefix[1]
                    items = T.ceildiv(T.max(0, count - first), lanes)
                    if items > 0:
                        T.copy(assist[expert_idx : expert_idx + 1, :], bias)
                    for item in T.serial(items):
                        row = start + first + item * lanes
                        _dequant_swiglu_row(
                            raw,
                            bias,
                            token_scale,
                            row,
                            width,
                            ri0,
                            ri1,
                            high,
                            low,
                            projected,
                            gb,
                            ub,
                            gate,
                            up,
                            exponential,
                            gated,
                        )
                        T.tile.cast(a, gated, "CAST_NONE", width)
                        T.tile.abs(absolute, a)
                        T.reduce_max(absolute, maximum, dim=1)
                        T.tile.max(maximum, maximum, 1e-12)
                        T.tile.fill(divisor, 127.0)
                        div_127_fp32(maximum, divisor, scale, scale_negative, scale_residual)
                        scale_high[0, 0] = T.reinterpret(
                            "float32",
                            T.bitwise_and(
                                T.reinterpret("uint32", scale[0, 0]), T.uint32(4294963200)
                            ),
                        )
                        T.tile.sub(scale_low, scale, scale_high)
                        T.tile.broadcast(scale_tile, scale)
                        div_fp32_scalar(
                            a,
                            scale_tile,
                            scale_high,
                            scale_low,
                            a,
                            residual,
                            absolute,
                            quotient_low,
                            product,
                        )
                        T.tile.min(a, a, 127.0)
                        T.tile.max(a, a, -127.0)
                        T.tile.round(a, a, width)
                        T.tile.cast(rounded, a[0, :], "CAST_RINT", width)
                        T.copy(scale, scales[row : row + 1])
                        T.tile.mul(hi_f, rounded, 0.0625)
                        T.tile.cast(hi4, hi_f, "CAST_FLOOR", width)
                        T.tile.cast(hi_f, hi4, "CAST_NONE", width)
                        T.tile.mul(hi_f, hi_f, 16.0)
                        T.tile.sub(rounded, rounded, hi_f)
                        T.tile.add(rounded, rounded, -8.0)
                        T.tile.cast(lo4, rounded, "CAST_NONE", width)
                        T.set_flag("V", "MTE3", 7)
                        T.wait_flag("V", "MTE3", 7)
                        T.reinterpretcast(hi_b, hi4, "int8_t")
                        T.reinterpretcast(lo_b, lo4, "int8_t")
                        T.copy(hi_b, packed[row * 2, :])
                        T.copy(lo_b, packed[row * 2 + 1, :])
                        T.set_flag("MTE3", "V", 6)
                        T.wait_flag("MTE3", "V", 6)
                    prefix[0] = start + count

    return main.with_attr("orinfer.explicit_cross_sync", True).with_attr(
        "orinfer.component_sources", COMPONENT_SOURCES
    )
