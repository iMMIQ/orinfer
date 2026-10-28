"""Model packet boundaries preserving checkpoint FP32 and packed row layouts."""

import math

import tilelang.language as T

from tools.operators.ascend.vector_math import div_127_fp32, div_fp32_scalar, rsqrt_fp32


def causal_indices(rows, capacity, ratio=1):
    """Build the full-coverage causal index view on device from token positions."""
    length = capacity // ratio
    block = 128
    tiles = (length + block - 1) // block

    @T.prim_func
    def main(
        positions: T.Tensor((rows,), "int32"),
        indices: T.Tensor((rows, length), "int32"),
        lengths: T.Tensor((rows,), "int32"),
    ):
        with T.Kernel(T.ceildiv(rows * tiles, 2), is_npu=True) as (cid, vid):
            ids = T.alloc_ub((block,), "float32")
            mask = T.alloc_ub((block // 8,), "uint8")
            out = T.alloc_ub((block,), "int32")
            length_value = T.alloc_ub((8,), "int32")
            index = cid * 2 + vid
            row, tile = index // tiles, index % tiles
            if row < rows:
                count = (positions[row] + 1) // ratio
                T.tile.createvecindex(ids, T.float32(0.0))
                T.tile.add(ids, ids, T.cast(tile * block, "float32"))
                T.tile.compare(mask, ids, T.cast(count, "float32"), "LT")
                T.tile.select(ids, mask, ids, -1.0, "VSEL_TENSOR_SCALAR_MODE")
                T.tile.cast(out, ids, "CAST_RINT", block)
                valid = T.min(block, length - tile * block)
                T.copy(out[:valid], indices[row, tile * block : tile * block + valid])
                if tile == 0:
                    length_value[0] = count
                    T.copy(length_value[:1], lengths[row : row + 1])

    return main


def norm(rows, width, eps=1e-20):
    @T.prim_func
    def main(
        x: T.Tensor((rows, width), "bfloat16"),
        weight: T.Tensor((width,), "float32"),
        out: T.Tensor((rows, width), "bfloat16"),
        precise: T.Tensor((rows, width), "float32"),
    ):
        with T.Kernel(T.ceildiv(rows, 2), is_npu=True) as (cid, vid):
            xb = T.alloc_ub((1, width), "bfloat16")
            a = T.alloc_ub((1, width), "float32")
            w = T.alloc_ub((1, width), "float32")
            squared = T.alloc_ub((1, width), "float32")
            variance = T.alloc_ub((1, 1), "float32")
            inverse = T.alloc_ub((1, 1), "float32")
            correction = T.alloc_ub((1, 1), "float32")
            y = T.alloc_ub((1, width), "bfloat16")
            row = cid * 2 + vid
            if row < rows:
                T.copy(x[row, :], xb)
                T.copy(weight[:], w)
                T.tile.cast(a, xb, "CAST_NONE", width)
                T.tile.mul(squared, a, a)
                T.reduce_sum(squared, variance, dim=1)
                T.tile.mul(variance, variance, 1.0 / width)
                T.tile.add(variance, variance, eps)
                rsqrt_fp32(variance, inverse, correction)
                T.tile.mul(a, a, inverse[0, 0])
                T.tile.mul(a, a, w)
                T.copy(a, precise[row, :])
                T.tile.cast(y, a, "CAST_RINT", width)
                T.copy(y, out[row, :])

    return main


def norm_quant(rows, width, eps=1e-20):
    """Fused zero-centered RMSNorm and per-row INT8 quantization.

    Bit-exact with the separate norm -> quantize pair: the normalized FP32
    row rounds through BF16 exactly as the standalone kernels do before the
    activation scale is fitted.
    """

    @T.prim_func
    def main(
        x: T.Tensor((rows, width), "bfloat16"),
        weight: T.Tensor((width,), "float32"),
        out: T.Tensor((rows, width), "int8"),
        scales: T.Tensor((rows,), "float32"),
    ):
        with T.Kernel(T.ceildiv(rows, 2), is_npu=True) as (cid, vid):
            xb = T.alloc_ub((1, width), "bfloat16")
            a = T.alloc_ub((1, width), "float32")
            w = T.alloc_ub((1, width), "float32")
            squared = T.alloc_ub((1, width), "float32")
            variance = T.alloc_ub((1, 1), "float32")
            inverse = T.alloc_ub((1, 1), "float32")
            correction = T.alloc_ub((1, 1), "float32")
            y = T.alloc_ub((1, width), "bfloat16")
            absolute = T.alloc_ub((1, width), "float32")
            maximum = T.alloc_ub((1, 1), "float32")
            scale = T.alloc_ub((1, 1), "float32")
            divisor = T.alloc_ub((1, 1), "float32")
            scale_residual = T.alloc_ub((1, 1), "float32")
            scale_negative = T.alloc_ub((1, 1), "float32")
            scale_high = T.alloc_ub((1, 1), "float32")
            scale_low = T.alloc_ub((1, 1), "float32")
            scale_tile = T.alloc_ub((1, width), "float32")
            residual = T.alloc_ub((1, width), "float32")
            quotient_low = T.alloc_ub((1, width), "float32")
            product = T.alloc_ub((1, width), "float32")
            rounded = T.alloc_ub((1, width), "float16")
            yq = T.alloc_ub((1, width), "int8")
            row = cid * 2 + vid
            if row < rows:
                T.copy(x[row, :], xb)
                T.copy(weight[:], w)
                T.tile.cast(a, xb, "CAST_NONE", width)
                T.tile.mul(squared, a, a)
                T.reduce_sum(squared, variance, dim=1)
                T.tile.mul(variance, variance, 1.0 / width)
                T.tile.add(variance, variance, eps)
                rsqrt_fp32(variance, inverse, correction)
                T.tile.mul(a, a, inverse[0, 0])
                T.tile.mul(a, a, w)
                T.tile.cast(y, a, "CAST_RINT", width)
                T.tile.cast(a, y, "CAST_NONE", width)
                T.tile.abs(absolute, a)
                T.reduce_max(absolute, maximum, dim=1)
                T.tile.max(maximum, maximum, 1e-12)
                T.tile.fill(divisor, 127.0)
                div_127_fp32(maximum, divisor, scale, scale_negative, scale_residual)
                scale_high[0, 0] = T.reinterpret(
                    "float32",
                    T.bitwise_and(T.reinterpret("uint32", scale[0, 0]), T.uint32(4294963200)),
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
                T.tile.cast(rounded, a, "CAST_RINT", width)
                T.tile.cast(yq, rounded, "CAST_RINT", width)
                T.copy(yq, out[row : row + 1, :])
                T.copy(scale, scales[row : row + 1])

    return main


def convert(rows, width, source_dtype, destination_dtype, scale=1.0):
    block = math.gcd(width, 256)
    blocks = width // block

    @T.prim_func
    def main(
        x: T.Tensor((rows, width), source_dtype),
        out: T.Tensor((rows, width), destination_dtype),
    ):
        with T.Kernel(T.ceildiv(rows * blocks, 2), is_npu=True) as (cid, vid):
            a = T.alloc_ub((block,), source_dtype)
            f = T.alloc_ub((block,), "float32")
            y = T.alloc_ub((block,), destination_dtype)
            index = cid * 2 + vid
            row = index // blocks
            start = index % blocks * block
            if row < rows:
                T.copy(x[row, start : start + block], a)
                if source_dtype == "float32":
                    T.tile.add(f, a, 0.0)
                else:
                    T.tile.cast(f, a, "CAST_NONE", block)
                T.tile.mul(f, f, scale)
                if destination_dtype == "float32":
                    T.tile.add(y, f, 0.0)
                else:
                    T.tile.cast(y, f, "CAST_RINT", block)
                T.copy(y, out[row, start : start + block])

    return main


def partial_rope(rows, heads, dimension, inverse=False, vector_indices=True):
    rotary, half = 64, 32
    sign = -1.0 if inverse else 1.0

    @T.prim_func
    def main(
        x: T.Tensor((rows, heads, dimension), "bfloat16"),
        cosine: T.Tensor((rows, half), "float32"),
        sine: T.Tensor((rows, half), "float32"),
        out: T.Tensor((rows, heads, dimension), "bfloat16"),
    ):
        with T.Kernel(T.ceildiv(rows * heads, 2), is_npu=True) as (cid, vid):
            source = T.alloc_ub((dimension,), "bfloat16")
            xb = T.alloc_ub((rotary,), "bfloat16")
            a = T.alloc_ub((rotary,), "float32")
            even = T.alloc_ub((half,), "float32")
            odd = T.alloc_ub((half,), "float32")
            co = T.alloc_ub((half,), "float32")
            si = T.alloc_ub((half,), "float32")
            product = T.alloc_ub((half,), "float32")
            result = T.alloc_ub((2, half), "float32")
            offsets = T.alloc_ub((rotary,), "uint32")
            lane = T.alloc_ub((rotary,), "float32")
            half_index = T.alloc_ub((rotary,), "float32")
            floor_index = T.alloc_ub((rotary,), "int32")
            y = T.alloc_ub((rotary,), "bfloat16")
            index = cid * 2 + vid
            row = index // heads
            head = index % heads
            if row < rows:
                T.copy(x[row, head, :], source)
                T.copy(source[dimension - rotary :], xb)
                T.copy(cosine[row, :], co)
                T.copy(sine[row, :], si)
                T.tile.mul(si, si, sign)
                T.tile.cast(a, xb, "CAST_NONE", rotary)
                T.tile.gather_mask(even, a, "P0101")
                T.tile.gather_mask(odd, a, "P1010")
                T.tile.mul(result[0, :], even, co)
                T.tile.mul(product, odd, si)
                T.tile.sub(result[0, :], result[0, :], product)
                T.tile.mul(result[1, :], odd, co)
                T.tile.mul(product, even, si)
                T.tile.add(result[1, :], result[1, :], product)
                if vector_indices:
                    T.tile.createvecindex(lane, 0.0)
                    T.tile.mul(half_index, lane, 0.5)
                    T.tile.cast(floor_index, half_index, "CAST_FLOOR", rotary)
                    T.tile.cast(half_index, floor_index, "CAST_NONE", rotary)
                    T.tile.mul(a, half_index, 2.0)
                    T.tile.sub(lane, lane, a)
                    T.tile.mul(lane, lane, float(half))
                    T.tile.add(lane, lane, half_index)
                    T.tile.mul(lane, lane, 4.0)
                    T.tile.cast(floor_index, lane, "CAST_RINT", rotary)
                    T.reinterpretcast(offsets, floor_index, "uint32_t")
                else:
                    for i in T.Parallel(rotary):
                        offsets[i] = T.cast((i % 2 * half + i // 2) * 4, "uint32")
                T.tile.gather(a, result, offsets, 0)
                T.tile.cast(y, a, "CAST_RINT", rotary)
                T.copy(source[: dimension - rotary], out[row, head, : dimension - rotary])
                T.copy(y, out[row, head, dimension - rotary :])

    return main


def gather_attention(rows, capacity, compressed_capacity, compressed=True):
    selected = 640 if compressed else 128

    @T.prim_func
    def main(
        swa: T.Tensor((capacity, 512), "bfloat16"),
        long_kv: T.Tensor((compressed_capacity, 512), "bfloat16"),
        swa_indices: T.Tensor((rows, 128), "int32"),
        long_indices: T.Tensor((rows, 512), "int32"),
        out: T.Tensor((rows, selected, 512), "bfloat16"),
        mask: T.Tensor((rows, selected), "int32"),
    ):
        with T.Kernel(T.ceildiv(rows * selected, 2), is_npu=True) as (cid, vid):
            values = T.alloc_ub((512,), "bfloat16")
            flag = T.alloc_ub((8,), "int32")
            index = cid * 2 + vid
            row = index // selected
            slot = index % selected
            if row < rows:
                T.tile.fill(values, 0.0)
                flag[0] = -1
                if slot < 128:
                    source = swa_indices[row, slot]
                    if source >= 0 and source < capacity:
                        T.copy(swa[source, :], values)
                        flag[0] = 0
                else:
                    if compressed:
                        source = long_indices[row, slot - 128]
                        if source >= 0 and source < compressed_capacity:
                            T.copy(long_kv[source, :], values)
                            flag[0] = 0
                T.set_flag("S", "MTE3", 6)
                T.wait_flag("S", "MTE3", 6)
                T.copy(values, out[row, slot, :])
                T.copy(flag[:1], mask[row, slot : slot + 1])
                # A bounded physical block handles successive query rows.
                # Both producers must wait until DMA has consumed their UB.
                T.set_flag("MTE3", "V", 7)
                T.wait_flag("MTE3", "V", 7)
                T.set_flag("MTE3", "S", 7)
                T.wait_flag("MTE3", "S", 7)

    return main


def swiglu_packed(rows, width, limit=10.0):
    @T.prim_func
    def main(
        packed: T.Tensor((rows, 2 * width), "bfloat16"),
        out: T.Tensor((rows, width), "bfloat16"),
    ):
        with T.Kernel(T.ceildiv(rows, 2), is_npu=True) as (cid, vid):
            gb = T.alloc_ub((width,), "bfloat16")
            ub = T.alloc_ub((width,), "bfloat16")
            g = T.alloc_ub((width,), "float32")
            u = T.alloc_ub((width,), "float32")
            exp = T.alloc_ub((width,), "float32")
            y = T.alloc_ub((width,), "bfloat16")
            row = cid * 2 + vid
            if row < rows:
                T.copy(packed[row, :width], gb)
                T.copy(packed[row, width:], ub)
                T.tile.cast(g, gb, "CAST_NONE", width)
                T.tile.cast(u, ub, "CAST_NONE", width)
                T.tile.min(g, g, limit)
                T.tile.min(u, u, limit)
                T.tile.max(u, u, -limit)
                T.tile.mul(exp, g, -1.0)
                T.tile.exp(exp, exp)
                T.tile.add(exp, exp, 1.0)
                T.tile.div(g, g, exp)
                T.tile.mul(u, u, g)
                T.tile.cast(y, u, "CAST_RINT", width)
                T.copy(y, out[row, :])

    return main


def swiglu_packed_quant(rows, width, limit=10.0):
    """Fused packed SiLU-and-up gate with per-row INT8 quantization.

    Bit-exact with the separate swiglu_packed -> quantize pair: the gated FP32
    row rounds through BF16 exactly as the standalone kernels do. The gated
    row moves into the quantization layout through an in-UB copy, never
    touching global memory.
    """

    @T.prim_func
    def main(
        packed: T.Tensor((rows, 2 * width), "bfloat16"),
        out: T.Tensor((rows, width), "int8"),
        scales: T.Tensor((rows,), "float32"),
    ):
        with T.Kernel(T.ceildiv(rows, 2), is_npu=True) as (cid, vid):
            gb = T.alloc_ub((width,), "bfloat16")
            ub = T.alloc_ub((width,), "bfloat16")
            g = T.alloc_ub((width,), "float32")
            u = T.alloc_ub((width,), "float32")
            exp = T.alloc_ub((width,), "float32")
            y = T.alloc_ub((width,), "bfloat16")
            yb = T.alloc_ub((1, width), "bfloat16")
            a = T.alloc_ub((1, width), "float32")
            absolute = T.alloc_ub((1, width), "float32")
            maximum = T.alloc_ub((1, 1), "float32")
            scale = T.alloc_ub((1, 1), "float32")
            divisor = T.alloc_ub((1, 1), "float32")
            scale_residual = T.alloc_ub((1, 1), "float32")
            scale_negative = T.alloc_ub((1, 1), "float32")
            scale_high = T.alloc_ub((1, 1), "float32")
            scale_low = T.alloc_ub((1, 1), "float32")
            scale_tile = T.alloc_ub((1, width), "float32")
            residual = T.alloc_ub((1, width), "float32")
            quotient_low = T.alloc_ub((1, width), "float32")
            product = T.alloc_ub((1, width), "float32")
            rounded = T.alloc_ub((1, width), "float16")
            yq = T.alloc_ub((1, width), "int8")
            row = cid * 2 + vid
            if row < rows:
                T.copy(packed[row, :width], gb)
                T.copy(packed[row, width:], ub)
                T.tile.cast(g, gb, "CAST_NONE", width)
                T.tile.cast(u, ub, "CAST_NONE", width)
                T.tile.min(g, g, limit)
                T.tile.min(u, u, limit)
                T.tile.max(u, u, -limit)
                T.tile.mul(exp, g, -1.0)
                T.tile.exp(exp, exp)
                T.tile.add(exp, exp, 1.0)
                T.tile.div(g, g, exp)
                T.tile.mul(u, u, g)
                T.tile.cast(y, u, "CAST_RINT", width)
                T.copy(y[:], yb[0, :])
                T.tile.cast(a, yb, "CAST_NONE", width)
                T.tile.abs(absolute, a)
                T.reduce_max(absolute, maximum, dim=1)
                T.tile.max(maximum, maximum, 1e-12)
                T.tile.fill(divisor, 127.0)
                div_127_fp32(maximum, divisor, scale, scale_negative, scale_residual)
                scale_high[0, 0] = T.reinterpret(
                    "float32",
                    T.bitwise_and(T.reinterpret("uint32", scale[0, 0]), T.uint32(4294963200)),
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
                T.tile.cast(rounded, a, "CAST_RINT", width)
                T.tile.cast(yq, rounded, "CAST_RINT", width)
                T.copy(yq, out[row : row + 1, :])
                T.copy(scale, scales[row : row + 1])

    return main


def add_shared(rows, width):
    block = math.gcd(width, 256)
    blocks = width // block

    @T.prim_func
    def main(
        routed: T.Tensor((rows, width), "float32"),
        shared: T.Tensor((rows, width), "bfloat16"),
        out: T.Tensor((rows, width), "bfloat16"),
    ):
        with T.Kernel(T.ceildiv(rows * blocks, 2), is_npu=True) as (cid, vid):
            r = T.alloc_ub((block,), "float32")
            rb = T.alloc_ub((block,), "bfloat16")
            sb = T.alloc_ub((block,), "bfloat16")
            s = T.alloc_ub((block,), "float32")
            y = T.alloc_ub((block,), "bfloat16")
            index = cid * 2 + vid
            row = index // blocks
            start = index % blocks * block
            if row < rows:
                T.copy(routed[row, start : start + block], r)
                T.copy(shared[row, start : start + block], sb)
                T.tile.cast(rb, r, "CAST_RINT", block)
                T.tile.cast(r, rb, "CAST_NONE", block)
                T.tile.cast(s, sb, "CAST_NONE", block)
                T.tile.add(r, r, s)
                T.tile.cast(y, r, "CAST_RINT", block)
                T.copy(y, out[row, start : start + block])

    return main


def split(rows, left, right):
    block = math.gcd(math.gcd(left, right), 256)
    blocks = (left + right) // block

    @T.prim_func
    def main(
        packed: T.Tensor((rows, left + right), "bfloat16"),
        first: T.Tensor((rows, left), "bfloat16"),
        second: T.Tensor((rows, right), "bfloat16"),
    ):
        with T.Kernel(T.ceildiv(rows * blocks, 2), is_npu=True) as (cid, vid):
            a = T.alloc_ub((block,), "bfloat16")
            index = cid * 2 + vid
            row = index // blocks
            start = index % blocks * block
            if row < rows:
                T.copy(packed[row, start : start + block], a)
                if start < left:
                    T.copy(a, first[row, start : start + block])
                else:
                    T.copy(a, second[row, start - left : start - left + block])

    return main
