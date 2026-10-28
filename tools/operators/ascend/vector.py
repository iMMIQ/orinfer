"""A2 vector kernels. Python is used only for offline compilation and validation.

All reductions and activation arithmetic use FP32. BF16 stores are explicit
model boundaries; the integer projection inputs retain signed A8 precision.
"""

import math

import tilelang.language as T

from tools.operators.ascend.vector_math import div_127_fp32, div_fp32_scalar, rsqrt_fp32


def rms_norm(rows, width, eps=1e-20):
    @T.prim_func
    def main(
        x: T.Tensor((rows, width), "bfloat16"),
        weight: T.Tensor((width,), "bfloat16"),
        out: T.Tensor((rows, width), "bfloat16"),
    ):
        with T.Kernel(T.ceildiv(rows, 2), is_npu=True) as (cid, vid):
            xb = T.alloc_ub((1, width), "bfloat16")
            wb = T.alloc_ub((1, width), "bfloat16")
            a = T.alloc_ub((1, width), "float32")
            w = T.alloc_ub((1, width), "float32")
            square = T.alloc_ub((1, width), "float32")
            total = T.alloc_ub((1, 1), "float32")
            root = T.alloc_ub((1, 1), "float32")
            correction = T.alloc_ub((1, 1), "float32")
            inverse = T.alloc_ub((1, width), "float32")
            y = T.alloc_ub((1, width), "bfloat16")
            row = cid * 2 + vid
            if row < rows:
                T.copy(x[row : row + 1, :], xb)
                T.copy(weight[:], wb)
                T.tile.cast(a, xb, "CAST_NONE", width)
                T.tile.cast(w, wb, "CAST_NONE", width)
                T.tile.mul(square, a, a)
                T.reduce_sum(square, total, dim=1)
                T.tile.div(total, total, float(width))
                T.tile.add(total, total, eps)
                rsqrt_fp32(total, root, correction)
                T.tile.broadcast(inverse, root)
                T.tile.mul(a, a, inverse)
                T.tile.mul(a, a, w)
                T.tile.cast(y, a, "CAST_RINT", width)
                T.copy(y, out[row : row + 1, :])

    return main


def quantize(rows, width, fp16_query_scale=False):
    scale_dtype = "float16" if fp16_query_scale else "float32"

    @T.prim_func
    def main(
        x: T.Tensor((rows, width), "bfloat16"),
        out: T.Tensor((rows, width), "int8"),
        scales: T.Tensor((rows,), scale_dtype),
    ):
        with T.Kernel(T.ceildiv(rows, 2), is_npu=True) as (cid, vid):
            xb = T.alloc_ub((1, width), "bfloat16")
            a = T.alloc_ub((1, width), "float32")
            absolute = T.alloc_ub((1, width), "float32")
            maximum = T.alloc_ub((1, 1), "float32")
            scale = T.alloc_ub((1, 1), "float32")
            divisor = T.alloc_ub((1, 1), "float32")
            scale_residual = T.alloc_ub((1, 1), "float32")
            scale_negative = T.alloc_ub((1, 1), "float32")
            scale_high = T.alloc_ub((1, 1), "float32")
            scale_low = T.alloc_ub((1, 1), "float32")
            half_scale = T.alloc_ub((1, 1), "float16")
            scale_tile = T.alloc_ub((1, width), "float32")
            residual = T.alloc_ub((1, width), "float32")
            quotient_low = T.alloc_ub((1, width), "float32")
            product = T.alloc_ub((1, width), "float32")
            rounded = T.alloc_ub((1, width), "float16")
            y = T.alloc_ub((1, width), "int8")
            row = cid * 2 + vid
            if row < rows:
                T.copy(x[row : row + 1, :], xb)
                T.tile.cast(a, xb, "CAST_NONE", width)
                T.tile.abs(absolute, a)
                T.reduce_max(absolute, maximum, dim=1)
                T.tile.max(maximum, maximum, 1e-12)
                T.tile.fill(divisor, 127.0)
                div_127_fp32(maximum, divisor, scale, scale_negative, scale_residual)
                if fp16_query_scale:
                    T.tile.cast(half_scale, scale, "CAST_RINT", 1)
                    T.tile.cast(scale, half_scale, "CAST_NONE", 1)
                    T.tile.max(scale, scale, 2.0**-24)
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
                T.tile.cast(y, rounded, "CAST_RINT", width)
                T.copy(y, out[row : row + 1, :])
                if fp16_query_scale:
                    T.tile.cast(half_scale, scale, "CAST_RINT", 1)
                    T.copy(half_scale, scales[row : row + 1])
                else:
                    T.copy(scale, scales[row : row + 1])

    return main


def swiglu(rows, width, limit=10.0):
    @T.prim_func
    def main(
        gate: T.Tensor((rows, width), "bfloat16"),
        up: T.Tensor((rows, width), "bfloat16"),
        out: T.Tensor((rows, width), "bfloat16"),
    ):
        with T.Kernel(T.ceildiv(rows, 2), is_npu=True) as (cid, vid):
            gb = T.alloc_ub((1, width), "bfloat16")
            ub = T.alloc_ub((1, width), "bfloat16")
            g = T.alloc_ub((1, width), "float32")
            u = T.alloc_ub((1, width), "float32")
            exponential = T.alloc_ub((1, width), "float32")
            y = T.alloc_ub((1, width), "bfloat16")
            row = cid * 2 + vid
            if row < rows:
                T.copy(gate[row : row + 1, :], gb)
                T.copy(up[row : row + 1, :], ub)
                T.tile.cast(g, gb, "CAST_NONE", width)
                T.tile.cast(u, ub, "CAST_NONE", width)
                T.tile.min(g, g, limit)
                T.tile.min(u, u, limit)
                T.tile.max(u, u, -limit)
                T.tile.mul(exponential, g, -1.0)
                T.tile.exp(exponential, exponential)
                T.tile.add(exponential, exponential, 1.0)
                T.tile.div(g, g, exponential)
                T.tile.mul(u, g, u)
                T.tile.cast(y, u, "CAST_RINT", width)
                T.copy(y, out[row : row + 1, :])

    return main


def swiglu_quant(rows, width, limit=10.0):
    """Fused SiLU-and-up gate with per-row INT8 quantization.

    Bit-exact with the separate swiglu -> quantize pair: the gated FP32 row
    rounds through BF16 exactly as the standalone kernels do.
    """

    @T.prim_func
    def main(
        gate: T.Tensor((rows, width), "bfloat16"),
        up: T.Tensor((rows, width), "bfloat16"),
        out: T.Tensor((rows, width), "int8"),
        scales: T.Tensor((rows,), "float32"),
    ):
        with T.Kernel(T.ceildiv(rows, 2), is_npu=True) as (cid, vid):
            gb = T.alloc_ub((1, width), "bfloat16")
            ub = T.alloc_ub((1, width), "bfloat16")
            g = T.alloc_ub((1, width), "float32")
            u = T.alloc_ub((1, width), "float32")
            exponential = T.alloc_ub((1, width), "float32")
            y = T.alloc_ub((1, width), "bfloat16")
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
                T.copy(gate[row, :], gb)
                T.copy(up[row, :], ub)
                T.tile.cast(g, gb, "CAST_NONE", width)
                T.tile.cast(u, ub, "CAST_NONE", width)
                T.tile.min(g, g, limit)
                T.tile.min(u, u, limit)
                T.tile.max(u, u, -limit)
                T.tile.mul(exponential, g, -1.0)
                T.tile.exp(exponential, exponential)
                T.tile.add(exponential, exponential, 1.0)
                T.tile.div(g, g, exponential)
                T.tile.mul(u, g, u)
                T.tile.cast(y, u, "CAST_RINT", width)
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


def mhc_split_sinkhorn(rows, iterations=20, eps=1e-6, vector_rows=True):
    """Padded row strides: 24 mixes in 32 lanes, four gates in eight lanes."""

    @T.prim_func
    def main(
        mixes: T.Tensor((rows, 32), "float32"),
        scale: T.Tensor((8,), "float32"),
        base: T.Tensor((32,), "float32"),
        pre: T.Tensor((rows, 8), "float32"),
        post: T.Tensor((rows, 8), "float32"),
        combination: T.Tensor((rows, 16), "float32"),
    ):
        with T.Kernel(T.ceildiv(rows, 2), is_npu=True) as (cid, vid):
            m = T.alloc_ub((32,), "float32")
            s = T.alloc_ub((8,), "float32")
            b = T.alloc_ub((32,), "float32")
            expanded_scale = T.alloc_ub((32,), "float32")
            p = T.alloc_ub((8,), "float32")
            q = T.alloc_ub((8,), "float32")
            # A2 vector rows require 32-byte UB alignment, including 4x4 HC.
            a = T.alloc_ub((4, 8), "float32")
            r = T.alloc_ub((8,), "float32")
            c = T.alloc_ub((1, 8), "float32")
            row_divisor = T.alloc_ub((4, 8), "float32")
            column_divisor = T.alloc_ub((4, 8), "float32")
            indices = T.alloc_ub((32,), "float32")
            integer_indices = T.alloc_ub((32,), "int32")
            offsets = T.alloc_ub((32,), "uint32")
            row = cid * 2 + vid
            if row < rows:
                T.copy(mixes[row, :], m)
                T.copy(scale, s)
                T.copy(base, b)
                T.tile.fill(expanded_scale, 0.0)
                for i in T.serial(4):
                    expanded_scale[i] = s[0]
                    expanded_scale[4 + i] = s[1]
                for i in T.serial(16):
                    expanded_scale[8 + i] = s[2]
                T.tile.mul(m, m, expanded_scale)
                T.tile.add(m, m, b)
                T.tile.fill(p, 0.0)
                T.tile.fill(q, 0.0)
                for i in T.serial(4):
                    p[i] = m[i]
                    q[i] = m[4 + i]
                T.tile.sigmoid(p, p)
                T.tile.sigmoid(q, q)
                T.tile.add(p, p, eps)
                T.tile.mul(q, q, 2.0)
                for i in T.Parallel(4):
                    p[4 + i] = 0.0
                    q[4 + i] = 0.0
                T.tile.fill(a, 0.0)
                for i in T.serial(4):
                    for j in T.serial(4):
                        a[i, j] = m[8 + i * 4 + j]
                if vector_rows:
                    T.tile.createvecindex(indices, 0.0)
                    T.tile.mul(indices, indices, 0.125)
                    T.tile.cast(integer_indices, indices, "CAST_FLOOR", 32)
                    T.tile.cast(indices, integer_indices, "CAST_NONE", 32)
                    T.tile.mul(indices, indices, 4.0)
                    T.tile.cast(integer_indices, indices, "CAST_RINT", 32)
                    T.reinterpretcast(offsets, integer_indices, "uint32_t")
                T.reduce_max(a, r, dim=1, real_shape=[4, 4])
                if vector_rows:
                    T.tile.gather(row_divisor, r, offsets, 0)
                else:
                    for i in T.serial(4):
                        T.tile.fill(row_divisor[i, :], r[i])
                T.tile.sub(a, a, row_divisor)
                T.tile.exp(a, a)
                T.reduce_sum(a, r, dim=1, real_shape=[4, 4])
                if vector_rows:
                    T.tile.gather(row_divisor, r, offsets, 0)
                else:
                    for i in T.serial(4):
                        T.tile.fill(row_divisor[i, :], r[i])
                T.tile.div(a, a, row_divisor)
                T.tile.add(a, a, eps)
                T.reduce_sum(a, c, dim=0)
                T.tile.add(c, c, eps)
                T.tile.broadcast(column_divisor, c)
                T.tile.div(a, a, column_divisor)
                for _ in T.serial(iterations - 1):
                    T.reduce_sum(a, r, dim=1, real_shape=[4, 4])
                    T.tile.add(r, r, eps)
                    if vector_rows:
                        T.tile.gather(row_divisor, r, offsets, 0)
                    else:
                        for i in T.serial(4):
                            T.tile.fill(row_divisor[i, :], r[i])
                    T.tile.div(a, a, row_divisor)
                    T.reduce_sum(a, c, dim=0)
                    T.tile.add(c, c, eps)
                    T.tile.broadcast(column_divisor, c)
                    T.tile.div(a, a, column_divisor)
                T.copy(p, pre[row, :])
                T.copy(q, post[row, :])
                for i in T.serial(4):
                    T.copy(a[i, :4], combination[row, i * 4 : i * 4 + 4])

    return main


def mhc_expand(rows, width):
    block = math.gcd(width, 256)
    blocks = width // block

    @T.prim_func
    def main(
        x: T.Tensor((rows, width), "bfloat16"),
        out: T.Tensor((rows, 4, width), "bfloat16"),
    ):
        with T.Kernel(T.ceildiv(rows, 2) * blocks, is_npu=True) as (cid, vid):
            a = T.alloc_ub((block,), "bfloat16")
            row = cid // blocks * 2 + vid
            column = cid % blocks * block
            if row < rows:
                T.copy(x[row, column : column + block], a)
                for path in T.serial(4):
                    T.copy(a, out[row, path, column : column + block])

    return main


def mhc_pre_apply(rows, width):
    block = math.gcd(width, 256)
    blocks = width // block

    @T.prim_func
    def main(
        residual: T.Tensor((rows, 4, width), "bfloat16"),
        mix: T.Tensor((rows, 8), "float32"),
        out: T.Tensor((rows, width), "bfloat16"),
    ):
        with T.Kernel(T.ceildiv(rows, 2) * blocks, is_npu=True) as (cid, vid):
            xb = T.alloc_ub((4, block), "bfloat16")
            x = T.alloc_ub((4, block), "float32")
            weights = T.alloc_ub((8,), "float32")
            a = T.alloc_ub((block,), "float32")
            y = T.alloc_ub((block,), "bfloat16")
            row = cid // blocks * 2 + vid
            column = cid % blocks * block
            if row < rows:
                T.copy(residual[row, :, column : column + block], xb)
                T.copy(mix[row, :], weights)
                T.tile.cast(x, xb, "CAST_NONE", 4 * block)
                for path in T.serial(4):
                    T.tile.mul(x[path, :], x[path, :], weights[path])
                T.tile.add(a, x[0, :], x[1, :])
                T.tile.add(a, a, x[2, :])
                T.tile.add(a, a, x[3, :])
                T.tile.cast(y, a, "CAST_RINT", block)
                T.copy(y, out[row, column : column + block])

    return main


def mhc_post(rows, width):
    block = math.gcd(width, 256)
    blocks = width // block

    @T.prim_func
    def main(
        layer: T.Tensor((rows, width), "bfloat16"),
        residual: T.Tensor((rows, 4, width), "bfloat16"),
        post: T.Tensor((rows, 8), "float32"),
        combination: T.Tensor((rows, 16), "float32"),
        out: T.Tensor((rows, 4, width), "bfloat16"),
    ):
        with T.Kernel(T.ceildiv(rows, 2) * blocks, is_npu=True) as (cid, vid):
            xb = T.alloc_ub((4, block), "bfloat16")
            lb = T.alloc_ub((block,), "bfloat16")
            x = T.alloc_ub((4, block), "float32")
            l = T.alloc_ub((block,), "float32")
            p = T.alloc_ub((8,), "float32")
            c = T.alloc_ub((16,), "float32")
            temporary = T.alloc_ub((block,), "float32")
            a = T.alloc_ub((block,), "float32")
            term = T.alloc_ub((block,), "float32")
            y = T.alloc_ub((block,), "bfloat16")
            row = cid // blocks * 2 + vid
            column = cid % blocks * block
            if row < rows:
                T.copy(residual[row, :, column : column + block], xb)
                T.copy(layer[row, column : column + block], lb)
                T.copy(post[row, :], p)
                T.copy(combination[row, :], c)
                T.tile.cast(x, xb, "CAST_NONE", 4 * block)
                T.tile.cast(l, lb, "CAST_NONE", block)
                for path in T.serial(4):
                    T.tile.mul(a, x[0, :], c[path])
                    for source in T.serial(1, 4):
                        T.tile.mul(temporary, x[source, :], c[source * 4 + path])
                        T.tile.add(a, a, temporary)
                    T.tile.mul(term, l, p[path])
                    T.tile.add(a, term, a)
                    T.tile.cast(y, a, "CAST_RINT", block)
                    T.copy(y, out[row, path, column : column + block])

    return main


def mhc_post_pre(rows, width, tile=256):
    """Keep the post BF16 boundary, then collapse using the next pre-mix."""
    if tile not in (256, 512, 1024, 1280, 2560):
        raise ValueError("mHC post/pre tile must fit the A2 unified buffer")
    block = math.gcd(width, tile)
    blocks = width // block

    @T.prim_func
    def main(
        layer: T.Tensor((rows, width), "bfloat16"),
        residual: T.Tensor((rows, 4, width), "bfloat16"),
        post: T.Tensor((rows, 8), "float32"),
        combination: T.Tensor((rows, 16), "float32"),
        out: T.Tensor((rows, 4, width), "bfloat16"),
        mix: T.Tensor((rows, 8), "float32"),
        collapsed: T.Tensor((rows, width), "bfloat16"),
    ):
        with T.Kernel(T.ceildiv(rows, 2) * blocks, is_npu=True) as (cid, vid):
            xb = T.alloc_ub((4, block), "bfloat16")
            lb = T.alloc_ub((block,), "bfloat16")
            x = T.alloc_ub((4, block), "float32")
            l = T.alloc_ub((block,), "float32")
            p = T.alloc_ub((8,), "float32")
            c = T.alloc_ub((16,), "float32")
            temporary = T.alloc_ub((block,), "float32")
            a = T.alloc_ub((block,), "float32")
            term = T.alloc_ub((block,), "float32")
            y = T.alloc_ub((block,), "bfloat16")
            weights = T.alloc_ub((8,), "float32")
            weighted = T.alloc_ub((4, block), "float32")
            row, column = cid // blocks * 2 + vid, cid % blocks * block
            if row < rows:
                T.copy(residual[row, :, column : column + block], xb)
                T.copy(layer[row, column : column + block], lb)
                T.copy(post[row, :], p)
                T.copy(mix[row, :], weights)
                T.copy(combination[row, :], c)
                T.tile.cast(x, xb, "CAST_NONE", 4 * block)
                T.tile.cast(l, lb, "CAST_NONE", block)
                for path in T.serial(4):
                    T.tile.mul(a, x[0, :], c[path])
                    for source in T.serial(1, 4):
                        T.tile.mul(temporary, x[source, :], c[source * 4 + path])
                        T.tile.add(a, a, temporary)
                    T.tile.mul(term, l, p[path])
                    T.tile.add(a, term, a)
                    T.tile.cast(y, a, "CAST_RINT", block)
                    T.copy(y, out[row, path, column : column + block])
                    T.tile.cast(weighted[path, :], y, "CAST_NONE", block)
                    T.tile.mul(weighted[path, :], weighted[path, :], weights[path])
                T.tile.add(a, weighted[0, :], weighted[1, :])
                T.tile.add(a, a, weighted[2, :])
                T.tile.add(a, a, weighted[3, :])
                T.tile.cast(y, a, "CAST_RINT", block)
                T.copy(y, collapsed[row, column : column + block])

    return main


def mhc_prenorm_partials(rows, width, splits=8, lanes=32):
    """FP32 fn arithmetic, split over K; no BF16 approximation of HC weights.

    The padded fn has 24 valid rows. Output lane 24 stores sum(x*x), lanes
    25..31 are reserved zeros. Splitting output lanes distributes the same
    per-lane K reduction across more cores without changing its arithmetic.
    The reduction kernel owns the RMS boundary.
    """
    reduction = 4 * width
    chunk = math.gcd(reduction // splits, 256)
    if reduction % splits:
        raise ValueError("mHC K must be divisible by split count")
    loops = reduction // splits // chunk
    if lanes not in (8, 16, 32):
        raise ValueError("mHC output lane tile must be 8, 16 or 32")
    groups = 32 // lanes

    @T.prim_func
    def main(
        residual: T.Tensor((rows, reduction), "bfloat16"),
        fn: T.Tensor((32, reduction), "float32"),
        out: T.Tensor((rows, splits, 32), "float32"),
    ):
        with T.Kernel(T.ceildiv(rows * splits * groups, 2), is_npu=True) as (cid, vid):
            xb = T.alloc_ub((1, chunk), "bfloat16")
            x = T.alloc_ub((1, chunk), "float32")
            square = T.alloc_ub((1, chunk), "float32")
            sq = T.alloc_ub((1, 1), "float32")
            sq_acc = T.alloc_ub((1, 1), "float32")
            broadcast = T.alloc_ub((lanes, chunk), "float32")
            w = T.alloc_ub((lanes, chunk), "float32")
            product = T.alloc_ub((lanes, chunk), "float32")
            partial = T.alloc_ub((lanes, 1), "float32")
            total = T.alloc_ub((lanes, 1), "float32")
            index = cid * 2 + vid
            row = index // (splits * groups)
            split = index // groups % splits
            group = index % groups
            if row < rows:
                T.tile.fill(total, 0.0)
                T.tile.fill(sq_acc, 0.0)
                for k in T.serial(loops):
                    offset = (split * loops + k) * chunk
                    T.copy(residual[row : row + 1, offset : offset + chunk], xb)
                    T.copy(fn[group * lanes : (group + 1) * lanes, offset : offset + chunk], w)
                    T.tile.cast(x, xb, "CAST_NONE", chunk)
                    T.tile.mul(square, x, x)
                    T.reduce_sum(square, sq, dim=1)
                    T.tile.add(sq_acc, sq_acc, sq)
                    T.tile.broadcast(broadcast, x)
                    T.tile.mul(product, w, broadcast)
                    T.reduce_sum(product, partial, dim=1)
                    T.tile.add(total, total, partial)
                if group == 24 // lanes:
                    total[24 % lanes, 0] = sq_acc[0, 0]
                    for i in T.Parallel(7):
                        total[24 % lanes + 1 + i, 0] = 0.0
                T.copy(total, out[row, split, group * lanes : (group + 1) * lanes])

    return main


def mhc_prenorm_reduce(rows, width, splits=8, eps=1e-20):
    @T.prim_func
    def main(
        partials: T.Tensor((rows, splits, 32), "float32"),
        out: T.Tensor((rows, 32), "float32"),
    ):
        with T.Kernel(T.ceildiv(rows, 2), is_npu=True) as (cid, vid):
            p = T.alloc_ub((splits, 32), "float32")
            total = T.alloc_ub((1, 32), "float32")
            inverse = T.alloc_ub((1, 1), "float32")
            variance = T.alloc_ub((1, 1), "float32")
            correction = T.alloc_ub((1, 1), "float32")
            row = cid * 2 + vid
            if row < rows:
                T.copy(partials[row, :, :], p)
                T.reduce_sum(p, total, dim=0)
                variance[0, 0] = total[0, 24]
                T.tile.div(variance, variance, float(4 * width))
                T.tile.add(variance, variance, eps)
                rsqrt_fp32(variance, inverse, correction)
                T.tile.mul(total, total, inverse[0, 0])
                for i in T.Parallel(8):
                    total[0, 24 + i] = 0.0
                T.copy(total, out[row, :])

    return main


def mhc_head_mix(rows, eps=1e-6):
    @T.prim_func
    def main(
        mixes: T.Tensor((rows, 8), "float32"),
        scale: T.Tensor((8,), "float32"),
        base: T.Tensor((8,), "float32"),
        out: T.Tensor((rows, 8), "float32"),
    ):
        with T.Kernel(T.ceildiv(rows, 2), is_npu=True) as (cid, vid):
            x = T.alloc_ub((8,), "float32")
            s = T.alloc_ub((8,), "float32")
            b = T.alloc_ub((8,), "float32")
            row = cid * 2 + vid
            if row < rows:
                T.copy(mixes[row, :], x)
                T.copy(scale, s)
                T.copy(base, b)
                T.tile.mul(x, x, s[0])
                T.tile.add(x, x, b)
                T.tile.sigmoid(x, x)
                T.tile.add(x, x, eps)
                for i in T.serial(4):
                    x[4 + i] = 0.0
                T.copy(x, out[row, :])

    return main
