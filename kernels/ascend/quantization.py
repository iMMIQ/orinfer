"""FP32 A8 row arithmetic shared by direct and routed activation transports."""

import tilelang.language as T

from kernels.ascend.vector_math import div_127_fp32, div_fp32_scalar


def row_quantizer(width, fp16_query_scale=False):
    @T.macro
    def apply(xb, y, scale, half_scale):
        a = T.alloc_ub((1, width), "float32")
        absolute = T.alloc_ub((1, width), "float32")
        maximum = T.alloc_ub((1, 1), "float32")
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
        T.tile.cast(a, xb, "CAST_NONE", width)
        T.tile.abs(absolute, a)
        T.reduce_max(absolute, maximum, dim=1)
        T.tile.max(maximum, maximum, 1e-12)
        T.tile.fill(divisor, 127.0)
        div_127_fp32(maximum, divisor, scale, scale_negative, scale_residual)
        if fp16_query_scale:
            # The pinned emitter truncates FP32 literals. Converting
            # through FP16 preserves this exact subnormal minimum.
            T.tile.fill(half_scale, 2.0**-24)
            T.tile.cast(scale_negative, half_scale, "CAST_NONE", 1)
            T.tile.cast(half_scale, scale, "CAST_RINT", 1)
            T.tile.cast(scale, half_scale, "CAST_NONE", 1)
            T.tile.max(scale, scale, scale_negative)
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

    return apply
