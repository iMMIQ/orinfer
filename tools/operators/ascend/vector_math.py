"""FP32 vector math shared by normalization, mHC and routing on A2."""

import tilelang.language as T


@T.macro
def rsqrt_fp32(source, destination, correction):
    """Refine A2's hardware estimate twice; source and destination are disjoint."""
    T.tile.rsqrt(destination, source)
    for _ in T.serial(2):
        T.tile.mul(correction, destination, destination)
        T.tile.mul(correction, correction, source)
        T.tile.mul(correction, correction, -0.5)
        T.tile.add(correction, correction, 1.5)
        T.tile.mul(destination, destination, correction)


@T.macro
def div_fp32_scalar(
    numerator,
    denominator,
    denominator_high,
    denominator_low,
    destination,
    residual,
    high,
    low,
    product,
):
    """Correct division by a split scalar without fused multiply-add.

    The scalar denominator is split into <=12-bit parts by the caller. Splitting
    the quotient makes each of the four products exact in FP32. This preserves
    the small residual that decides dynamic A8 half-integer rounding.
    """
    T.tile.add(residual, numerator, 0.0)
    T.tile.div(destination, numerator, denominator)
    T.tile.mul(high, destination, 4097.0)
    T.tile.sub(low, high, destination)
    T.tile.sub(high, high, low)
    T.tile.sub(low, destination, high)
    T.tile.mul(product, high, denominator_high[0, 0])
    T.tile.sub(residual, residual, product)
    T.tile.mul(product, high, denominator_low[0, 0])
    T.tile.sub(residual, residual, product)
    T.tile.mul(product, low, denominator_high[0, 0])
    T.tile.sub(residual, residual, product)
    T.tile.mul(product, low, denominator_low[0, 0])
    T.tile.sub(residual, residual, product)
    T.tile.div(residual, residual, denominator)
    T.tile.add(destination, destination, residual)


@T.macro
def div_127_fp32(numerator, divisor, destination, high, low):
    """Correct scale division without relying on fused FP32 multiply-add.

    Split the quotient into <=12-bit parts. Their products by the 7-bit integer
    127 are exact, so subtraction recovers the residual hidden by a rounded
    full-width product. All buffers are disjoint; divisor contains 127.
    """
    T.tile.div(destination, numerator, divisor)
    T.tile.mul(high, destination, 4097.0)
    T.tile.sub(low, high, destination)
    T.tile.sub(high, high, low)
    T.tile.sub(low, destination, high)
    T.tile.mul(high, high, 127.0)
    T.tile.mul(low, low, 127.0)
    T.tile.sub(high, numerator, high)
    T.tile.sub(low, high, low)
    T.tile.div(low, low, divisor)
    T.tile.add(destination, destination, low)


@T.macro
def div_fp32_rows(
    numerator,
    denominator,
    denominator_high,
    denominator_low,
    destination,
    residual,
    high,
    low,
    product,
):
    """Apply the scalar refiner to rows with broadcast split denominators."""
    T.tile.add(residual, numerator, 0.0)
    T.tile.div(destination, numerator, denominator)
    T.tile.mul(high, destination, 4097.0)
    T.tile.sub(low, high, destination)
    T.tile.sub(high, high, low)
    T.tile.sub(low, destination, high)
    T.tile.mul(product, high, denominator_high)
    T.tile.sub(residual, residual, product)
    T.tile.mul(product, high, denominator_low)
    T.tile.sub(residual, residual, product)
    T.tile.mul(product, low, denominator_high)
    T.tile.sub(residual, residual, product)
    T.tile.mul(product, low, denominator_low)
    T.tile.sub(residual, residual, product)
    T.tile.div(residual, residual, denominator)
    T.tile.add(destination, destination, residual)
