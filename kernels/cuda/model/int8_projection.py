"""Shared row-scaled dense INT8 projections for model adapters.

Only the final INT32 accumulator is rescaled. Weights remain stored INT8,
including the output head; no persistent FP16 or expanded expert copy exists.
"""

import tilelang.language as T

from orinfer_build.compiler.cuda.common import orin_jit
from kernels.shared.rows import row_count
from kernels.cuda.model.integer_core import dp4a, tensor_core_tiles, tile_coordinates


@orin_jit
def int8_projection(
    M: int,
    N: int,
    K: int,
    output_dtype: str = "float16",
    block_m: int = 16,
    block_n: int = 64,
    block_k: int = 64,
    group_m: int = 0,
    dynamic_rows: bool = False,
    output_stride: int | None = None,
    tail_m: int = 0,
    cache_scales: bool = False,
):
    if any(type(v) is not int or v <= 0 for v in (M, N, K)) or K % 64 or K * 128 * 128 > 2**31 - 1:
        raise ValueError("Invalid INT8 projection geometry")
    if output_dtype not in ("float16", "float32"):
        raise ValueError("Invalid INT8 projection output dtype")
    if block_m not in (16, 32, 64, 128, 256):
        raise ValueError("Invalid INT8 row tile")
    if block_n not in (64, 128) or block_k not in (64, 128) or K % block_k:
        raise ValueError("Invalid INT8 column/reduction tile")
    if type(cache_scales) is not bool:
        raise ValueError("Invalid INT8 scale-cache choice")
    if type(group_m) is not int or group_m not in (0, 1, 2, 4, 8):
        raise ValueError("Invalid INT8 row grouping")
    M = row_count(M, dynamic_rows)
    stride = N if output_stride is None else output_stride
    if type(stride) is not int or stride < N:
        raise ValueError("INT8 output stride must contain the complete output row")
    nt = (N + block_n - 1) // block_n
    mt, project = tensor_core_tiles(
        M, N, K, block_m, block_n, block_k, tail_m=tail_m, cache_scales=cache_scales
    )

    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.int8),
        Weight: T.Tensor((N, K), T.int8),
        WeightScale: T.Tensor((N,), T.float16),
        TokenScale: T.Tensor((M,), T.float16),
        Output: T.Tensor((M, stride), output_dtype),
    ):
        with T.Kernel(mt * nt, threads=128) as pid:
            by, bx = tile_coordinates(pid, mt, nt, group_m)
            project(A, Weight, Weight, WeightScale, WeightScale, TokenScale, Output, by, bx)

    return main


@orin_jit
def int8_small_projection(
    M: int,
    N: int,
    K: int,
    output_dtype: str = "float16",
    dynamic_rows: bool = False,
    block_m: int | None = None,
    block_n: int = 4,
    output_stride: int | None = None,
):
    """Row-tiled DP4A projections; packed views alias the original INT8.

    Each warp reduces one output, reusing its weight words across four token
    rows. INT32 accumulation is exact; rescaling keeps the Tensor Core path's
    FP32 multiply order and final output rounding.
    """
    if any(type(v) is not int or v <= 0 for v in (M, N, K)) or K % 128 or K * 128 * 128 > 2**31 - 1:
        raise ValueError("Invalid small INT8 projection geometry")
    if output_dtype not in ("float16", "float32"):
        raise ValueError("Invalid small INT8 projection output dtype")
    bm, bn = min(M, 4) if block_m is None else block_m, block_n
    if not 1 <= bm <= 8 or bn not in (4, 8, 16):
        raise ValueError("Invalid INT8 DP4A tile")
    M = row_count(M, dynamic_rows)
    stride = N if output_stride is None else output_stride
    if type(stride) is not int or stride < N:
        raise ValueError("INT8 output stride must contain the complete output row")

    project = dp4a(M, N, K, bm, bn)

    @T.prim_func
    def main(
        A: T.Tensor((M, K // 4), T.int32),
        Weight: T.Tensor((N, K // 4), T.int32),
        WeightScale: T.Tensor((N,), T.float16),
        TokenScale: T.Tensor((M,), T.float16),
        Output: T.Tensor((M, stride), output_dtype),
    ):
        with T.Kernel(T.ceildiv(M, bm), T.ceildiv(N, bn), threads=128) as (by, bx):
            project(A, Weight, Weight, WeightScale, WeightScale, TokenScale, Output, by, bx)

    return main
