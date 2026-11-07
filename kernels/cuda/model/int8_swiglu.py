"""Shared-expert W8 gate/up and SwiGLU, preserving FP16 projection rounding."""

import tilelang.language as T

from orinfer_build.compiler.cuda.common import orin_jit
from kernels.shared.rows import row_count
from kernels.cuda.model.integer_core import dp4a, tensor_core


@orin_jit
def int8_swiglu(
    M: int,
    N: int,
    K: int,
    block_n: int = 64,
    dynamic_rows: bool = False,
    *,
    algorithm: str = "tensor_core",
    block_m: int = 16,
):
    if any(type(v) is not int or v <= 0 for v in (M, N, K)) or K % 128:
        raise ValueError("Invalid decode INT8 SwiGLU geometry")
    if K * 128 * 128 > 2**31 - 1 or block_n not in (4, 8, 16, 32, 64, 128):
        raise ValueError("Invalid INT8 SwiGLU accumulator/tile")
    if algorithm not in ("dp4a", "tensor_core"):
        raise ValueError("Invalid INT8 SwiGLU algorithm")
    packed = algorithm == "dp4a"
    if packed and (not 1 <= block_m <= 8 or block_n not in (4, 8, 16, 32, 64)):
        raise ValueError("Invalid DP4A SwiGLU tile")
    if not packed and (block_m not in (16, 32) or block_n not in (64, 128)):
        raise ValueError("Invalid Tensor Core SwiGLU tile")
    width = K // 4 if packed else K
    M = row_count(M, dynamic_rows)

    project = (
        dp4a(M, N, K, block_m, block_n, dual=True)
        if packed
        else tensor_core(M, N, K, block_m, block_n, 128, dual=True)
    )

    @T.prim_func
    def main(
        A: T.Tensor((M, width), T.int32 if packed else T.int8),
        Gate: T.Tensor((N, width), T.int32 if packed else T.int8),
        Up: T.Tensor((N, width), T.int32 if packed else T.int8),
        GateScale: T.Tensor((N,), T.float16),
        UpScale: T.Tensor((N,), T.float16),
        TokenScale: T.Tensor((M,), T.float16),
        Output: T.Tensor((M, N), T.float16),
    ):
        with T.Kernel(T.ceildiv(N, block_n), T.ceildiv(M, block_m), threads=128) as (
            bx,
            by,
        ):
            project(A, Gate, Up, GateScale, UpScale, TokenScale, Output, by, bx)

    return main
