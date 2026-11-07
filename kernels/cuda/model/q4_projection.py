"""Row-scaled scalar Q4 indices decoded to INT8 for full-K accumulation.

Each nibble stores three magnitude bits and one sign bit. A fixed symmetric
16-point integer codebook allows two byte-permute instructions to unpack four
weights without group scale operations inside the dot-product loop.
"""

import tilelang.language as T
from orinfer_build.compiler.cuda.common import orin_jit
from kernels.shared.rows import row_count
from kernels.cuda.model.integer_core import dp4a, tensor_core_tiles, tile_coordinates


MAGNITUDES = (3, 10, 19, 31, 46, 67, 93, 127)


def decode_source(magnitudes=MAGNITUDES):
    if len(magnitudes) != 8 or any(type(v) is not int or not 0 < v <= 127 for v in magnitudes):
        raise ValueError("Q4 needs eight positive INT8 magnitudes")
    if tuple(sorted(set(magnitudes))) != tuple(magnitudes):
        raise ValueError("Q4 magnitudes must be strictly increasing")
    lo = sum(v << (8 * i) for i, v in enumerate(magnitudes[:4]))
    hi = sum(v << (8 * i) for i, v in enumerate(magnitudes[4:]))
    return f"""
__device__ __forceinline__ unsigned q4_decode(unsigned p) {{
    unsigned v = __byte_perm({lo}u, {hi}u, p & 0x7777u);
    unsigned signs = __byte_perm(0u, 0xffffffffu, (p >> 1) & 0x4444u);
    return (v ^ signs) + (signs & 0x01010101u);
}}
"""


def validate_shape(M, N, K):
    if min(M, N, K) <= 0 or K % 128 or K * 128 * 127 > 2**31 - 1:
        raise ValueError("Q4 projection requires block128 geometry and bounded INT32 accumulation")


@orin_jit
def q4_gemv(
    M,
    N,
    K,
    BN=16,
    output_dtype="float16",
    magnitudes=MAGNITUDES,
    *,
    dynamic_rows=False,
    block_m=None,
    output_stride=None,
):
    validate_shape(M, N, K)
    BM = min(M, 8) if block_m is None else block_m
    if type(BM) is not int or not 1 <= BM <= 8 or BN not in (4, 8, 16):
        raise ValueError("Q4 DP4A requires BM=1..8 and BN=4/8/16")
    M = row_count(M, dynamic_rows)
    stride = N if output_stride is None else output_stride
    if type(stride) is not int or stride < N:
        raise ValueError("Q4 output stride must contain the complete output row")
    source = decode_source(magnitudes)

    project = dp4a(M, N, K, BM, BN, codec="q4", source=source)

    @T.prim_func
    def main(
        A: T.Tensor((M, K // 4), T.int32),
        W: T.Tensor((N, K // 4), T.uint16),
        S: T.Tensor((N,), T.float16),
        AS: T.Tensor((M,), T.float16),
        Out: T.Tensor((M, stride), output_dtype),
    ):
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=128) as (bx, by):
            project(A, W, W, S, S, AS, Out, by, bx)

    return main


def q4_decode(
    rows,
    outputs,
    inputs,
    *,
    algorithm,
    block_m,
    block_n,
    block_k=128,
    stages=2,
    output_dtype="float16",
    dynamic_rows=False,
    output_stride=None,
    tail_m=0,
    group_m=0,
):
    """One AOT entry point; tiling selects DP4A or Tensor Core at build time."""
    if algorithm == "dp4a":
        if tail_m or group_m:
            raise ValueError("Tail/group scheduling requires the INT8 Tensor Core path")
        return q4_gemv.__wrapped__(
            rows,
            outputs,
            inputs,
            block_n,
            output_dtype,
            dynamic_rows=dynamic_rows,
            block_m=block_m,
            output_stride=output_stride,
        )
    if algorithm == "tensor_core":
        return q4_projection.__wrapped__(
            rows,
            outputs,
            inputs,
            block_n,
            block_k,
            output_dtype,
            dynamic_rows=dynamic_rows,
            block_m=block_m,
            stages=stages,
            output_stride=output_stride,
            tail_m=tail_m,
            group_m=group_m,
        )
    raise ValueError("Unknown Q4 decode implementation")


@orin_jit
def q4_projection(
    M,
    N,
    K,
    BN=64,
    BK=128,
    output_dtype="float16",
    magnitudes=MAGNITUDES,
    *,
    block_m=8,
    dynamic_rows=False,
    stages=2,
    output_stride=None,
    tail_m=0,
    group_m=0,
):
    validate_shape(M, N, K)
    if BN not in (32, 64, 128) or BK % 128 or K % BK:
        raise ValueError("Invalid Q4 Tensor Core tile")
    if block_m not in (8, 16, 32, 64, 128):
        raise ValueError("Invalid Q4 row tile")
    if type(stages) is not int or stages not in (1, 2):
        raise ValueError("Invalid Q4 pipeline depth")
    if type(group_m) is not int or group_m not in (0, 1, 2, 4, 8):
        raise ValueError("Invalid Q4 row grouping")
    BM = block_m
    M = row_count(M, dynamic_rows)
    stride = N if output_stride is None else output_stride
    if type(stride) is not int or stride < N:
        raise ValueError("Q4 output stride must contain the complete output row")
    threads = 64 if BN == 32 else 128
    source = decode_source(magnitudes)

    tiles, project = tensor_core_tiles(
        M, N, K, BM, BN, BK, stages=stages, codec="q4", source=source, tail_m=tail_m
    )

    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.int8),
        W: T.Tensor((N, K // 4), T.uint16),
        S: T.Tensor((N,), T.float16),
        AS: T.Tensor((M,), T.float16),
        Out: T.Tensor((M, stride), output_dtype),
    ):
        if group_m:
            with T.Kernel(tiles * T.ceildiv(N, BN), threads=threads) as pid:
                by, bx = tile_coordinates(pid, tiles, T.ceildiv(N, BN), group_m)
                project(A, W, W, S, S, AS, Out, by, bx)
        else:
            with T.Kernel(tiles, T.ceildiv(N, BN), threads=threads) as (by, bx):
                project(A, W, W, S, S, AS, Out, by, bx)

    return main


@orin_jit
def unpack_q4(N: int, K: int, magnitudes=MAGNITUDES):
    """Expand one projection into caller-owned, temporary INT8 storage.

    Packed INT32 stores preserve the exact signed integer codebook. Scales and
    rotations stay with the consuming projection; no new quantization occurs.
    """
    validate_shape(1, N, K)
    source = decode_source(magnitudes)
    words = N * (K // 4)

    @T.prim_func
    def main(
        W: T.Tensor((N, K // 4), T.uint16),
        Output: T.Tensor((N, K // 4), T.int32),
    ):
        with T.Kernel(T.ceildiv(words, 256), threads=128) as block:
            T.import_source(source)
            for offset in T.Parallel(256):
                word = block * 256 + offset
                if word < words:
                    n, k = word // (K // 4), word % (K // 4)
                    Output[n, k] = T.cast(
                        T.call_pure_extern("uint32", "q4_decode", T.cast(W[n, k], T.uint32)),
                        T.int32,
                    )

    return main
