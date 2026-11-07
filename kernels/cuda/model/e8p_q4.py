"""Two-stage E8P projections with exact INT8 or direct INT4 schedules."""

import tilelang
import tilelang.language as T

from kernels.cuda.model.integer_core import (
    dp4a,
    prepare_e8p_book,
    tensor_core_tiles,
    tile_coordinates,
)
from kernels.cuda.model.q4_projection import validate_shape
from kernels.shared.rows import row_count
from orinfer_build.compiler.cuda.common import orin_jit
from orinfer_build.quantization.e8p_q4 import validate_coefficients


def decode_source(coefficients):
    a, b = validate_coefficients(coefficients)
    coarse_scale = 4 if (a, b) == (4, 1) else 1
    # Bias each decoded byte by 112 (coarse) and 16 (fine). Their sum is
    # q+128 in [1,255], so a single packed addition cannot carry between
    # bytes. The final XOR converts offset binary back to signed INT8.
    coarse_positive, coarse_negative = (
        ("0x749c8c7cu", "0x744c5c6cu") if (a, b) == (4, 1) else ("0x042c1c0cu", "0x04dcecfcu")
    )
    fine_positive, fine_negative = (
        ("0x111b1713u", "0x11070b0fu") if (a, b) == (4, 1) else ("0x010b0703u", "0x01f7fbffu")
    )
    combine = (
        """
    return (x + y) ^ 0x80808080u;
"""
        if (a, b) == (4, 1)
        else f"""
    // Each independent 16-bit lane holds q+128 in [1,255].
    unsigned xb = x ^ 0x80808080u, yb = y ^ 0x80808080u;
    unsigned lo = {a}u * (xb & 0x00ff00ffu) + {b}u * (yb & 0x00ff00ffu)
                  - {a + b - 1}u * 0x00800080u;
    unsigned hi = {a}u * ((xb >> 8) & 0x00ff00ffu) + {b}u * ((yb >> 8) & 0x00ff00ffu)
                  - {a + b - 1}u * 0x00800080u;
    return ((lo & 0x00ff00ffu) | ((hi & 0x00ff00ffu) << 8)) ^ 0x80808080u;
"""
    )
    return f"""
__device__ __forceinline__ unsigned e8p_q4_flags(unsigned code) {{
    unsigned parity = __popc(code) & 1u;
    unsigned signs = code ^ parity;
    unsigned lo = ((signs & 0x11u) << 2u) | ((signs & 0x22u) << 9u);
    unsigned hi = (((signs >> 2u) & 0x11u) << 2u) | (((signs >> 2u) & 0x22u) << 9u);
    // Reserve both parity shifts; sign flags occupy only nibble bit two.
    return (parity * 0x0a0a0a0au) | lo | (hi << 16u);
}}
__device__ __forceinline__ unsigned e8p_q4_selector_flags(unsigned basis, unsigned flags, unsigned half, unsigned scale) {{
    unsigned selector = (basis ^ (flags & 0x44444444u)) >> (half * 16u);
    unsigned shift = flags & (scale == 4u ? 0x08080808u : 0x02020202u);
    unsigned result;
    // Selector nibbles contain only bits 0..2. PRMT avoids the redundant
    // selector mask inserted by __byte_perm. Keep subtraction rather than
    // parity XOR so zero-valued basis coordinates retain their exact rule.
    // Scale is a call-site constant. Fold coarse x4 into the permute constants
    // and its parity shift instead of expanding and multiplying every quad.
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(result)
        : "r"((scale == 4u ? {coarse_positive} : {fine_positive}) - shift),
          "r"((scale == 4u ? {coarse_negative} : {fine_negative}) - shift), "r"(selector));
    return result;
}}
__device__ __forceinline__ unsigned e8p_q4_combine(unsigned x, unsigned y) {{
    {combine}
}}
__device__ __forceinline__ unsigned e8p_q4_decode_flags(unsigned b0, unsigned b1, unsigned f0, unsigned f1, unsigned half) {{
    return e8p_q4_combine(e8p_q4_selector_flags(b0, f0, half, {coarse_scale}u),
                          e8p_q4_selector_flags(b1, f1, half, 1u));
}}
"""


def validate_packed_k(K, packed_k):
    if type(packed_k) is not int or packed_k not in (0, 64, 128) or (packed_k and K % packed_k):
        raise ValueError("Invalid compressed K layout")


@orin_jit
def embedding(capacity, vocab, hidden):
    """Gather and inverse-rotate selected rows of the shared K128 Q4 bank."""
    from kernels.cuda.model.rotation_a8 import block_hadamard

    validate_shape(capacity, vocab, hidden)
    m = row_count(capacity, True)
    source = decode_source((4, 1))
    butterfly = block_hadamard(128)

    @T.prim_func
    def main(
        IDs: T.Tensor((m,), T.int32),
        Weight: T.Tensor((hidden // 128, vocab, 16), T.uint32),
        Book: T.Tensor((256, 1), T.uint32),
        Scale: T.Tensor((vocab,), T.float16),
        Signs: T.Tensor((hidden,), T.int8),
        Output: T.Tensor((m, hidden), T.float16),
    ):
        with T.Kernel(m, hidden // 128, threads=128) as (row, block):
            T.import_source(source)
            book = T.alloc_shared((512, 1), T.uint32)
            current = T.alloc_shared((128,), T.float32)
            values = T.alloc_fragment((128,), T.float32)
            T.annotate_layout(
                {
                    values: tilelang.Fragment(
                        (128,), forward_thread_fn=lambda k: k, forward_index_fn=lambda k: 0
                    )
                }
            )
            prepare_e8p_book(Book, book)
            T.sync_threads()
            for k in T.Parallel(128):
                values[k] = 0
                if 0 <= IDs[row] and IDs[row] < vocab:
                    code = Weight[block, IDs[row], k // 8]
                    word = T.call_pure_extern(
                        "uint32",
                        "e8p_q4_decode_flags",
                        book[(code >> 8) & 255, 0],
                        book[code >> 24, 0],
                        book[256 + (code & 255), 0],
                        book[256 + ((code >> 16) & 255), 0],
                        (k // 4) % 2,
                    )
                    values[k] = T.cast(T.cast((word >> ((k % 4) * 8)) & 255, T.int8), T.float32)
            butterfly(values, current)
            for k in T.Parallel(128):
                Output[row, block * 128 + k] = T.if_then_else(
                    0 <= IDs[row] and IDs[row] < vocab,
                    (values[k] * 0.08838834764831845)
                    * T.cast(Scale[IDs[row]], T.float32)
                    * T.cast(Signs[block * 128 + k], T.float32),
                    0,
                )

    return main


@orin_jit
def projection(
    M,
    N,
    K,
    *,
    coefficients=(4, 1),
    algorithm="tensor_core",
    block_m=8,
    block_n=64,
    block_k=128,
    stages=2,
    dynamic_rows=False,
    output_dtype="float16",
    output_stride=None,
    packed_k=0,
    basis_nonzero=False,
    split_k=1,
    tail_m=0,
    group_m=0,
    cache_scales=False,
):
    validate_shape(M, N, K)
    source = decode_source(coefficients)
    stride = N if output_stride is None else output_stride
    if type(stride) is not int or stride < N:
        raise ValueError("Output stride must contain the entire row")
    validate_packed_k(K, packed_k)
    if type(basis_nonzero) is not bool:
        raise ValueError("Invalid E8P basis contract")
    if type(split_k) is not int or split_k not in (1, 2, 4):
        raise ValueError("Invalid E8P K partition")
    if type(group_m) is not int or group_m not in (0, 1, 2, 4, 8):
        raise ValueError("Invalid E8P row grouping")
    if algorithm != "tensor_core" and (tail_m or group_m):
        raise ValueError("Tail/group scheduling requires the INT8 Tensor Core path")
    if type(cache_scales) is not bool or (cache_scales and algorithm != "tensor_core"):
        raise ValueError("Scale caching requires the INT8 Tensor Core path")
    if algorithm != "int4_direct" and split_k != 1:
        raise ValueError("K partition requires direct INT4")
    if algorithm == "tensor_core":
        if block_m not in (8, 16, 32, 64, 96, 128, 192, 256) or block_n not in (32, 64, 128):
            raise ValueError("Invalid E8P-Q4 Tensor Core tile")
        if block_k <= 0 or block_k % 128 or K % block_k or stages not in (1, 2):
            raise ValueError("Invalid E8P-Q4 K tile or pipeline")
    elif algorithm == "dp4a":
        if not 1 <= block_m <= 8 or block_n not in (4, 8, 16):
            raise ValueError("Invalid E8P-Q4 DP4A tile")
    elif algorithm == "int4_direct":
        if not packed_k or not basis_nonzero:
            raise ValueError("Direct INT4 requires a verified nonzero basis and K-major weights")
        if (
            block_m != 8
            or block_n not in (32, 64, 128)
            or block_k != 128
            or stages != 2
            or K % (128 * split_k)
            or block_n * split_k > 256
        ):
            raise ValueError("Invalid direct E8P-Q4 tile")
    else:
        raise ValueError("Unknown E8P-Q4 projection algorithm")
    m = row_count(M, dynamic_rows)
    activation_width = K // 4 if algorithm == "dp4a" else K
    activation_dtype = T.int32 if algorithm == "dp4a" else T.int8
    if algorithm == "int4_direct":
        from kernels.cuda.model.e8p_direct import direct_tensor_core

        threads = block_n * split_k
        project = direct_tensor_core(
            m,
            N,
            K,
            block_m,
            block_n,
            stages=stages,
            packed_k=packed_k,
            coefficients=coefficients,
            split_k=split_k,
        )
    else:
        threads = 64 if algorithm == "tensor_core" and block_n == 32 else 128
        if algorithm == "tensor_core":
            tiles, project = tensor_core_tiles(
                m,
                N,
                K,
                block_m,
                block_n,
                block_k,
                stages=stages,
                codec="e8p_q4",
                source=source,
                packed_k=packed_k,
                tail_m=tail_m,
                cache_scales=cache_scales,
            )
        else:
            project = dp4a(
                m, N, K, block_m, block_n, codec="e8p_q4", source=source, packed_k=packed_k
            )
    if algorithm != "tensor_core":
        tiles = T.ceildiv(m, block_m)

    @T.prim_func
    def main(
        A: T.Tensor((m, activation_width), activation_dtype),
        W: T.Tensor((K // packed_k, N, packed_k // 8) if packed_k else (N, K // 8), T.uint32),
        Book: T.Tensor((256, 1), T.uint32),
        S: T.Tensor((N,), T.float16),
        AS: T.Tensor((m,), T.float16),
        Out: T.Tensor((m, stride), output_dtype),
    ):
        if group_m:
            with T.Kernel(tiles * T.ceildiv(N, block_n), threads=threads) as pid:
                by, bx = tile_coordinates(pid, tiles, T.ceildiv(N, block_n), group_m)
                project(A, W, Book, S, S, AS, Out, by, bx)
        else:
            with T.Kernel(tiles, T.ceildiv(N, block_n), threads=threads) as (by, bx):
                project(A, W, Book, S, S, AS, Out, by, bx)

    return main


@orin_jit
def unpack(N, K, *, coefficients=(4, 1), packed_k=0):
    validate_shape(1, N, K)
    validate_packed_k(K, packed_k)
    source = decode_source(coefficients)

    @T.prim_func
    def main(
        W: T.Tensor((K // packed_k, N, packed_k // 8) if packed_k else (N, K // 8), T.uint32),
        Book: T.Tensor((256, 1), T.uint32),
        Output: T.Tensor((N, K // 4), T.uint32),
    ):
        with T.Kernel(T.ceildiv(N * K // 4, 256), threads=128) as block:
            T.import_source(source)
            book = T.alloc_shared((512, 1), T.uint32)
            prepare_e8p_book(Book, book)
            for i in T.Parallel(256):
                word = block * 256 + i
                if word < N * K // 4:
                    row, col = word // (K // 4), word % (K // 4)
                    vector = col // 2
                    code = (
                        W[vector // (packed_k // 8), row, vector % (packed_k // 8)]
                        if packed_k
                        else W[row, vector]
                    )
                    Output[row, col] = T.call_pure_extern(
                        "uint32",
                        "e8p_q4_decode_flags",
                        book[(code >> 8) & 255, 0],
                        book[code >> 24, 0],
                        book[256 + (code & 255), 0],
                        book[256 + ((code >> 16) & 255), 0],
                        col % 2,
                    )

    return main
