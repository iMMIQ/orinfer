"""Compile-time integer projection components shared across codecs and fusions.

Only weight loading depends on the codec. Scheduling owns accumulation and
tails; the epilogue owns scale order and the FP16 gate/up rounding boundary.
Macros inline into the caller, preserving its ABI and runtime row parameter.
"""

import tilelang
import tilelang.language as T


DP4A_SOURCE = """
__device__ __forceinline__ int orinfer_dp4a(int a, int b, int c) {
    return __dp4a(a, b, c);
}
"""


def rescale(value, weight_scale, token_scale):
    return (T.cast(value, T.float32) * T.cast(weight_scale, T.float32)) * T.cast(
        token_scale, T.float32
    )


def swiglu(gate, up):
    g = T.cast(T.cast(gate, T.float16), T.float32)
    u = T.cast(T.cast(up, T.float16), T.float32)
    return g / (1.0 + T.exp(-g)) * u


def tile_coordinates(pid, rows, columns, group_m):
    """Group neighbouring row tiles while reusing a bounded activation slab."""
    if group_m == 1:
        return pid // columns, pid % columns
    if group_m:
        first = (pid // (group_m * columns)) * group_m
        actual = T.min(rows - first, group_m)
        return first + (pid % (group_m * columns)) % actual, (pid % (group_m * columns)) // actual
    return pid % rows, pid // rows


def tensor_core_tiles(m, n, k, bm, bn, bk, *, tail_m=0, grouped=False, **options):
    """Use the same full-K arithmetic for large full tiles and smaller tails."""
    if (
        type(tail_m) is not int
        or tail_m not in (0, 8, 16, 32, 64)
        or (tail_m and (tail_m >= bm or bm % tail_m))
    ):
        raise ValueError("Invalid integer projection tail tile")
    bulk = tensor_core(m, n, k, bm, bn, bk, grouped=grouped, **options)
    if tail_m:
        tail = tensor_core(m, n, k, tail_m, bn, bk, grouped=grouped, **options)
    tiles = T.ceildiv(m, bm)

    if grouped:

        @T.macro
        def grouped_project(A, W, U, WS, US, AS, Out, by, bx, expert, first, count):
            if tail_m:
                if count <= tail_m:
                    tail(A, W, U, WS, US, AS, Out, 0, bx, expert, first, count)
                else:
                    bulk(A, W, U, WS, US, AS, Out, 0, bx, expert, first, count)
            else:
                bulk(A, W, U, WS, US, AS, Out, by, bx, expert, first, count)

        return tiles, grouped_project

    @T.macro
    def project(A, W, U, WS, US, AS, Out, by, bx):
        if tail_m:
            if (by + 1) * bm > m and m % bm <= tail_m:
                tail(A, W, U, WS, US, AS, Out, by * (bm // tail_m), bx)
            else:
                bulk(A, W, U, WS, US, AS, Out, by, bx)
        else:
            bulk(A, W, U, WS, US, AS, Out, by, bx)

    return tiles, project


@T.macro
def prepare_e8p_book(source, book):
    # Hoist the 256 possible sign/parity combinations out of every K tile.
    # The additional 1 KiB is block-local, never a persistent weight cache.
    for i in T.Parallel(256):
        book[i, 0] = source[i, 0]
        book[i + 256, 0] = T.call_pure_extern("uint32", "e8p_q4_flags", T.cast(i, T.uint32))


def weight_loader(codec, n, bn, bk, source, *, packed_k=0, grouped=False, wait_for_encoded=True):
    if codec not in ("w8", "q4", "e8p_q4"):
        raise ValueError("Unsupported integer weight codec")

    @T.macro
    def load(W, Book, tile, bx, kg, expert):
        if codec == "w8":
            if grouped:
                T.copy(W[expert, bx * bn, kg * bk], tile)
            else:
                T.copy(W[bx * bn, kg * bk], tile)
        elif codec == "q4":
            T.import_source(source)
            decoded = T.alloc_fragment((bn, bk // 4), T.uint32)
            for j, k in T.Parallel(bn, bk // 4):
                decoded[j, k] = 0
                if bx * bn + j < n:
                    decoded[j, k] = T.call_pure_extern(
                        "uint32", "q4_decode", T.cast(W[bx * bn + j, kg * (bk // 4) + k], T.uint32)
                    )
            for j, k in T.Parallel(bn, bk):
                tile[j, k] = T.cast((decoded[j, k // 4] >> ((k % 4) * 8)) & 255, T.int8)
        else:
            T.import_source(source)
            if packed_k == 128:
                # Stage contiguous compressed slabs alongside A. The pipeline
                # can prefetch the next slab while this one is decoded/MMA'd;
                # the resident weight representation remains compressed.
                encoded = T.alloc_shared((bk // 128, bn, 16), T.uint32)
                for part in T.unroll(bk // 128):
                    if grouped:
                        T.copy(
                            W[expert, kg * (bk // 128) + part, bx * bn : bx * bn + bn, 0:16],
                            encoded[part, :, :],
                        )
                    else:
                        T.copy(
                            W[kg * (bk // 128) + part, bx * bn : bx * bn + bn, 0:16],
                            encoded[part, :, :],
                        )
                if wait_for_encoded:
                    # A's multi-stage wait may leave this newest copy pending
                    # when the scalar decoder starts consuming the slab.
                    T.ptx_wait_group(0)
                    T.sync_threads()
            decoded = T.alloc_fragment((bn, bk // 4), T.uint32)
            for j, k in T.Parallel(bn, bk // 4):
                decoded[j, k] = 0
                if bx * bn + j < n:
                    vector = kg * (bk // 8) + k // 2
                    if packed_k == 128:
                        code = encoded[k // 32, j, (k // 2) % 16]
                    elif packed_k:
                        code = W[vector // (packed_k // 8), bx * bn + j, vector % (packed_k // 8)]
                    else:
                        code = W[bx * bn + j, vector]
                    decoded[j, k] = T.call_pure_extern(
                        "uint32",
                        "e8p_q4_decode_flags",
                        Book[(code >> 8) & 255, 0],
                        Book[code >> 24, 0],
                        Book[256 + (code & 255), 0],
                        Book[256 + ((code >> 16) & 255), 0],
                        k % 2,
                    )
            for j, k in T.Parallel(bn, bk):
                tile[j, k] = T.cast((decoded[j, k // 4] >> ((k % 4) * 8)) & 255, T.int8)

    return load


def tensor_core(
    m,
    n,
    k,
    bm,
    bn,
    bk,
    *,
    stages=2,
    codec="w8",
    source="",
    dual=False,
    packed_k=0,
    grouped=False,
    cache_scales=False,
):
    if codec == "e8p_q4" and dual:
        raise ValueError("E8P-Q4 reserves the secondary operand for its shared basis")
    if grouped and (
        codec not in ("w8", "e8p_q4") or (codec == "e8p_q4" and packed_k != 128) or dual
    ):
        raise ValueError("Unsupported grouped integer operands")
    # A single-stage pipeline already waits for every copy. Preserve that
    # schedule instead of adding another wait/barrier to its hot K loop.
    load = weight_loader(
        codec,
        n,
        bn,
        bk,
        source,
        packed_k=packed_k,
        grouped=grouped,
        wait_for_encoded=stages > 1,
    )
    # Keep Q4's weight-left MMA orientation and W8's activation-left layout.
    left = codec != "w8"

    @T.macro
    def core(A, W, U, WS, US, AS, Out, by, bx, expert, first, m):
        a = T.alloc_shared((bm, bk), T.int8)
        w = T.alloc_shared((bn, bk), T.int8)
        acc = T.alloc_fragment((bn, bm) if left else (bm, bn), T.int32)
        if cache_scales:
            # Broadcast layouts follow the accumulator: each thread loads its
            # distinct column/row scales once and reuses them across outputs.
            # Widening FP16 scales is exact; rescale retains both FP32 multiplies.
            weight_scales = T.alloc_fragment((bn,), T.float32)
            token_scales = T.alloc_fragment((bm,), T.float32)
            for j in T.Parallel(bn):
                weight_scales[j] = 0
                if bx * bn + j < n:
                    weight_scales[j] = T.cast(
                        WS[expert, bx * bn + j] if grouped else WS[bx * bn + j], T.float32
                    )
            for i in T.Parallel(bm):
                token_scales[i] = 0
                if by * bm + i < m:
                    token_scales[i] = T.cast(AS[first + by * bm + i], T.float32)
            if dual:
                up_scales = T.alloc_fragment((bn,), T.float32)
                for j in T.Parallel(bn):
                    up_scales[j] = 0
                    if bx * bn + j < n:
                        up_scales[j] = T.cast(US[bx * bn + j], T.float32)
        if dual:
            u = T.alloc_shared((bn, bk), T.int8)
            up = T.alloc_fragment((bn, bm) if left else (bm, bn), T.int32)
            T.clear(up)
        if codec == "e8p_q4":
            T.import_source(source)
            book = T.alloc_shared((512, 1), T.uint32)
            prepare_e8p_book(U, book)
        T.clear(acc)
        for kg in T.Pipelined(k // bk, num_stages=stages):
            T.copy(A[first + by * bm, kg * bk], a)
            if codec == "e8p_q4":
                load(W, book, w, bx, kg, expert)
            else:
                load(W, W, w, bx, kg, expert)
            if dual:
                load(U, U, u, bx, kg, expert)
            if left:
                T.gemm(w, a, acc, transpose_B=True)
            else:
                T.gemm(a, w, acc, transpose_B=True)
            if dual:
                if left:
                    T.gemm(u, a, up, transpose_B=True)
                else:
                    T.gemm(a, u, up, transpose_B=True)
        for i, j in T.Parallel(bm, bn):
            if by * bm + i < m and bx * bn + j < n:
                if cache_scales:
                    scale = weight_scales[j]
                    token_scale = token_scales[i]
                else:
                    scale = WS[expert, bx * bn + j] if grouped else WS[bx * bn + j]
                    token_scale = AS[first + by * bm + i]
                value = rescale(acc[j, i] if left else acc[i, j], scale, token_scale)
                if dual:
                    other_scale = up_scales[j] if cache_scales else US[bx * bn + j]
                    other = rescale(up[j, i] if left else up[i, j], other_scale, token_scale)
                    Out[first + by * bm + i, bx * bn + j] = swiglu(value, other)
                else:
                    Out[first + by * bm + i, bx * bn + j] = value

    @T.macro
    def project(A, W, U, WS, US, AS, Out, by, bx):
        core(A, W, U, WS, US, AS, Out, by, bx, 0, 0, m)

    return core if grouped else project


def dp4a(m, n, k, bm, bn, *, codec="w8", source="", dual=False, packed_k=0):
    if codec not in ("w8", "q4", "e8p_q4"):
        raise ValueError("Unsupported integer weight codec")
    if dual and codec != "w8":
        raise ValueError("The dual DP4A ABI requires two W8 operands")

    @T.macro
    def project(A, W, U, WS, US, AS, Out, by, bx):
        T.import_source(DP4A_SOURCE + source)
        acc = T.alloc_fragment((bm, bn, 32), T.int32)
        total = T.alloc_fragment((bm, bn), T.int32)
        if dual:
            up = T.alloc_fragment((bm, bn, 32), T.int32)
            up_total = T.alloc_fragment((bm, bn), T.int32)
            T.annotate_layout(
                {
                    up: tilelang.Fragment(
                        (bm, bn, 32),
                        forward_thread_fn=lambda i, j, l: (j % 4) * 32 + l,
                        forward_index_fn=lambda i, j, l: i * (bn // 4) + j // 4,
                    )
                }
            )
            T.clear(up)
        else:
            weight = T.alloc_fragment((bn, 32), T.int32 if codec == "w8" else T.uint32)
            T.annotate_layout(
                {
                    weight: tilelang.Fragment(
                        (bn, 32),
                        forward_thread_fn=lambda j, l: (j % 4) * 32 + l,
                        forward_index_fn=lambda j, l: j // 4,
                    )
                }
            )
        if codec == "e8p_q4":
            book = T.alloc_shared((512, 1), T.uint32)
            prepare_e8p_book(U, book)
        T.annotate_layout(
            {
                acc: tilelang.Fragment(
                    (bm, bn, 32),
                    forward_thread_fn=lambda i, j, l: (j % 4) * 32 + l,
                    forward_index_fn=lambda i, j, l: i * (bn // 4) + j // 4,
                )
            }
        )
        T.clear(acc)
        for kg in T.serial(k // 128):
            if not dual:
                for j, l in T.Parallel(bn, 32):
                    weight[j, l] = 0
                    if bx * bn + j < n:
                        if codec == "w8":
                            weight[j, l] = W[bx * bn + j, kg * 32 + l]
                        elif codec == "q4":
                            weight[j, l] = T.call_pure_extern(
                                "uint32", "q4_decode", T.cast(W[bx * bn + j, kg * 32 + l], T.uint32)
                            )
                        else:
                            vector = kg * 16 + l // 2
                            code = (
                                W[vector // (packed_k // 8), bx * bn + j, vector % (packed_k // 8)]
                                if packed_k
                                else W[bx * bn + j, vector]
                            )
                            weight[j, l] = T.call_pure_extern(
                                "uint32",
                                "e8p_q4_decode_flags",
                                book[(code >> 8) & 255, 0],
                                book[code >> 24, 0],
                                book[256 + (code & 255), 0],
                                book[256 + ((code >> 16) & 255), 0],
                                l % 2,
                            )
            for i, j, l in T.Parallel(bm, bn, 32):
                if by * bm + i < m:
                    if dual:
                        if bx * bn + j < n:
                            acc[i, j, l] = T.call_pure_extern(
                                "int32",
                                "orinfer_dp4a",
                                A[by * bm + i, kg * 32 + l],
                                W[bx * bn + j, kg * 32 + l],
                                acc[i, j, l],
                            )
                            up[i, j, l] = T.call_pure_extern(
                                "int32",
                                "orinfer_dp4a",
                                A[by * bm + i, kg * 32 + l],
                                U[bx * bn + j, kg * 32 + l],
                                up[i, j, l],
                            )
                    else:
                        acc[i, j, l] = T.call_pure_extern(
                            "int32",
                            "orinfer_dp4a",
                            A[by * bm + i, kg * 32 + l],
                            T.cast(weight[j, l], T.int32),
                            acc[i, j, l],
                        )
        T.reduce_sum(acc, total, dim=2)
        if dual:
            T.reduce_sum(up, up_total, dim=2)
        for i, j in T.Parallel(bm, bn):
            if by * bm + i < m and bx * bn + j < n:
                value = rescale(total[i, j], WS[bx * bn + j], AS[by * bm + i])
                if dual:
                    other = rescale(up_total[i, j], US[bx * bn + j], AS[by * bm + i])
                    Out[by * bm + i, bx * bn + j] = swiglu(value, other)
                else:
                    Out[by * bm + i, bx * bn + j] = value

    return project
