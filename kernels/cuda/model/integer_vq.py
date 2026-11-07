"""Integer E8P tiles expanded in shared memory for SM87 INT8 MMA.

Reference ABI: group-major indices, integer table, output-row scale, A8
activation and token scale. Full K is accumulated in INT32. Tables can be
shared across experts; no persistent expanded W8 or floating inner-K scales.
"""

import tilelang
import tilelang.language as T

from orinfer_build.compiler.cuda.common import orin_jit


SOURCE = r"""
__device__ __forceinline__ unsigned int integer_e8p_selector_quad(unsigned int code, unsigned int basis, unsigned int half) {
    unsigned int parity = __popc(code & 255u) & 1u;
    unsigned int s = (code ^ parity) >> (half * 2u);
    unsigned int flags = ((s & 0x11u) << 2u) | ((s & 0x22u) << 9u);
    unsigned int selector = ((basis >> (half * 16u)) & 0x7777u) ^ flags;
    unsigned int shift = parity * 0x02020202u;
    // Both constants retain the parity-dependent zero entry. Each byte of
    // the nonzero entries is at least two, so subtracting shift has no
    // inter-byte borrow. The final byte wraps to -1 for the odd-parity zero.
    return __byte_perm(0x010b0703u - shift, 0x01f7fbffu - shift, selector);
}
__device__ __forceinline__ unsigned int integer_e8p_quad(unsigned int code, unsigned int basis, unsigned int half) {
    unsigned int parity = __popc(code & 255u) & 1u;
    unsigned int s = ((code & 255u) ^ parity) >> (half * 2u);
    unsigned int flags = (s & 1u) | ((s & 16u) << 4u) | ((s & 2u) << 15u) | ((s & 32u) << 19u);
    unsigned int mask = flags * 255u;
    // Ordinary word arithmetic with each byte's carry confined to its own
    // sign bit. CUDA's vadd4/vsub4 intrinsics expand into costly emulation on
    // SM87; these expressions preserve even zero-valued diagnostic tables.
    unsigned int negative = ((~basis & 0x7f7f7f7fu) + 0x01010101u) ^ (~basis & 0x80808080u);
    unsigned int oriented = basis ^ ((basis ^ negative) & mask);
    unsigned int plus = oriented | 0x01010101u; // every E8P basis byte is even
    unsigned int minus = (((oriented & 0x7f7f7f7fu) | 0x80808080u) - 0x01010101u) ^ (~oriented & 0x80808080u);
    return parity ? minus : plus;
}
__device__ __forceinline__ unsigned int integer_e8p_parity(unsigned int code) { return __popc(code & 255u) & 1u; }
__device__ __forceinline__ unsigned int integer_e8p_sign_quad(unsigned int code, unsigned int parity, unsigned int positive, unsigned int negative, unsigned int half) {
    unsigned int s = ((code & 255u) ^ parity) >> (half * 2u);
    unsigned int flags = (s & 1u) | ((s & 16u) << 4u) | ((s & 2u) << 15u) | ((s & 32u) << 19u);
    unsigned int mask = flags * 255u;
    return positive ^ ((positive ^ negative) & mask);
}
__device__ __forceinline__ unsigned int integer_e8p_sign_quad_prmt(unsigned int code, unsigned int parity, unsigned int positive, unsigned int negative, unsigned int half) {
    unsigned int s = (code ^ parity) >> (half * 2u);
    unsigned int selector = 0x3210u | ((s & 0x11u) << 2u) | ((s & 0x22u) << 9u);
    unsigned int value;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(value) : "r"(positive), "r"(negative), "r"(selector));
    return value;
}

"""


@orin_jit
def _compile(
    rows,
    E,
    N,
    K,
    kind,
    shared_table,
    grouped,
    tiles,
    M,
    BN,
    patched,
    shortbook,
    num_stages,
    BM,
    byte_permute,
    selectorbook,
    group_n,
    packed_k,
    dynamic_rows=False,
):
    if (
        kind != "e8p"
        or any(type(x) is not int or x <= 0 for x in (rows, E, N, K, tiles))
        or K % 128
        or K * 128 * 128 > 2**31 - 1
    ):
        raise ValueError("Invalid integer VQ geometry")
    if BN not in (64, 128):
        raise ValueError("Invalid VQ output tile")
    if BM not in (8, 16, 32, 64):
        raise ValueError("Invalid VQ row tile")
    if num_stages not in (1, 2):
        raise ValueError("Invalid VQ decoder/pipeline")
    if type(byte_permute) is not bool or byte_permute and not shortbook:
        raise ValueError("Byte permutation requires the E8P short table")
    if type(selectorbook) is not bool or selectorbook and not shortbook:
        raise ValueError("Selector decoding requires a validated E8P short basis")
    if type(group_n) is not bool:
        raise ValueError("Invalid routed tile order")
    if type(packed_k) is not int or packed_k not in (64, 128):
        raise ValueError("Invalid integer VQ code layout")
    D = 8
    PD = T.uint16
    TE = 1 if shared_table else E
    entries, words = (256, 1) if selectorbook else (512, 4) if shortbook else (256, D // 4)
    G = 128
    from kernels.shared.rows import row_count

    rows = row_count(rows, dynamic_rows)

    @T.macro
    def project(A, P, Book, Patch, WS, AS, C, expert, offset, count, bx):
        T.import_source(SOURCE)
        # Small routed batches feed MMA directly from register fragments.
        # Keep packed-code prefetch and the LUT in shared memory, but avoid
        # staging expanded INT8 weights and activations through ldmatrix.
        if BM == 8 and shortbook:
            a = T.alloc_fragment((BM, G), T.int8)
            b = T.alloc_fragment((BN, G), T.int8)
        else:
            a = T.alloc_shared((BM, G), T.int8)
            b = T.alloc_shared((BN, G), T.int8)
        packed = T.alloc_shared((BN, G // D), PD)
        table = T.alloc_shared((entries, words), T.uint32)
        patch = T.alloc_shared((BN,), T.uint16)
        decoded = T.alloc_fragment((BN, G // 4), T.uint32)
        # Put short token batches on MMA's eight-column dimension. Decoding
        # and stored weights stay unchanged; only the integer product is
        # transposed, then written in the original token/channel order.
        acc = T.alloc_fragment((BN, BM) if BM == 8 else (BM, BN), T.int32)
        T.copy(Book[0 if shared_table else expert, 0, 0], table)
        T.clear(acc)
        for kg in T.Pipelined(K // G, num_stages=num_stages):
            if (BM == 32 or selectorbook and BM == 16) and num_stages == 1:
                # MMA rows are independent, and only rows below count are
                # stored. Extra rows may read the following expert's inputs;
                # T.copy masks the end of the allocation. This permits vector
                # asynchronous copies instead of conditional byte stores.
                T.copy(A[offset, kg * G], a)
            else:
                for m, k in T.Parallel(BM, G):
                    a[m, k] = 0
                    if m < count:
                        a[m, k] = A[offset + m, kg * G + k]
            if packed_k == 64:
                # Two contiguous K64 rows reconstruct the original K128 tile.
                for half in T.unroll(2):
                    T.copy(
                        P[expert, kg * 2 + half, bx * BN : (bx + 1) * BN, 0:8],
                        packed[:, half * 8 : (half + 1) * 8],
                    )
            else:
                T.copy(P[expert, kg, bx * BN, 0], packed)
            if patched:
                # Scalar metadata reads avoid mixing two asynchronous producer
                # stages in TileLang's software-pipeline dependency graph.
                for n in T.Parallel(BN):
                    patch[n] = 0
                    if bx * BN + n < N:
                        patch[n] = Patch[expert, kg, bx * BN + n]
            # Decode each four-byte word once in registers. The separate,
            # layout-aware shared write lets TileLang vectorize four bytes;
            # a scalar alloc_var in the byte loop prevents that on SM87.
            for n, q in T.Parallel(BN, G // 4):
                code = T.cast(packed[n, q // 2], T.uint32)
                if selectorbook:
                    basis = table[T.cast(code >> 8, T.int32), 0]
                    decoded[n, q] = T.call_pure_extern(
                        "uint32",
                        "integer_e8p_selector_quad",
                        code,
                        basis,
                        T.cast(q % 2, T.uint32),
                    )
                elif shortbook:
                    parity = T.call_pure_extern("uint32", "integer_e8p_parity", code)
                    index = T.cast((code >> 8) + parity * 256, T.int32)
                    positive = table[index, (q % 2) * 2]
                    negative = table[index, (q % 2) * 2 + 1]
                    decoded[n, q] = T.call_pure_extern(
                        "uint32",
                        "integer_e8p_sign_quad_prmt" if byte_permute else "integer_e8p_sign_quad",
                        code,
                        parity,
                        positive,
                        negative,
                        T.cast(q % 2, T.uint32),
                    )
                else:
                    word = table[T.cast(code >> 8, T.int32), q % 2]
                    decoded[n, q] = T.call_pure_extern(
                        "uint32", "integer_e8p_quad", code, word, T.cast(q % 2, T.uint32)
                    )
                if patched:
                    shift = T.cast((patch[n] & 3) * 8, T.uint32)
                    replacement = (decoded[n, q] & ~(T.uint32(255) << shift)) | (
                        T.cast((patch[n] >> 7) & 255, T.uint32) << shift
                    )
                    decoded[n, q] = T.if_then_else(
                        (patch[n] & 32768) != 0 and (patch[n] & 127) // 4 == q,
                        replacement,
                        decoded[n, q],
                    )
            for n, k in T.Parallel(BN, G):
                b[n, k] = T.cast((decoded[n, k // 4] >> ((k % 4) * 8)) & 255, T.int8)
            if BM == 8:
                T.gemm(b, a, acc, transpose_B=True)
            else:
                T.gemm(a, b, acc, transpose_B=True)
        for m, n in T.Parallel(BM, BN):
            if m < count and bx * BN + n < N:
                C[offset + m, bx * BN + n] = (
                    T.cast(acc[n, m] if BM == 8 else acc[m, n], T.float32)
                    * T.cast(WS[expert, bx * BN + n], T.float32)
                ) * T.cast(AS[offset + m], T.float32)

    @T.prim_func
    def direct(
        A: T.Tensor((rows, K), T.int8),
        P: T.Tensor((E, K // packed_k, N, packed_k // D), PD),
        Book: T.Tensor((TE, entries, words), T.uint32),
        Patch: T.Tensor((E, K // G, N) if patched else (1,), T.uint16),
        WS: T.Tensor((E, N), T.float16),
        AS: T.Tensor((rows,), T.float16),
        C: T.Tensor((rows, N), T.float16),
    ):
        with T.Kernel(T.ceildiv(M, BM), T.ceildiv(N, BN), E, threads=128) as (by, bx, e):
            project(A, P, Book, Patch, WS, AS, C, e, e * M + by * BM, T.min(BM, M - by * BM), bx)

    @T.prim_func
    def routed(
        A: T.Tensor((rows, K), T.int8),
        P: T.Tensor((E, K // packed_k, N, packed_k // D), PD),
        Book: T.Tensor((TE, entries, words), T.uint32),
        Patch: T.Tensor((E, K // G, N) if patched else (1,), T.uint16),
        WS: T.Tensor((E, N), T.float16),
        AS: T.Tensor((rows,), T.float16),
        Counts: T.Tensor((E,), T.int32),
        Offsets: T.Tensor((E,), T.int32),
        TileExpert: T.Tensor((tiles,), T.int32),
        TileRow: T.Tensor((tiles,), T.int32),
        TileCount: T.Tensor((1,), T.int32),
        C: T.Tensor((rows, N), T.float16),
    ):
        # Adjacent output tiles reuse one expert's routed activations in L2.
        with T.Kernel(
            T.ceildiv(N, BN) if group_n else tiles,
            tiles if group_n else T.ceildiv(N, BN),
            threads=128,
        ) as (first, second):
            tile = second if group_n else first
            bx = first if group_n else second
            meta = T.alloc_local((4,), T.int32)
            if tile < TileCount[0]:
                meta[0] = TileExpert[tile]
                meta[1] = TileRow[tile]
                meta[2] = Offsets[meta[0]] + meta[1]
                meta[3] = T.min(BM, Counts[meta[0]] - meta[1])
                project(A, P, Book, Patch, WS, AS, C, meta[0], meta[2], meta[3], bx)

    return routed if grouped else direct


def integer_vq(
    E,
    M,
    N,
    K,
    *,
    kind,
    shared_table=False,
    block_n=64,
    patched=False,
    shortbook=False,
    num_stages=2,
    block_m=16,
    byte_permute=False,
    selectorbook=False,
    packed_k=128,
):
    return _compile(
        E * M,
        E,
        N,
        K,
        kind,
        shared_table,
        False,
        1,
        M,
        block_n,
        patched,
        shortbook,
        num_stages,
        block_m,
        byte_permute,
        selectorbook,
        False,
        packed_k,
    )


def integer_vq_grouped(
    rows,
    E,
    tiles,
    N,
    K,
    *,
    kind,
    shared_table=False,
    block_n=64,
    patched=False,
    shortbook=False,
    num_stages=2,
    block_m=16,
    byte_permute=False,
    selectorbook=False,
    warp_decode=False,
    split_k=1,
    parity_xor=False,
    group_n=False,
    packed_k=128,
    dynamic_rows=False,
):
    if type(group_n) is not bool:
        raise ValueError("Invalid routed tile order")
    if type(parity_xor) is not bool or parity_xor and not warp_decode:
        raise ValueError("XOR parity requires warp decode")
    if warp_decode:
        if dynamic_rows:
            raise ValueError("Warp decode uses its own fixed-row activation layout")
        if (
            kind != "e8p"
            or not shared_table
            or not selectorbook
            or not shortbook
            or patched
            or block_m != 8
            or group_n
        ):
            raise ValueError("Warp decode requires unpatched shared E8P selectors and BM8")
        from kernels.cuda.model.e8p_decode import e8p_decode_grouped

        return e8p_decode_grouped(
            rows,
            E,
            tiles,
            N,
            K,
            block_n=block_n,
            split_k=split_k,
            parity_xor=parity_xor,
            packed_k=packed_k,
        )
    if split_k != 1:
        raise ValueError("Split-K requires warp decode")
    return _compile(
        rows,
        E,
        N,
        K,
        kind,
        shared_table,
        True,
        tiles,
        1,
        block_n,
        patched,
        shortbook,
        num_stages,
        block_m,
        byte_permute,
        selectorbook,
        group_n,
        packed_k,
        dynamic_rows,
    )


@orin_jit
def integer_e8p_gemv(E: int, routes: int, N: int, K: int, *, shared_input: bool, block_n: int = 8):
    """Single-token routes directly index packed banks, using exact INT8 DP4A."""
    assert E > 0 and routes > 0 and N > 0 and K % 128 == 0 and block_n in (4, 8, 16)
    assert K * 128 * 128 < 2**31
    rows = 1 if shared_input else routes

    @T.prim_func
    def main(
        A: T.Tensor((rows, K // 4), T.int32),
        P: T.Tensor((E, K // 128, N, 16), T.uint16),
        Book: T.Tensor((1, 256, 2), T.uint32),
        WS: T.Tensor((E, N), T.float16),
        AS: T.Tensor((rows,), T.float16),
        IDs: T.Tensor((1, routes), T.int32),
        C: T.Tensor((routes, N), T.float16),
    ):
        with T.Kernel(T.ceildiv(N, block_n), routes, threads=128) as (bx, route):
            T.import_source(
                SOURCE
                + "\n__device__ __forceinline__ int flash_dp4a(int a, int b, int c) { return __dp4a(a,b,c); }\n"
            )
            table = T.alloc_shared((256, 2), T.uint32)
            acc = T.alloc_fragment((block_n, 32), T.int32)
            total = T.alloc_fragment((block_n,), T.int32)
            T.annotate_layout(
                {
                    acc: tilelang.Fragment(
                        (block_n, 32),
                        forward_thread_fn=lambda n, k: (n % 4) * 32 + k,
                        forward_index_fn=lambda n, k: n // 4,
                    )
                }
            )
            T.copy(Book[0, 0, 0], table)
            T.clear(acc)
            expert = IDs[0, route]
            row = 0 if shared_input else route
            for kg in T.serial(K // 128):
                for n, k in T.Parallel(block_n, 32):
                    if bx * block_n + n < N:
                        code = T.cast(P[expert, kg, bx * block_n + n, k // 2], T.uint32)
                        word = T.call_pure_extern(
                            "uint32",
                            "integer_e8p_quad",
                            code,
                            table[code >> 8, k % 2],
                            T.cast(k % 2, T.uint32),
                        )
                        acc[n, k] = T.call_pure_extern(
                            "int32",
                            "flash_dp4a",
                            A[row, kg * 32 + k],
                            T.cast(word, T.int32),
                            acc[n, k],
                        )
            T.reduce_sum(acc, total, dim=1)
            for n in T.Parallel(block_n):
                if bx * block_n + n < N:
                    C[route, bx * block_n + n] = (
                        T.cast(total[n], T.float32)
                        * T.cast(WS[expert, bx * block_n + n], T.float32)
                    ) * T.cast(AS[row], T.float32)

    return main


@orin_jit
def rotate_activation(M, K, swiglu=False):
    if type(M) is not int or M <= 0 or type(K) is not int or K <= 0 or K % 128:
        raise ValueError("Invalid block128 activation rotation")
    width = K * 2 if swiglu else K

    @T.prim_func
    def main(
        X: T.Tensor((M, width), T.float16),
        Signs: T.Tensor((K,), T.int8),
        Y: T.Tensor((M, K), T.float16),
    ):
        with T.Kernel(M, threads=128) as row:
            current = T.alloc_shared((K,), T.float32)
            temporary = T.alloc_shared((K,), T.float32)
            # Scalar signed-byte conversion avoids packed int8 -> float4
            # lowering through plain char, whose signedness is target-dependent.
            for k in T.Parallel(K, coalesced_width=T.int32(1)):
                if swiglu:
                    g = T.cast(X[row, k], T.float32)
                    e = T.exp(-T.abs(g))
                    sigmoid = T.if_then_else(g >= 0, 1 / (1 + e), e / (1 + e))
                    value = T.cast(
                        T.cast((g * sigmoid) * T.cast(X[row, k + K], T.float32), T.float16),
                        T.float32,
                    )
                    current[k] = value * T.cast(Signs[k], T.float32)
                else:
                    current[k] = T.cast(X[row, k], T.float32) * T.cast(Signs[k], T.float32)
            T.sync_threads()
            for stage in T.unroll(7):
                for k in T.Parallel(K):
                    temporary[k] = current[k ^ (1 << stage)] + current[k] * (
                        1 - 2 * ((k >> stage) & 1)
                    )
                T.sync_threads()
                T.copy(temporary, current)
                T.sync_threads()
            for k in T.Parallel(K):
                Y[row, k] = current[k] * 0.08838834764831845

    return main
