"""SM87 grouped Q2_0 x A8 projections for pre-dispatched expert rows.

P[E,N,K/64*18] retains the original Q2_0 blocks: a little-endian FP16
scale followed by 16 bytes, each containing four consecutive two-bit codes.
Each weight is scale * (code - 1). Scales may be negative. No global W8
representation or weight requantization is used.

A[E,M,K] is contiguous INT8; S[E,M,K/64] contains FP16 activation scales.
The caller quantizes groups of 64, pads unused expert rows with zero, and
owns routing and mixture reduction. q2a8_indexed reads the original bank using
INT32 expert IDs, including separate gate/up banks without reordering weights.
IDs outside [0, bank_experts) produce zero output for inactive slots. C[E,M,N]
is FP16; each group uses INT32 MMA, then FP32 rescaling and accumulation.
Buffers must be disjoint.
"""
import tilelang.language as T

from tools.operators.common import orin_jit


SOURCE = r'''
__device__ __forceinline__ unsigned q2a8_expand_word(unsigned bits) {
    unsigned result = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        result |= (unsigned(int((bits >> (2*i)) & 3u) - 1) & 255u) << (8*i);
    return result;
}
__device__ __forceinline__ unsigned q2a8_load_word(
    const signed char* values, int offset) {
    return *reinterpret_cast<const unsigned*>(values + offset);
}
__device__ __forceinline__ float q2a8_shuffle_scale(float value, int lane) {
    return __shfl_sync(0xffffffffu, value, lane);
}
'''


def _implementation(E, M, N, K, implementation):
    if any(type(value) is not int or value <= 0 for value in (E, M, N, K)):
        raise ValueError("Q2A8 dimensions must be positive integers")
    if K % 64 or N % 64:
        raise ValueError("Q2A8 requires K and N divisible by 64")
    if implementation == "auto":
        implementation = "register" if M <= 8 else "shared"
    if implementation not in ("register", "shared"):
        raise ValueError(f"Unknown Q2A8 implementation: {implementation}")
    return _register if implementation == "register" else _shared


def q2a8(E: int, M: int, N: int, K: int, *, implementation: str = "auto"):
    """Build (A, P, S, C), with pre-dispatched weights and rows.

    Auto selects register MMA for M<=8, shared-memory MMA otherwise. This is
    a starting policy from the 2560/640 expert shapes; callers can override it
    when profiling other shapes. Use the generated host ABI for Rust launches.
    """
    return _implementation(E, M, N, K, implementation)(E, M, N, K, E, False, False)


def q2a8_indexed(bank_experts: int, E: int, M: int, N: int, K: int, *,
                  split_gate_up: bool = False, implementation: str = "auto"):
    """Build (A, P, U, IDs, S, C) without gathering expert weights.

    A/S/C use the same dispatch-slot shapes as q2a8. IDs[E] maps each slot to
    the immutable bank. P/U[bank_experts,N/(2 if split else 1),K/64*18] retain
    source bytes. With split_gate_up=True, C's first/second halves come from
    P/U respectively; N must be divisible by 128. Otherwise U is ignored and
    may alias P. Invalid IDs produce zeros; buffers otherwise remain disjoint.
    IDs and activation rows may change between graph replays at stable addresses.
    """
    build = _implementation(E, M, N, K, implementation)
    if type(bank_experts) is not int or bank_experts <= 0:
        raise ValueError("Q2A8 bank size must be a positive integer")
    if type(split_gate_up) is not bool or (split_gate_up and N % 128):
        raise ValueError("Split gate/up requires a boolean flag and N divisible by 128")
    return build(E, M, N, K, bank_experts, True, split_gate_up)


@orin_jit
def q2a8_repack(E: int, N: int, K: int):
    """Load-time (RowMajor[E,N,K/64*18], GroupMajor[E,K/64,N,18]).

    This is a byte-preserving transpose. Release the source allocation after
    conversion, before loading the next bank; the output replaces its resident
    representation. Never call it in the per-token execution plan or alias the
    input/output. Temporary shared memory contains packed Q2 bytes, not W8.
    """
    _implementation(E, 16, N, K, 'shared')
    groups = K // 64
    @T.prim_func
    def main(Source: T.Tensor((E, N, groups * 18), T.uint8),
             Destination: T.Tensor((E, groups, N, 18), T.uint8)):
        with T.Kernel(T.ceildiv(N, 16), T.ceildiv(groups, 8), E, threads=128) as (nx, gx, expert):
            packed = T.alloc_shared((16, 8, 18), T.uint8)
            for n, g, byte in T.Parallel(16, 8, 18):
                if nx * 16 + n < N and gx * 8 + g < groups:
                    packed[n, g, byte] = Source[expert, nx * 16 + n, (gx * 8 + g) * 18 + byte]
            for g, n, byte in T.Parallel(8, 16, 18):
                if nx * 16 + n < N and gx * 8 + g < groups:
                    Destination[expert, gx * 8 + g, nx * 16 + n, byte] = packed[n, g, byte]
    return main


@orin_jit
def q2a8_grouped(assignments: int, bank_experts: int, tiles: int, N: int, K: int,
                split_gate_up: bool = False, implementation: str = "shared",
                layout: str = "row_major"):
    """Compact grouped projection over GPU-generated 16-row expert tiles.

    ABI (A, P, U, S, Counts, RowOffsets, TileExpert, TileRow, TileCount, C).
    A[assignments,K] INT8, S[assignments,K/64] FP16, C[assignments,N] FP16;
    P/U have the same original bank layout as q2a8_indexed. All routing buffers
    are INT32, generated by moe.expert_* kernels. Only live tile rows are read
    or written; empty experts never read weights. No expanded global W8.

    group_major uses P/U[B,K/64,N/(2 if split else 1),18], a lossless block
    transpose performed once at load time. It retains every original scale and
    code byte while coalescing adjacent output rows of each K group.
    """
    _implementation(assignments, 16, N, K, implementation)
    if any(type(x) is not int or x <= 0 for x in (bank_experts, tiles)):
        raise ValueError('Invalid Q2A8 grouped bank/tile dimensions')
    if type(split_gate_up) is not bool or (split_gate_up and N % 128):
        raise ValueError('Invalid grouped gate/up dimensions')
    if layout not in ('row_major', 'group_major'):
        raise ValueError('Invalid Q2A8 grouped weight layout')
    B, BM, BN = bank_experts, 16, 64
    weight_rows = N // 2 if split_gate_up else N
    weight_shape = ((B, K // 64, weight_rows, 18) if layout == 'group_major'
                    else (B, weight_rows, K // 64 * 18))

    @T.macro
    def copy_block(Weight, packed, expert, n, group):
        if layout == 'group_major':
            T.copy(Weight[expert, group, n, 0], packed)
        else:
            T.copy(Weight[expert, n, group * 18], packed)

    @T.macro
    def shared_projection(A, P, U, S, Counts, RowOffsets, TileExpert, TileRow, TileCount, C):
        with T.Kernel(tiles, N // BN, threads=128) as (tile, bx):
            metadata = T.alloc_local((4,), T.int32)
            a = T.alloc_shared((BM, 64), T.int8)
            b = T.alloc_shared((BN, 64), T.int8)
            packed = T.alloc_shared((BN, 18), T.uint8)
            scales = T.alloc_shared((BN,), T.float16)
            acc = T.alloc_fragment((BM, BN), T.int32)
            total = T.alloc_fragment((BM, BN), T.float32)
            if tile < TileCount[0]:
                metadata[0] = TileExpert[tile]
                metadata[1] = TileRow[tile]
                metadata[2] = RowOffsets[metadata[0]] + metadata[1]
                metadata[3] = T.min(BM, Counts[metadata[0]] - metadata[1])
                expert = metadata[0]
                offset = metadata[2]
                count = metadata[3]
                T.clear(total)
                for group in T.Pipelined(K // 64, num_stages=2):
                    T.copy(A[offset, group * 64], a)
                    if split_gate_up:
                        if bx * BN < N // 2:
                            copy_block(P, packed, expert, bx * BN, group)
                        else:
                            copy_block(U, packed, expert, bx * BN - N // 2, group)
                    else:
                        copy_block(P, packed, expert, bx * BN, group)
                    for n in T.Parallel(BN):
                        bits = T.cast(packed[n, 0], T.uint16) | (T.cast(packed[n, 1], T.uint16) << 8)
                        scales[n] = T.reinterpret(T.float16, bits)
                    for n, k in T.Parallel(BN, 64):
                        b[n, k] = T.cast((packed[n, 2 + k // 4] >> ((k % 4) * 2)) & 3, T.int8) - 1
                    T.clear(acc)
                    T.gemm(a, b, acc, transpose_B=True)
                    for m, n in T.Parallel(BM, BN):
                        if m < count:
                            total[m, n] += (T.cast(acc[m, n], T.float32) * T.cast(scales[n], T.float32)) * T.cast(S[offset + m, group], T.float32)
                for m, n in T.Parallel(BM, BN):
                    if m < count:
                        C[offset + m, bx * BN + n] = total[m, n]

    @T.macro
    def read_byte(P, U, expert, n, byte):
        if layout == 'group_major':
            if split_gate_up:
                return T.if_then_else(n < N // 2, P[expert, byte // 18, n % (N // 2), byte % 18],
                                      U[expert, byte // 18, n % (N // 2), byte % 18])
            else:
                return P[expert, byte // 18, n, byte % 18]
        else:
            if split_gate_up:
                return T.if_then_else(n < N // 2, P[expert, n % (N // 2), byte],
                                      U[expert, n % (N // 2), byte])
            else:
                return P[expert, n, byte]

    @T.macro
    def register_projection(A, P, U, S, Counts, RowOffsets, TileExpert, TileRow, TileCount, C):
        with T.Kernel(tiles, N // BN, threads=128) as (tile, bx):
            T.import_source(SOURCE)
            metadata = T.alloc_local((4,), T.int32)
            tx = T.get_thread_binding()
            warp, lane = tx // 32, tx % 32
            row, tid = lane // 4, lane % 4
            ar = T.alloc_local((4,), T.uint32)
            br = T.alloc_local((2,), T.uint32)
            acc = T.alloc_local((4,), T.int32)
            total = T.alloc_local((2, 4), T.float32)
            if tile < TileCount[0]:
                metadata[0] = TileExpert[tile]
                metadata[1] = TileRow[tile]
                metadata[2] = RowOffsets[metadata[0]] + metadata[1]
                metadata[3] = T.min(BM, Counts[metadata[0]] - metadata[1])
                expert = metadata[0]
                offset = metadata[2]
                count = metadata[3]
                for part in T.unroll(2):
                    for c in T.unroll(4):
                        total[part, c] = 0.0
                for group in T.serial(K // 64):
                    for part in T.unroll(2):
                        n = bx * 64 + warp * 16 + part * 8 + row
                        base = group * 18
                        bits = T.cast(read_byte(P, U, expert, n, base), T.uint16) | (T.cast(read_byte(P, U, expert, n, base + 1), T.uint16) << 8)
                        scale = T.cast(T.reinterpret(T.float16, bits), T.float32)
                        for c in T.unroll(4):
                            acc[c] = 0
                        for ki in T.unroll(2):
                            for b in T.unroll(2):
                                br[b] = T.call_pure_extern("uint32", "q2a8_expand_word", T.cast(read_byte(P, U, expert, n, base + 2 + ki * 8 + b * 4 + tid), T.uint32))
                            for a in T.unroll(4):
                                r = row + (a % 2) * 8
                                ar[a] = 0
                                if r < count:
                                    position = (offset + r) * K + group * 64 + ki * 32 + tid * 4 + (a // 2) * 16
                                    ar[a] = T.call_pure_extern("uint32", "q2a8_load_word", A.data, position)
                            T.ptx_mma("int32", "m16n8k32", "row", "col", "int8", "int8", "int32", ar.data, 0, br.data, 0, acc.data, 0, False)
                        for c in T.unroll(4):
                            weight_scale = T.call_extern("float32", "q2a8_shuffle_scale", scale, (tid * 2 + c % 2) * 4)
                            r = row + (c // 2) * 8
                            if r < count:
                                total[part, c] += (T.cast(acc[c], T.float32) * weight_scale) * T.cast(S[offset + r, group], T.float32)
                for part in T.unroll(2):
                    for c in T.unroll(4):
                        r = row + (c // 2) * 8
                        n = bx * 64 + warp * 16 + part * 8 + tid * 2 + c % 2
                        if r < count:
                            C[offset + r, n] = total[part, c]

    @T.prim_func
    def main(A: T.Tensor((assignments, K), T.int8),
             P: T.Tensor(weight_shape, T.uint8),
             U: T.Tensor(weight_shape, T.uint8),
             S: T.Tensor((assignments, K // 64), T.float16),
             Counts: T.Tensor((B,), T.int32),
             RowOffsets: T.Tensor((B,), T.int32),
             TileExpert: T.Tensor((tiles,), T.int32),
             TileRow: T.Tensor((tiles,), T.int32),
             TileCount: T.Tensor((1,), T.int32),
             C: T.Tensor((assignments, N), T.float16)):
        if implementation == "register":
            register_projection(A, P, U, S, Counts, RowOffsets, TileExpert, TileRow, TileCount, C)
        else:
            shared_projection(A, P, U, S, Counts, RowOffsets, TileExpert, TileRow, TileCount, C)
    return main


@orin_jit
def _shared(E, M, N, K, B, indexed, split):
    BM, BN = 16, 64

    @T.macro
    def project(A, P, U, IDs, S, C):
        with T.Kernel(T.ceildiv(M, BM), N // BN, E, threads=128) as (by, bx, ex):
            expert = IDs[ex] if indexed else ex
            a = T.alloc_shared((BM, 64), T.int8)
            b = T.alloc_shared((BN, 64), T.int8)
            packed = T.alloc_shared((BN, 18), T.uint8)
            scales = T.alloc_shared((BN,), T.float16)
            acc = T.alloc_fragment((BM, BN), T.int32)
            total = T.alloc_fragment((BM, BN), T.float32)
            if expert >= 0 and expert < B:
                T.clear(total)
                for group in T.Pipelined(K // 64, num_stages=2):
                    T.copy(A[ex, by * BM, group * 64], a)
                    if split:
                        if bx * BN < N // 2:
                            T.copy(P[expert, bx * BN, group * 18], packed)
                        else:
                            T.copy(U[expert, bx * BN - N // 2, group * 18], packed)
                    else:
                        T.copy(P[expert, bx * BN, group * 18], packed)
                    for n in T.Parallel(BN):
                        bits = T.cast(packed[n, 0], T.uint16) | (T.cast(packed[n, 1], T.uint16) << 8)
                        scales[n] = T.reinterpret(T.float16, bits)
                    for n, k in T.Parallel(BN, 64):
                        b[n, k] = T.cast((packed[n, 2 + k // 4] >> ((k % 4) * 2)) & 3, T.int8) - 1
                    T.clear(acc)
                    T.gemm(a, b, acc, transpose_B=True)
                    for m, n in T.Parallel(BM, BN):
                        if by * BM + m < M:
                            total[m, n] += (T.cast(acc[m, n], T.float32) * T.cast(scales[n], T.float32)) * T.cast(S[ex, by * BM + m, group], T.float32)
                T.copy(total, C[ex, by * BM, bx * BN])
            else:
                for m, n in T.Parallel(BM, BN):
                    if by * BM + m < M:
                        C[ex, by * BM + m, bx * BN + n] = 0.0

    if indexed:
        @T.prim_func
        def main(A: T.Tensor((E, M, K), T.int8),
                 P: T.Tensor((B, N // 2 if split else N, K // 64 * 18), T.uint8),
                 U: T.Tensor((B, N // 2 if split else N, K // 64 * 18), T.uint8),
                 IDs: T.Tensor((E,), T.int32),
                 S: T.Tensor((E, M, K // 64), T.float16),
                 C: T.Tensor((E, M, N), T.float16)):
            project(A, P, U, IDs, S, C)
    else:
        @T.prim_func
        def main(A: T.Tensor((E, M, K), T.int8),
                 P: T.Tensor((E, N, K // 64 * 18), T.uint8),
                 S: T.Tensor((E, M, K // 64), T.float16),
                 C: T.Tensor((E, M, N), T.float16)):
            project(A, P, P, S, S, C)
    return main


@orin_jit
def _register(E, M, N, K, B, indexed, split):
    @T.macro
    def read_byte(P, U, expert, n, byte):
        if split:
            return T.if_then_else(n < N // 2, P[expert, n % (N // 2), byte],
                                  U[expert, n % (N // 2), byte])
        else:
            return P[expert, n, byte]

    @T.macro
    def project(A, P, U, IDs, S, C):
        with T.Kernel(N // 64, T.ceildiv(M, 16), E, threads=128) as (bx, my, ex):
            T.import_source(SOURCE)
            expert = IDs[ex] if indexed else ex
            tx = T.get_thread_binding()
            warp, lane = tx // 32, tx % 32
            row, tid = lane // 4, lane % 4
            ar = T.alloc_local((4,), T.uint32)
            br = T.alloc_local((2,), T.uint32)
            acc = T.alloc_local((4,), T.int32)
            total = T.alloc_local((2, 4), T.float32)
            if expert >= 0 and expert < B:
                for part in T.unroll(2):
                    for c in T.unroll(4):
                        total[part, c] = 0.0
                for group in T.serial(K // 64):
                    for part in T.unroll(2):
                        n = bx * 64 + warp * 16 + part * 8 + row
                        base = group * 18
                        bits = T.cast(read_byte(P, U, expert, n, base), T.uint16) | (T.cast(read_byte(P, U, expert, n, base + 1), T.uint16) << 8)
                        scale = T.cast(T.reinterpret(T.float16, bits), T.float32)
                        for c in T.unroll(4):
                            acc[c] = 0
                        for ki in T.unroll(2):
                            for b in T.unroll(2):
                                br[b] = T.call_pure_extern("uint32", "q2a8_expand_word", T.cast(read_byte(P, U, expert, n, base + 2 + ki * 8 + b * 4 + tid), T.uint32))
                            for a in T.unroll(4):
                                r = my * 16 + row + (a % 2) * 8
                                ar[a] = 0
                                if r < M:
                                    offset = (ex * M + r) * K + group * 64 + ki * 32 + tid * 4 + (a // 2) * 16
                                    ar[a] = T.call_pure_extern("uint32", "q2a8_load_word", A.data, offset)
                            T.ptx_mma("int32", "m16n8k32", "row", "col", "int8", "int8", "int32", ar.data, 0, br.data, 0, acc.data, 0, False)
                        for c in T.unroll(4):
                            # This warp collective is impure to prevent the compiler
                            # moving it under the row-tail predicate. Every lane runs.
                            weight_scale = T.call_extern("float32", "q2a8_shuffle_scale", scale, (tid * 2 + c % 2) * 4)
                            r = my * 16 + row + (c // 2) * 8
                            if r < M:
                                total[part, c] += (T.cast(acc[c], T.float32) * weight_scale) * T.cast(S[ex, r, group], T.float32)
                for part in T.unroll(2):
                    for c in T.unroll(4):
                        r = my * 16 + row + (c // 2) * 8
                        n = bx * 64 + warp * 16 + part * 8 + tid * 2 + c % 2
                        if r < M:
                            C[ex, r, n] = total[part, c]
            else:
                for part in T.unroll(2):
                    for c in T.unroll(4):
                        r = my * 16 + row + (c // 2) * 8
                        n = bx * 64 + warp * 16 + part * 8 + tid * 2 + c % 2
                        if r < M:
                            C[ex, r, n] = 0.0

    if indexed:
        @T.prim_func
        def main(A: T.Tensor((E, M, K), T.int8),
                 P: T.Tensor((B, N // 2 if split else N, K // 64 * 18), T.uint8),
                 U: T.Tensor((B, N // 2 if split else N, K // 64 * 18), T.uint8),
                 IDs: T.Tensor((E,), T.int32),
                 S: T.Tensor((E, M, K // 64), T.float16),
                 C: T.Tensor((E, M, N), T.float16)):
            project(A, P, U, IDs, S, C)
    else:
        @T.prim_func
        def main(A: T.Tensor((E, M, K), T.int8),
                 P: T.Tensor((E, N, K // 64 * 18), T.uint8),
                 S: T.Tensor((E, M, K // 64), T.float16),
                 C: T.Tensor((E, M, N), T.float16)):
            project(A, P, P, S, S, C)
    return main
