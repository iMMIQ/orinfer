"""Dynamic N/K adjacent-U4 -> temporary W8; finite weights, SM87 only.

P/S/Z plus offline rowWS are resident. W8 is an explicit per-call workspace.
The inline table is CTA-local shared memory, never a persistent global LUT.
"""
import tilelang
import tilelang.language as T


@T.macro
def weight_code(code, zero, scale, row_scale):
    # This explicit FP16 multiplication is the source W4 dequant boundary.
    weight = T.cast(T.cast(T.cast(code, T.int32) - T.cast(zero, T.int32), T.float16) * scale, T.float16)
    ratio = T.call_extern("float32", "__fdiv_rn", T.cast(weight, T.float32), T.cast(row_scale, T.float32))
    return T.cast(T.max(-127.0, T.min(127.0, T.round(ratio))), T.int8)


@tilelang.jit(out_idx=[], execution_backend="nvrtc",
              target={"kind": "cuda", "arch": "sm_87"})
def _compile(BN: int, BK: int, inline: bool):
    n, k = T.dynamic("N"), T.dynamic("K")

    @T.prim_func
    def main(W8: T.Tensor((n, k), T.int8),
             P: T.Tensor((n, k // 2), T.uint8),
             S: T.Tensor((n, k // 128), T.float16),
             Z: T.Tensor((n, k // 128), T.int8),
             WS: T.Tensor((n,), T.float16)):
        with T.Kernel(T.ceildiv(n, BN), T.ceildiv(k, BK), threads=256) as (bx, by):
            if inline:
                packed = T.alloc_shared((BN, BK // 2), T.uint8)
                scales = T.alloc_shared((BN, BK // 128), T.float16)
                zeros = T.alloc_shared((BN, BK // 128), T.int8)
                rows = T.alloc_shared((BN,), T.float16)
                table = T.alloc_shared((BK // 128, BN, 4), T.int32)
                for i, j in T.Parallel(BN, BK // 2):
                    packed[i, j] = 0
                    if bx * BN + i < n and by * (BK // 2) + j < k // 2:
                        packed[i, j] = P[bx * BN + i, by * (BK // 2) + j]
                for i, g in T.Parallel(BN, BK // 128):
                    scales[i, g] = 0.0
                    zeros[i, g] = 0
                    if bx * BN + i < n and by * (BK // 128) + g < k // 128:
                        scales[i, g] = S[bx * BN + i, by * (BK // 128) + g]
                        zeros[i, g] = Z[bx * BN + i, by * (BK // 128) + g]
                for i in T.Parallel(BN):
                    rows[i] = 1.0
                    if bx * BN + i < n:
                        rows[i] = WS[bx * BN + i]
                for g, i, word_idx in T.Parallel(BK // 128, BN, 4):
                    word = T.alloc_var(T.int32)
                    word = 0
                    for c in T.unroll(4):
                        code8 = T.cast(weight_code(word_idx * 4 + c, zeros[i, g], scales[i, g], rows[i]), T.int32)
                        word = word | ((code8 & 255) << (c * 8))
                    table[g, i, word_idx] = word
                for i, j in T.Parallel(BN, BK):
                    if bx * BN + i < n and by * BK + j < k:
                        code4 = (T.cast(packed[i, j // 2], T.int32) >> ((j % 2) * 4)) & 15
                        word8 = table[j // 128, i, code4 // 4]
                        W8[bx * BN + i, by * BK + j] = T.cast((word8 >> ((code4 % 4) * 8)) & 255, T.int8)
            else:
                for i, j in T.Parallel(BN, BK):
                    row, col = bx * BN + i, by * BK + j
                    if row < n and col < k:
                        code4 = (T.cast(P[row, col // 2], T.int32) >> ((col % 2) * 4)) & 15
                        W8[row, col] = weight_code(code4, Z[row, col // 128], S[row, col // 128], WS[row])
    return main


def w4_to_temporary_w8(*, route="inline", BN=64, BK=256):
    """Build (P,S,Z,WS,W8) with dynamic N/K, K a positive multiple of128.

    Source W_half=half((q-Z)*S), WS=half(max(abs(W_half))/127) with
    floor2^-24, all-zero row WS=1; code=clamp(rint_even(W_half/WS),-127,127).
    Caller computes offline WS from every actual packed row, not all16 possible
    U4 values. Updating weights requires compatible row metadata. No allocator,
    persistent W8, full global LUT or scale preparation occurs in this kernel.
    """
    assert route in ("inline", "direct")
    assert BN in (16, 32, 64) and BK in (128, 256, 512)
    kernel = _compile(BN, BK, route == "inline")
    kernel.adapter.kernels = dict(kernel.adapter.kernels)
    return kernel


@tilelang.jit(out_idx=[], execution_backend="nvrtc",
              target={"kind": "cuda", "arch": "sm_87"})
def _compile_aligned(N: int, K: int, BN: int, BK: int):
    @T.prim_func
    def main(P: T.Tensor((N, K // 2), T.uint8),
             S: T.Tensor((N, K // 128), T.float16),
             Z: T.Tensor((N, K // 128), T.int8),
             WS: T.Tensor((N,), T.float16),
             W8: T.Tensor((N, K), T.int8)):
        with T.Kernel(N // BN, K // BK, threads=256) as (bx, by):
            packed = T.alloc_shared((BN, BK // 2), T.uint8)
            scales = T.alloc_shared((BN, BK // 128), T.float16)
            zeros = T.alloc_shared((BN, BK // 128), T.int8)
            rows = T.alloc_shared((BN,), T.float16)
            table = T.alloc_shared((BK // 128, BN, 4), T.int32)
            # Compile-time aligned extents avoid guarded initialize/load loops
            # and the per-iteration barriers present in the general tail path.
            T.copy(P[bx * BN, by * (BK // 2)], packed)
            T.copy(S[bx * BN, by * (BK // 128)], scales)
            T.copy(Z[bx * BN, by * (BK // 128)], zeros)
            T.copy(WS[bx * BN], rows)
            for g, i, word_idx in T.Parallel(BK // 128, BN, 4):
                word = T.alloc_var(T.int32)
                word = 0
                for c in T.unroll(4):
                    code8 = T.cast(weight_code(word_idx * 4 + c, zeros[i, g], scales[i, g], rows[i]), T.int32)
                    word = word | ((code8 & 255) << (c * 8))
                table[g, i, word_idx] = word
            for i, j in T.Parallel(BN, BK):
                code4 = (T.cast(packed[i, j // 2], T.int32) >> ((j % 2) * 4)) & 15
                word8 = table[j // 128, i, code4 // 4]
                W8[bx * BN + i, by * BK + j] = T.cast((word8 >> ((code4 % 4) * 8)) & 255, T.int8)
    return main


def w4_to_temporary_w8_aligned(N, K, *, BN=64, BK=256):
    """Shape-specialized version of the same FP16/strict-RNE code contract.

    Requires N % BN == K % BK == 0; use the general kernel for tails.
    PrimFunc order is P/S/Z/WS/W8. Actual exported ABI remains authoritative.
    """
    assert N > 0 and K > 0 and BN in (16, 32, 64) and BK in (128, 256, 512)
    assert N % BN == 0 and K % BK == 0
    kernel = _compile_aligned(N, K, BN, BK)
    kernel.adapter.kernels = dict(kernel.adapter.kernels)
    return kernel


@tilelang.jit(out_idx=[], execution_backend='nvrtc',
              target={'kind': 'cuda', 'arch': 'sm_87'})
def _compile_warp(N: int, K: int, BK: int):
    @T.prim_func
    def main(PP: T.Tensor((N//64, K//128, 128, 8), T.uint32),
             S: T.Tensor((N, K//128), T.float16),
             Z: T.Tensor((N, K//128), T.int8),
             WS: T.Tensor((N,), T.float16),
             W8: T.Tensor((N, K), T.int8)):
        with T.Kernel(N//64, K//BK, threads=256) as (bx, by):
            packed = T.alloc_shared((BK//128, 128, 8), T.uint32)
            scales = T.alloc_shared((64, BK//128), T.float16)
            zeros = T.alloc_shared((64, BK//128), T.int8)
            rows = T.alloc_shared((64,), T.float16)
            table = T.alloc_shared((BK//128, 64, 4), T.int32)
            T.copy(PP[bx, by*(BK//128), 0, 0], packed)
            T.copy(S[bx*64, by*(BK//128)], scales)
            T.copy(Z[bx*64, by*(BK//128)], zeros)
            T.copy(WS[bx*64], rows)
            for g, i, word_idx in T.Parallel(BK//128, 64, 4):
                lut_word = T.alloc_var(T.int32)
                lut_word = 0
                for c in T.unroll(4):
                    code = T.cast(weight_code(word_idx*4+c, zeros[i, g], scales[i, g], rows[i]), T.int32)
                    lut_word = lut_word | ((code & 255) << (c*8))
                table[g, i, word_idx] = lut_word
            for i, j in T.Parallel(64, BK):
                lane = (i//16)*32 + (i%8)*4 + ((j%8)//2)
                shift = ((i%16//8)*4 + ((j%16)//8)*2 + (j%2))*4
                nibble = (packed[j//128, lane, (j%128)//16] >> shift) & 15
                word8 = table[j//128, i, nibble//4]
                W8[bx*64+i, by*BK+j] = T.cast((word8 >> ((nibble%4)*8)) & 255, T.int8)
    return main


def w4_warp_to_temporary_w8(N, K, *, BK=256):
    """Same strict op29 math, reading the single warp-MMA packed W4 copy."""
    assert N > 0 and N % 64 == 0 and K > 0 and K % BK == 0
    assert BK in (128, 256, 512)
    kernel = _compile_warp(N, K, BK)
    kernel.adapter.kernels = dict(kernel.adapter.kernels)
    return kernel


def launch(kernel, P, S, Z, WS, W8, *, stream):
    """Caller supplies explicit stream and disjoint contiguous stable buffers."""
    n, k = W8.shape
    assert n > 0 and k > 0 and k % 128 == 0
    assert P.shape == (n, k // 2) and S.shape == Z.shape == (n, k // 128)
    assert WS.shape == (n,)
    return kernel.adapter.func(W8, P, S, Z, WS, stream=stream)
