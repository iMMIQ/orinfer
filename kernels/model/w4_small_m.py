"""Small-M W4A16 projections sharing one packed-weight load across token rows.

The packed weights are identical to w4_decode_register_mma. M is a sequence
length, not a request batch; this kernel only computes independent linear
projections. Causal attention and GDN state updates must be supplied by the
execution plan. Unused MMA rows are zero, and split-K outputs remain FP32.
"""
import tilelang.language as T
from kernels.operators.op03_ffn_gate_up import _orin_jit, _PAIR_SOURCE


@_orin_jit
def w4_small_m(M: int, N: int, K: int, SPLIT=1,
               output_dtype='float16', TILE_N=64, output_layout='flat', weight_layout='f16'):
    assert 1 <= M <= 2048
    assert TILE_N in (64, 128, 256)
    assert N % TILE_N == 0 and K % (128 * SPLIT) == 0
    assert output_dtype in ('float16', 'float32')
    assert output_layout in ('flat', 'qkvz')
    assert weight_layout in ('f16', 'i8')
    if output_layout == 'qkvz':
        assert N == 16384 and SPLIT == 1 and output_dtype == 'float16'
    blocks = TILE_N // 64

    @T.macro
    def multiply(A, PP, S, Z, bx, sk, my, acc):
        T.import_source(_PAIR_SOURCE)
        tx = T.get_thread_binding()
        lane, warp = tx % 32, tx // 32
        row, tid = lane // 4, lane % 4
        packed = T.alloc_local((blocks, 8), T.uint32)
        native = T.alloc_local((2, 2), T.uint32)
        scales = T.alloc_local((blocks * 2,), T.float16)
        zeros = T.alloc_local((blocks * 2,), T.int8)
        ar = T.alloc_local((4,), T.uint32)
        br = T.alloc_local((2,), T.uint32)
        for i in T.unroll(blocks * 8):
            acc[i] = 0.0
        for i in T.unroll(4):
            ar[i] = 0
        for kg in T.serial(K // 128 // SPLIT):
            gk = sk * (K // 128 // SPLIT) + kg
            for block in T.unroll(blocks):
                if weight_layout == 'f16':
                    for vector in T.unroll(2):
                        for j in T.vectorized(4):
                            packed[block, vector * 4 + j] = PP[bx * blocks + block, gk, tx, vector * 4 + j]
                else:
                    # Reconstruct F16 fragments directly from the resident I8
                    # layout. No second packed copy or full dequantized B.
                    for ki32 in T.unroll(4):
                        for ni in T.vectorized(2):
                            native[0, ni] = PP[bx * blocks + block, gk, (tx // 4) * 4 + tid // 2, ki32 * 2 + ni]
                            native[1, ni] = PP[bx * blocks + block, gk, (tx // 4) * 4 + tid // 2 + 2, ki32 * 2 + ni]
                        for half in T.unroll(2):
                            shift = half * 16 + (tid % 2) * 8
                            packed[block, ki32 * 2 + half] = 0
                            for ni in T.unroll(2):
                                pair0 = (native[0, ni] >> shift) & 255
                                pair1 = (native[1, ni] >> shift) & 255
                                packed[block, ki32 * 2 + half] = packed[block, ki32 * 2 + half] | ((pair0 | (pair1 << 8)) << (ni * 16))
            for part in T.unroll(blocks * 2):
                col = bx * TILE_N + warp * 16 + (part // 2) * 64 + (part % 2) * 8 + row
                scales[part] = S[col, gk]
                zeros[part] = Z[col, gk]
            for ki in T.unroll(8):
                for half in T.unroll(2):
                    if my * 16 + row + half * 8 < M:
                        col = gk * 128 + ki * 16 + tid * 2
                        ar[half] = (T.cast(T.reinterpret(T.uint16, A[my * 16 + row + half * 8, col]), T.uint32)
                                    | (T.cast(T.reinterpret(T.uint16, A[my * 16 + row + half * 8, col + 1]), T.uint32) << 16))
                        ar[half + 2] = (T.cast(T.reinterpret(T.uint16, A[my * 16 + row + half * 8, col + 8]), T.uint32)
                                        | (T.cast(T.reinterpret(T.uint16, A[my * 16 + row + half * 8, col + 9]), T.uint32) << 16))
                for part in T.unroll(blocks * 2):
                    br[0] = T.call_pure_extern('uint32', 'op03_deq_pair',
                        T.cast((packed[part // 2, ki] >> ((part % 2) * 16)) & 255, T.uint8), scales[part], zeros[part])
                    br[1] = T.call_pure_extern('uint32', 'op03_deq_pair',
                        T.cast((packed[part // 2, ki] >> ((part % 2) * 16 + 8)) & 255, T.uint8), scales[part], zeros[part])
                    T.ptx_mma('float32', 'm16n8k16', 'row', 'col', 'fp16', 'fp16', 'fp32',
                              ar.data, 0, br.data, 0, acc.data, part * 4, False)

    if output_layout == 'qkvz':
        @T.prim_func
        def main(A: T.Tensor((M, K), T.float16),
                 PP: T.Tensor((N // 64, K // 128, 128, 8), T.uint32),
                 S: T.Tensor((N, K // 128), T.float16),
                 Z: T.Tensor((N, K // 128), T.int8),
                 QKV: T.Tensor((M, 10240), T.float16),
                 ZOUT: T.Tensor((M, 6144), T.float16)):
            with T.Kernel(N // TILE_N, T.ceildiv(M,16), threads=128) as (bx, my):
                acc = T.alloc_local((blocks * 8,), T.float32)
                multiply(A, PP, S, Z, bx, 0, my, acc)
                tx = T.get_thread_binding()
                row = (tx % 32) // 4
                for half in T.unroll(2):
                    if my * 16 + row + half * 8 < M:
                        for part in T.unroll(blocks * 2):
                            for j in T.unroll(2):
                                col = bx * TILE_N + (tx // 32) * 16 + (part // 2) * 64 + (part % 2) * 8 + (tx % 4) * 2 + j
                                if col < 10240:
                                    QKV[my * 16 + row + half * 8, col] = acc[part * 4 + half * 2 + j]
                                else:
                                    ZOUT[my * 16 + row + half * 8, col - 10240] = acc[part * 4 + half * 2 + j]
    else:
        @T.prim_func
        def main(A: T.Tensor((M, K), T.float16),
                 PP: T.Tensor((N // 64, K // 128, 128, 8), T.uint32),
                 S: T.Tensor((N, K // 128), T.float16),
                 Z: T.Tensor((N, K // 128), T.int8),
                 O: T.Tensor((SPLIT, M, N), output_dtype)):
            with T.Kernel(N // TILE_N, SPLIT, T.ceildiv(M,16), threads=128) as (bx, sk, my):
                acc = T.alloc_local((blocks * 8,), T.float32)
                multiply(A, PP, S, Z, bx, sk, my, acc)
                tx = T.get_thread_binding()
                row = (tx % 32) // 4
                for half in T.unroll(2):
                    if my * 16 + row + half * 8 < M:
                        for part in T.unroll(blocks * 2):
                            for j in T.unroll(2):
                                col = bx * TILE_N + (tx // 32) * 16 + (part // 2) * 64 + (part % 2) * 8 + (tx % 4) * 2 + j
                                O[sk, my * 16 + row + half * 8, col] = acc[part * 4 + half * 2 + j]
    return main
