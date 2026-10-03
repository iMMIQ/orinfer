"""Strict row-W8 expansion from the lossless I8-fragment packed W4 layout.

Only nibble addressing differs from op29's F16-fragment reader. Inputs share
one persistent W4 copy with the short-M fused and M1 decode readers; output
is caller-owned transient workspace. Source dequant and RNE are unchanged.
"""
import tilelang.language as T
from kernels.operators.op03_ffn_gate_up import _orin_jit
from kernels.operators.op29_w4_to_temporary_w8 import weight_code


@_orin_jit
def w4_i8_to_temporary_w8(N: int, K: int, BK=256):
    assert N > 0 and N % 64 == 0
    assert BK in (128, 256, 512) and K > 0 and K % BK == 0

    @T.prim_func
    def i8_expand(PP: T.Tensor((N//64, K//128, 128, 8), T.uint32),
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
                lane = (i//16)*32 + (i%8)*4 + ((j%16)//4)
                word = ((j%128)//32)*2 + (i%16)//8
                shift = ((j%32)//16)*16 + (j%4)*4
                nibble = (packed[j//128, lane, word] >> shift) & 15
                word8 = table[j//128, i, nibble//4]
                W8[bx*64+i, by*BK+j] = T.cast((word8 >> ((nibble%4)*8)) & 255, T.int8)
    return i8_expand
