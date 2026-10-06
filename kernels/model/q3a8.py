"""Native Q3_K x A8 projection and Q3_K embedding lookup on SM87.

The 110-byte/256-weight representation remains resident. Q3_K sub-block
codes q in [-4,3] and integer scales s in [-32,31] can be combined exactly
as -q*s in [-128,124], paired with the negated super-block FP16 scale. This
uses signed INT8 Tensor Cores without requantizing weights or keeping W8.
Activation scales use groups of 64; accumulators and rescaling are INT32/F32.
"""
import tilelang.language as T

from tools.operators.common import orin_jit


@T.macro
def integer_weight(packed, n, position):
    group = position // 16
    low = (T.cast(packed[n, 96 + group % 8], T.int32) >> (4 * (group // 8))) & 15
    high = (T.cast(packed[n, 104 + group % 4], T.int32) >> (2 * (group // 4))) & 3
    scale = (low | (high << 4)) - 32
    pair = (position % 128) // 32
    code = (T.cast(packed[n, 32 + (position // 128) * 32 + position % 32], T.int32) >> (2 * pair)) & 3
    high_bit = (T.cast(packed[n, position % 32], T.int32) >> ((position // 128) * 4 + pair)) & 1
    signed = code - T.if_then_else(high_bit == 0, 4, 0)
    return -(signed * scale)


@orin_jit
def q3a8(M: int, N: int, K: int, block_m: int = 16):
    """Build (A[M,K], P[N,K/256*110], S[M,K/64], C[M,N]).

    INT8/U8/FP16/FP16, disjoint buffers. M tails are supported; N must be a
    multiple of 64, K a multiple of 256. Scale signs/zeroes are preserved.
    """
    if any(type(x) is not int or x <= 0 for x in (M, N, K)) or N % 64 or K % 256 or block_m not in (16,32,64):
        raise ValueError('Invalid Q3A8 dimensions')
    @T.prim_func
    def main(A: T.Tensor((M, K), T.int8), P: T.Tensor((N, K//256*110), T.uint8),
             S: T.Tensor((M, K//64), T.float16), C: T.Tensor((M, N), T.float16)):
        with T.Kernel(T.ceildiv(M,block_m), N//64, threads=128) as (by,bx):
            a = T.alloc_shared((block_m,64), T.int8)
            b = T.alloc_shared((64,64), T.int8)
            packed = T.alloc_shared((64,110), T.uint8)
            scales = T.alloc_shared((64,), T.float16)
            acc = T.alloc_fragment((block_m,64), T.int32)
            total = T.alloc_fragment((block_m,64), T.float32)
            T.clear(total)
            for superblock in T.Pipelined(K//256, num_stages=2):
                T.copy(P[bx*64,superblock*110],packed)
                for n in T.Parallel(64):
                    bits = T.cast(packed[n,108],T.uint16) | (T.cast(packed[n,109],T.uint16) << 8)
                    scales[n] = -T.reinterpret(T.float16,bits)
                for part in T.unroll(4):
                    T.copy(A[by*block_m,superblock*256+part*64],a)
                    for n,k in T.Parallel(64,64):
                        b[n,k] = integer_weight(packed,n,part*64+k)
                    T.clear(acc)
                    T.gemm(a,b,acc,transpose_B=True)
                    for m,n in T.Parallel(block_m,64):
                        if by*block_m+m < M:
                            total[m,n] += (T.cast(acc[m,n],T.float32)*T.cast(scales[n],T.float32))*T.cast(S[by*block_m+m,superblock*4+part],T.float32)
            T.copy(total,C[by*block_m,bx*64])
    return main


@orin_jit
def q3_embedding(M: int, H: int, vocab: int):
    """Build (IDs[M], P[vocab,H/256*110], Output[M,H]), I32/U8/FP16.

    Invalid token IDs return zero rows. No full dequantized embedding table is
    materialized; only the requested rows are read and converted.
    """
    if any(type(x) is not int or x <= 0 for x in (M,H,vocab)) or H % 256:
        raise ValueError('Invalid Q3 embedding dimensions')
    @T.prim_func
    def main(IDs: T.Tensor((M,),T.int32), P: T.Tensor((vocab,H//256*110),T.uint8),
             Output: T.Tensor((M,H),T.float16)):
        with T.Kernel(M,H//256,threads=128) as (row,group):
            packed = T.alloc_shared((1,110),T.uint8)
            token = T.alloc_local((1,),T.int32)
            scale = T.alloc_local((1,),T.float32)
            token[0] = IDs[row]
            if token[0] >= 0 and token[0] < vocab:
                T.copy(P[token[0],group*110],packed)
                bits = T.cast(packed[0,108],T.uint16) | (T.cast(packed[0,109],T.uint16) << 8)
                scale[0] = -T.cast(T.reinterpret(T.float16,bits),T.float32)
                for k in T.Parallel(256):
                    Output[row,group*256+k] = T.cast(integer_weight(packed,0,k),T.float32)*scale[0]
            else:
                for k in T.Parallel(256):
                    Output[row,group*256+k] = 0.0
    return main
