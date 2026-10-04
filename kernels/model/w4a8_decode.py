"""Small-M SM87 INT8 MMA from one resident I8-fragment W4 representation.

row mode generates the exact existing row-W8 codes in warp-local registers;
group mode accumulates each 128-element group with its original FP16 scale.
Neither materializes W8 globally nor adds persistent weight metadata. Split-K
partials are FP32; the caller performs the final reduction before FP16 output.
"""
import tilelang.language as T
from tools.operators.common import orin_jit


SOURCE = r'''
#include <cuda_fp16.h>
#include <tl_templates/cuda/instruction/mma.h>
__device__ __forceinline__ unsigned orin_decode_lut_word(
    int first, half_t scale, signed char zero, half_t row_scale) {
    unsigned word = 0;
    __half s = *reinterpret_cast<__half*>(&scale);
    float rs = __half2float(*reinterpret_cast<__half*>(&row_scale));
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        __half w = __hmul(__int2half_rn(first+c-int(zero)), s);
        int code = __float2int_rn(__fdiv_rn(__half2float(w), rs));
        code = max(-127, min(127, code));
        word |= (unsigned(code) & 255u) << (c*8);
    }
    return word;
}
__device__ __forceinline__ unsigned orin_decode_lut_quartet(unsigned x, unsigned word) {
    unsigned result = 0;
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        unsigned q = (x >> (c*4)) & 15u;
        unsigned entry = __shfl_sync(0xffffffffu, word, q/4, 4);
        result |= ((entry >> ((q%4)*8)) & 255u) << (c*8);
    }
    return result;
}
__device__ __forceinline__ unsigned orin_decode_group_quartet(unsigned x, signed char zero) {
    unsigned result = 0;
    #pragma unroll
    for (int c = 0; c < 4; ++c)
        result |= (unsigned(int((x >> (c*4)) & 15u)-int(zero)) & 255u) << (c*8);
    return result;
}
__device__ __forceinline__ float orin_decode_output_scale(half_t scale, int column) {
    float value = __half2float(*reinterpret_cast<__half*>(&scale));
    return __shfl_sync(0xffffffffu, value, column*4);
}
'''


@orin_jit
def w4a8_decode(M, N: int, K: int, SPLIT=1, *, mode='group', TILE_N=64,
                TILE_M=16, output_dtype='float16', shared_a=None, activation_group=None):
    assert M is None or 1 <= M <= 2048
    shared_a = M is None if shared_a is None else shared_a
    M = T.dynamic('M') if M is None else M
    assert mode in ('row', 'group')
    assert activation_group is None or (mode=='group' and activation_group==128)
    assert TILE_N in (64, 128) and TILE_M in (16, 32, 64)
    assert N % TILE_N == 0 and K % (128*SPLIT) == 0
    assert output_dtype in ('float16', 'float32')
    assert SPLIT == 1 or output_dtype == 'float32'
    blocks, row_tiles = TILE_N//64, TILE_M//16

    @T.prim_func
    def main(A: T.Tensor((M, K//4), T.uint32),
             PP: T.Tensor((N//64, K//128, 128, 8), T.uint32),
             S: T.Tensor((N, K//128), T.float16),
             Z: T.Tensor((N, K//128), T.int8),
             WS: T.Tensor((N,), T.float16),
             AS: T.Tensor((M, K//128 if activation_group else 1), T.float16),
             O: T.Tensor((SPLIT, M, N), output_dtype)):
        with T.Kernel(N//TILE_N, SPLIT, T.ceildiv(M,TILE_M), threads=128) as (bx,sk,my):
            T.import_source(SOURCE)
            tx = T.get_thread_binding()
            warp, lane = tx//32, tx%32
            row, tid = lane//4, lane%4
            packed = T.alloc_local((blocks,8), T.uint32)
            scales = T.alloc_local((blocks*2,), T.float16)
            zeros = T.alloc_local((blocks*2,), T.int8)
            tables = T.alloc_local((blocks*2,), T.uint32)
            activation_scales = T.alloc_local((row_tiles*2,), T.float32)
            ar = T.alloc_local((4,), T.uint32)
            br = T.alloc_local((2,), T.uint32)
            acc = T.alloc_local((row_tiles*blocks*8,), T.int32)
            total = T.alloc_local((row_tiles*blocks*8,), T.float32)
            shared = T.alloc_shared((TILE_M,32), T.uint32)
            for j in T.unroll(row_tiles*blocks*8):
                acc[j] = 0
                total[j] = 0.0
            for kg in T.serial(K//128//SPLIT):
                gk = sk*(K//128//SPLIT)+kg
                if activation_group:
                    for ri in T.unroll(row_tiles*2):
                        activation_scales[ri] = 0.0
                        r = my*TILE_M+(ri//2)*16+row+(ri%2)*8
                        if r < M:
                            activation_scales[ri] = T.cast(AS[r,gk],T.float32)
                if shared_a:
                    for r,c in T.Parallel(TILE_M,32):
                        shared[r,c] = 0
                        if my*TILE_M+r < M:
                            shared[r,c] = A[my*TILE_M+r,gk*32+(c ^ ((r%8)*4))]
                    T.sync_threads()
                for block in T.unroll(blocks):
                    for vector in T.unroll(2):
                        for j in T.vectorized(4):
                            packed[block,vector*4+j] = PP[bx*blocks+block,gk,tx,vector*4+j]
                for part in T.unroll(blocks*2):
                    col = bx*TILE_N+warp*16+(part//2)*64+(part%2)*8+row
                    scales[part] = S[col,gk]
                    zeros[part] = Z[col,gk]
                    if mode == 'row':
                        tables[part] = T.call_pure_extern('uint32','orin_decode_lut_word',
                            tid*4,scales[part],zeros[part],WS[col])
                for ki in T.unroll(4):
                    for part in T.unroll(blocks*2):
                        for half in T.unroll(2):
                            quartet = (packed[part//2,ki*2+part%2]>>(half*16))&65535
                            if mode == 'row':
                                br[half] = T.call_pure_extern('uint32','orin_decode_lut_quartet',quartet,tables[part])
                            else:
                                br[half] = T.call_pure_extern('uint32','orin_decode_group_quartet',quartet,zeros[part])
                        for mi in T.unroll(row_tiles):
                            for ai in T.unroll(4):
                                ar[ai] = 0
                                r = my*TILE_M+mi*16+row+(ai%2)*8
                                if shared_a:
                                    ar[ai] = shared[mi*16+row+(ai%2)*8,(ki*8+tid+(ai//2)*4)^((row%8)*4)]
                                elif r < M:
                                    ar[ai] = A[r,gk*32+ki*8+tid+(ai//2)*4]
                            T.ptx_mma('int32','m16n8k32','row','col','int8','int8','int32',
                                ar.data,0,br.data,0,acc.data,mi*blocks*8+part*4,False)
                if mode == 'group':
                    for mi in T.unroll(row_tiles):
                        for part in T.unroll(blocks*2):
                            for ci in T.unroll(4):
                                index = mi*blocks*8+part*4+ci
                                scale = T.call_pure_extern('float32','orin_decode_output_scale',scales[part],tid*2+ci%2)
                                if activation_group:
                                    total[index] += (T.cast(acc[index],T.float32)*scale)*activation_scales[mi*2+ci//2]
                                else:
                                    total[index] += T.cast(acc[index],T.float32)*scale
                                acc[index] = 0
                if shared_a:
                    T.sync_threads()
            for mi in T.unroll(row_tiles):
                for part in T.unroll(blocks*2):
                    for ci in T.unroll(4):
                        r = my*TILE_M+mi*16+row+(ci//2)*8
                        col = bx*TILE_N+warp*16+(part//2)*64+(part%2)*8+tid*2+ci%2
                        if r < M:
                            index = mi*blocks*8+part*4+ci
                            if mode == 'row':
                                O[sk,r,col] = T.cast(acc[index],T.float32)*T.cast(AS[r,0],T.float32)*T.cast(WS[col],T.float32)
                            elif activation_group:
                                O[sk,r,col] = total[index]
                            else:
                                O[sk,r,col] = total[index]*T.cast(AS[r,0],T.float32)
    return main
