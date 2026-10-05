"""SM87 projection candidates adapted from local prefill_layout/kernels.py.
Logical weights P[N,K/2] are adjacent low/high U4 pairs; scales and unsigned
zeros are N x K/128. Dynamic M masks load/store tails; temporary W8 is per call.
"""
from functools import wraps
import torch
import tilelang
import tilelang.language as T

def orin_jit(function):
    compiled = tilelang.jit(out_idx=[], execution_backend="nvrtc", target={"kind":"cuda","arch":"sm_87"})(function)
    @wraps(function)
    def build(*args, **kwargs):
        kernel = compiled(*args, **kwargs)
        kernel.adapter.kernels = dict(kernel.adapter.kernels)
        def launch(*inputs, stream=None):
            if stream is None:
                stream = torch.cuda.current_stream(inputs[0].device).cuda_stream
            return kernel.adapter.func(*inputs, stream=stream)
        kernel.torch_function = launch
        return kernel
    return build

PAIR_SOURCE = r'''
#include <cuda_fp16.h>
#include <tl_templates/cuda/instruction/mma.h>
__device__ __forceinline__ unsigned int deq_u4_pair(unsigned char x, half_t scale,
                                                  signed char zero) {
    unsigned int bits = 0x64006400u | (x & 15u) | ((x & 240u) << 12);
    __half2 values = *reinterpret_cast<__half2*>(&bits);
    __half2 offset = __half2half2(__int2half_rn(1024 + int(zero)));
    __half native_scale = *reinterpret_cast<__half*>(&scale);
    __half2 result = __hmul2(__hsub2(values, offset), __half2half2(native_scale));
    return *reinterpret_cast<unsigned int*>(&result);
}
'''


@orin_jit
def activation_q8(M: int, K: int, block: int = 256):
    @T.prim_func
    def kernel(A: T.Tensor((M, K), T.float16), Q: T.Tensor((M, K), T.int8),
               S: T.Tensor((M,), T.float16)):
        with T.Kernel(M, threads=256) as row:
            x = T.alloc_fragment((K,), T.float32)
            absx = T.alloc_fragment((K,), T.float32)
            maximum = T.alloc_fragment((1,), T.float32)
            scale = T.alloc_fragment((1,), T.float16)
            for j in T.Parallel(K):
                x[j] = A[row, j]
                absx[j] = T.abs(x[j])
            T.reduce_max(absx, maximum, dim=0)
            scale[0] = T.if_then_else(maximum[0] > 0, T.max(maximum[0] / 127, 2**-24), 1)
            S[row] = scale[0]
            for j in T.Parallel(K):
                Q[row, j] = T.cast(T.max(-127.0, T.min(127.0, T.round(x[j] / T.cast(scale[0], T.float32)))), T.int8)
    return kernel


@orin_jit
def expand_weight_q8(N: int, K: int):
    """One-pass temporary per-channel W8 view; intentionally adds rounding."""
    @T.prim_func
    def kernel(P: T.Tensor((N, K // 2), T.uint8),
               S: T.Tensor((N, K // 128), T.float16),
               Z: T.Tensor((N, K // 128), T.int8),
               B: T.Tensor((N, K), T.int8), BS: T.Tensor((N,), T.float16)):
        with T.Kernel(N, threads=256) as row:
            weights = T.alloc_fragment((K,), T.float32)
            absolute = T.alloc_fragment((K,), T.float32)
            maximum = T.alloc_fragment((1,), T.float32)
            scale = T.alloc_fragment((1,), T.float16)
            for j in T.Parallel(K):
                code = (T.cast(P[row, j // 2], T.int32) >> ((j % 2) * 4)) & 15
                weights[j] = T.cast(T.cast(code - T.cast(Z[row, j // 128], T.int32), T.float16) * S[row, j // 128], T.float32)
                absolute[j] = T.abs(weights[j])
            T.reduce_max(absolute, maximum, dim=0)
            scale[0] = T.max(maximum[0] / 127, 2**-24)
            BS[row] = scale[0]
            for j in T.Parallel(K):
                B[row, j] = T.cast(T.min(127.0, T.max(-127.0, T.round(weights[j] / T.cast(scale[0], T.float32)))), T.int8)
    return kernel


@orin_jit
def expand_weight_q8_precomputed(N: int, K: int, BN: int = 16, BK: int = 256):
    """Use an offline row scale, keeping expansion register usage bounded."""
    @T.prim_func
    def kernel(P: T.Tensor((N, K // 2), T.uint8),
               S: T.Tensor((N, K // 128), T.float16),
               Z: T.Tensor((N, K // 128), T.int8),
               BS: T.Tensor((N,), T.float16), B: T.Tensor((N, K), T.int8)):
        with T.Kernel(N // BN, T.ceildiv(K, BK), threads=256) as (bx, bk):
            for i, j in T.Parallel(BN, BK):
                row = bx * BN + i
                col = bk * BK + j
                if col < K:
                    code = (T.cast(P[row, col // 2], T.int32) >> ((col % 2) * 4)) & 15
                    weight = T.cast(T.cast(code - T.cast(Z[row, col // 128], T.int32), T.float16) * S[row, col // 128], T.float32)
                    B[row, col] = T.cast(T.min(127.0, T.max(-127.0, T.round(weight / T.cast(BS[row], T.float32)))), T.int8)
    return kernel


@orin_jit
def prefill(M: int, N: int, K: int, layout: str = "nk", scale_layout: str = "gn",
            route: str = "w4a16", BM: int = 64, BN: int = 64, BK: int = 128,
            stages: int = 2, threads: int = 128, swizzle: int = 0, fast_decode: bool = False,
            packed_swizzle: bool = False, grid_order: str = "nfirst", transpose_compute: bool = False,
            min_blocks: int = 1):
    assert K % 128 == 0 and N % BN == 0 and K % BK == 0
    assert layout in ("nk", "kn", "n16k32", "n32k128") and scale_layout in ("ng", "gn")
    assert route in ("w4a16", "w4a8")
    assert route != "w4a8" or BK in (32, 64, 128)
    micro_n, micro_k = (32, 128) if layout == "n32k128" else (16, 32)
    assert layout in ("nk", "kn") or BK % micro_k == 0
    packed_shape = (N, K // 2) if layout == "nk" else ((K // 2, N) if layout == "kn" else (N // micro_n, K // micro_k, micro_n, micro_k // 2))
    packed_tile = (BN, BK // 2) if layout == "nk" else ((BK // 2, BN) if layout == "kn" else (BN // micro_n, BK // micro_k, micro_n, micro_k // 2))
    scale_shape = (N, K // 128) if scale_layout == "ng" else (K // 128, N)
    a_dtype = T.float16 if route == "w4a16" else T.int8
    grid_x, grid_y = (N // BN, T.ceildiv(M, BM)) if grid_order == "nfirst" else (T.ceildiv(M, BM), N // BN)
    accum_shape = (BN, BM) if transpose_compute else (BM, BN)

    @T.prim_func
    def kernel(A: T.Tensor((M, K), a_dtype), P: T.Tensor(packed_shape, T.uint8),
               S: T.Tensor(scale_shape, T.float16), Z: T.Tensor(scale_shape, T.int8),
               AS: T.Tensor((M,), T.float16), C: T.Tensor((M, N), T.float16)):
        with T.Kernel(grid_x, grid_y, threads=threads) as (gx, gy):
            T.annotate_min_blocks_per_sm(min_blocks)
            bx = gx if grid_order == "nfirst" else gy
            by = gy if grid_order == "nfirst" else gx
            a = T.alloc_shared((BM, BK), a_dtype)
            packed = T.alloc_shared(packed_tile, T.uint8)
            b = T.alloc_shared((BN, BK), a_dtype)
            scale = T.alloc_shared((BN,), T.float16)
            zero = T.alloc_shared((BN,), T.int8)
            accum = T.alloc_fragment(accum_shape, T.float32)
            integer = T.alloc_fragment(accum_shape, T.int32)
            if packed_swizzle and layout == "kn":
                T.annotate_layout({packed: tilelang.Layout((BK // 2, BN), lambda i, j: (i, j ^ ((i % 16) * 4)))})
            if fast_decode and route == "w4a16":
                T.import_source(PAIR_SOURCE)
            if swizzle:
                T.use_swizzle(panel_size=swizzle)
            T.clear(accum)
            for ko in T.Pipelined(K // BK, num_stages=stages):
                T.copy(A[by * BM, ko * BK], a)
                if layout == "nk":
                    T.copy(P[bx * BN, ko * BK // 2], packed)
                elif layout == "kn":
                    T.copy(P[ko * BK // 2, bx * BN], packed)
                else:
                    T.copy(P[bx * BN // micro_n, ko * BK // micro_k, 0, 0], packed)
                for i in T.Parallel(BN):
                    if scale_layout == "ng":
                        scale[i] = S[bx * BN + i, ko * BK // 128]
                        zero[i] = Z[bx * BN + i, ko * BK // 128]
                    else:
                        scale[i] = S[ko * BK // 128, bx * BN + i]
                        zero[i] = Z[ko * BK // 128, bx * BN + i]
                if fast_decode and route == "w4a16":
                    for i, j in T.Parallel(BN, BK // 2):
                        if layout == "nk":
                            byte = packed[i, j]
                        elif layout == "kn":
                            byte = packed[j, i]
                        else:
                            byte = packed[i // micro_n, j // (micro_k // 2), i % micro_n, j % (micro_k // 2)]
                        pair = T.call_pure_extern("uint32", "deq_u4_pair", byte, scale[i], zero[i])
                        b[i, j * 2] = T.reinterpret(T.float16, T.cast(pair & 65535, T.uint16))
                        b[i, j * 2 + 1] = T.reinterpret(T.float16, T.cast(pair >> 16, T.uint16))
                else:
                    for i, j in T.Parallel(BN, BK):
                        if layout == "nk":
                            byte = T.cast(packed[i, j // 2], T.int32)
                        elif layout == "kn":
                            byte = T.cast(packed[j // 2, i], T.int32)
                        else:
                            byte = T.cast(packed[i // micro_n, j // micro_k, i % micro_n, (j % micro_k) // 2], T.int32)
                        code = (byte >> ((j % 2) * 4)) & 15
                        signed = code - T.cast(zero[i], T.int32)
                        if route == "w4a16":
                            b[i, j] = T.cast(signed, T.float16) * scale[i]
                        else:
                            b[i, j] = T.cast(signed, T.int8)
                if route == "w4a16":
                    if transpose_compute:
                        T.gemm(b, a, accum, transpose_B=True)
                    else:
                        T.gemm(a, b, accum, transpose_B=True)
                else:
                    if ko % (128 // BK) == 0:
                        T.clear(integer)
                    if transpose_compute:
                        T.gemm(b, a, integer, transpose_B=True)
                        if ko % (128 // BK) == (128 // BK) - 1:
                            for i, j in T.Parallel(BN, BM):
                                accum[i, j] += T.cast(integer[i, j], T.float32) * T.cast(scale[i], T.float32)
                    else:
                        T.gemm(a, b, integer, transpose_B=True)
                        if ko % (128 // BK) == (128 // BK) - 1:
                            for i, j in T.Parallel(BM, BN):
                                accum[i, j] += T.cast(integer[i, j], T.float32) * T.cast(scale[j], T.float32)
            if transpose_compute:
                for i, j in T.Parallel(BN, BM):
                    if route == "w4a8":
                        C[by * BM + j, bx * BN + i] = accum[i, j] * T.cast(AS[by * BM + j], T.float32)
                    else:
                        C[by * BM + j, bx * BN + i] = accum[i, j]
            else:
                if route == "w4a8":
                    for i, j in T.Parallel(BM, BN):
                        accum[i, j] *= T.cast(AS[by * BM + i], T.float32)
                T.copy(accum, C[by * BM, bx * BN])
    return kernel


@orin_jit
def fp16_gemm(M: int, N: int, K: int, BM: int = 64, BN: int = 64,
              BK: int = 64, stages: int = 3, threads: int = 128):
    @T.prim_func
    def kernel(A: T.Tensor((M, K), T.float16), B: T.Tensor((N, K), T.float16),
               C: T.Tensor((M, N), T.float16)):
        with T.Kernel(N // BN, T.ceildiv(M, BM), threads=threads) as (bx, by):
            a = T.alloc_shared((BM, BK), T.float16)
            b = T.alloc_shared((BN, BK), T.float16)
            accum = T.alloc_fragment((BM, BN), T.float32)
            T.clear(accum)
            for ko in T.Pipelined(K // BK, num_stages=stages):
                T.copy(A[by * BM, ko * BK], a)
                T.copy(B[bx * BN, ko * BK], b)
                T.gemm(a, b, accum, transpose_B=True)
            T.copy(accum, C[by * BM, bx * BN])
    return kernel


@orin_jit
def int8_gemm(M: int, N: int, K: int, BM: int = 128, BN: int = 128,
              BK: int = 128, stages: int = 3, threads: int = 256, min_blocks: int = 1,
              grid_order: str = 'nfirst', cache_policy: str = 'default'):
    """Compute ceiling: per-channel W8 permits scaling only in the epilogue."""
    assert grid_order in ('nfirst', 'mfirst', 'grouped4', 'grouped8',
                          'n1m2','n1m4','n1m8','n2m2','n2m4')
    assert cache_policy in ('default','a-last-b-first','a-last','b-first','b-last','a-first-b-last')
    grouped = grid_order in ('grouped4','grouped8','n1m2','n1m4','n1m8','n2m2','n2m4')
    nm, nn = (M+BM-1)//BM, N//BN
    gn = 1 if grid_order.startswith('n1m') else 2 if grid_order.startswith('n2m') else 4
    requested_m = int(grid_order[-1]) if grid_order.startswith(('n1m','n2m')) else 4 if grid_order=='grouped4' else 8
    gm = min(requested_m,nm) if grouped else 1
    if grouped:
        assert nn % gn == 0
    gx, gy = (nn*nm,1) if grouped else (nn,nm) if grid_order=='nfirst' else (nm,nn)
    @T.prim_func
    def kernel(A: T.Tensor((M, K), T.int8), B: T.Tensor((N, K), T.int8),
               AS: T.Tensor((M,), T.float16), BS: T.Tensor((N,), T.float16),
               C: T.Tensor((M, N), T.float16)):
        with T.Kernel(gx, gy, threads=threads) as (blockx, blocky):
            # The final M group can contain fewer tiles. Keep every valid
            # (M,N) tile exactly once, including non-aligned shape tails.
            if grouped:
                first_m = (blockx // (nn * gm)) * gm
                active_m = T.min(nm - first_m, gm)
                within = blockx % (nn * gm)
                bx = (within // (gn * active_m)) * gn + within % gn
                by = first_m + (within // gn) % active_m
            else:
                bx = blockx if grid_order == 'nfirst' else blocky
                by = blocky if grid_order == 'nfirst' else blockx
            T.annotate_min_blocks_per_sm(min_blocks)
            a = T.alloc_shared((BM, BK), T.int8)
            b = T.alloc_shared((BN, BK), T.int8)
            accum = T.alloc_fragment((BM, BN), T.int32)
            T.clear(accum)
            for ko in T.Pipelined(K // BK, num_stages=stages):
                T.copy(A[by * BM, ko * BK], a,
                       eviction_policy='evict_last' if cache_policy in ('a-last','a-last-b-first') else 'evict_first' if cache_policy=='a-first-b-last' else None)
                T.copy(B[bx * BN, ko * BK], b,
                       eviction_policy='evict_first' if cache_policy in ('b-first','a-last-b-first') else 'evict_last' if cache_policy in ('b-last','a-first-b-last') else None)
                T.gemm(a, b, accum, transpose_B=True)
            for i, j in T.Parallel(BM, BN):
                if by * BM + i < M:
                    C[by * BM + i, bx * BN + j] = T.cast(accum[i, j], T.float32) * T.cast(AS[by * BM + i], T.float32) * T.cast(BS[bx * BN + j], T.float32)
    return kernel


@orin_jit
def w4_gemv(M: int, N: int, K: int, BN: int = 16, BK: int = 256):
    """Streaming direct U4 scalar dot, one token/CTA with 16 output channels."""
    @T.prim_func
    def kernel(A: T.Tensor((M,K),T.float16), P: T.Tensor((N,K//2),T.uint8),
               S: T.Tensor((N,K//128),T.float16), Z: T.Tensor((N,K//128),T.int8),
               C: T.Tensor((M,N),T.float16)):
        with T.Kernel(N//BN,M,threads=256) as (bx,row):
            product=T.alloc_fragment((BN,BK),T.float32)
            part=T.alloc_fragment((BN,),T.float32)
            accum=T.alloc_fragment((BN,),T.float32)
            T.clear(accum)
            for ko in T.serial(K//BK):
                for i,j in T.Parallel(BN,BK):
                    col=ko*BK+j
                    code=(T.cast(P[bx*BN+i,col//2],T.int32)>>((col%2)*4))&15
                    weight=T.cast(T.cast(code-T.cast(Z[bx*BN+i,col//128],T.int32),T.float16)*S[bx*BN+i,col//128],T.float16)
                    product[i,j]=T.cast(weight,T.float32)*T.cast(A[row,col],T.float32)
                T.reduce_sum(product,part,dim=1)
                for i in T.Parallel(BN): accum[i]+=part[i]
            for i in T.Parallel(BN): C[row,bx*BN+i]=accum[i]
    return kernel


@orin_jit
def weight_q8_lut(N: int, K: int, BN: int = 16):
    """Temporary group/code lookup, rebuilt per projection to avoid W8 residency."""
    @T.prim_func
    def kernel(S: T.Tensor((N,K//128),T.float16), Z: T.Tensor((N,K//128),T.int8),
               BS: T.Tensor((N,),T.float16), L: T.Tensor((N,K//128,16),T.int8)):
        with T.Kernel(N//BN,K//128,threads=256) as (bx,g):
            for i,code in T.Parallel(BN,16):
                row=bx*BN+i
                weight=T.cast(T.cast(code-T.cast(Z[row,g],T.int32),T.float16)*S[row,g],T.float32)
                L[row,g,code]=T.cast(T.min(127.0,T.max(-127.0,T.round(weight/T.cast(BS[row],T.float32)))),T.int8)
    return kernel


@orin_jit
def expand_weight_q8_lut(N: int, K: int, BN: int = 16, BK: int = 256):
    @T.prim_func
    def kernel(P: T.Tensor((N,K//2),T.uint8), L: T.Tensor((N,K//128,16),T.int8),
               B: T.Tensor((N,K),T.int8)):
        with T.Kernel(N//BN,K//BK,threads=256) as (bx,bk):
            lookup=T.alloc_shared((BN,BK//128,16),T.int8)
            packed=T.alloc_shared((BN,BK//2),T.uint8)
            T.copy(L[bx*BN,bk*(BK//128),0],lookup)
            T.copy(P[bx*BN,bk*(BK//2)],packed)
            thread=T.get_thread_binding()
            for offset in T.serial(BN*BK//256):
                i=(offset*256+thread)//BK
                j=(offset*256+thread)%BK
                code=(T.cast(packed[i,j//2],T.int32)>>((j%2)*4))&15
                B[bx*BN+i,bk*BK+j]=lookup[i,j//128,code]
    return kernel


@orin_jit
def w4_splitk(M: int, N: int, K: int, SPLIT: int = 8, BM: int = 16, BN: int = 64, BK: int = 128):
    assert K % (BK*SPLIT) == 0
    @T.prim_func
    def kernel(A: T.Tensor((M,K),T.float16), P: T.Tensor((N,K//2),T.uint8),
               S: T.Tensor((N,K//128),T.float16), Z: T.Tensor((N,K//128),T.int8),
               O: T.Tensor((SPLIT,M,N),T.float32)):
        with T.Kernel(N//BN,T.ceildiv(M,BM),SPLIT,threads=128) as (bx,by,sk):
            T.import_source(PAIR_SOURCE)
            a=T.alloc_shared((BM,BK),T.float16)
            p=T.alloc_shared((BN,BK//2),T.uint8)
            b=T.alloc_shared((BN,BK),T.float16)
            scale=T.alloc_shared((BN,),T.float16)
            zero=T.alloc_shared((BN,),T.int8)
            accum=T.alloc_fragment((BM,BN),T.float32)
            T.clear(accum)
            for ki in T.Pipelined(K//BK//SPLIT,num_stages=2):
                ko=sk*(K//BK//SPLIT)+ki
                T.copy(A[by*BM,ko*BK],a)
                T.copy(P[bx*BN,ko*(BK//2)],p)
                for i in T.Parallel(BN):
                    scale[i]=S[bx*BN+i,ko]
                    zero[i]=Z[bx*BN+i,ko]
                for i,j in T.Parallel(BN,BK//2):
                    pair=T.call_pure_extern('uint32','deq_u4_pair',p[i,j],scale[i],zero[i])
                    b[i,j*2]=T.reinterpret(T.float16,T.cast(pair&65535,T.uint16))
                    b[i,j*2+1]=T.reinterpret(T.float16,T.cast(pair>>16,T.uint16))
                T.gemm(a,b,accum,transpose_B=True)
            T.copy(accum,O[sk,by*BM,bx*BN])
    return kernel


@orin_jit
def splitk_reduce(M: int, N: int, SPLIT: int = 8):
    @T.prim_func
    def kernel(P: T.Tensor((SPLIT,M,N),T.float32), O: T.Tensor((M,N),T.float16)):
        with T.Kernel(T.ceildiv(M*N,256),threads=256) as bx:
            thread=T.get_thread_binding()
            index=bx*256+thread
            accum=T.alloc_local((1,),T.float32)
            accum[0]=0
            if index < M*N:
                for sk in T.serial(SPLIT): accum[0]+=P[sk,index//N,index%N]
                O[index//N,index%N]=accum[0]
    return kernel


@orin_jit
def expand_weight_q8_inline_lut(N: int, K: int, BN: int = 64, BK: int = 256):
    """Adapt mature Marlin LUT method to explicitly checked logical NK U4 ABI."""
    @T.prim_func
    def kernel(P: T.Tensor((N,K//2),T.uint8), S: T.Tensor((N,K//128),T.float16),
               Z: T.Tensor((N,K//128),T.int8), BS: T.Tensor((N,),T.float16), B: T.Tensor((N,K),T.int8)):
        with T.Kernel(N//BN,K//BK,threads=256) as (bx,bk):
            packed=T.alloc_shared((BN,BK//2),T.uint8)
            scale=T.alloc_shared((BN,BK//128),T.float16)
            zero=T.alloc_shared((BN,BK//128),T.int8)
            rowscale=T.alloc_shared((BN,),T.float16)
            table=T.alloc_shared((BK//128,BN,4),T.int32)
            T.copy(P[bx*BN,bk*(BK//2)],packed)
            T.copy(S[bx*BN,bk*(BK//128)],scale)
            T.copy(Z[bx*BN,bk*(BK//128)],zero)
            T.copy(BS[bx*BN],rowscale)
            for g,i,word_idx in T.Parallel(BK//128,BN,4):
                word=T.alloc_var(T.int32)
                word=0
                for c in T.unroll(4):
                    weight=T.cast(T.cast(word_idx*4+c-T.cast(zero[i,g],T.int32),T.float16)*scale[i,g],T.float32)
                    code8=T.cast(T.min(127.0,T.max(-127.0,T.round(weight/T.cast(rowscale[i],T.float32)))),T.int32)
                    word=word|((code8&255)<<(c*8))
                table[g,i,word_idx]=word
            for i,j in T.Parallel(BN,BK):
                code4=(T.cast(packed[i,j//2],T.int32)>>((j%2)*4))&15
                word8=table[j//128,i,code4//4]
                B[bx*BN+i,bk*BK+j]=T.cast((word8>>((code4%4)*8))&255,T.int8)
    return kernel
