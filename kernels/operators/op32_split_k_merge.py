"""Allocation-free SM87 split-K final reduction.

Partial[S,M,N] is contiguous FP32. S/N are compile-time, M may be symbolic.
Residual is consumed only after the complete reduction; native residual math
rounds the projection to FP16 before its FP32 residual addition (op02 contract).
The caller owns distinct inputs/outputs and passes the active stream per call.
"""
import tilelang.language as T
from tools.operators.common import orin_jit


def _dimensions(M, N, SPLIT, threads):
    if N <= 0 or SPLIT <= 0 or threads not in (128, 256, 512):
        raise ValueError('positive N/SPLIT and threads 128/256/512 required')
    if isinstance(M, int) and M <= 0:
        raise ValueError('M must be positive')
    return T.dynamic('M') if M is None else M


@orin_jit
def _merge(M, N, SPLIT, output_dtype, threads):
    @T.prim_func
    def kernel(P: T.Tensor((SPLIT, M, N), T.float32),
               O: T.Tensor((M, N), output_dtype)):
        with T.Kernel(T.ceildiv(M * N, threads), threads=threads) as bx:
            index = bx * threads + T.get_thread_binding()
            value = T.alloc_local((1,), T.float32)
            value[0] = 0.0
            if index < M * N:
                for sk in T.unroll(SPLIT):
                    value[0] += P[sk, index // N, index % N]
                O[index // N, index % N] = value[0]
    return kernel


@orin_jit
def _merge_residual(M, N, SPLIT, residual_dtype, threads):
    @T.prim_func
    def kernel(P: T.Tensor((SPLIT, M, N), T.float32),
               R: T.Tensor((M, N), residual_dtype),
               RO: T.Tensor((M, N), T.float32)):
        with T.Kernel(T.ceildiv(M * N, threads), threads=threads) as bx:
            index = bx * threads + T.get_thread_binding()
            value = T.alloc_local((1,), T.float32)
            value[0] = 0.0
            if index < M * N:
                for sk in T.unroll(SPLIT):
                    value[0] += P[sk, index // N, index % N]
                # Preserve the native projection FP16 store/load rounding point.
                projection = T.cast(T.cast(value[0], T.float16), T.float32)
                RO[index // N, index % N] = projection + T.cast(R[index // N, index % N], T.float32)
    return kernel


def split_k_merge(M=None, N=5120, SPLIT=8, output_dtype='float16', threads=256):
    """Build (partialF32, output), ordered FP32 sum followed by one output cast.

    M=None creates a single runtime-M cubin. FP16/FP32 outputs are supported.
    No allocations or residual writes into partial are performed.
    """
    rows = _dimensions(M, N, SPLIT, threads)
    if output_dtype not in ('float16', 'float32'):
        raise ValueError('output_dtype must be float16 or float32')
    return _merge(rows, N, SPLIT, output_dtype, threads)


def split_k_merge_residual(M=None, N=5120, SPLIT=8,
                           residual_dtype='float32', threads=256):
    """Build (partialF32, residual, ROF32) with native projection rounding.

    RO = FP32(FP16(ordered_FP32_sum(partial))) + FP32(R). R may be FP16/FP32.
    This consumes the residual exactly once, after all splits, and exposes no
    unrounded-projection alternative that could silently change op02 math.
    RO can replace op02's residual-add result only when normalization consumes
    RO directly; passing RO into op02 as X would add the residual a second time.
    """
    rows = _dimensions(M, N, SPLIT, threads)
    if residual_dtype not in ('float16', 'float32'):
        raise ValueError('residual_dtype must be float16 or float32')
    return _merge_residual(rows, N, SPLIT, residual_dtype, threads)
