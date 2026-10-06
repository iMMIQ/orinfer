"""Qwen4 PLE gating and causal dilated convolution.

CPU lookup supplies embedding features; key/value projections and grouped norms
use the projection and HC normalization kernels. These operators preserve the
materialized low-precision boundaries of the unfused reference path. History
is chronological, request-owned, and includes (taps-1)*dilation rows.
"""
import tilelang.language as T

from tools.operators.common import orin_jit
from kernels.model.hyperconnection import _dtype


@orin_jit
def ple_gate(M: int, H: int, streams: int = 4, dtype: str = 'float16'):
    """(NormedKey[M,streams,H], NormedQuery, Value[M,H], Gated)."""
    _dtype(dtype)
    if any(type(x) is not int or x <= 0 for x in (M, H, streams)):
        raise ValueError('Invalid PLE gate dimensions')
    width = 1 << (H - 1).bit_length()
    @T.prim_func
    def main(Key: T.Tensor((M, streams, H), dtype),
             Query: T.Tensor((M, streams, H), dtype),
             Value: T.Tensor((M, H), dtype),
             Gated: T.Tensor((M, streams, H), dtype)):
        with T.Kernel(M, streams, threads=256) as (row, branch):
            product = T.alloc_fragment((width,), T.float32)
            total = T.alloc_fragment((1,), T.float32)
            gate = T.alloc_local((1,), T.float32)
            for j in T.Parallel(width, coalesced_width=T.int32(1)):
                if j < H:
                    product[j] = T.cast(T.cast(T.cast(Key[row, branch, j], T.float32) *
                                              T.cast(Query[row, branch, j], T.float32), dtype), T.float32)
                else:
                    product[j] = 0.0
            T.reduce_sum(product, total, dim=0)
            raw = T.cast(T.cast(T.cast(T.cast(total[0], dtype), T.float32) / (H**0.5), dtype), T.float32)
            # abs/clamp, sqrt, sign multiplication and sigmoid are separate
            # materializations in the reference, including the exact zero gate.
            absolute = T.cast(T.cast(T.max(T.abs(raw), 1e-6), dtype), T.float32)
            root = T.cast(T.cast(T.sqrt(absolute), dtype), T.float32)
            sign = T.if_then_else(raw > 0.0, 1.0, T.if_then_else(raw < 0.0, -1.0, 0.0))
            signed = T.cast(T.cast(root * sign, dtype), T.float32)
            gate[0] = T.cast(T.cast(1.0 / (1.0 + T.exp(-signed)), dtype), T.float32)
            for j in T.Parallel(H, coalesced_width=T.int32(1)):
                Gated[row, branch, j] = gate[0] * T.cast(Value[row, j], T.float32)
    return main


@orin_jit
def ple_conv(M: int, C: int, taps: int = 4, dilation: int = 3, dtype: str = 'float16'):
    """(Normed[M,C], Gated[M,C], State[history,C], Weight[C,taps], Output).

    Weight is supplied as FP16 coefficients by the execution plan, converted
    to stream dtype in registers. State is read-only here; ple_history advances it separately so
    prefill blocks cannot race against old-history readers. Output is the
    PLE addition (gated value + SiLU(conv)), before the outer residual addition.
    """
    _dtype(dtype)
    if any(type(x) is not int or x <= 0 for x in (M, C, taps, dilation)) or taps < 2:
        raise ValueError('Invalid PLE convolution dimensions')
    history = (taps - 1) * dilation
    @T.prim_func
    def main(Normed: T.Tensor((M, C), dtype), Gated: T.Tensor((M, C), dtype),
             State: T.Tensor((history, C), dtype),
             Weight: T.Tensor((C, taps), T.float16), Output: T.Tensor((M, C), dtype)):
        with T.Kernel(M, T.ceildiv(C, 256), threads=128) as (row, block):
            acc = T.alloc_fragment((256,), T.float32)
            T.clear(acc)
            for tap in T.unroll(taps):
                position = row - (taps - 1 - tap) * dilation
                for j in T.Parallel(256, coalesced_width=T.int32(1)):
                    col = block * 256 + j
                    if col < C:
                        value = T.if_then_else(position >= 0, Normed[position, col], State[history + position, col])
                        weight = T.cast(T.cast(Weight[col, tap], dtype), T.float32)
                        acc[j] += T.cast(value, T.float32) * weight
            for j in T.Parallel(256, coalesced_width=T.int32(1)):
                col = block * 256 + j
                if col < C:
                    v = T.cast(T.cast(acc[j], dtype), T.float32)
                    activated = T.cast(T.cast(v / (1.0 + T.exp(-v)), dtype), T.float32)
                    Output[row, col] = T.cast(Gated[row, col], T.float32) + activated
    return main


@orin_jit
def ple_history(M: int, C: int, history: int = 9, dtype: str = 'float16'):
    """(Normed[M,C], State[history,C], NextState), permitting State aliasing.

    One thread owns each channel and copies time rows in increasing order.
    Old-state source rows are higher than destination rows, so an in-place
    shift cannot overwrite any value before its read. Normed remains disjoint.
    """
    _dtype(dtype)
    if any(type(x) is not int or x <= 0 for x in (M, C, history)):
        raise ValueError('Invalid PLE history dimensions')
    @T.prim_func
    def main(Normed: T.Tensor((M, C), dtype), State: T.Tensor((history, C), dtype),
             NextState: T.Tensor((history, C), dtype)):
        with T.Kernel(T.ceildiv(C, 256), threads=128) as block:
            for j in T.Parallel(256, coalesced_width=T.int32(1)):
                col = block * 256 + j
                if col < C:
                    for row in T.serial(history):
                        position = M - history + row
                        NextState[row, col] = T.if_then_else(position >= 0, Normed[position, col], State[M + row, col])
    return main
