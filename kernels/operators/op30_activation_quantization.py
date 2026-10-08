"""SM87 dynamic-M A8 and bit-exact FP16 SwiGLU -> A8 fusion candidates.

No compensation or model quality policy is implemented here. All buffers are
explicit; callers own validation, stable addresses and CUDA stream selection.
"""

import tilelang
import tilelang.language as T

from kernels.operators.op04_swiglu import swiglu_fp16


@T.macro
def quantized_code(value, scale):
    # Explicit correctly rounded FP32 division avoids reciprocal approximation
    # changing integer ties. T.round lowers to nearbyintf (nearest, ties-even).
    ratio = T.call_extern("float32", "__fdiv_rn", value, T.cast(scale, T.float32))
    return T.cast(T.max(-127.0, T.min(127.0, T.round(ratio))), T.int8)


@tilelang.jit(out_idx=[], execution_backend="nvrtc", target={"kind": "cuda", "arch": "sm_87"})
def _compile(K: int, group: int, masked: bool, fused: bool, threads: int):
    rows = T.dynamic("rows")
    groups = (K + group - 1) // group
    tile = ((group + threads - 1) // threads) * threads

    @T.prim_func
    def main(
        X: T.Tensor((rows, (2 * K if fused else K)), T.float16),
        Mask: T.Tensor((K,), T.uint8),
        Q: T.Tensor((rows, K), T.int8),
        S: T.Tensor((rows, groups), T.float16),
    ):
        with T.Kernel(rows, groups, threads=threads) as (row, block):
            value = T.alloc_fragment((tile,), T.float32)
            absolute = T.alloc_fragment((tile,), T.float32)
            maximum = T.alloc_fragment((1,), T.float32)
            scale = T.alloc_fragment((1,), T.float16)
            for j in T.Parallel(tile):
                col = block * group + j
                value[j] = 0.0
                if j < group and col < K:
                    if fused:
                        # Do not quantize FP32 SwiGLU directly: this FP16 cast
                        # is the exact materialization boundary used by op04.
                        value[j] = T.cast(swiglu_fp16(X[row, col], X[row, K + col]), T.float32)
                    else:
                        value[j] = T.cast(X[row, col], T.float32)
                    if masked:
                        if Mask[col] != 0:
                            value[j] = 0.0
                absolute[j] = T.abs(value[j])
            T.reduce_max(absolute, maximum, dim=0)
            scale[0] = T.if_then_else(
                maximum[0] > 0.0,
                T.max(T.call_extern("float32", "__fdiv_rn", maximum[0], 127.0), 2**-24),
                1.0,
            )
            S[row, block] = scale[0]
            for j in T.Parallel(tile):
                col = block * group + j
                if j < group and col < K:
                    Q[row, col] = quantized_code(value[j], scale[0])

    return main


def activation_quantization(
    K: int, group_size: int | None = None, *, masked: bool = False, threads: int = 256
):
    """Build dynamic-M (X, Mask, Q, S), FP16/uint8/int8/FP16.

    group_size=None means per-token. S[M,ceildiv(K,group_size)] is contiguous.
    Mask[K] is ignored unless masked=True; any nonzero byte excludes a column
    from amax and writes code zero. No high precision correction is performed.
    Finite FP16 inputs only. All-zero (or all-excluded) group scale=1; otherwise
    scale=half(max(amax/127,2**-24)), code=clamp(rint(x/float(scale)),-127,127).
    """
    return _build(K, group_size, masked, False, threads)


def swiglu_activation_quantization(
    K: int = 17408, group_size: int | None = None, *, masked: bool = False, threads: int = 256
):
    """Build the same ABI with split-layout X[M,2*K]=[gate|up].

    The imported op04 macro returns FP16 before the FP32 amax/code arithmetic.
    This path must match op04 split -> activation_quantization bit for bit.
    """
    return _build(K, group_size, masked, True, threads)


def _build(K, group_size, masked, fused, threads):
    assert isinstance(K, int) and K > 0
    group = K if group_size is None else group_size
    assert isinstance(group, int) and 0 < group <= K
    assert threads in (128, 256, 512)
    kernel = _compile(K, group, masked, fused, threads)
    kernel.adapter.kernels = dict(kernel.adapter.kernels)
    return kernel


def launch(kernel, X, Mask, Q, S, *, stream):
    """Launch on the caller's current explicit stream; inputs/outputs disjoint."""
    return kernel.adapter.func(X, Mask, Q, S, stream=stream)
