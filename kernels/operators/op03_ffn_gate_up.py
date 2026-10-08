"""SM87 W4A16 gate/up projection, explicit-output and dynamic-M TileLang API.

P[N,K/2] stores adjacent low/high U4. S[N,K/128] is FP16 and
Z[N,K/128] contains numeric int8 zero points 0..15. Output is gate then up.
Only lossless offline packing is assumed; no activation quantization or native GEMM.
"""

from functools import wraps
import torch
import tilelang
import tilelang.language as T

N_GATE_UP = 34816
K_HIDDEN = 5120


def _orin_jit(function):
    compiled = tilelang.jit(
        out_idx=[], execution_backend="nvrtc", target={"kind": "cuda", "arch": "sm_87"}
    )(function)

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


_PAIR_SOURCE = r"""
#include <cuda_fp16.h>
#include <tl_templates/cuda/instruction/mma.h>
__device__ __forceinline__ unsigned int op03_deq_pair(unsigned char x, half_t scale,
                                                    signed char zero) {
    unsigned int bits = 0x64006400u | (x & 15u) | ((x & 240u) << 12);
    __half2 values = *reinterpret_cast<__half2*>(&bits);
    __half2 offset = __half2half2(__int2half_rn(1024 + int(zero)));
    __half native_scale = *reinterpret_cast<__half*>(&scale);
    __half2 result = __hmul2(__hsub2(values, offset), __half2half2(native_scale));
    return *reinterpret_cast<unsigned int*>(&result);
}
"""


@_orin_jit
def ffn_gate_up(
    M,
    N: int = N_GATE_UP,
    K: int = K_HIDDEN,
    implementation: str = "shared",
    BM: int = 16,
    BN: int = 64,
    BK: int = 128,
    stages: int = 2,
    threads: int = 128,
):
    """Build kernel(A,P,S,Z,C), with optional explicit CUDA stream.

    M may be T.dynamic('M'); dimensions are inferred from A at each invocation.
    shared: validated shared packed/metadata baseline.
    register: packed bytes and metadata live in per-thread fragment registers,
              removing their shared stores/loads before shared FP16 MMA.
    Neither implementation allocates a global workspace or changes weight math.
    """
    assert implementation in ("shared", "register")
    assert K % 128 == 0 and K % BK == 0 and BK in (64, 128)
    assert N % BN == 0 and BM >= 16
    register = implementation == "register"

    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), T.float16),
        P: T.Tensor((N, K // 2), T.uint8),
        S: T.Tensor((N, K // 128), T.float16),
        Z: T.Tensor((N, K // 128), T.int8),
        C: T.Tensor((M, N), T.float16),
    ):
        with T.Kernel(N // BN, T.ceildiv(M, BM), threads=threads) as (bx, by):
            T.import_source(_PAIR_SOURCE)
            a = T.alloc_shared((BM, BK), T.float16)
            b = T.alloc_shared((BN, BK), T.float16)
            packed = (
                T.alloc_fragment((BN, BK // 2), T.uint8)
                if register
                else T.alloc_shared((BN, BK // 2), T.uint8)
            )
            scale = (
                T.alloc_fragment((BN,), T.float16) if register else T.alloc_shared((BN,), T.float16)
            )
            zero = T.alloc_fragment((BN,), T.int8) if register else T.alloc_shared((BN,), T.int8)
            accum = T.alloc_fragment((BM, BN), T.float32)
            T.clear(accum)
            for ko in T.Pipelined(K // BK, num_stages=stages):
                T.copy(A[by * BM, ko * BK], a)
                T.copy(P[bx * BN, ko * BK // 2], packed)
                for i in T.Parallel(BN):
                    scale[i] = S[bx * BN + i, ko * BK // 128]
                    zero[i] = Z[bx * BN + i, ko * BK // 128]
                for i, j in T.Parallel(BN, BK // 2):
                    pair = T.call_pure_extern(
                        "uint32", "op03_deq_pair", packed[i, j], scale[i], zero[i]
                    )
                    b[i, j * 2] = T.reinterpret(T.float16, T.cast(pair & 65535, T.uint16))
                    b[i, j * 2 + 1] = T.reinterpret(T.float16, T.cast(pair >> 16, T.uint16))
                T.gemm(a, b, accum, transpose_B=True)
            T.copy(accum, C[by * BM, bx * BN])

    return kernel
