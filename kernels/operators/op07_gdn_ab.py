"""High precision GDN small projection: X[M,5120] @ W_ab[96,5120].T.

W_ab and Y use a first (48 columns), b second (48 columns), unlike native
vLLM in_proj_ba which emits b first. No bias or gate functions here.
"""

from functools import wraps
import torch
import tilelang
import tilelang.language as T

K_HIDDEN = 5120
HEADS = 48
N_AB = 96


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


@_orin_jit
def gdn_ab_simt(
    M,
    dtype: str = "float16",
    output_dtype: str = "float16",
    N: int = N_AB,
    K: int = K_HIDDEN,
    threads: int = 256,
):
    """Build explicit-output (X,W_ab,Y), dynamic M allowed; one CTA per scalar.

    Products and parallel reduction use FP32; final store rounds to output dtype.
    No workspace, global atomic, quantization, or native GEMM.
    """
    assert dtype in ("float16", "bfloat16")
    assert output_dtype in ("float16", "bfloat16", "float32")

    @T.prim_func
    def kernel(
        X: T.Tensor((M, K), dtype), W_ab: T.Tensor((N, K), dtype), Y: T.Tensor((M, N), output_dtype)
    ):
        with T.Kernel(N, M, threads=threads) as (n, m):
            products = T.alloc_fragment((K,), T.float32)
            total = T.alloc_fragment((1,), T.float32)
            for k in T.Parallel(K):
                if dtype == "bfloat16":
                    # Exact BF16 expansion avoids compiler vector BF16 cast temporaries.
                    xv = T.reinterpret(
                        T.float32, T.cast(T.reinterpret(T.uint16, X[m, k]), T.uint32) << 16
                    )
                    wv = T.reinterpret(
                        T.float32, T.cast(T.reinterpret(T.uint16, W_ab[n, k]), T.uint32) << 16
                    )
                    products[k] = xv * wv
                else:
                    products[k] = T.cast(X[m, k], T.float32) * T.cast(W_ab[n, k], T.float32)
            T.reduce_sum(products, total, dim=0)
            if T.get_thread_binding() == 0:
                Y[m, n] = total[0]

    return kernel


@_orin_jit
def gdn_ab_tensorcore(
    M,
    dtype: str = "float16",
    output_dtype: str = "float16",
    N: int = N_AB,
    K: int = K_HIDDEN,
    BM: int = 16,
    BN: int = 64,
    BK: int = 64,
    stages: int = 2,
    threads: int = 128,
):
    """Build explicit-output (X,W_ab,Y); M and N tile tails are masked.

    FP16/BF16 Tensor Core products, FP32 accumulation, one final output cast.
    N96/BN64 exercises a 32-column tail. M may be T.dynamic('M').
    """
    assert dtype in ("float16", "bfloat16")
    assert output_dtype in ("float16", "bfloat16", "float32")
    assert K % BK == 0

    @T.prim_func
    def kernel(
        X: T.Tensor((M, K), dtype), W_ab: T.Tensor((N, K), dtype), Y: T.Tensor((M, N), output_dtype)
    ):
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=threads) as (bx, by):
            x = T.alloc_shared((BM, BK), dtype)
            w = T.alloc_shared((BN, BK), dtype)
            accum = T.alloc_fragment((BM, BN), T.float32)
            T.clear(accum)
            for ko in T.Pipelined(K // BK, num_stages=stages):
                T.copy(X[by * BM, ko * BK], x)
                T.copy(W_ab[bx * BN, ko * BK], w)
                T.gemm(x, w, accum, transpose_B=True)
            T.copy(accum, Y[by * BM, bx * BN])

    return kernel


def launch(kernel, x, w_ab, y, stream=None):
    """Use the current stream at each invocation, including graph capture."""
    if stream is None:
        stream = torch.cuda.current_stream(x.device).cuda_stream
    return kernel(x, w_ab, y, stream=stream)
