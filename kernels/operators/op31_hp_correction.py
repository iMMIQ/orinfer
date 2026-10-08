"""SM87 selected-column FP16 correction with FP32 masked base, dynamic M.

Base must EXCLUDE every selected column (masked A8 GEMM). W_hp is the
prepacked SAME FP16-dequantized W4 selected columns; no model policy here.
"""

import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[], execution_backend="nvrtc", target={"kind": "cuda", "arch": "sm_87"})
def _compile(N: int, K: int, O: int, dtype: str, BM: int, BN: int):
    M = T.dynamic("M")
    BK = 32

    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.float16),
        Idx: T.Tensor((O,), T.int32),
        W_hp: T.Tensor((N, O), T.float16),
        Base: T.Tensor((M, N), T.float32),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=128) as (bx, by):
            if O > 0:
                a = T.alloc_shared((BM, BK), T.float16)
                b = T.alloc_shared((BN, BK), T.float16)
                accum = T.alloc_fragment((BM, BN), T.float32)
                T.clear(accum)
                for tile in T.serial(T.ceildiv(O, BK)):
                    for i, j in T.Parallel(BM, BK):
                        a[i, j] = 0.0
                        if by * BM + i < M and tile * BK + j < O:
                            # Real branch prevents padded indices reading A[:,0]
                            # (which may be NaN); padding is zero, never a load.
                            a[i, j] = A[by * BM + i, Idx[tile * BK + j]]
                    for i, j in T.Parallel(BN, BK):
                        b[i, j] = 0.0
                        if bx * BN + i < N and tile * BK + j < O:
                            b[i, j] = W_hp[bx * BN + i, tile * BK + j]
                    T.gemm(a, b, accum, transpose_B=True)
                for i, j in T.Parallel(BM, BN):
                    if by * BM + i < M and bx * BN + j < N:
                        C[by * BM + i, bx * BN + j] = accum[i, j] + Base[by * BM + i, bx * BN + j]
            else:
                for i, j in T.Parallel(BM, BN):
                    if by * BM + i < M and bx * BN + j < N:
                        C[by * BM + i, bx * BN + j] = Base[by * BM + i, bx * BN + j]

    return main


def hp_correction(
    N: int, K: int, O: int = 32, *, output_dtype: str = "float16", small: bool = False
):
    """Build (A, Idx, W_hp, Base, C); M is symbolic. Output is FP16/FP32.

    C = cast_output(Base_F32 + sum_o(float(A[:,Idx[o]])*float(W_hp[:,o]))).
    O=0 is legal. All inputs/outputs disjoint. O need not be MMA aligned.
    Call validate_indices once per new index VALUES before launch/capture;
    stable-address graph mutations must uphold that validation contract.
    """
    if not (
        isinstance(N, int)
        and N > 0
        and isinstance(K, int)
        and K > 0
        and isinstance(O, int)
        and 0 <= O <= K
    ):
        raise ValueError("Require N,K>0 and 0<=O<=K")
    if output_dtype not in ("float16", "float32"):
        raise ValueError("output_dtype must be float16 or float32")
    kernel = _compile(N, K, O, output_dtype, 16 if small else 64, 64 if small else 128)
    kernel.adapter.kernels = dict(kernel.adapter.kernels)
    kernel.op31_spec = (N, K, O, output_dtype)
    return kernel


def validate_indices(indices, K: int):
    """Offline/prepare validation; may synchronize when supplied CUDA indices."""
    if str(indices.dtype) != "torch.int32" or indices.ndim != 1 or not indices.is_contiguous():
        raise ValueError("Idx must be contiguous int32[O]")
    values = indices.detach().cpu().tolist()
    if len(values) > K or len(set(values)) != len(values) or any(v < 0 or v >= K for v in values):
        raise ValueError("Idx must contain unique indices in [0,K)")
    return values


def launch(kernel, A, Idx, W_hp, Base, C, *, stream, base_is_masked: bool):
    """Explicit stream; caller certifies masked base, prevalidated Idx values.

    Metadata/alias checks are graph safe. A numerical tensor cannot establish
    its own provenance: unmasked base is rejected unless caller falsely asserts
    base_is_masked=True. Do not add correction to an unmasked A8 GEMM.
    """
    if base_is_masked is not True:
        raise ValueError("Base must be from GEMM excluding selected columns")
    N, K, O, dtype = kernel.op31_spec
    if A.ndim != 2 or len(A) <= 0:
        raise ValueError("A must be nonempty [M,K]")
    specs = [
        (A, (len(A), K), "torch.float16"),
        (Idx, (O,), "torch.int32"),
        (W_hp, (N, O), "torch.float16"),
        (Base, (len(A), N), "torch.float32"),
        (C, (len(A), N), "torch." + dtype),
    ]
    ranges = []
    for tensor, shape, dt in specs:
        if (
            tuple(tensor.shape) != shape
            or str(tensor.dtype) != dt
            or not tensor.is_contiguous()
            or not tensor.is_cuda
            or tensor.device != A.device
        ):
            raise ValueError("Tensor shape/dtype/layout/device violates op31 ABI")
        if tensor.numel():
            start = tensor.data_ptr()
            end = start + tensor.numel() * tensor.element_size()
            if any(start < hi and lo < end for lo, hi in ranges):
                raise ValueError("op31 buffers must not alias")
            ranges.append((start, end))
    return kernel.adapter.func(A, Idx, W_hp, Base, C, stream=stream)


@tilelang.jit(out_idx=[], execution_backend="nvrtc", target={"kind": "cuda", "arch": "sm_87"})
def masked_base_gemm(N: int, K: int, *, small: bool = False):
    """F32-output adaptation of frozen candidates.int8_gemm for chain tests.

    No FP16 base rounding. Production callers may supply any compatible masked
    F32 base implementation. This helper performs no masking itself: Q must
    contain zeros in every selected column, from op30 masked quantization.
    """
    M = T.dynamic("M")
    BM, BN, threads = (16, 64, 128) if small else (256, 128, 256)

    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.int8),
        B: T.Tensor((N, K), T.int8),
        AS: T.Tensor((M, 1), T.float16),
        BS: T.Tensor((N,), T.float16),
        Base: T.Tensor((M, N), T.float32),
    ):
        with T.Kernel(N // BN, T.ceildiv(M, BM), threads=threads) as (bx, by):
            a = T.alloc_shared((BM, 128), T.int8)
            b = T.alloc_shared((BN, 128), T.int8)
            acc = T.alloc_fragment((BM, BN), T.int32)
            T.clear(acc)
            for ko in T.Pipelined(K // 128, num_stages=2):
                T.copy(A[by * BM, ko * 128], a)
                T.copy(B[bx * BN, ko * 128], b)
                T.gemm(a, b, acc, transpose_B=True)
            for i, j in T.Parallel(BM, BN):
                if by * BM + i < M:
                    Base[by * BM + i, bx * BN + j] = (
                        T.cast(acc[i, j], T.float32)
                        * T.cast(AS[by * BM + i, 0], T.float32)
                        * T.cast(BS[bx * BN + j], T.float32)
                    )

    return main
