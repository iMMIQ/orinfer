"""Shared-expert W8 gate/up and SwiGLU, preserving FP16 projection rounding."""

import tilelang
import tilelang.language as T

from tools.operators.common import orin_jit
from kernels.model.rows import row_count


@orin_jit
def int8_swiglu(M: int, N: int, K: int, block_n: int = 64, dynamic_rows: bool = False):
    if any(type(v) is not int or v <= 0 for v in (M, N, K)) or M > 8 or K % 128:
        raise ValueError("Invalid small-batch INT8 SwiGLU geometry")
    if K * 128 * 128 > 2**31 - 1 or block_n not in (4, 8, 16, 32, 64):
        raise ValueError("Invalid INT8 SwiGLU accumulator/tile")
    if M != 1 and block_n != 64:
        raise ValueError("Batched INT8 SwiGLU requires a 64-column tile")
    packed = M == 1
    width = K // 4 if packed else K
    M = row_count(M, dynamic_rows)

    @T.prim_func
    def main(
        A: T.Tensor((M, width), T.int32 if packed else T.int8),
        Gate: T.Tensor((N, width), T.int32 if packed else T.int8),
        Up: T.Tensor((N, width), T.int32 if packed else T.int8),
        GateScale: T.Tensor((N,), T.float16),
        UpScale: T.Tensor((N,), T.float16),
        TokenScale: T.Tensor((M,), T.float16),
        Output: T.Tensor((M, N), T.float16),
    ):
        with T.Kernel(T.ceildiv(N, block_n), threads=128) as bx:
            if packed:
                T.import_source(
                    "\n__device__ __forceinline__ int orinfer_swiglu_dp4a(int a, int b, int c) { return __dp4a(a,b,c); }\n"
                )
                gate_acc = T.alloc_fragment((block_n, 32), T.int32)
                up_acc = T.alloc_fragment((block_n, 32), T.int32)
                gate_total = T.alloc_fragment((block_n,), T.int32)
                up_total = T.alloc_fragment((block_n,), T.int32)
                T.annotate_layout(
                    {
                        gate_acc: tilelang.Fragment(
                            (block_n, 32),
                            forward_thread_fn=lambda n, k: (n % 4) * 32 + k,
                            forward_index_fn=lambda n, k: n // 4,
                        ),
                        up_acc: tilelang.Fragment(
                            (block_n, 32),
                            forward_thread_fn=lambda n, k: (n % 4) * 32 + k,
                            forward_index_fn=lambda n, k: n // 4,
                        ),
                    }
                )
                T.clear(gate_acc)
                T.clear(up_acc)
                for kg in T.serial(K // 128):
                    for n, k in T.Parallel(block_n, 32):
                        if bx * block_n + n < N:
                            gate_acc[n, k] = T.call_pure_extern(
                                "int32",
                                "orinfer_swiglu_dp4a",
                                A[0, kg * 32 + k],
                                Gate[bx * block_n + n, kg * 32 + k],
                                gate_acc[n, k],
                            )
                            up_acc[n, k] = T.call_pure_extern(
                                "int32",
                                "orinfer_swiglu_dp4a",
                                A[0, kg * 32 + k],
                                Up[bx * block_n + n, kg * 32 + k],
                                up_acc[n, k],
                            )
                T.reduce_sum(gate_acc, gate_total, dim=1)
                T.reduce_sum(up_acc, up_total, dim=1)
                for n in T.Parallel(block_n):
                    if bx * block_n + n < N:
                        g = T.cast(
                            T.cast(
                                (
                                    T.cast(gate_total[n], T.float32)
                                    * T.cast(GateScale[bx * block_n + n], T.float32)
                                )
                                * T.cast(TokenScale[0], T.float32),
                                T.float16,
                            ),
                            T.float32,
                        )
                        u = T.cast(
                            T.cast(
                                (
                                    T.cast(up_total[n], T.float32)
                                    * T.cast(UpScale[bx * block_n + n], T.float32)
                                )
                                * T.cast(TokenScale[0], T.float32),
                                T.float16,
                            ),
                            T.float32,
                        )
                        Output[0, bx * block_n + n] = g / (1.0 + T.exp(-g)) * u
            else:
                a = T.alloc_shared((16, 128), T.int8)
                gate = T.alloc_shared((block_n, 128), T.int8)
                up = T.alloc_shared((block_n, 128), T.int8)
                gate_acc = T.alloc_fragment((16, block_n), T.int32)
                up_acc = T.alloc_fragment((16, block_n), T.int32)
                T.clear(gate_acc)
                T.clear(up_acc)
                for kg in T.Pipelined(K // 128, num_stages=2):
                    T.copy(A[0, kg * 128], a)
                    T.copy(Gate[bx * block_n, kg * 128], gate)
                    T.copy(Up[bx * block_n, kg * 128], up)
                    T.gemm(a, gate, gate_acc, transpose_B=True)
                    T.gemm(a, up, up_acc, transpose_B=True)
                for i, j in T.Parallel(16, block_n):
                    if i < M and bx * block_n + j < N:
                        g = T.cast(
                            T.cast(
                                (
                                    T.cast(gate_acc[i, j], T.float32)
                                    * T.cast(GateScale[bx * block_n + j], T.float32)
                                )
                                * T.cast(TokenScale[i], T.float32),
                                T.float16,
                            ),
                            T.float32,
                        )
                        u = T.cast(
                            T.cast(
                                (
                                    T.cast(up_acc[i, j], T.float32)
                                    * T.cast(UpScale[bx * block_n + j], T.float32)
                                )
                                * T.cast(TokenScale[i], T.float32),
                                T.float16,
                            ),
                            T.float32,
                        )
                        Output[i, bx * block_n + j] = g / (1.0 + T.exp(-g)) * u

    return main
