"""SM87 GDN chunk output, explicit FP32 SIMT math and caller-owned buffers.

Q has 16 heads, other operands have 48; kh=vh//3 avoids head expansion.
QK already contains q_scale. Only the entering-state term scales Q.
"""
import math

import tilelang
import tilelang.language as T

HK, HV, DK, DV = 16, 48, 128, 128


@tilelang.jit(out_idx=[], execution_backend="nvrtc",
              target={"kind": "cuda", "arch": "sm_87"})
def _compile(bt: int, q_scale: float, q_dtype: str, output_dtype: str,
             token_tile: int, value_tile: int, output_layout: str):
    batch, chunks = T.dynamic("batch"), T.dynamic("chunks")
    y_shape = ((batch, HV, chunks, bt, DV) if output_layout == "headmajor"
               else (batch, chunks * bt, HV, DV))

    @T.prim_func
    def main(Q: T.Tensor((batch, HK, chunks, bt, DK), q_dtype),
             G: T.Tensor((batch, HV, chunks, bt), "float32"),
             QK: T.Tensor((batch, HV, chunks, bt, bt), "float32"),
             Senter: T.Tensor((batch, HV, chunks, DK, DV), "float32"),
             R: T.Tensor((batch, HV, chunks, bt, DV), "float32"),
             Y: T.Tensor(y_shape, output_dtype)):
        # Flatten the original value/token/BHC CTA order into gridX. CUDA
        # gridY/Z are capped at 65535, which B*48*C can exceed for BT16.
        with T.Kernel(batch * HV * chunks * (DV // value_tile) * (bt // token_tile),
                      threads=128) as block:
            bv = block % (DV // value_tile)
            br = (block // (DV // value_tile)) % (bt // token_tile)
            bhc = block // ((DV // value_tile) * (bt // token_tile))
            b = bhc // (HV * chunks)
            h, c = (bhc // chunks) % HV, bhc % chunks
            qs = T.alloc_shared((token_tile, DK), "float32")
            state = T.alloc_shared((DK, value_tile), "float32")
            residual = T.alloc_shared((bt, value_tile), "float32")
            qk = T.alloc_shared((token_tile, bt), "float32")
            entering_sum = T.alloc_local((1,), "float32")
            within_sum = T.alloc_local((1,), "float32")
            for i, k in T.Parallel(token_tile, DK):
                qs[i, k] = (T.cast(Q[b, h // 3, c, br * token_tile + i, k], "float32")
                            * q_scale) * T.exp(G[b, h, c, br * token_tile + i])
            for k, j in T.Parallel(DK, value_tile):
                state[k, j] = Senter[b, h, c, k, bv * value_tile + j]
            for t, j in T.Parallel(bt, value_tile):
                residual[t, j] = R[b, h, c, t, bv * value_tile + j]
            for i, t in T.Parallel(token_tile, bt):
                qk[i, t] = QK[b, h, c, br * token_tile + i, t]
            T.sync_threads()
            # Each output has one owner. No scalar thread-zero guard can
            # reverse-infer a serialized fragment layout for these loops.
            for i, j in T.Parallel(token_tile, value_tile):
                entering_sum[0] = 0.0
                within_sum[0] = 0.0
                for k in T.serial(DK):
                    entering_sum[0] += qs[i, k] * state[k, j]
                for t in T.serial(bt):
                    within_sum[0] += qk[i, t] * residual[t, j]
                if output_layout == "headmajor":
                    Y[b, h, c, br * token_tile + i, bv * value_tile + j] = (
                        entering_sum[0] + within_sum[0])
                else:
                    Y[b, c * bt + br * token_tile + i, h, bv * value_tile + j] = (
                        entering_sum[0] + within_sum[0])
    return main


def gdn_chunk_output(*, q_scale, bt=64, q_dtype="float16",
                     output_dtype="float16", token_tile=8, value_tile=16,
                     output_layout="headmajor"):
    """Compile positive dynamic B/C; contiguous logical tensors, no workspace.

    Q: [B,16,C,BT,128], G: FP32[B,48,C,BT], QK: FP32[B,48,C,BT,BT],
    Senter: FP32[B,48,C,128,128] in [K,V] order, R/Y: [B,48,C,BT,128].
    G is cumulative natural-log decay. Invalid Q/QK/R rows are zero.
    q_scale is required: 128**-.5 for unscaled Q; 1 for pre-scaled FP32 Q.
    All products, exp and sums are FP32 SIMT. FP16 Y rounds only at store.
    """
    if type(bt) is not int or bt not in (16, 32, 64):
        raise ValueError("BT must be 16, 32 or 64")
    if q_dtype not in ("float16", "float32") or output_dtype not in ("float16", "float32"):
        raise ValueError("Q/Y dtype must be float16 or float32")
    if isinstance(q_scale, bool) or not isinstance(q_scale, (int, float)) or not math.isfinite(q_scale) or q_scale <= 0:
        raise ValueError("q_scale must be an explicit finite positive number")
    if (token_tile, value_tile) not in ((4, 32), (8, 16), (16, 8)):
        raise ValueError("unsupported 128-thread output tile")
    if output_layout not in ("headmajor", "tokenmajor"):
        raise ValueError("output_layout must be headmajor or tokenmajor")
    kernel = _compile(bt, float(q_scale), q_dtype, output_dtype, token_tile, value_tile, output_layout)
    kernel.adapter.kernels = dict(kernel.adapter.kernels)
    kernel.op16_spec = (bt, q_dtype, output_dtype, output_layout)
    return kernel


def launch(kernel, q, cumulative_g, causal_qk, entering_states, residuals,
           output, *, stream):
    """Validate metadata, then launch on the explicit caller/capture stream."""
    bt, q_dtype, y_dtype, layout = kernel.op16_spec
    tensors = (q, cumulative_g, causal_qk, entering_states, residuals, output)
    assert all(x.is_contiguous() and x.is_cuda for x in tensors), "all operands must be CUDA contiguous"
    assert all(x.device == q.device for x in tensors), "all operands must share a CUDA device"
    b, hk, c, qb, dk = q.shape
    assert b > 0 and c > 0 and (hk, qb, dk) == (HK, bt, DK)
    y_shape = (b, HV, c, bt, DV) if layout == "headmajor" else (b, c * bt, HV, DV)
    shapes = ((b, HV, c, bt), (b, HV, c, bt, bt),
              (b, HV, c, DK, DV), (b, HV, c, bt, DV), y_shape)
    assert all(tuple(x.shape) == shape for x, shape in zip(tensors[1:], shapes))
    dtypes = (q_dtype, "float32", "float32", "float32", "float32", y_dtype)
    assert all(str(x.dtype) == "torch." + dtype for x, dtype in zip(tensors, dtypes))
    ranges = sorted((x.data_ptr(), x.data_ptr() + x.numel() * x.element_size()) for x in tensors)
    assert all(left[1] <= right[0] for left, right in zip(ranges, ranges[1:])), "operands must not alias"
    return kernel.adapter.func(*tensors, stream=stream)
