"""Independent FP32 SIMT attention oracle for offline tuning and validation."""

import tilelang.language as T
from orinfer_build.compiler.cuda.common import orin_jit
from kernels.cuda.model.kv_int8 import HALF2_SOURCE, int8_kv_tile


@orin_jit
def attention_prefill_int8(
    batch: int,
    tq: int,
    tkv: int,
    input_layout: str = "token_major",
    output_layout: str = "token_major",
    kv_layout: str = "token_major",
    gate_mode: str = "native_fp16",
    block_m: int = 32,
    block_n: int = 32,
    threads: int = 128,
    exp_mode: str = "precise",
    dynamic_rows: bool = False,
    query_heads: int = 24,
    kv_heads: int = 4,
    dimension: int = 256,
    gated: bool = True,
):
    """Build (Q,K,V,RawGate,Positions,Lengths,Y), all explicit tensors.

    Q/G/Y token_major [B,Tq,24,256] or head_major [B,24,Tq,256].
    K/V token_major INT8 [B,Tkv,4,256], FP16 group-64 scales.
    Metadata int32 [B,Tq], [B]. FP16 compute, scale=1/16, qh//6 GQA.
    native_fp16 rounds sigmoid and normalized attention before multiplication.
    fp32_fused is a separately identified candidate with only output rounding.
    """
    assert (batch is None or batch > 0) and tq > 0 and tkv > 0
    if dynamic_rows:
        if batch != 1:
            raise ValueError("Compact dynamic attention rows require one causal segment")
        from kernels.shared.rows import row_count

        tq = row_count(tq, True)
    batch = T.dynamic("batch") if batch is None else batch
    assert input_layout in ("token_major", "head_major")
    assert output_layout in ("token_major", "head_major")
    assert kv_layout == "token_major"
    assert gate_mode in ("native_fp16", "fp32_fused")
    assert block_m in (16, 32, 64, 128) and block_n in (32, 64)
    assert exp_mode in ("precise", "fast")

    @T.macro
    def probability_exp(value):
        return T.call_extern("float32", "__expf", value) if exp_mode == "fast" else T.exp(value)

    qs = (
        (batch, tq, query_heads * dimension)
        if input_layout == "token_major"
        else (batch, query_heads, tq, dimension)
    )
    ys = (
        (batch, tq, query_heads, dimension)
        if output_layout == "token_major"
        else (batch, query_heads, tq, dimension)
    )
    ks = (
        (batch, tkv, kv_heads * (dimension // 4))
        if kv_layout == "token_major"
        else (batch, kv_heads, tkv, dimension)
    )

    load_kv = int8_kv_tile(block_n, dimension, threads=threads)

    @T.prim_func
    def kernel(
        Q: T.Tensor(qs, T.float16),
        K: T.Tensor(ks, T.uint32),
        V: T.Tensor(ks, T.uint32),
        Gate: T.Tensor(qs, T.float16),
        Positions: T.Tensor((batch, tq), T.int32),
        Lengths: T.Tensor((batch,), T.int32),
        Y: T.Tensor(ys, T.float16),
        KS: T.Tensor((batch, tkv, kv_heads * (dimension // 64)), T.float16),
        VS: T.Tensor((batch, tkv, kv_heads * (dimension // 64)), T.float16),
    ):
        with T.Kernel(T.ceildiv(tq, block_m), query_heads, batch, threads=threads) as (qb, h, b):
            T.import_source(HALF2_SOURCE)
            q = T.alloc_shared((block_m, dimension), T.float16)
            k = T.alloc_shared((block_n, dimension), T.float16)
            v = T.alloc_shared((block_n, dimension), T.float16)
            p = T.alloc_shared((block_m, block_n), T.float16)
            score = T.alloc_fragment((block_m, block_n), T.float32)
            out = T.alloc_fragment((block_m, dimension), T.float32)
            maximum = T.alloc_fragment((block_m,), T.float32)
            previous = T.alloc_fragment((block_m,), T.float32)
            correction = T.alloc_fragment((block_m,), T.float32)
            denom = T.alloc_fragment((block_m,), T.float32)
            rowsum = T.alloc_fragment((block_m,), T.float32)
            pos = T.alloc_fragment((block_m,), T.int32)
            maxpos = T.alloc_fragment((1,), T.int32)
            T.fill(maximum, -1e30)
            T.clear(denom)
            T.clear(out)
            for i in T.Parallel(block_m):
                pos[i] = T.if_then_else(qb * block_m + i < tq, Positions[b, qb * block_m + i], -1)
            T.reduce_max(pos, maxpos, dim=0)
            if input_layout == "token_major":
                T.copy(Q[b, qb * block_m, h * dimension], q, prefer_instruction="sync")
            else:
                T.copy(Q[b, h, qb * block_m, 0], q, prefer_instruction="sync")
            for kb in T.serial(T.ceildiv(T.max(0, T.min(Lengths[b], maxpos[0] + 1)), block_n)):
                load_kv(
                    K,
                    V,
                    KS,
                    VS,
                    k,
                    v,
                    b,
                    h // (query_heads // kv_heads),
                    kb * block_n,
                    T.min(Lengths[b], tkv),
                )
                T.sync_threads()
                T.clear(score)
                T.gemm(q, k, score, transpose_B=True)
                for i in T.Parallel(block_m):
                    previous[i] = maximum[i]
                for i, j in T.Parallel(block_m, block_n):
                    score[i, j] = T.if_then_else(
                        kb * block_n + j < Lengths[b] and kb * block_n + j <= pos[i],
                        score[i, j] * (dimension**-0.5),
                        -1e30,
                    )
                T.reduce_max(score, maximum, dim=1, clear=False)
                for i, j in T.Parallel(block_m, block_n):
                    score[i, j] = T.if_then_else(
                        kb * block_n + j < Lengths[b] and kb * block_n + j <= pos[i],
                        probability_exp(score[i, j] - maximum[i]),
                        0,
                    )
                    p[i, j] = score[i, j]
                T.reduce_sum(score, rowsum, dim=1)
                for i in T.Parallel(block_m):
                    correction[i] = probability_exp(previous[i] - maximum[i])
                    denom[i] = denom[i] * correction[i] + rowsum[i]
                for i, d in T.Parallel(block_m, dimension):
                    out[i, d] *= correction[i]
                T.gemm(p, v, out)
            for i, d in T.Parallel(block_m, dimension):
                if qb * block_m + i < tq:
                    if input_layout == "token_major":
                        raw = T.cast(Gate[b, qb * block_m + i, h * dimension + d], T.float32)
                    else:
                        raw = T.cast(Gate[b, h, qb * block_m + i, d], T.float32)
                    sig = 1 / (1 + T.exp(-raw)) if gated else 1.0
                    attn = T.if_then_else(denom[i] > 0, out[i, d] / T.max(denom[i], 1e-30), 0)
                    if gate_mode == "native_fp16":
                        result = T.cast(T.cast(attn, T.float16), T.float32) * T.cast(
                            T.cast(sig, T.float16), T.float32
                        )
                    else:
                        result = attn * sig
                    if output_layout == "token_major":
                        Y[b, qb * block_m + i, h, d] = result
                    else:
                        Y[b, h, qb * block_m + i, d] = result

    return kernel
