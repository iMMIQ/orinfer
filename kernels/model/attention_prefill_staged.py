"""Vector Q/K/V staging; same layouts, math and state as op21.

Flattened declarations expose identical token-major physical storage.
"""

import tilelang.language as T
from tools.operators.common import orin_jit


def validate_metadata(positions, lengths, tkv):
    """CPU scheduler contract: empty rows are legal and produce zero.

    Positions may be unsorted. -1 denotes a padded/empty query. KV positions
    are contiguous absolute positions 0..length-1, not chunk-relative indices.
    """
    if len(positions) != len(lengths) or not positions:
        raise ValueError("metadata batch mismatch")
    width = len(positions[0])
    if width < 1 or any(len(row) != width for row in positions):
        raise ValueError("positions must be nonempty rectangular")
    for row, length in zip(positions, lengths):
        if type(length) is not int or not 0 <= length <= tkv:
            raise ValueError("length outside contiguous KV allocation")
        if any(type(p) is not int or p < -1 or p > 2147483646 for p in row):
            raise ValueError("invalid absolute query position")


@orin_jit
def attention_prefill_staged(
    batch: int,
    tq: int,
    tkv: int,
    input_layout: str = "token_major",
    output_layout: str = "token_major",
    kv_layout: str = "head_major",
    gate_mode: str = "native_fp16",
    block_m: int = 32,
    block_n: int = 32,
    threads: int = 128,
    exp_mode: str = "precise",
    num_stages: int = 0,
    interior_mask: bool = False,
    contiguous_queries: bool = False,
):
    """Build (Q,K,V,RawGate,Positions,Lengths,Y), all explicit tensors.

    Q/G/Y token_major [B,Tq,24,256] or head_major [B,24,Tq,256].
    K/V head_major [B,4,Tkv,256] or token_major [B,Tkv,4,256].
    Metadata int32 [B,Tq], [B]. FP16 data, scale=1/16, qh//6 GQA.
    native_fp16 rounds sigmoid and normalized attention before multiplication.
    fp32_fused is a separately identified candidate with only output rounding.
    contiguous_queries requires nondecreasing positions within each query tile.
    Async stages require K/V zero-filled from Lengths through the next block_n
    boundary; dequant_prefill_kv(pad_to=block_n) provides that padding.
    """
    assert (batch is None or batch > 0) and tq > 0 and tkv > 0
    batch = T.dynamic("batch") if batch is None else batch
    assert input_layout in ("token_major", "head_major")
    assert output_layout in ("token_major", "head_major")
    assert kv_layout in ("token_major", "head_major")
    assert gate_mode in ("native_fp16", "fp32_fused")
    assert block_m in (16, 32, 64, 128) and block_n in (32, 64)
    assert exp_mode in ("precise", "fast")
    assert num_stages in (0, 1, 2)
    assert num_stages == 0 or contiguous_queries
    assert num_stages == 0 or tkv % block_n == 0

    @T.macro
    def probability_exp(value):
        return T.call_extern("float32", "__expf", value) if exp_mode == "fast" else T.exp(value)

    @T.macro
    def consume(
        q,
        k,
        v,
        p,
        score,
        out,
        maximum,
        previous,
        correction,
        denom,
        rowsum,
        pos,
        minpos,
        kb,
        length,
    ):
        T.sync_threads()
        T.clear(score)
        T.gemm(q, k, score, transpose_B=True)
        for i in T.Parallel(block_m):
            previous[i] = maximum[i]
        if interior_mask and (kb + 1) * block_n <= T.min(length, minpos[0] + 1):
            for i, j in T.Parallel(block_m, block_n):
                score[i, j] *= 0.0625
        else:
            for i, j in T.Parallel(block_m, block_n):
                score[i, j] = T.if_then_else(
                    kb * block_n + j < length and kb * block_n + j <= pos[i],
                    score[i, j] * 0.0625,
                    -1e30,
                )
        T.reduce_max(score, maximum, dim=1, clear=False)
        if interior_mask and (kb + 1) * block_n <= T.min(length, minpos[0] + 1):
            for i, j in T.Parallel(block_m, block_n):
                score[i, j] = probability_exp(score[i, j] - maximum[i])
                p[i, j] = score[i, j]
        else:
            for i, j in T.Parallel(block_m, block_n):
                score[i, j] = T.if_then_else(
                    kb * block_n + j < length and kb * block_n + j <= pos[i],
                    probability_exp(score[i, j] - maximum[i]),
                    0,
                )
                p[i, j] = score[i, j]
        T.reduce_sum(score, rowsum, dim=1)
        for i in T.Parallel(block_m):
            correction[i] = probability_exp(previous[i] - maximum[i])
            denom[i] = denom[i] * correction[i] + rowsum[i]
        for i, d in T.Parallel(block_m, 256):
            out[i, d] *= correction[i]
        T.gemm(p, v, out)

    qs = (batch, tq, 6144) if input_layout == "token_major" else (batch, 24, tq, 256)
    ys = (batch, tq, 24, 256) if output_layout == "token_major" else (batch, 24, tq, 256)
    ks = (batch, tkv, 1024) if kv_layout == "token_major" else (batch, 4, tkv, 256)

    @T.prim_func
    def kernel(
        Q: T.Tensor(qs, T.float16),
        K: T.Tensor(ks, T.float16),
        V: T.Tensor(ks, T.float16),
        Gate: T.Tensor(qs, T.float16),
        Positions: T.Tensor((batch, tq), T.int32),
        Lengths: T.Tensor((batch,), T.int32),
        Y: T.Tensor(ys, T.float16),
    ):
        with T.Kernel(T.ceildiv(tq, block_m), 24, batch, threads=threads) as (qb, h, b):
            q = T.alloc_shared((block_m, 256), T.float16)
            k = T.alloc_shared((block_n, 256), T.float16)
            v = T.alloc_shared((block_n, 256), T.float16)
            p = T.alloc_shared((block_m, block_n), T.float16)
            score = T.alloc_fragment((block_m, block_n), T.float32)
            out = T.alloc_fragment((block_m, 256), T.float32)
            maximum = T.alloc_fragment((block_m,), T.float32)
            previous = T.alloc_fragment((block_m,), T.float32)
            correction = T.alloc_fragment((block_m,), T.float32)
            denom = T.alloc_fragment((block_m,), T.float32)
            rowsum = T.alloc_fragment((block_m,), T.float32)
            pos = T.alloc_fragment((block_m,), T.int32)
            maxpos = T.alloc_fragment((1,), T.int32)
            minpos = T.alloc_fragment((1,), T.int32)
            T.fill(maximum, -1e30)
            T.clear(denom)
            T.clear(out)
            for i in T.Parallel(block_m):
                pos[i] = T.if_then_else(qb * block_m + i < tq, Positions[b, qb * block_m + i], -1)
            T.reduce_max(pos, maxpos, dim=0)
            T.reduce_min(pos, minpos, dim=0)
            if input_layout == "token_major":
                T.copy(Q[b, qb * block_m, h * 256], q, prefer_instruction="sync")
            else:
                T.copy(Q[b, h, qb * block_m, 0], q, prefer_instruction="sync")
            # Pipelined variants require zero-filled K/V through the final
            # block_n boundary (dequant_prefill_kv(pad_to=block_n) contract).
            for kb in T.Pipelined(
                T.ceildiv(
                    T.max(
                        0,
                        T.min(
                            Lengths[b],
                            Positions[b, T.min(tq - 1, (qb + 1) * block_m - 1)] + 1
                            if contiguous_queries
                            else maxpos[0] + 1,
                        ),
                    ),
                    block_n,
                ),
                num_stages=num_stages,
            ):
                if num_stages > 0:
                    if kv_layout == "token_major":
                        T.copy(K[b, kb * block_n, (h // 6) * 256], k, prefer_instruction="cp_async")
                        T.copy(V[b, kb * block_n, (h // 6) * 256], v, prefer_instruction="cp_async")
                    else:
                        T.copy(K[b, h // 6, kb * block_n, 0], k, prefer_instruction="cp_async")
                        T.copy(V[b, h // 6, kb * block_n, 0], v, prefer_instruction="cp_async")
                elif (kb + 1) * block_n <= Lengths[b] and (kb + 1) * block_n <= tkv:
                    if kv_layout == "token_major":
                        T.copy(K[b, kb * block_n, (h // 6) * 256], k, prefer_instruction="sync")
                        T.copy(V[b, kb * block_n, (h // 6) * 256], v, prefer_instruction="sync")
                    else:
                        T.copy(K[b, h // 6, kb * block_n, 0], k, prefer_instruction="sync")
                        T.copy(V[b, h // 6, kb * block_n, 0], v, prefer_instruction="sync")
                else:
                    for j, d in T.Parallel(block_n, 256):
                        if kb * block_n + j < Lengths[b] and kb * block_n + j < tkv:
                            if kv_layout == "token_major":
                                k[j, d] = K[b, kb * block_n + j, (h // 6) * 256 + d]
                                v[j, d] = V[b, kb * block_n + j, (h // 6) * 256 + d]
                            else:
                                k[j, d] = K[b, h // 6, kb * block_n + j, d]
                                v[j, d] = V[b, h // 6, kb * block_n + j, d]
                        else:
                            k[j, d] = 0.0
                            v[j, d] = 0.0
                consume(
                    q,
                    k,
                    v,
                    p,
                    score,
                    out,
                    maximum,
                    previous,
                    correction,
                    denom,
                    rowsum,
                    pos,
                    minpos,
                    kb,
                    Lengths[b],
                )
            for i, d in T.Parallel(block_m, 256):
                if qb * block_m + i < tq:
                    if input_layout == "token_major":
                        raw = T.cast(Gate[b, qb * block_m + i, h * 256 + d], T.float32)
                    else:
                        raw = T.cast(Gate[b, h, qb * block_m + i, d], T.float32)
                    sig = 1 / (1 + T.exp(-raw))
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
