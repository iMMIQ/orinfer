"""FP32 online-softmax paged decode for Q24/KV4/D256, SM87.

Production arithmetic is TileLang. Caller owns all contiguous buffers, passes
the capture-current stream, and validates metadata after every update.
"""

import tilelang
import tilelang.language as T
from tools.operators.common import orin_jit


def validate_host_metadata(page_table, seq_lengths, query_positions, num_pages, block_size=128):
    """Scheduler policy: reject malformed/illegal metadata; empty attention=0.

    -1 query position is the explicit no-valid-token sentinel. Shared immutable
    prefix pages are allowed. Validate ALL pages in each declared sequence.
    """
    if any(type(v) is not int or v <= 0 for v in (num_pages, block_size)):
        raise ValueError("invalid cache dimensions")
    if (
        not page_table
        or len(page_table) != len(seq_lengths)
        or len(page_table) != len(query_positions)
    ):
        raise ValueError("empty or mismatched batch")
    width = len(page_table[0])
    if width <= 0 or any(len(row) != width for row in page_table):
        raise ValueError("nonrectangular page table")
    if width * block_size > 2147483647:
        raise ValueError("sequence capacity exceeds int32")
    for row, length, pos in zip(page_table, seq_lengths, query_positions):
        if type(length) is not int or not 0 <= length <= width * block_size:
            raise ValueError("sequence length out of bounds")
        if type(pos) is not int or pos < -1 or (pos >= length and pos != -1):
            raise ValueError("query position out of bounds")
        for page in row[: (length + block_size - 1) // block_size]:
            if type(page) is not int or not 0 <= page < num_pages:
                raise ValueError("physical page out of bounds")
    return True


def _check(max_pages, num_pages, block_size, tile_tokens, nsplits):
    if any(type(v) is not int or v <= 0 for v in (max_pages, num_pages, block_size)):
        raise ValueError("invalid cache dimensions")
    if tile_tokens not in (16, 32, 64) or nsplits not in (1, 2, 4, 8, 16):
        raise ValueError("unsupported tile/splits")
    if max_pages * block_size > 2147483647:
        raise ValueError("sequence capacity exceeds int32")


@orin_jit
def _compile(
    max_pages: int,
    num_pages: int,
    block_size: int,
    tile_tokens: int,
    nsplits: int,
    partials: bool,
    gate_mode: str = "native_fp16",
):
    batch = T.dynamic("batch")

    @T.macro
    def online(Q, K, V, Pages, SeqLen, QueryPos, b, h, split, acc, m, l):
        products = T.alloc_fragment((tile_tokens, 256), T.float32)
        scores = T.alloc_fragment((tile_tokens,), T.float32)
        probs = T.alloc_fragment((tile_tokens,), T.float32)
        maximum = T.alloc_fragment((1,), T.float32)
        new_m = T.alloc_fragment((1,), T.float32)
        alpha = T.alloc_fragment((1,), T.float32)
        tile_sum = T.alloc_fragment((1,), T.float32)
        tile_out = T.alloc_fragment((256,), T.float32)
        T.annotate_layout(
            {
                products: tilelang.Fragment(
                    (tile_tokens, 256),
                    forward_thread_fn=lambda i, d: (i % 4) * 32 + d % 32,
                    forward_index_fn=lambda i, d: (i // 4) * 8 + d // 32,
                ),
                scores: tilelang.Fragment(
                    (tile_tokens,),
                    forward_thread_fn=lambda i, rep: (i % 4) * 32 + rep,
                    forward_index_fn=lambda i: i // 4,
                    replicate=32,
                ),
                probs: tilelang.Fragment(
                    (tile_tokens,),
                    forward_thread_fn=lambda i, rep: (i % 4) * 32 + rep,
                    forward_index_fn=lambda i: i // 4,
                    replicate=32,
                ),
                tile_out: tilelang.Fragment(
                    (256,),
                    forward_thread_fn=lambda d, rep: rep * 32 + d % 32,
                    forward_index_fn=lambda d: d // 32,
                    replicate=4,
                ),
                acc: tilelang.Fragment(
                    (256,), forward_thread_fn=lambda d: d % 128, forward_index_fn=lambda d: d // 128
                ),
                m: tilelang.Fragment((1,), forward_thread_fn=lambda j, rep: rep, replicate=128),
                l: tilelang.Fragment((1,), forward_thread_fn=lambda j, rep: rep, replicate=128),
                maximum: tilelang.Fragment(
                    (1,), forward_thread_fn=lambda j, rep: rep, replicate=128
                ),
                new_m: tilelang.Fragment((1,), forward_thread_fn=lambda j, rep: rep, replicate=128),
                alpha: tilelang.Fragment((1,), forward_thread_fn=lambda j, rep: rep, replicate=128),
                tile_sum: tilelang.Fragment(
                    (1,), forward_thread_fn=lambda j, rep: rep, replicate=128
                ),
            }
        )
        T.clear(acc)
        m[0] = -T.infinity(T.float32)
        l[0] = 0.0
        valid = T.max(0, T.min(T.min(SeqLen[b], QueryPos[b] + 1), max_pages * block_size))
        width = T.ceildiv(valid, nsplits)
        start = split * width
        end = T.min(start + width, valid)
        for tile in T.serial(T.ceildiv(T.max(0, end - start), tile_tokens)):
            for i, d in T.Parallel(tile_tokens, 256):
                token = start + tile * tile_tokens + i
                products[i, d] = 0.0
                if token < end:
                    page = Pages[b, token // block_size]
                    if page >= 0 and page < num_pages:
                        products[i, d] = T.cast(Q[b, h, d], T.float32) * T.cast(
                            K[page, token % block_size, h // 6, d], T.float32
                        )
            T.reduce_sum(products, scores, dim=1)
            for i in T.Parallel(tile_tokens):
                token = start + tile * tile_tokens + i
                if token < end:
                    page = Pages[b, token // block_size]
                    if page >= 0 and page < num_pages:
                        scores[i] *= 0.0625
                    else:
                        scores[i] = -T.infinity(T.float32)
                else:
                    scores[i] = -T.infinity(T.float32)
            T.reduce_max(scores, maximum, dim=0)
            new_m[0] = T.max(m[0], maximum[0])
            alpha[0] = 0.0
            if l[0] > 0:
                alpha[0] = T.exp(m[0] - new_m[0])
            for i in T.Parallel(tile_tokens):
                probs[i] = 0.0
                if scores[i] != -T.infinity(T.float32):
                    probs[i] = T.exp(scores[i] - new_m[0])
            T.reduce_sum(probs, tile_sum, dim=0)
            for i, d in T.Parallel(tile_tokens, 256):
                token = start + tile * tile_tokens + i
                products[i, d] = 0.0
                if token < end:
                    page = Pages[b, token // block_size]
                    if page >= 0 and page < num_pages:
                        products[i, d] = probs[i] * T.cast(
                            V[page, token % block_size, h // 6, d], T.float32
                        )
            T.reduce_sum(products, tile_out, dim=0)
            for d in T.Parallel(256):
                acc[d] = acc[d] * alpha[0] + tile_out[d]
            l[0] = l[0] * alpha[0] + tile_sum[0]
            m[0] = new_m[0]

    if partials:

        @T.prim_func
        def kernel(
            Q: T.Tensor((batch, 24, 256), T.float16),
            K: T.Tensor((num_pages, block_size, 4, 256), T.float16),
            V: T.Tensor((num_pages, block_size, 4, 256), T.float16),
            Pages: T.Tensor((batch, max_pages), T.int32),
            SeqLen: T.Tensor((batch,), T.int32),
            QueryPos: T.Tensor((batch,), T.int32),
            M: T.Tensor((batch, 24, nsplits), T.float32),
            L: T.Tensor((batch, 24, nsplits), T.float32),
            O: T.Tensor((batch, 24, nsplits, 256), T.float32),
        ):
            with T.Kernel(24, nsplits, batch, threads=128) as (h, s, b):
                acc = T.alloc_fragment((256,), T.float32)
                m = T.alloc_fragment((1,), T.float32)
                l = T.alloc_fragment((1,), T.float32)
                online(Q, K, V, Pages, SeqLen, QueryPos, b, h, s, acc, m, l)
                T.copy(acc, O[b, h, s, :])
                if T.get_thread_binding() == 0:
                    M[b, h, s] = m[0]
                    L[b, h, s] = l[0]
    else:

        @T.prim_func
        def kernel(
            Q: T.Tensor((batch, 24, 256), T.float16),
            K: T.Tensor((num_pages, block_size, 4, 256), T.float16),
            V: T.Tensor((num_pages, block_size, 4, 256), T.float16),
            Pages: T.Tensor((batch, max_pages), T.int32),
            SeqLen: T.Tensor((batch,), T.int32),
            QueryPos: T.Tensor((batch,), T.int32),
            RawGate: T.Tensor((batch, 24, 256), T.float16),
            Y: T.Tensor((batch, 24, 256), T.float16),
        ):
            with T.Kernel(24, batch, threads=128) as (h, b):
                acc = T.alloc_fragment((256,), T.float32)
                m = T.alloc_fragment((1,), T.float32)
                l = T.alloc_fragment((1,), T.float32)
                online(Q, K, V, Pages, SeqLen, QueryPos, b, h, 0, acc, m, l)
                for d in T.Parallel(256):
                    Y[b, h, d] = 0.0
                    if l[0] > 0:
                        if gate_mode == "native_fp16":
                            attention = T.cast(acc[d] / l[0], T.float16)
                            gate = T.cast(
                                1.0 / (1.0 + T.exp(-T.cast(RawGate[b, h, d], T.float32))), T.float16
                            )
                            Y[b, h, d] = T.cast(attention, T.float32) * T.cast(gate, T.float32)
                        else:
                            Y[b, h, d] = (acc[d] / l[0]) / (
                                1.0 + T.exp(-T.cast(RawGate[b, h, d], T.float32))
                            )

    return kernel


def paged_attention_decode(
    max_pages, num_pages, block_size=128, tile_tokens=32, gate_mode="native_fp16"
):
    """Build dynamic B. API: Q,K,V,Pages,SeqLen,QueryPos,RawGate,Y.

    Native output = half(half(attention_fp32)*half(sigmoid(raw_gate))).
    Fully empty attention writes zero. Buffers disjoint, metadata CPU validated.
    """
    _check(max_pages, num_pages, block_size, tile_tokens, 1)
    if gate_mode not in ("native_fp16", "fp32_fused"):
        raise ValueError("unsupported gate rounding")
    return _compile(max_pages, num_pages, block_size, tile_tokens, 1, False, gate_mode)


def paged_attention_partials(max_pages, num_pages, nsplits=4, block_size=128, tile_tokens=32):
    """API Q,K,V,Pages,SeqLen,QueryPos,M,L,O; O is UNNORMALIZED FP32.

    Splits use ceil(valid/S), valid=min(seqLen,queryPos+1). No gate. Empty
    partition m=-inf,l=0,o=0. Merge and normalization belong to op28.
    """
    _check(max_pages, num_pages, block_size, tile_tokens, nsplits)
    return _compile(max_pages, num_pages, block_size, tile_tokens, nsplits, True)


@orin_jit
def _compile_gqa(
    max_pages: int, num_pages: int, block_size: int, block_n: int, nsplits: int, partials: bool
):
    """Explicit KV reuse candidate: six Q heads share each staged KV tile.

    QK and PV accumulate FP32 on FP16 tensorcores. Softmax/max/denominator are
    FP32, but PV probability operands round FP16, as in FlashAttention. This
    extra rounding is separate from the strict FP32 SIMT baseline.
    """
    batch = T.dynamic("batch")

    @T.macro
    def online(Q, K, V, Pages, SeqLen, QueryPos, b, kh, split, out, maximum, denom):
        q = T.alloc_shared((16, 256), T.float16)
        k = T.alloc_shared((block_n, 256), T.float16)
        v = T.alloc_shared((block_n, 256), T.float16)
        p = T.alloc_shared((16, block_n), T.float16)
        scores = T.alloc_fragment((16, block_n), T.float32)
        previous = T.alloc_fragment((16,), T.float32)
        correction = T.alloc_fragment((16,), T.float32)
        rowsum = T.alloc_fragment((16,), T.float32)
        T.clear(out)
        T.clear(denom)
        T.fill(maximum, -T.infinity(T.float32))
        for i, d in T.Parallel(16, 256):
            q[i, d] = 0.0
            if i < 6:
                q[i, d] = Q[b, kh * 6 + i, d]
        valid = T.max(0, T.min(T.min(SeqLen[b], QueryPos[b] + 1), max_pages * block_size))
        width = T.ceildiv(valid, nsplits)
        start = split * width
        end = T.min(start + width, valid)
        for tile in T.serial(T.ceildiv(T.max(0, end - start), block_n)):
            for j, d in T.Parallel(block_n, 256):
                token = start + tile * block_n + j
                k[j, d] = 0.0
                v[j, d] = 0.0
                if token < end:
                    page = Pages[b, token // block_size]
                    if page >= 0 and page < num_pages:
                        k[j, d] = K[page, token % block_size, kh, d]
                        v[j, d] = V[page, token % block_size, kh, d]
            T.clear(scores)
            T.gemm(q, k, scores, transpose_B=True)
            for i in T.Parallel(16):
                previous[i] = maximum[i]
            for i, j in T.Parallel(16, block_n):
                token = start + tile * block_n + j
                if i < 6 and token < end:
                    page = Pages[b, token // block_size]
                    if page >= 0 and page < num_pages:
                        scores[i, j] *= 0.0625
                    else:
                        scores[i, j] = -1e30
                else:
                    scores[i, j] = -1e30
            T.reduce_max(scores, maximum, dim=1, clear=False)
            for i, j in T.Parallel(16, block_n):
                token = start + tile * block_n + j
                if i < 6 and token < end:
                    page = Pages[b, token // block_size]
                    if page >= 0 and page < num_pages:
                        scores[i, j] = T.exp(scores[i, j] - maximum[i])
                    else:
                        scores[i, j] = 0.0
                else:
                    scores[i, j] = 0.0
                p[i, j] = scores[i, j]
            T.reduce_sum(scores, rowsum, dim=1)
            for i in T.Parallel(16):
                correction[i] = T.exp(previous[i] - maximum[i])
                denom[i] = denom[i] * correction[i] + rowsum[i]
            for i, d in T.Parallel(16, 256):
                out[i, d] *= correction[i]
            T.gemm(p, v, out)

    if partials:

        @T.prim_func
        def kernel(
            Q: T.Tensor((batch, 24, 256), T.float16),
            K: T.Tensor((num_pages, block_size, 4, 256), T.float16),
            V: T.Tensor((num_pages, block_size, 4, 256), T.float16),
            Pages: T.Tensor((batch, max_pages), T.int32),
            SeqLen: T.Tensor((batch,), T.int32),
            QueryPos: T.Tensor((batch,), T.int32),
            M: T.Tensor((batch, 24, nsplits), T.float32),
            L: T.Tensor((batch, 24, nsplits), T.float32),
            O: T.Tensor((batch, 24, nsplits, 256), T.float32),
        ):
            with T.Kernel(4, nsplits, batch, threads=128) as (kh, s, b):
                out = T.alloc_fragment((16, 256), T.float32)
                maximum = T.alloc_fragment((16,), T.float32)
                denom = T.alloc_fragment((16,), T.float32)
                online(Q, K, V, Pages, SeqLen, QueryPos, b, kh, s, out, maximum, denom)
                for i, d in T.Parallel(16, 256):
                    if i < 6:
                        O[b, kh * 6 + i, s, d] = out[i, d]
                for i in T.Parallel(16):
                    if i < 6:
                        M[b, kh * 6 + i, s] = T.if_then_else(
                            denom[i] > 0, maximum[i], -T.infinity(T.float32)
                        )
                        L[b, kh * 6 + i, s] = denom[i]
    else:

        @T.prim_func
        def kernel(
            Q: T.Tensor((batch, 24, 256), T.float16),
            K: T.Tensor((num_pages, block_size, 4, 256), T.float16),
            V: T.Tensor((num_pages, block_size, 4, 256), T.float16),
            Pages: T.Tensor((batch, max_pages), T.int32),
            SeqLen: T.Tensor((batch,), T.int32),
            QueryPos: T.Tensor((batch,), T.int32),
            RawGate: T.Tensor((batch, 24, 256), T.float16),
            Y: T.Tensor((batch, 24, 256), T.float16),
        ):
            with T.Kernel(4, batch, threads=128) as (kh, b):
                out = T.alloc_fragment((16, 256), T.float32)
                maximum = T.alloc_fragment((16,), T.float32)
                denom = T.alloc_fragment((16,), T.float32)
                online(Q, K, V, Pages, SeqLen, QueryPos, b, kh, 0, out, maximum, denom)
                for i, d in T.Parallel(16, 256):
                    if i < 6:
                        Y[b, kh * 6 + i, d] = 0.0
                        if denom[i] > 0:
                            attn = T.cast(out[i, d] / denom[i], T.float16)
                            gate = T.cast(
                                1 / (1 + T.exp(-T.cast(RawGate[b, kh * 6 + i, d], T.float32))),
                                T.float16,
                            )
                            Y[b, kh * 6 + i, d] = T.cast(attn, T.float32) * T.cast(gate, T.float32)

    return kernel


def paged_attention_decode_gqa(max_pages, num_pages, block_size=128, block_n=64):
    """Same direct API, explicit six-head KV reuse / FP16 PV operand candidate."""
    _check(max_pages, num_pages, block_size, 32, 1)
    if block_n not in (32, 64):
        raise ValueError("unsupported tensorcore tile")
    return _compile_gqa(max_pages, num_pages, block_size, block_n, 1, False)


def paged_attention_partials_gqa(max_pages, num_pages, nsplits=4, block_size=128, block_n=64):
    """Same M/L/O contract; only PV probability operands additionally round FP16."""
    _check(max_pages, num_pages, block_size, 32, nsplits)
    if block_n not in (32, 64):
        raise ValueError("unsupported tensorcore tile")
    return _compile_gqa(max_pages, num_pages, block_size, block_n, nsplits, True)
