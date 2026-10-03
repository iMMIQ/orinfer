"""SM87 full-vocabulary FP32 normalization and explicit queried scores.

No penalties/temperature transformation; teacher-forced evaluation consumes raw
logits. Partial + merge buffers are caller-owned; no Torch computation here.
"""
import tilelang.language as T
from tools.operators.common import orin_jit

VOCAB = 248320
FP32_MAX = 3.4028234663852886e38


def validate_query_ids(ids, vocab=VOCAB):
    """CPU scheduler validation before upload; duplicates are legal."""
    if type(vocab) is not int or not 0 < vocab <= 2147483647:
        raise ValueError('vocab must be positive int32')
    if not isinstance(ids, (list, tuple)) or not ids:
        raise ValueError('query IDs must be a nonempty rectangular CPU matrix')
    q = None
    for row in ids:
        if not isinstance(row, (list, tuple)) or not row:
            raise ValueError('query rows must be nonempty')
        if q is None:
            q = len(row)
        if len(row) != q:
            raise ValueError('query matrix must be rectangular')
        if any(type(i) is not int or not 0 <= i < vocab for i in row):
            raise ValueError('every query ID must be an integer in [0,vocab)')
    return tuple(tuple(row) for row in ids)


@orin_jit
def probability_partials(dtype='float16', vocab=VOCAB, chunk=4096, threads=256):
    """(Logits[M,V], Partial[M,ceil(V/chunk),3] FP32).

    Partial[...,0:3] = finite max, sum(exp(x-max)), nonfinite count. Nonfinite
    logits are replaced by zero for safe arithmetic and always flag the row.
    Tail positions contribute neither mass nor invalid counts.
    """
    assert dtype in ('float16', 'bfloat16', 'float32')
    assert vocab > 0 and chunk > 0 and chunk % threads == 0
    rows = T.dynamic('rows')
    blocks = (vocab + chunk - 1) // chunk

    @T.prim_func
    def kernel(Logits: T.Tensor((rows, vocab), dtype),
               Partial: T.Tensor((rows, blocks, 3), T.float32)):
        with T.Kernel(blocks, rows, threads=threads) as (block, row):
            values = T.alloc_fragment((chunk,), T.float32)
            mass = T.alloc_fragment((chunk,), T.float32)
            bad = T.alloc_fragment((chunk,), T.float32)
            maximum = T.alloc_fragment((1,), T.float32)
            total = T.alloc_fragment((1,), T.float32)
            invalid = T.alloc_fragment((1,), T.float32)
            for j in T.Parallel(chunk):
                col = block * chunk + j
                values[j] = -FP32_MAX
                bad[j] = 0.0
                if col < vocab:
                    values[j] = T.cast(Logits[row, col], T.float32)
                    # Explicit isfinite survives comparison canonicalization;
                    # mark before replacing value (vectorized conditional writes
                    # can otherwise re-evaluate the condition after replacement).
                    bad[j] = T.if_then_else(T.call_extern('bool', 'isfinite', values[j]), 0.0, 1.0)
                    values[j] = T.if_then_else(bad[j] > 0.0, 0.0, values[j])
            T.reduce_max(values, maximum, dim=0)
            for j in T.Parallel(chunk):
                mass[j] = 0.0
                if block * chunk + j < vocab:
                    mass[j] = T.exp(values[j] - maximum[0])
            T.reduce_sum(mass, total, dim=0)
            T.reduce_sum(bad, invalid, dim=0)
            Partial[row, block, 0] = maximum[0]
            Partial[row, block, 1] = total[0]
            Partial[row, block, 2] = invalid[0]
    return kernel


@orin_jit
def probability_merge(dtype='float16', vocab=VOCAB, chunk=4096, threads=128):
    """(Logits, IDs[M,Q] i32, Partial, LSE[M], LP[M,Q], P[M,Q], RS[M], QS[M,Q]).

    Dynamic M/Q; FP32 LSE/logprob/prob. RS=1 rejects NaN/Inf anywhere in row.
    QS bit1=invalid row, bit2=out-of-range ID, bit4=unrepresentable FP32 logprob.
    Rejected scores explicitly use (-inf,0); rejected row LSE=0 sentinel.
    Every query must inspect status. Bit4 may occur for finite FP32 logits with
    range greater than FP32_MAX. No out-of-range device read occurs.
    """
    assert dtype in ('float16', 'bfloat16', 'float32') and vocab > 0
    rows, queries = T.dynamic('rows'), T.dynamic('queries')
    blocks = (vocab + chunk - 1) // chunk
    tile = ((blocks + threads - 1) // threads) * threads

    @T.prim_func
    def kernel(Logits: T.Tensor((rows, vocab), dtype),
               IDs: T.Tensor((rows, queries), T.int32),
               Partial: T.Tensor((rows, blocks, 3), T.float32),
               LSE: T.Tensor((rows,), T.float32),
               LP: T.Tensor((rows, queries), T.float32),
               P: T.Tensor((rows, queries), T.float32),
               RS: T.Tensor((rows,), T.int32),
               QS: T.Tensor((rows, queries), T.int32)):
        with T.Kernel(rows, threads=threads) as row:
            maxima = T.alloc_fragment((tile,), T.float32)
            mass = T.alloc_fragment((tile,), T.float32)
            bad = T.alloc_fragment((tile,), T.float32)
            maximum = T.alloc_fragment((1,), T.float32)
            total = T.alloc_fragment((1,), T.float32)
            invalid = T.alloc_fragment((1,), T.float32)
            logtotal = T.alloc_fragment((1,), T.float32)
            for j in T.Parallel(tile):
                maxima[j] = -FP32_MAX
                bad[j] = 0.0
                if j < blocks:
                    maxima[j] = Partial[row, j, 0]
                    bad[j] = Partial[row, j, 2]
            T.reduce_max(maxima, maximum, dim=0)
            for j in T.Parallel(tile):
                mass[j] = 0.0
                if j < blocks:
                    mass[j] = Partial[row, j, 1] * T.exp(maxima[j] - maximum[0])
            T.reduce_sum(mass, total, dim=0)
            T.reduce_sum(bad, invalid, dim=0)
            logtotal[0] = T.log(total[0])
            RS[row] = T.if_then_else(invalid[0] > 0.0, 1, 0)
            LSE[row] = T.if_then_else(invalid[0] > 0.0, 0.0, maximum[0] + logtotal[0])
            for q in T.Parallel(queries):
                LP[row, q] = float("-inf")
                P[row, q] = 0.0
                QS[row, q] = T.if_then_else(invalid[0] > 0.0, 1, 0)
                if IDs[row, q] < 0 or IDs[row, q] >= vocab:
                    QS[row, q] = QS[row, q] + 2
                else:
                    if invalid[0] == 0.0:
                        score = (T.cast(Logits[row, IDs[row, q]], T.float32) - maximum[0]) - logtotal[0]
                        if T.abs(score) <= FP32_MAX:
                            LP[row, q] = score
                            P[row, q] = T.exp(score)
                        else:
                            QS[row, q] = 4
    return kernel


def launch(partials, merge, logits, ids, partial, lse, logprobs, probs,
           row_status, query_status, *, stream):
    """Two ordered launches on explicit current/capture stream; no allocation.

    All buffers disjoint, contiguous and correctly typed. Shapes remain fixed
    for each captured graph; addresses stable, input content may change.
    """
    partials(logits, partial, stream=stream)
    merge(logits, ids, partial, lse, logprobs, probs, row_status, query_status, stream=stream)
