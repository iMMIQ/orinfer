"""Exact FP32 sparse attention arithmetic on A2 Vector cores.

The row gather is a separate common kernel. This bounded implementation consumes
the union of selected compressed KV rows and SWA rows; request/cache ownership
and causal selection remain the model adapter's responsibility.
"""

import tilelang.language as T


def sparse_attention(rows, heads, selected, dimension=512, scale=None):
    block = 16
    if selected % block or dimension % 32:
        raise ValueError("Sparse attention requires S%16=0 and D%32=0")
    scale = dimension**-0.5 if scale is None else scale

    @T.prim_func
    def main(
        query: T.Tensor((rows, heads, dimension), "bfloat16"),
        gathered_kv: T.Tensor((rows, selected, dimension), "bfloat16"),
        indices: T.Tensor((rows, selected), "int32"),
        sink: T.Tensor((heads,), "float32"),
        out: T.Tensor((rows, heads, dimension), "bfloat16"),
    ):
        with T.Kernel(T.ceildiv(rows * heads, 2), is_npu=True) as (cid, vid):
            qb = T.alloc_ub((1, dimension), "bfloat16")
            q = T.alloc_ub((1, dimension), "float32")
            kb = T.alloc_ub((block, dimension), "bfloat16")
            k = T.alloc_ub((block, dimension), "float32")
            product = T.alloc_ub((block, dimension), "float32")
            ids = T.alloc_ub((block,), "int32")
            scores = T.alloc_ub((block,), "float32")
            # Compare/select operate on 256-byte vectors. Keep the arithmetic
            # at 16 scores and pad only the mask operation to 64 FP32 lanes.
            ids_float = T.alloc_ub((64,), "float32")
            mask = T.alloc_ub((8,), "uint8")
            masked_scores = T.alloc_ub((64,), "float32")
            probabilities = T.alloc_ub((block, 1), "float32")
            maximum = T.alloc_ub((1, 1), "float32")
            block_maximum = T.alloc_ub((1, 1), "float32")
            updated_maximum = T.alloc_ub((1, 1), "float32")
            alpha = T.alloc_ub((1, 1), "float32")
            denominator = T.alloc_ub((1, 1), "float32")
            block_sum = T.alloc_ub((1, 1), "float32")
            accumulator = T.alloc_ub((1, dimension), "float32")
            block_output = T.alloc_ub((1, dimension), "float32")
            y = T.alloc_ub((1, dimension), "bfloat16")
            index = cid * 2 + vid
            row = index // heads
            head = index % heads
            if row < rows:
                T.copy(query[row, head, :], qb)
                T.tile.cast(q, qb, "CAST_NONE", dimension)
                # Include the zero-value sink from the start. This also makes
                # all-masked rows well defined without inf-inf or host fallback.
                maximum[0, 0] = sink[head]
                T.tile.fill(denominator, 1.0)
                T.tile.fill(accumulator, 0.0)
                for start in T.serial(selected // block):
                    T.copy(gathered_kv[row, start * block : (start + 1) * block, :], kb)
                    T.copy(indices[row, start * block : (start + 1) * block], ids)
                    T.tile.cast(k, kb, "CAST_NONE", block * dimension)
                    T.tile.broadcast(product, q)
                    T.tile.mul(product, product, k)
                    T.reduce_sum(product, scores, dim=1)
                    T.tile.mul(scores, scores, scale)
                    T.tile.fill(ids_float, -1.0)
                    T.tile.cast(ids_float, ids, "CAST_NONE", block)
                    T.tile.compare(mask, ids_float, -1.0, "GT")
                    T.tile.fill(masked_scores, -T.infinity("float32"))
                    T.copy(scores, masked_scores[:block])
                    T.tile.select(
                        masked_scores,
                        mask,
                        masked_scores,
                        -T.infinity("float32"),
                        "VSEL_TENSOR_SCALAR_MODE",
                    )
                    T.copy(masked_scores[:block], scores)
                    T.reduce_max(scores, block_maximum[0, :], dim=0)
                    T.tile.max(updated_maximum, maximum, block_maximum)
                    T.tile.sub(alpha, maximum, updated_maximum)
                    T.tile.exp(alpha, alpha)
                    T.tile.sub(scores, scores, updated_maximum[0, 0])
                    T.tile.exp(scores, scores)
                    T.reduce_sum(scores, block_sum[0, :], dim=0)
                    T.tile.mul(denominator, denominator, alpha)
                    T.tile.add(denominator, denominator, block_sum)
                    T.copy(scores[:], probabilities[:, 0])
                    T.tile.broadcast(product, probabilities)
                    T.tile.mul(product, product, k)
                    T.reduce_sum(product, block_output, dim=0)
                    T.tile.mul(accumulator, accumulator, alpha[0, 0])
                    T.tile.add(accumulator, accumulator, block_output)
                    maximum[0, 0] = updated_maximum[0, 0]
                T.tile.div(accumulator, accumulator, denominator[0, 0])
                T.tile.cast(y, accumulator, "CAST_RINT", dimension)
                T.copy(y, out[row, head, :])

    return main
