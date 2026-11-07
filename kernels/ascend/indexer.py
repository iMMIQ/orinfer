"""Paged sparse-index logits: sum_h weight_h * relu(q_h @ key), in FP32."""

import tilelang.language as T


def logits(rows, length, table_rows, heads=4, dimension=128, scale=1.0):
    block = 16
    blocks = (length + block - 1) // block
    weight_stride = (heads + 7) // 8 * 8

    @T.prim_func
    def main(
        query: T.Tensor((rows, heads, dimension), "bfloat16"),
        keys: T.Tensor((table_rows, dimension), "bfloat16"),
        weights: T.Tensor((rows, weight_stride), "float32"),
        indices: T.Tensor((rows, length), "int32"),
        out: T.Tensor((rows, length), "float32"),
        errors: T.Tensor((rows, blocks), "int32"),
    ):
        with T.Kernel(T.ceildiv(rows * blocks, 2), is_npu=True) as (cid, vid):
            qb = T.alloc_ub((heads, dimension), "bfloat16")
            q = T.alloc_ub((heads, dimension), "float32")
            kb = T.alloc_ub((block, dimension), "bfloat16")
            k = T.alloc_ub((block, dimension), "float32")
            product = T.alloc_ub((block, dimension), "float32")
            w = T.alloc_ub((weight_stride,), "float32")
            ids = T.alloc_ub((block,), "int32")
            partial = T.alloc_ub((block,), "float32")
            total = T.alloc_ub((block,), "float32")
            flag = T.alloc_ub((8,), "int32")
            index = cid * 2 + vid
            row = index // blocks
            tile = index % blocks
            if row < rows:
                start = tile * block
                valid = T.min(block, length - start)
                T.copy(query[row, :, :], qb)
                T.copy(weights[row, :], w)
                T.tile.fill(ids, -1)
                T.copy(indices[row, start : start + valid], ids[:valid])
                T.tile.fill(kb, 0.0)
                flag[0] = 0
                for i in T.serial(block):
                    slot = ids[i]
                    if slot >= 0 and slot < table_rows:
                        T.copy(keys[slot, :], kb[i, :])
                    else:
                        if slot < -1 or slot >= table_rows:
                            flag[0] = 1
                T.tile.cast(q, qb, "CAST_NONE", heads * dimension)
                T.tile.cast(k, kb, "CAST_NONE", block * dimension)
                T.tile.fill(total, 0.0)
                for head in T.serial(heads):
                    T.tile.broadcast(product, q[head : head + 1, :])
                    T.tile.mul(product, product, k)
                    T.reduce_sum(product, partial, dim=1)
                    T.tile.mul(partial, partial, scale)
                    T.tile.max(partial, partial, 0.0)
                    T.tile.mul(partial, partial, w[head])
                    T.tile.add(total, total, partial)
                for i in T.serial(block):
                    if ids[i] < 0 or ids[i] >= table_rows:
                        total[i] = -T.infinity("float32")
                T.copy(total[:valid], out[row, start : start + valid])
                T.copy(flag[:1], errors[row, tile : tile + 1])

    return main


def quantized_logits(rows, length, table_rows, heads=32, dimension=128):
    """QLI arithmetic: exact INT8 dot, FP16 Q/K scales and scaled FP16 weights."""
    block = 16
    blocks = (length + block - 1) // block
    weights_scale = dimension**-0.5 * heads**-0.5

    @T.prim_func
    def main(
        query: T.Tensor((rows, heads, dimension), "int8"),
        query_scale: T.Tensor((rows, heads), "float16"),
        keys: T.Tensor((table_rows, dimension), "int8"),
        key_scale: T.Tensor((table_rows,), "float16"),
        weights: T.Tensor((rows, heads), "bfloat16"),
        indices: T.Tensor((rows, length), "int32"),
        out: T.Tensor((rows, length), "float32"),
    ):
        with T.Kernel(T.ceildiv(rows * blocks, 2), is_npu=True) as (cid, vid):
            qb = T.alloc_ub((heads, dimension), "int8")
            qh = T.alloc_ub((heads, dimension), "float16")
            q = T.alloc_ub((heads, dimension), "float32")
            kb = T.alloc_ub((block, dimension), "int8")
            kb_zero = T.alloc_ub((block, dimension // 2), "int16")
            kh = T.alloc_ub((block, dimension), "float16")
            k = T.alloc_ub((block, dimension), "float32")
            product = T.alloc_ub((block, dimension), "float32")
            wb = T.alloc_ub((heads,), "bfloat16")
            wh = T.alloc_ub((heads,), "float16")
            w = T.alloc_ub((heads,), "float32")
            qsb = T.alloc_ub((heads,), "float16")
            qs = T.alloc_ub((heads,), "float32")
            ksb = T.alloc_ub((block,), "float16")
            ks = T.alloc_ub((block,), "float32")
            ids = T.alloc_ub((block,), "int32")
            ids_float = T.alloc_ub((1, block), "float32")
            last_id = T.alloc_ub((1, 1), "float32")
            partial = T.alloc_ub((block,), "float32")
            total = T.alloc_ub((block,), "float32")
            index = cid * 2 + vid
            row = index // blocks
            tile = index % blocks
            if row < rows:
                start = tile * block
                valid = T.min(block, length - start)
                T.tile.fill(ids, -1)
                T.copy(indices[row, start : start + valid], ids[:valid])
                T.tile.cast(ids_float, ids, "CAST_NONE", block)
                T.reduce_max(ids_float, last_id, dim=1)
                # A completely masked tile has no score computation. Retain
                # the original arithmetic for every tile with a real index.
                if last_id[0, 0] >= 0.0:
                    T.copy(query[row, :, :], qb)
                    T.copy(query_scale[row, :], qsb)
                    T.copy(weights[row, :], wb)
                    # A2 has no INT8 Duplicate. Clear the FP16 alias, then copy only
                    # real keys; invalid positions never participate in a score.
                    T.reinterpretcast(kb, kb_zero, "int8_t")
                    T.tile.fill(kb_zero, 0)
                    for i in T.serial(block):
                        slot = ids[i]
                        if slot >= 0 and slot < table_rows:
                            T.copy(keys[slot, :], kb[i, :])
                            ksb[i] = key_scale[slot]
                        else:
                            ksb[i] = T.float16(0.0)
                    T.tile.cast(qh, qb, "CAST_NONE", heads * dimension)
                    T.tile.cast(q, qh, "CAST_NONE", heads * dimension)
                    T.tile.cast(kh, kb, "CAST_NONE", block * dimension)
                    T.tile.cast(k, kh, "CAST_NONE", block * dimension)
                    T.tile.cast(qs, qsb, "CAST_NONE", heads)
                    T.tile.cast(ks, ksb, "CAST_NONE", block)
                    T.tile.cast(w, wb, "CAST_NONE", heads)
                    T.tile.mul(w, w, weights_scale)
                    T.tile.cast(wh, w, "CAST_RINT", heads)
                    T.tile.cast(w, wh, "CAST_NONE", heads)
                    T.tile.fill(total, 0.0)
                    for head in T.serial(heads):
                        query_factor = qs[head]
                        weight_factor = w[head]
                        T.tile.broadcast(product, q[head : head + 1, :])
                        T.tile.mul(product, product, k)
                        T.reduce_sum(product, partial, dim=1)
                        T.tile.max(partial, partial, 0.0)
                        T.tile.mul(partial, partial, query_factor)
                        T.tile.mul(partial, partial, ks)
                        T.tile.mul(partial, partial, weight_factor)
                        T.tile.add(total, total, partial)
                    for i in T.serial(block):
                        if ids[i] < 0 or ids[i] >= table_rows:
                            total[i] = -T.infinity("float32")
                else:
                    T.tile.fill(total, -T.infinity("float32"))
                T.copy(total[:valid], out[row, start : start + valid])

    return main
