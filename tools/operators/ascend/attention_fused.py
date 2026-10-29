"""A2 tiled SWA/compressed prefill with online softmax and paired-core exchange.

Only one score/probability/output tile per logical query block is exchanged
through GM/L2. The complete score/probability matrices are never materialized.
The cache inputs retain the existing BF16 compute boundary after INT8 decoding.
"""

import tilelang.language as T


def prefill(rows, capacity, ratio=2, compressed=True, tile_m=32, tile_k=128, reuse_kv=False):
    if (
        capacity % 256
        or ratio not in (1, 2)
        or tile_m not in (16, 32, 64)
        or tile_k not in (128, 256)
        or (compressed and (capacity // ratio) % tile_k)
    ):
        raise ValueError("Fused A2 attention requires aligned C, H8/D512 and M16/32/64, K128")
    heads, dimension = 8, 512
    half = tile_m // 2
    blocks = T.ceildiv(rows * heads, tile_m)
    long_length = capacity // ratio

    @T.prim_func
    def main(
        query: T.Tensor((rows * heads, dimension), "bfloat16"),
        swa: T.Tensor((capacity, dimension), "bfloat16"),
        long_kv: T.Tensor((long_length, dimension), "bfloat16"),
        positions: T.Tensor((rows,), "int32"),
        selected: T.Tensor((rows, long_length), "int32"),
        sink: T.Tensor((heads,), "float32"),
        output: T.Tensor((rows * heads, dimension), "bfloat16"),
        score_tile: T.Tensor((blocks, tile_m, tile_k), "float32"),
        probability_tile: T.Tensor((blocks, tile_m, tile_k), "bfloat16"),
        output_tile: T.Tensor((blocks, tile_m, dimension), "float32"),
    ):
        with T.Kernel(blocks, is_npu=True) as (cid, vid):
            first = cid * tile_m
            valid = T.min(tile_m, rows * heads - first)
            begin = T.max(0, (positions[first // heads] - 127) // tile_k)
            end = positions[T.min(rows - 1, (first + tile_m - 1) // heads)] + 1
            short_tiles = T.ceildiv(end, tile_k) - begin
            long_tiles = T.ceildiv(end // ratio, tile_k) if compressed else 0
            with T.Scope("C"):
                q = T.alloc_L1((tile_m, dimension if reuse_kv else 128), "bfloat16")
                key = T.alloc_L1((tile_k, dimension if reuse_kv else 128), "bfloat16")
                value = T.alloc_L1((tile_k, 1 if reuse_kv else dimension), "bfloat16")
                prob = T.alloc_L1((tile_m, tile_k), "bfloat16")
                dots = T.alloc_L0C((tile_m, tile_k), "float32")
                product = T.alloc_L0C((tile_m, dimension), "float32")
                if reuse_kv:
                    T.copy(query[first : first + valid, :], q[:valid, :])
                for tile in T.serial(short_tiles + long_tiles):
                    index = T.if_then_else(
                        tile < short_tiles, (begin + tile) * tile_k, (tile - short_tiles) * tile_k
                    )
                    if reuse_kv:
                        if tile < short_tiles:
                            T.copy(swa[index : index + tile_k, :], key)
                        else:
                            T.copy(long_kv[index : index + tile_k, :], key)
                        T.gemm_v0(
                            q,
                            key,
                            dots,
                            transpose_B=True,
                            init=True,
                            kL0Size=64 if tile_k == 256 else 128,
                        )
                    else:
                        for k in T.serial(dimension // 128):
                            T.copy(
                                query[first : first + valid, k * 128 : (k + 1) * 128], q[:valid, :]
                            )
                            if tile < short_tiles:
                                T.copy(swa[index : index + tile_k, k * 128 : (k + 1) * 128], key)
                            else:
                                T.copy(
                                    long_kv[index : index + tile_k, k * 128 : (k + 1) * 128], key
                                )
                            T.gemm_v0(
                                q,
                                key,
                                dots,
                                transpose_B=True,
                                init=(k == 0),
                                kL0Size=64 if tile_k == 256 else 128,
                            )
                    T.copy(dots, score_tile[cid, :, :])
                    T.set_cross_flag("FIX", 0)
                    T.wait_cross_flag(1)
                    T.copy(probability_tile[cid, :, :], prob)
                    if reuse_kv:
                        T.gemm_v0(prob, key, product, init=True)
                    else:
                        if tile < short_tiles:
                            T.copy(swa[index : index + tile_k, :], value)
                        else:
                            T.copy(long_kv[index : index + tile_k, :], value)
                        T.gemm_v0(prob, value, product, init=True)
                    T.copy(product, output_tile[cid, :, :])
                    T.set_cross_flag("FIX", 2)
                    T.wait_cross_flag(3)
            with T.Scope("V"):
                score = T.alloc_ub((half, tile_k), "float32")
                probability = T.alloc_ub((half, tile_k), "bfloat16")
                accumulator = T.alloc_ub((half, dimension), "float32")
                partial = T.alloc_ub((half, dimension), "float32")
                score_broadcast = T.alloc_ub((half, tile_k if tile_k == 128 else 1), "float32")
                maximum = T.alloc_ub((half, 1), "float32")
                previous = T.alloc_ub((half, 1), "float32")
                scale = T.alloc_ub((half, 1), "float32")
                denominator = T.alloc_ub((half, 1), "float32")
                total = T.alloc_ub((half, 1), "float32")
                indices = T.alloc_ub((tile_k,), "int32")
                index_values = T.alloc_ub((tile_k,), "float32")
                lanes = T.alloc_ub((tile_k,), "float32")
                mask = T.alloc_ub((tile_k // 8,), "uint8")
                out = T.alloc_ub((half, dimension), "bfloat16")
                start = first + vid * half
                T.tile.fill(accumulator, 0.0)
                T.tile.fill(denominator, 1.0)
                for h in T.serial(half):
                    maximum[h, 0] = sink[(start + h) % heads]
                for tile in T.serial(short_tiles + long_tiles):
                    index = T.if_then_else(
                        tile < short_tiles, (begin + tile) * tile_k, (tile - short_tiles) * tile_k
                    )
                    T.wait_cross_flag(0)
                    T.copy(score_tile[cid, vid * half : (vid + 1) * half, :], score)
                    T.tile.mul(score, score, dimension**-0.5)
                    T.tile.createvecindex(lanes, 0.0)
                    T.tile.add(lanes, lanes, T.cast(index, "float32"))
                    for token in T.serial(half // heads):
                        row = T.min(rows - 1, start // heads + token)
                        if tile < short_tiles:
                            T.tile.compare(mask, lanes, T.cast(positions[row] + 1, "float32"), "LT")
                            for head in T.unroll(heads):
                                h = token * heads + head
                                T.tile.select(
                                    score[h, :],
                                    mask,
                                    score[h, :],
                                    -T.infinity("float32"),
                                    "VSEL_TENSOR_SCALAR_MODE",
                                )
                            T.tile.compare(
                                mask, lanes, T.cast(positions[row] - 128, "float32"), "GT"
                            )
                        else:
                            T.copy(selected[row, index : index + tile_k], indices)
                            T.tile.cast(index_values, indices, "CAST_NONE", tile_k)
                            T.tile.compare(mask, index_values, 0.0, "GT")
                        for head in T.unroll(heads):
                            h = token * heads + head
                            T.tile.select(
                                score[h, :],
                                mask,
                                score[h, :],
                                -T.infinity("float32"),
                                "VSEL_TENSOR_SCALAR_MODE",
                            )
                    T.copy(maximum, previous)
                    T.reduce_max(score, total, dim=1)
                    T.tile.max(maximum, maximum, total)
                    T.tile.sub(scale, previous, maximum)
                    T.tile.exp(scale, scale)
                    if tile_k == 128:
                        T.tile.broadcast(score_broadcast, maximum)
                        T.tile.sub(score, score, score_broadcast)
                    else:
                        for h in T.serial(half):
                            T.tile.sub(score[h, :], score[h, :], maximum[h, 0])
                    T.tile.exp(score, score)
                    T.reduce_sum(score, total, dim=1)
                    T.tile.mul(denominator, denominator, scale)
                    T.tile.add(denominator, denominator, total)
                    T.tile.cast(probability, score, "CAST_RINT", half * tile_k)
                    T.copy(probability, probability_tile[cid, vid * half : (vid + 1) * half, :])
                    T.set_cross_flag("MTE3", 1)
                    T.wait_cross_flag(2)
                    T.tile.broadcast(partial, scale)
                    T.tile.mul(accumulator, accumulator, partial)
                    T.copy(output_tile[cid, vid * half : (vid + 1) * half, :], partial)
                    T.tile.add(accumulator, accumulator, partial)
                    T.set_cross_flag("MTE2", 3)
                for h in T.serial(half):
                    scale[h, 0] = 1.0 / denominator[h, 0]
                T.tile.broadcast(partial, scale)
                T.tile.mul(accumulator, accumulator, partial)
                T.tile.cast(out, accumulator, "CAST_RINT", half * dimension)
                if start < rows * heads:
                    tail = T.min(half, rows * heads - start)
                    T.copy(out[:tail, :], output[start : start + tail, :])
                T.pipe_barrier("ALL")

    return main.with_attr("orinfer.coupled_resources", True)
