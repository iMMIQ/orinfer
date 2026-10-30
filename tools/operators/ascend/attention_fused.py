"""A2 tiled SWA/compressed attention with normalized BF16 probabilities.

Each query block caches its live score tiles, then feeds normalized BF16
probabilities to a continuous FP32 PV accumulator. Cube and Vector use the same
paired-core computation for every query length; cache inputs retain the BF16
compute boundary after INT8 decoding.
"""

import tilelang.language as T


def score_tiles(capacity, ratio, compressed, tile_m, tile_k):
    # A window can straddle tile boundaries; include the complete query block.
    short = (127 + tile_m // 8 + 2 * tile_k - 2) // tile_k
    return short + (capacity // ratio // tile_k if compressed else 0)


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
    output_rows = half // 4
    blocks = T.ceildiv(rows * heads, tile_m)
    long_length = capacity // ratio
    tiles = score_tiles(capacity, ratio, compressed, tile_m, tile_k)

    @T.macro
    def masked_scores(
        scores,
        positions,
        selected,
        score,
        lanes,
        mask,
        indices,
        index_values,
        cid,
        vid,
        tile,
        start,
        index,
        short_tiles,
    ):
        T.copy(
            scores[cid, vid * half : (vid + 1) * half, tile * tile_k : (tile + 1) * tile_k], score
        )
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
                T.tile.compare(mask, lanes, T.cast(positions[row] - 128, "float32"), "GT")
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

    @T.prim_func
    def main(
        query: T.Tensor((rows * heads, dimension), "bfloat16"),
        swa: T.Tensor((capacity, dimension), "bfloat16"),
        long_kv: T.Tensor((long_length, dimension), "bfloat16"),
        positions: T.Tensor((rows,), "int32"),
        selected: T.Tensor((rows, long_length), "int32"),
        sink: T.Tensor((heads,), "float32"),
        output: T.Tensor((rows * heads, dimension), "bfloat16"),
        score_tile: T.Tensor((blocks, tile_m, tiles * tile_k), "float32"),
        probability_tile: T.Tensor((blocks, tile_m, tile_k), "bfloat16"),
        output_tile: T.Tensor((blocks, tile_m, dimension), "float32"),
    ):
        with T.Kernel(blocks, is_npu=True) as (cid, vid):
            # The live regions are data-dependent. Declare their conservative
            # tensor bounds rather than inferring unions containing phase-local
            # query indices outside their defining LetStmt scope.
            T.reads(
                query[0 : rows * heads, 0:dimension],
                swa[0:capacity, 0:dimension],
                long_kv[0:long_length, 0:dimension],
                positions[0:rows],
                selected[0:rows, 0:long_length],
                sink[0:heads],
                score_tile[0:blocks, 0:tile_m, 0 : tiles * tile_k],
                probability_tile[0:blocks, 0:tile_m, 0:tile_k],
                output_tile[0:blocks, 0:tile_m, 0:dimension],
            )
            T.writes(
                output[0 : rows * heads, 0:dimension],
                score_tile[0:blocks, 0:tile_m, 0 : tiles * tile_k],
                probability_tile[0:blocks, 0:tile_m, 0:tile_k],
                output_tile[0:blocks, 0:tile_m, 0:dimension],
            )
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
                for tile in T.serial(tiles):
                    if tile < short_tiles + long_tiles:
                        index = T.if_then_else(
                            tile < short_tiles,
                            (begin + tile) * tile_k,
                            (tile - short_tiles) * tile_k,
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
                                    query[first : first + valid, k * 128 : (k + 1) * 128],
                                    q[:valid, :],
                                )
                                if tile < short_tiles:
                                    T.copy(
                                        swa[index : index + tile_k, k * 128 : (k + 1) * 128], key
                                    )
                                else:
                                    T.copy(
                                        long_kv[index : index + tile_k, k * 128 : (k + 1) * 128],
                                        key,
                                    )
                                T.gemm_v0(
                                    q,
                                    key,
                                    dots,
                                    transpose_B=True,
                                    init=(k == 0),
                                    kL0Size=64 if tile_k == 256 else 128,
                                )
                        T.copy(dots, score_tile[cid, :, tile * tile_k : (tile + 1) * tile_k])
                        T.set_cross_flag("FIX", 0)
                        T.wait_cross_flag(1)
                # Consume the last statistics acknowledgement before Vector
                # reuses flag 1 for the first normalized probability tile.
                T.set_cross_flag("FIX", 0)
                T.wait_cross_flag(1)
                T.set_cross_flag("FIX", 0)
                for tile in T.serial(tiles):
                    if tile < short_tiles + long_tiles:
                        index = T.if_then_else(
                            tile < short_tiles,
                            (begin + tile) * tile_k,
                            (tile - short_tiles) * tile_k,
                        )
                        T.wait_cross_flag(1)
                        T.copy(probability_tile[cid, :, :], prob)
                        if reuse_kv:
                            if tile < short_tiles:
                                T.copy(swa[index : index + tile_k, :], key)
                            else:
                                T.copy(long_kv[index : index + tile_k, :], key)
                            T.gemm_v0(prob, key, product, init=(tile == 0))
                        else:
                            if tile < short_tiles:
                                T.copy(swa[index : index + tile_k, :], value)
                            else:
                                T.copy(long_kv[index : index + tile_k, :], value)
                            T.gemm_v0(prob, value, product, init=(tile == 0))
                        if tile + 1 == short_tiles + long_tiles:
                            T.copy(product, output_tile[cid, :, :])
                        T.set_cross_flag("FIX", 2)
                        T.wait_cross_flag(3)
            with T.Scope("V"):
                score = T.alloc_ub((half, tile_k), "float32")
                probability = T.alloc_ub((half, tile_k), "bfloat16")
                partial = T.alloc_ub((half, dimension), "float32")
                score_broadcast = T.alloc_ub((half, tile_k), "float32")
                maximum = T.alloc_ub((half, 1), "float32")
                short_sum = T.alloc_ub((half, 1), "float32")
                long_sum = T.alloc_ub((half, 1), "float32")
                scale = T.alloc_ub((half, 1), "float32")
                denominator = T.alloc_ub((half, 1), "float32")
                total = T.alloc_ub((half, 1), "float32")
                indices = T.alloc_ub((tile_k,), "int32")
                index_values = T.alloc_ub((tile_k,), "float32")
                lanes = T.alloc_ub((tile_k,), "float32")
                mask = T.alloc_ub((tile_k // 8,), "uint8")
                out = T.alloc_ub((output_rows, dimension), "bfloat16")
                start = first + vid * half
                T.tile.fill(short_sum, 0.0)
                T.tile.fill(long_sum, 0.0)
                for h in T.serial(half):
                    maximum[h, 0] = sink[(start + h) % heads]
                for tile in T.serial(tiles):
                    if tile < short_tiles + long_tiles:
                        index = T.if_then_else(
                            tile < short_tiles,
                            (begin + tile) * tile_k,
                            (tile - short_tiles) * tile_k,
                        )
                        T.wait_cross_flag(0)
                        masked_scores(
                            score_tile,
                            positions,
                            selected,
                            score,
                            lanes,
                            mask,
                            indices,
                            index_values,
                            cid,
                            vid,
                            tile,
                            start,
                            index,
                            short_tiles,
                        )
                        T.reduce_max(score, total, dim=1)
                        T.tile.max(maximum, maximum, total)
                        T.set_flag("V", "MTE3", 6)
                        T.wait_flag("V", "MTE3", 6)
                        T.set_cross_flag("MTE3", 1)
                T.wait_cross_flag(0)
                for tile in T.serial(tiles):
                    if tile < short_tiles + long_tiles:
                        index = T.if_then_else(
                            tile < short_tiles,
                            (begin + tile) * tile_k,
                            (tile - short_tiles) * tile_k,
                        )
                        masked_scores(
                            score_tile,
                            positions,
                            selected,
                            score,
                            lanes,
                            mask,
                            indices,
                            index_values,
                            cid,
                            vid,
                            tile,
                            start,
                            index,
                            short_tiles,
                        )
                        T.tile.broadcast(score_broadcast, maximum)
                        T.tile.sub(score, score, score_broadcast)
                        T.tile.exp(score, score)
                        T.reduce_sum(score, total, dim=1)
                        if tile < short_tiles:
                            T.tile.add(short_sum, short_sum, total)
                        else:
                            T.tile.add(long_sum, long_sum, total)
                        T.set_flag("V", "MTE3", 6)
                        T.wait_flag("V", "MTE3", 6)
                        T.copy(
                            score,
                            score_tile[
                                cid,
                                vid * half : (vid + 1) * half,
                                tile * tile_k : (tile + 1) * tile_k,
                            ],
                        )
                        T.set_flag("MTE3", "V", 7)
                        T.wait_flag("MTE3", "V", 7)
                T.tile.add(denominator, short_sum, long_sum)
                for h in T.serial(half):
                    scale[h, 0] = sink[(start + h) % heads] - maximum[h, 0]
                T.tile.exp(scale, scale)
                T.tile.add(denominator, denominator, scale)
                for h in T.serial(half):
                    scale[h, 0] = 1.0 / denominator[h, 0]
                T.set_flag("V", "MTE3", 6)
                T.wait_flag("V", "MTE3", 6)
                T.set_cross_flag("MTE3", 1)
                T.wait_cross_flag(0)
                for tile in T.serial(tiles):
                    if tile < short_tiles + long_tiles:
                        T.copy(
                            score_tile[
                                cid,
                                vid * half : (vid + 1) * half,
                                tile * tile_k : (tile + 1) * tile_k,
                            ],
                            score,
                        )
                        T.tile.broadcast(score_broadcast, scale)
                        T.tile.mul(score, score, score_broadcast)
                        T.tile.cast(probability, score, "CAST_RINT", half * tile_k)
                        T.copy(probability, probability_tile[cid, vid * half : (vid + 1) * half, :])
                        T.set_cross_flag("MTE3", 1)
                        T.wait_cross_flag(2)
                        if tile + 1 == short_tiles + long_tiles:
                            T.copy(output_tile[cid, vid * half : (vid + 1) * half, :], partial)
                            for group in T.serial(4):
                                offset = start + group * output_rows
                                if offset < rows * heads:
                                    T.tile.cast(
                                        out,
                                        partial[group * output_rows : (group + 1) * output_rows, :],
                                        "CAST_RINT",
                                        output_rows * dimension,
                                    )
                                    T.copy(out, output[offset : offset + output_rows, :])
                                    T.set_flag("MTE3", "V", 7)
                                    T.wait_flag("MTE3", "V", 7)
                        T.set_cross_flag("MTE2", 3)
                T.pipe_barrier("ALL")

    return main.with_attr("orinfer.coupled_resources", True)


def selected_int8(
    rows,
    capacity,
    compressed_capacity,
    compressed=True,
    swa_storage_rows=0,
    heads=64,
    tile_slots=16,
    reuse_long=False,
):
    """One paired-core selected-KV gather/QK/softmax/PV invocation per query.

    The producer and consumer of each query run within one kernel. Workspace
    spills bridge AIV/AIC storage and retain the original FP32 scores, full-row
    normalization and normalized BF16 probability boundary.
    """
    from tools.operators.ascend.attention_cube import softmax_row
    from tools.operators.ascend.kv_int8 import gather_tile

    selected = 640 if compressed else 128
    storage = swa_storage_rows or capacity
    if heads not in (16, 32, 64) or tile_slots not in (1, 4, 8, 16):
        raise ValueError("Fused selected attention requires H16/32/64 and legal KV slot tiles")
    if not 128 <= storage <= capacity or (reuse_long and not compressed):
        raise ValueError("Invalid selected attention cache/reuse geometry")
    dimension, tile_k, tile_d = 512, 128, 128
    half_heads, decode_groups = heads // 2, tile_slots * 8
    produced = 128 if reuse_long else selected

    @T.prim_func
    def main(
        query: T.Tensor((rows, heads, dimension), "bfloat16"),
        swa: T.Tensor((storage, 8, 64), "int8"),
        swa_scales: T.Tensor((storage, 8), "float16"),
        long_kv: T.Tensor((compressed_capacity, 8, 64), "int8"),
        long_scales: T.Tensor((compressed_capacity, 8), "float16"),
        swa_indices: T.Tensor((rows, 128), "int32"),
        long_indices: T.Tensor((rows, 512), "int32"),
        sink: T.Tensor((heads,), "float32"),
        output: T.Tensor((rows, heads, dimension), "bfloat16"),
        kv_workspace: T.Tensor((rows, selected, dimension), "bfloat16"),
        mask_workspace: T.Tensor((rows, selected), "int32"),
        score_workspace: T.Tensor((rows, heads, selected), "float32"),
        probability_workspace: T.Tensor((rows, heads, selected), "bfloat16"),
        output_workspace: T.Tensor((rows, heads, dimension), "float32"),
    ):
        with T.Kernel(rows, is_npu=True) as (cid, vid):
            T.reads(
                query[0:rows, :, :],
                swa[:, :, :],
                swa_scales[:, :],
                long_kv[:, :, :],
                long_scales[:, :],
                swa_indices[0:rows, :],
                long_indices[0:rows, :],
                sink[:],
                kv_workspace[0:rows, :, :],
                mask_workspace[0:rows, :],
                score_workspace[0:rows, :, :],
                probability_workspace[0:rows, :, :],
                output_workspace[0:rows, :, :],
            )
            T.writes(
                output[0:rows, :, :],
                kv_workspace[0:rows, :, :],
                mask_workspace[0:rows, :],
                score_workspace[0:rows, :, :],
                probability_workspace[0:rows, :, :],
                output_workspace[0:rows, :, :],
            )
            row = cid
            if row < rows:
                with T.Scope("C"):
                    query_l1 = T.alloc_L1((heads, dimension), "bfloat16")
                    key_l1 = T.alloc_L1((tile_k, dimension), "bfloat16")
                    probability_l1 = T.alloc_L1((heads, selected), "bfloat16")
                    value_l1 = T.alloc_L1((selected, tile_d), "bfloat16")
                    acc = T.alloc_L0C((heads, 128), "float32")
                    T.copy(query[row, :, :], query_l1)
                    T.wait_cross_flag(0)
                    for slot in T.serial(selected // tile_k):
                        T.copy(kv_workspace[row, slot * tile_k : (slot + 1) * tile_k, :], key_l1)
                        T.gemm_v0(query_l1, key_l1, acc, transpose_B=True, init=True, kL0Size=128)
                        T.copy(acc, score_workspace[row, :, slot * 128 : (slot + 1) * 128])
                    T.set_cross_flag("FIX", 1)
                    T.wait_cross_flag(2)
                    T.copy(probability_workspace[row, :, :], probability_l1)
                    for d in T.serial(dimension // tile_d):
                        T.copy(kv_workspace[row, :, d * tile_d : (d + 1) * tile_d], value_l1)
                        T.gemm_v0(probability_l1, value_l1, acc, init=True, kL0Size=128)
                        T.copy(acc, output_workspace[row, :, d * 128 : (d + 1) * 128])
                    T.set_cross_flag("FIX", 3)
                    T.wait_cross_flag(4)
                with T.Scope("V"):
                    q = T.alloc_ub((decode_groups, 64), "int8")
                    zeros = T.alloc_ub((decode_groups, 16), "int32")
                    half_values = T.alloc_ub((decode_groups, 64), "float16")
                    values = T.alloc_ub((decode_groups, 64), "float32")
                    half_scales = T.alloc_ub((decode_groups, 1), "float16")
                    scale = T.alloc_ub((decode_groups, 1), "float32")
                    broadcast = T.alloc_ub((decode_groups, 64), "float32")
                    decoded = T.alloc_ub((tile_slots, dimension), "bfloat16")
                    flag = T.alloc_ub((max(8, tile_slots),), "int32")
                    x = T.alloc_ub((1, selected), "float32")
                    ids = T.alloc_ub((1, selected), "int32")
                    valid = T.alloc_ub((1, selected), "float32")
                    mask = T.alloc_ub((selected // 8,), "uint8")
                    maximum = T.alloc_ub((1, 1), "float32")
                    denominator = T.alloc_ub((1, 1), "float32")
                    sink_value = T.alloc_ub((1, 1), "float32")
                    probability = T.alloc_ub((1, selected), "bfloat16")
                    partial = T.alloc_ub((half_heads, tile_d), "float32")
                    result = T.alloc_ub((half_heads, tile_d), "bfloat16")
                    for part in T.serial(produced // (2 * tile_slots)):
                        first = vid * (produced // 2) + part * tile_slots
                        gather_tile(
                            swa,
                            swa_scales,
                            long_kv,
                            long_scales,
                            swa_indices,
                            long_indices,
                            kv_workspace,
                            mask_workspace,
                            q,
                            zeros,
                            half_values,
                            values,
                            half_scales,
                            scale,
                            broadcast,
                            decoded,
                            flag,
                            row,
                            first,
                            capacity,
                            compressed_capacity,
                            compressed,
                            storage,
                            tile_slots,
                            True,
                            decode_groups,
                        )
                    T.set_cross_flag("MTE3", 0)
                    T.wait_cross_flag(1)
                    for lane in T.serial(half_heads):
                        head = vid * half_heads + lane
                        softmax_row(
                            score_workspace,
                            mask_workspace,
                            sink,
                            probability_workspace,
                            row,
                            head,
                            x,
                            ids,
                            valid,
                            mask,
                            maximum,
                            denominator,
                            sink_value,
                            probability,
                            selected,
                            dimension,
                            lane == 0,
                        )
                    T.set_cross_flag("MTE3", 2)
                    T.wait_cross_flag(3)
                    for d in T.serial(dimension // tile_d):
                        first_head = vid * half_heads
                        T.copy(
                            output_workspace[
                                row,
                                first_head : first_head + half_heads,
                                d * tile_d : (d + 1) * tile_d,
                            ],
                            partial,
                        )
                        T.tile.cast(result, partial, "CAST_RINT", half_heads * tile_d)
                        T.copy(
                            result,
                            output[
                                row,
                                first_head : first_head + half_heads,
                                d * tile_d : (d + 1) * tile_d,
                            ],
                        )
                    T.set_cross_flag("MTE3", 4)
                    T.pipe_barrier("ALL")

    return (
        main.with_attr("orinfer.coupled_resources", True)
        .with_attr("orinfer.explicit_cross_sync", True)
        .with_attr(
            "orinfer.component_sources",
            ["tools/operators/ascend/kv_int8.py", "tools/operators/ascend/attention_cube.py"],
        )
    )
