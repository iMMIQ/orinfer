"""Prefill attention with shared contiguous KV banks and packed token/head tiles."""

import tilelang.language as T


def selected_mask(rows, length, selected=512, explicit_sync=False, store_unroll=1):
    if length % 128:
        raise ValueError("Dense prefill attention requires a 128-aligned KV bank")
    if store_unroll not in (1, 8, 16, 32) or selected < 1 or selected % store_unroll:
        raise ValueError("Mask stores require complete groups of 1/8/16/32 indices")

    @T.prim_func
    def main(
        indices: T.Tensor((rows, selected), "int32"),
        mask: T.Tensor((rows, length), "int32"),
    ):
        with T.Kernel(T.ceildiv(rows, 2), is_npu=True) as (cid, vid):
            ids = T.alloc_ub((selected,), "int32")
            values = T.alloc_ub((length,), "int32")
            row = cid * 2 + vid
            if row < rows:
                T.tile.fill(values, 0)
                T.copy(indices[row, :], ids)
                T.set_flag("MTE2", "S", 6)
                T.wait_flag("MTE2", "S", 6)
                T.set_flag("V", "S", 6)
                T.wait_flag("V", "S", 6)
                for group in T.serial(selected // store_unroll):
                    for lane in T.unroll(store_unroll):
                        key = ids[group * store_unroll + lane]
                        if key >= 0 and key < length:
                            values[key] = 1
                T.set_flag("S", "MTE3", 6)
                T.wait_flag("S", "MTE3", 6)
                T.copy(values, mask[row, :])
                T.set_flag("MTE3", "V", 7)
                T.wait_flag("MTE3", "V", 7)

    # All transfers, Scalar stores and UB reuse have paired flags above.
    return main.with_attr("orinfer.explicit_sync", True) if explicit_sync else main


def qk(
    rows,
    capacity,
    ratio=2,
    compressed=True,
    heads=8,
    dimension=512,
    windowed=True,
    live_tiles=False,
    key_splits=1,
):
    long_length = capacity // ratio
    if capacity % 256 or ratio not in (1, 2) or heads != 8 or dimension != 512:
        raise ValueError("Packed prefill QK requires C%256=0, ratio=1/2 and H8/D512")
    if key_splits not in (1, 2, 4, 8):
        raise ValueError("QK key partitions must be 1/2/4/8")
    length = capacity + (long_length if compressed else 0)
    tiles = length // 128
    query_tiles = T.ceildiv(rows * heads, 128)
    grid = query_tiles * (key_splits if live_tiles else tiles)

    @T.macro
    def score_tile(query, swa, long_kv, scores, a, b, c, first, key, valid):
        for k in T.serial(dimension // 128):
            T.copy(query[first : first + valid, k * 128 : (k + 1) * 128], a[:valid, :])
            if key < capacity:
                T.copy(swa[key : key + 128, k * 128 : (k + 1) * 128], b)
            else:
                T.copy(long_kv[key - capacity : key - capacity + 128, k * 128 : (k + 1) * 128], b)
            T.gemm_v0(a, b, c, transpose_B=True, init=(k == 0))
        T.copy(c[:valid, :], scores[first : first + valid, key : key + 128])

    @T.prim_func
    def main(
        query: T.Tensor((rows * heads, dimension), "bfloat16"),
        swa: T.Tensor((capacity, dimension), "bfloat16"),
        long_kv: T.Tensor((long_length, dimension), "bfloat16"),
        positions: T.Tensor((rows,), "int32"),
        scores: T.Tensor((rows * heads, length), "float32"),
    ):
        with T.Kernel(grid, is_npu=True) as (cid, vid):
            with T.Scope("C"):
                a = T.alloc_L1((128, 128), "bfloat16")
                b = T.alloc_L1((128, 128), "bfloat16")
                c = T.alloc_L0C((128, 128), "float32")
                if live_tiles:
                    logical = cid // key_splits
                    half = T.ceildiv(query_tiles, 2)
                    # Fold early and late query blocks together to balance the
                    # causal prefix work across bounded physical grids.
                    group = T.if_then_else(
                        logical < half, logical, query_tiles - 1 - (logical - half)
                    )
                    first = group * 128
                    valid = T.min(128, rows * heads - first)
                    end = (
                        positions[T.min(rows - 1, (first + 127) // heads)] + 1
                        if windowed
                        else positions[rows - 1] + 1
                    )
                    begin = T.max(0, (positions[first // heads] - 127) // 128) if windowed else 0
                    swa_tiles = T.ceildiv(end, 128) - begin
                    long_tiles = T.ceildiv(end // ratio, 128) if compressed else 0
                    active = swa_tiles + long_tiles
                    chunk = T.ceildiv(active, key_splits)
                    start_tile = (cid % key_splits) * chunk
                    for tile in T.serial(
                        start_tile,
                        T.max(start_tile, T.min(active, (cid % key_splits + 1) * chunk)),
                    ):
                        key = T.if_then_else(
                            tile < swa_tiles,
                            (begin + tile) * 128,
                            capacity + (tile - swa_tiles) * 128,
                        )
                        score_tile(query, swa, long_kv, scores, a, b, c, first, key, valid)
                else:
                    first, key = cid // tiles * 128, cid % tiles * 128
                    valid = T.min(128, rows * heads - first)
                    end = (
                        positions[T.min(rows - 1, (first + 127) // heads)] + 1
                        if windowed
                        else positions[rows - 1] + 1
                    )
                    begin = positions[first // heads] - 127 if windowed else 0
                    live = T.if_then_else(key < capacity, end, capacity + end // ratio)
                    if key < live and (key >= capacity or key + 128 > begin):
                        score_tile(query, swa, long_kv, scores, a, b, c, first, key, valid)

    return main


def softmax(
    rows,
    capacity,
    ratio=2,
    compressed=True,
    heads=8,
    dimension=512,
    compact=False,
    live_buckets=False,
    tile_heads=1,
    bucket=-1,
):
    if bucket != -1:
        if (
            not compact
            or not live_buckets
            or ratio not in (1, 2)
            or bucket not in (0, 1, 2)
            or (not compressed and bucket != 0)
            or capacity % 256
            or not 256 <= capacity <= 8448
        ):
            raise ValueError("Head tiles require compact, live prefill buckets at aligned C<=8448")
        return _bucketed_softmax(
            rows, capacity, ratio, compressed, heads, dimension, tile_heads, bucket
        )
    if tile_heads != 1:
        raise ValueError("A multi-head softmax tile requires an explicit prefix bucket")
    if compact:
        return _compact_softmax(rows, capacity, ratio, compressed, heads, dimension, live_buckets)
    long_length = capacity // ratio
    length = capacity + (long_length if compressed else 0)
    if capacity % 256 or capacity > 8448:
        raise ValueError("Packed prefill softmax supports 256-aligned capacity up to 8448")

    @T.prim_func
    def main(
        scores: T.Tensor((rows * heads, length), "float32"),
        positions: T.Tensor((rows,), "int32"),
        selected: T.Tensor((rows, long_length), "int32"),
        sink: T.Tensor((heads,), "float32"),
        probabilities: T.Tensor((rows * heads, length), "bfloat16"),
    ):
        with T.Kernel(T.ceildiv(rows * heads, 2), is_npu=True) as (cid, vid):
            xs = T.alloc_ub((1, capacity), "float32")
            xl = T.alloc_ub((1, long_length), "float32")
            scratch = T.alloc_ub((1, capacity), "float32")
            ids = T.alloc_ub((1, long_length), "int32")
            mask = T.alloc_ub((capacity // 8,), "uint8")
            p = T.alloc_ub((1, capacity), "bfloat16")
            maximum = T.alloc_ub((1, 1), "float32")
            other = T.alloc_ub((1, 1), "float32")
            denominator = T.alloc_ub((1, 1), "float32")
            sink_value = T.alloc_ub((1, 1), "float32")
            flat = cid * 2 + vid
            row, head = flat // heads, flat % heads
            if row < rows:
                position = positions[row]
                T.copy(scores[flat : flat + 1, :capacity], xs)
                T.tile.mul(xs, xs, dimension**-0.5)
                T.tile.createvecindex(scratch, 0.0)
                T.tile.compare(mask, scratch, T.cast(position + 1, "float32"), "LT")
                T.tile.select(xs, mask, xs, -T.infinity("float32"), "VSEL_TENSOR_SCALAR_MODE")
                T.tile.compare(mask, scratch, T.cast(position - 128, "float32"), "GT")
                T.tile.select(xs, mask, xs, -T.infinity("float32"), "VSEL_TENSOR_SCALAR_MODE")
                T.reduce_max(xs, maximum, dim=1)
                if compressed:
                    T.copy(scores[flat : flat + 1, capacity:], xl)
                    T.copy(selected[row : row + 1, :], ids)
                    T.tile.cast(scratch[:, :long_length], ids, "CAST_NONE", long_length)
                    T.tile.compare(mask, scratch[:, :long_length], 0.0, "GT")
                    T.tile.mul(xl, xl, dimension**-0.5)
                    T.tile.select(xl, mask, xl, -T.infinity("float32"), "VSEL_TENSOR_SCALAR_MODE")
                    T.reduce_max(xl, other, dim=1)
                    T.tile.max(maximum, maximum, other)
                sink_value[0, 0] = sink[head]
                T.tile.max(maximum, maximum, sink_value)
                T.tile.sub(xs, xs, maximum[0, 0])
                T.tile.exp(xs, xs)
                T.reduce_sum(xs, denominator, dim=1)
                if compressed:
                    T.tile.sub(xl, xl, maximum[0, 0])
                    T.tile.exp(xl, xl)
                    T.reduce_sum(xl, other, dim=1)
                    T.tile.add(denominator, denominator, other)
                T.tile.sub(sink_value, sink_value, maximum)
                T.tile.exp(sink_value, sink_value)
                T.tile.add(denominator, denominator, sink_value)
                T.tile.div(xs, xs, denominator[0, 0])
                T.tile.cast(p, xs, "CAST_RINT", capacity)
                T.copy(p, probabilities[flat : flat + 1, :capacity])
                T.set_flag("MTE3", "V", 7)
                T.wait_flag("MTE3", "V", 7)
                if compressed:
                    T.tile.add(other, denominator, 0.0)
                    T.tile.div(xl, xl, other[0, 0])
                    T.tile.cast(p[:, :long_length], xl, "CAST_RINT", long_length)
                    T.copy(p[:, :long_length], probabilities[flat : flat + 1, capacity:])
                    T.set_flag("MTE3", "V", 7)
                    T.wait_flag("MTE3", "V", 7)

    return main


def _compact_softmax(rows, capacity, ratio, compressed, heads, dimension, live_buckets):
    if capacity % 256 or not 256 <= capacity <= 8448 or heads != 8 or dimension != 512:
        raise ValueError("Compact softmax requires aligned C<=8448 and H8/D512")
    long_length = capacity // ratio
    score_length = capacity + (long_length if compressed else 0)
    span = min(384, capacity)
    probability_length = span + (long_length if compressed else 0)
    small = min(long_length, 2048 // ratio)
    medium = min(long_length, 4096 // ratio)

    @T.macro
    def normalize(
        scores,
        positions,
        selected,
        sink,
        probabilities,
        flat,
        active_long,
        xs,
        xl,
        scratch,
        lanes,
        ids,
        mask,
        p,
        maximum,
        other,
        denominator,
        sink_value,
    ):
        row = flat // heads
        head = flat % heads
        position = positions[row]
        first_row = flat // 128 * 16
        start = T.max(0, (positions[first_row] - 127) // 128) * 128
        valid = T.min(span, capacity - start)
        T.tile.fill(xs, -T.infinity("float32"))
        T.copy(scores[flat : flat + 1, start : start + valid], xs[:, :valid])
        T.tile.mul(xs, xs, dimension**-0.5)
        T.tile.createvecindex(lanes, 0.0)
        T.tile.add(lanes, lanes, T.cast(start, "float32"))
        T.tile.compare(mask, lanes, T.cast(position + 1, "float32"), "LT")
        T.tile.select(xs, mask, xs, -T.infinity("float32"), "VSEL_TENSOR_SCALAR_MODE")
        T.tile.compare(mask, lanes, T.cast(position - 128, "float32"), "GT")
        T.tile.select(xs, mask, xs, -T.infinity("float32"), "VSEL_TENSOR_SCALAR_MODE")
        T.reduce_max(xs, maximum, dim=1)
        if compressed:
            T.copy(scores[flat : flat + 1, capacity : capacity + active_long], xl)
            T.copy(selected[row : row + 1, :active_long], ids[:, :active_long])
            T.tile.cast(scratch[:, :active_long], ids[:, :active_long], "CAST_NONE", active_long)
            T.tile.compare(mask, scratch[:, :active_long], 0.0, "GT")
            T.tile.mul(xl, xl, dimension**-0.5)
            T.tile.select(
                xl,
                mask,
                xl,
                -T.infinity("float32"),
                "VSEL_TENSOR_SCALAR_MODE",
            )
            T.reduce_max(xl, other, dim=1)
            T.tile.max(maximum, maximum, other)
        sink_value[0, 0] = sink[head]
        T.tile.max(maximum, maximum, sink_value)
        T.tile.sub(xs, xs, maximum[0, 0])
        T.tile.exp(xs, xs)
        T.reduce_sum(xs, denominator, dim=1)
        if compressed:
            T.tile.sub(xl, xl, maximum[0, 0])
            T.tile.exp(xl, xl)
            T.reduce_sum(xl, other, dim=1)
            T.tile.add(denominator, denominator, other)
        T.tile.sub(sink_value, sink_value, maximum)
        T.tile.exp(sink_value, sink_value)
        T.tile.add(denominator, denominator, sink_value)
        T.tile.div(xs, xs, denominator[0, 0])
        T.tile.cast(p[:, :span], xs, "CAST_RINT", span)
        T.copy(p[:, :span], probabilities[flat : flat + 1, :span])
        T.set_flag("MTE3", "V", 7)
        T.wait_flag("MTE3", "V", 7)
        if compressed:
            T.tile.add(other, denominator, 0.0)
            T.tile.div(xl, xl, other[0, 0])
            T.tile.cast(p[:, :active_long], xl, "CAST_RINT", active_long)
            T.copy(p[:, :active_long], probabilities[flat : flat + 1, span : span + active_long])
            T.set_flag("MTE3", "V", 7)
            T.wait_flag("MTE3", "V", 7)

    @T.prim_func
    def main(
        scores: T.Tensor((rows * heads, score_length), "float32"),
        positions: T.Tensor((rows,), "int32"),
        selected: T.Tensor((rows, long_length), "int32"),
        sink: T.Tensor((heads,), "float32"),
        probabilities: T.Tensor((rows * heads, probability_length), "bfloat16"),
    ):
        with T.Kernel(T.ceildiv(rows * heads, 2), is_npu=True) as (cid, vid):
            xs = T.alloc_ub((1, span), "float32")
            # Distinct full buffers keep the pinned compiler on its wide
            # reduction path; BufferRegion reductions only support <=64 lanes.
            xl = T.alloc_ub((1, long_length), "float32")
            xl_small = T.alloc_ub((1, small), "float32")
            xl_medium = T.alloc_ub((1, medium), "float32")
            scratch = T.alloc_ub((1, max(span, long_length)), "float32")
            lanes = T.alloc_ub((1, span), "float32")
            ids = T.alloc_ub((1, long_length), "int32")
            mask = T.alloc_ub((max(span, long_length) // 8,), "uint8")
            p = T.alloc_ub((1, max(span, long_length)), "bfloat16")
            maximum = T.alloc_ub((1, 1), "float32")
            other = T.alloc_ub((1, 1), "float32")
            denominator = T.alloc_ub((1, 1), "float32")
            sink_value = T.alloc_ub((1, 1), "float32")
            flat = cid * 2 + vid
            row = flat // heads
            if row < rows:
                end = positions[T.min(rows - 1, flat // 128 * 16 + 15)] + 1
                if live_buckets and compressed and long_length > small:
                    if end <= 2048:
                        normalize(
                            scores,
                            positions,
                            selected,
                            sink,
                            probabilities,
                            flat,
                            small,
                            xs,
                            xl_small,
                            scratch,
                            lanes,
                            ids,
                            mask,
                            p,
                            maximum,
                            other,
                            denominator,
                            sink_value,
                        )
                    elif end <= 4096:
                        normalize(
                            scores,
                            positions,
                            selected,
                            sink,
                            probabilities,
                            flat,
                            medium,
                            xs,
                            xl_medium,
                            scratch,
                            lanes,
                            ids,
                            mask,
                            p,
                            maximum,
                            other,
                            denominator,
                            sink_value,
                        )
                    else:
                        normalize(
                            scores,
                            positions,
                            selected,
                            sink,
                            probabilities,
                            flat,
                            long_length,
                            xs,
                            xl,
                            scratch,
                            lanes,
                            ids,
                            mask,
                            p,
                            maximum,
                            other,
                            denominator,
                            sink_value,
                        )
                else:
                    normalize(
                        scores,
                        positions,
                        selected,
                        sink,
                        probabilities,
                        flat,
                        long_length,
                        xs,
                        xl,
                        scratch,
                        lanes,
                        ids,
                        mask,
                        p,
                        maximum,
                        other,
                        denominator,
                        sink_value,
                    )

    return main


def pv(
    rows, capacity, ratio=2, compressed=True, heads=8, dimension=512, windowed=True, compact=False
):
    long_length = capacity // ratio
    span = min(384, capacity) if compact else capacity
    length = span + (long_length if compressed else 0)
    if capacity % 256 or heads != 8 or dimension != 512:
        raise ValueError("Packed prefill PV requires C%256=0 and H8/D512")
    if compact and not windowed:
        raise ValueError("Compact probabilities require the shared query-block window")

    @T.prim_func
    def main(
        probabilities: T.Tensor((rows * heads, length), "bfloat16"),
        swa: T.Tensor((capacity, dimension), "bfloat16"),
        long_kv: T.Tensor((long_length, dimension), "bfloat16"),
        positions: T.Tensor((rows,), "int32"),
        output: T.Tensor((rows * heads, dimension), "bfloat16"),
        workspace: T.Tensor((T.ceildiv(rows * heads, 128) * 128, dimension), "float32"),
    ):
        with T.Kernel(T.ceildiv(rows * heads, 128) * (dimension // 64), is_npu=True) as (cid, vid):
            first, column = cid // (dimension // 64) * 128, cid % (dimension // 64) * 64
            with T.Scope("C"):
                a = T.alloc_L1((128, 128), "bfloat16")
                b = T.alloc_L1((128, 64), "bfloat16")
                c = T.alloc_L0C((128, 64), "float32")
                valid = T.min(128, rows * heads - first)
                end = (
                    positions[T.min(rows - 1, (first + 127) // heads)] + 1
                    if windowed
                    else positions[rows - 1] + 1
                )
                begin = T.max(0, (positions[first // heads] - 127) // 128) if windowed else 0
                for k in T.serial(begin, T.ceildiv(end, 128)):
                    T.copy(
                        probabilities[
                            first : first + valid,
                            (k - (begin if compact else 0)) * 128 : (
                                k + 1 - (begin if compact else 0)
                            )
                            * 128,
                        ],
                        a[:valid, :],
                    )
                    T.copy(swa[k * 128 : (k + 1) * 128, column : column + 64], b)
                    T.gemm_v0(a, b, c, init=(k == begin))
                if compressed:
                    for k in T.serial(T.ceildiv(end // ratio, 128)):
                        T.copy(
                            probabilities[
                                first : first + valid, span + k * 128 : span + (k + 1) * 128
                            ],
                            a[:valid, :],
                        )
                        T.copy(long_kv[k * 128 : (k + 1) * 128, column : column + 64], b)
                        T.gemm_v0(a, b, c, init=False)
                T.copy(c[:valid, :], workspace[first : first + valid, column : column + 64])
            with T.Scope("V"):
                x = T.alloc_ub((64, 64), "float32")
                y = T.alloc_ub((64, 64), "bfloat16")
                start = first + vid * 64
                if start < rows * heads:
                    valid = T.min(64, rows * heads - start)
                    T.copy(workspace[start : start + valid, column : column + 64], x[:valid, :])
                    T.tile.cast(y, x, "CAST_RINT", 64 * 64)
                    T.copy(y[:valid, :], output[start : start + valid, column : column + 64])

    return main


def replay_softmax(
    rows,
    capacity,
    ratio=1,
    compressed=True,
    compact=True,
    live_buckets=True,
    tile_heads=4,
    bucket=0,
):
    if not compressed or not compact or not live_buckets or ratio != 1:
        raise ValueError("Decoder replay softmax requires the compact C1 attention contract")
    return _bucketed_softmax(
        rows, capacity, ratio, compressed, 8, 512, tile_heads, bucket, replay=True
    )


def _bucketed_softmax(
    rows, capacity, ratio, compressed, heads, dimension, tile_heads, bucket, replay=False
):
    """Normalize one prefix bucket with shared masks and scalar FP32 reciprocals."""
    if heads != 8 or dimension != 512 or tile_heads not in (1, 2, 4, 8):
        raise ValueError("Head tiles require H8/D512 and 1, 2, 4 or 8 heads")
    long_length = capacity // ratio
    score_length = capacity + (long_length if compressed else 0)
    span = min(384, capacity)
    probability_length = span + (long_length if compressed else 0)
    small, medium = min(long_length, 2048 // ratio), min(long_length, 4096 // ratio)

    active = (small, medium, long_length)[bucket]
    if tile_heads * max(span, active) > 8448:
        raise ValueError("Head tile exceeds the bounded A2 softmax UB workspace")

    @T.macro
    def normalize(scores, positions, selected, sink, probabilities, flat, active_long, lower_bound):
        xs = T.alloc_ub((tile_heads, span), "float32")
        xl = T.alloc_ub((tile_heads, active_long), "float32")
        lanes = T.alloc_ub((1, span), "float32")
        ids = T.alloc_ub((1, active_long), "int32")
        ids_float = T.alloc_ub((1, active_long), "float32")
        mask = T.alloc_ub((max(span, active_long) // 8,), "uint8")
        short_broadcast = T.alloc_ub((tile_heads, span), "float32")
        long_broadcast = T.alloc_ub((tile_heads, active_long), "float32")
        ps = T.alloc_ub((tile_heads, span), "bfloat16")
        pl = T.alloc_ub((tile_heads, active_long), "bfloat16")
        maximum = T.alloc_ub((tile_heads, 1), "float32")
        other = T.alloc_ub((tile_heads, 1), "float32")
        denominator = T.alloc_ub((tile_heads, 1), "float32")
        sink_value = T.alloc_ub((tile_heads, 1), "float32")
        inverse = T.alloc_ub((tile_heads, 1), "float32")
        row, head = flat // heads, flat % heads
        position = positions[row]
        first_row = row // 16 * 16
        start = T.max(0, (positions[first_row] - 127) // 128) * 128
        valid = T.min(span, capacity - start)
        T.tile.fill(xs, -T.infinity("float32"))
        T.copy(scores[flat : flat + tile_heads, start : start + valid], xs[:, :valid])
        T.tile.mul(xs, xs, dimension**-0.5)
        T.tile.createvecindex(lanes, 0.0)
        T.tile.add(lanes, lanes, T.cast(start, "float32"))
        T.tile.compare(mask, lanes, T.cast(position + 1, "float32"), "LT")
        for h in T.unroll(tile_heads):
            T.tile.select(
                xs[h : h + 1, :],
                mask,
                xs[h : h + 1, :],
                -T.infinity("float32"),
                "VSEL_TENSOR_SCALAR_MODE",
            )
        T.tile.compare(mask, lanes, T.cast(T.max(position - 128, lower_bound - 1), "float32"), "GT")
        for h in T.unroll(tile_heads):
            T.tile.select(
                xs[h : h + 1, :],
                mask,
                xs[h : h + 1, :],
                -T.infinity("float32"),
                "VSEL_TENSOR_SCALAR_MODE",
            )
        T.reduce_max(xs, maximum, dim=1)
        if compressed:
            T.copy(scores[flat : flat + tile_heads, capacity : capacity + active_long], xl)
            T.copy(selected[row : row + 1, :active_long], ids)
            T.tile.cast(ids_float, ids, "CAST_NONE", active_long)
            T.tile.compare(mask, ids_float, 0.0, "GT")
            T.tile.mul(xl, xl, dimension**-0.5)
            for h in T.unroll(tile_heads):
                T.tile.select(
                    xl[h : h + 1, :],
                    mask,
                    xl[h : h + 1, :],
                    -T.infinity("float32"),
                    "VSEL_TENSOR_SCALAR_MODE",
                )
            T.reduce_max(xl, other, dim=1)
            T.tile.max(maximum, maximum, other)
        T.copy(sink[head : head + tile_heads], sink_value)
        T.tile.max(maximum, maximum, sink_value)
        T.tile.broadcast(short_broadcast, maximum)
        T.tile.sub(xs, xs, short_broadcast)
        T.tile.exp(xs, xs)
        T.reduce_sum(xs, denominator, dim=1)
        if compressed:
            T.tile.broadcast(long_broadcast, maximum)
            T.tile.sub(xl, xl, long_broadcast)
            T.tile.exp(xl, xl)
            T.reduce_sum(xl, other, dim=1)
            T.tile.add(denominator, denominator, other)
        T.tile.sub(sink_value, sink_value, maximum)
        T.tile.exp(sink_value, sink_value)
        T.tile.add(denominator, denominator, sink_value)
        # Retain the original scalar FP32 reciprocal, then the same FP32 multiply.
        for h in T.serial(tile_heads):
            inverse[h, 0] = 1.0 / denominator[h, 0]
        T.tile.broadcast(short_broadcast, inverse)
        T.tile.mul(xs, xs, short_broadcast)
        T.tile.cast(ps, xs, "CAST_RINT", tile_heads * span)
        T.copy(ps, probabilities[flat : flat + tile_heads, :span])
        T.set_flag("MTE3", "V", 7)
        T.wait_flag("MTE3", "V", 7)
        if compressed:
            T.tile.broadcast(long_broadcast, inverse)
            T.tile.mul(xl, xl, long_broadcast)
            T.tile.cast(pl, xl, "CAST_RINT", tile_heads * active_long)
            T.copy(pl, probabilities[flat : flat + tile_heads, span : span + active_long])
            T.set_flag("MTE3", "V", 7)
            T.wait_flag("MTE3", "V", 7)

    @T.prim_func
    def main(
        scores: T.Tensor((rows * heads, score_length), "float32"),
        positions: T.Tensor((rows,), "int32"),
        selected: T.Tensor((rows, long_length), "int32"),
        sink: T.Tensor((heads,), "float32"),
        probabilities: T.Tensor((rows * heads, probability_length), "bfloat16"),
    ):
        with T.Kernel(T.ceildiv(rows * heads, tile_heads * 2), is_npu=True) as (cid, vid):
            flat = (cid * 2 + vid) * tile_heads
            row = flat // heads
            if row < rows:
                end = positions[T.min(rows - 1, row // 16 * 16 + 15)] + 1
                if not compressed or long_length == small:
                    if bucket == 0:
                        normalize(scores, positions, selected, sink, probabilities, flat, active, 0)
                elif bucket == 0:
                    if end <= 2048:
                        normalize(scores, positions, selected, sink, probabilities, flat, active, 0)
                elif bucket == 1:
                    if end > 2048:
                        if end <= 4096:
                            normalize(
                                scores, positions, selected, sink, probabilities, flat, active, 0
                            )
                else:
                    if end > 4096:
                        normalize(scores, positions, selected, sink, probabilities, flat, active, 0)

    @T.prim_func
    def replay_main(
        scores: T.Tensor((rows * heads, score_length), "float32"),
        positions: T.Tensor((rows,), "int32"),
        selected: T.Tensor((rows, long_length), "int32"),
        sink: T.Tensor((heads,), "float32"),
        probabilities: T.Tensor((rows * heads, probability_length), "bfloat16"),
        replay_start: T.Tensor((1,), "int32"),
    ):
        with T.Kernel(T.ceildiv(rows * heads, tile_heads * 2), is_npu=True) as (cid, vid):
            flat = (cid * 2 + vid) * tile_heads
            row = flat // heads
            if row < rows:
                end = positions[T.min(rows - 1, row // 16 * 16 + 15)] + 1
                if bucket == 0:
                    if end <= 2048:
                        normalize(
                            scores,
                            positions,
                            selected,
                            sink,
                            probabilities,
                            flat,
                            active,
                            replay_start[0],
                        )
                elif bucket == 1:
                    if end > 2048:
                        if end <= 4096:
                            normalize(
                                scores,
                                positions,
                                selected,
                                sink,
                                probabilities,
                                flat,
                                active,
                                replay_start[0],
                            )
                else:
                    if end > 4096:
                        normalize(
                            scores,
                            positions,
                            selected,
                            sink,
                            probabilities,
                            flat,
                            active,
                            replay_start[0],
                        )

    return replay_main if replay else main
