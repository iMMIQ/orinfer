"""Owner-addressed input continuation; all row families share these factories.

The host row map describes membership, never a sampled continuation token.
State is [pending token, next input position, accepted drafts, owner epoch].
CPU prefix/stop processing retains its own committed history.
"""

import tilelang.language as T


def tokens(rows, owners, drafts=16):
    @T.prim_func
    def main(
        mapping: T.Tensor((rows, 8), "int32"),
        state: T.Tensor((owners, 4), "int32"),
        proposals: T.Tensor((owners, drafts), "int32"),
        ids: T.Tensor((rows,), "int32"),
        positions: T.Tensor((rows,), "int32"),
    ):
        with T.Kernel(T.ceildiv(rows, 2), is_npu=True) as (cid, vid):
            row = cid * 2 + vid
            if row < rows:
                owner = mapping[row, 0]
                offset = mapping[row, 1]
                mode = mapping[row, 3]
                epoch = mapping[row, 6]
                values = T.alloc_ub((8,), "int32")
                values[0] = mapping[row, 5]
                values[1] = mapping[row, 4] + offset
                if mode != 0 and state[owner, 3] == epoch:
                    values[1] = state[owner, 1] + offset
                    values[0] = state[owner, 0]
                    if mode == 2 and offset > 0:
                        values[0] = proposals[owner, offset - 1]
                    if mode == 3 and offset > 0:
                        values[0] = 128799
                T.copy(values[:1], ids[row : row + 1])
                position = T.alloc_ub((8,), "int32")
                position[0] = values[1]
                T.copy(position[:1], positions[row : row + 1])

    return main


def coordinates(rows, context, draft=0, storage=128, source=False):
    @T.prim_func
    def main(
        mapping: T.Tensor((rows, 8), "int32"),
        positions: T.Tensor((rows,), "int32"),
        rope: T.Tensor((4, context, 64), "bfloat16"),
        base_cos: T.Tensor((rows, 64), "bfloat16"),
        base_sin: T.Tensor((rows, 64), "bfloat16"),
        long_cos: T.Tensor((rows, 64), "bfloat16"),
        long_sin: T.Tensor((rows, 64), "bfloat16"),
        source_cos: T.Tensor((rows, 64), "bfloat16"),
        source_sin: T.Tensor((rows, 64), "bfloat16"),
        slots: T.Tensor((rows,), "int32"),
        c2_slots: T.Tensor((rows,), "int32"),
        swa: T.Tensor((rows, 128), "int32"),
        source_swa: T.Tensor((rows, 128), "int32"),
        ring: T.Tensor((rows, 8), "int32"),
        premix: T.Tensor((rows, 8), "float32"),
    ):
        with T.Kernel(T.ceildiv(rows, 2), is_npu=True) as (cid, vid):
            row = cid * 2 + vid
            b = T.alloc_ub((1, 64), "bfloat16")
            sw = T.alloc_ub((1, 128), "int32")
            rs = T.alloc_ub((1, 8), "int32")
            mix = T.alloc_ub((1, 8), "float32")
            if row < rows:
                position = positions[row]
                offset = mapping[row, 1]
                base = position - offset
                for plane in T.unroll(4):
                    T.copy(rope[plane, position : position + 1, :], b)
                    if plane == 0:
                        T.copy(b, base_cos[row : row + 1, :])
                    elif plane == 1:
                        T.copy(b, base_sin[row : row + 1, :])
                    elif plane == 2:
                        T.copy(b, long_cos[row : row + 1, :])
                    else:
                        T.copy(b, long_sin[row : row + 1, :])
                even = position // 2 * 2
                T.copy(rope[2, even : even + 1, :], b)
                T.copy(b, source_cos[row : row + 1, :])
                T.copy(rope[3, even : even + 1, :], b)
                T.copy(b, source_sin[row : row + 1, :])
                rs[0, 0] = position
                T.copy(rs[0, :1], slots[row : row + 1])
                if draft != 0:
                    rs[0, 0] = offset
                else:
                    rs[0, 0] = T.if_then_else(position % 2 == 1, position // 2, -1)
                T.copy(rs[0, :1], c2_slots[row : row + 1])
                first = T.max(0, base - 128) if draft != 0 else T.max(0, position - 127)
                last = base if draft != 0 else position + 1
                for lane in T.serial(128):
                    previous = first + lane
                    sw[0, lane] = T.if_then_else(
                        previous < last, previous % storage if draft != 0 else previous, -1
                    )
                T.copy(sw, swa[row : row + 1, :])
                if source:
                    T.copy(sw, source_swa[row : row + 1, :])
                for lane in T.serial(8):
                    rs[0, lane] = 0
                    mix[0, lane] = 0.0
                rs[0, 0] = position
                rs[0, 1] = 1
                rs[0, 2] = T.if_then_else(offset == 0, -1, offset - 1)
                rs[0, 3] = 1
                rs[0, 4] = base + mapping[row, 2] - 1
                rs[0, 5] = 1
                mix[0, 0] = 1.0
                T.copy(rs, ring[row : row + 1, :])
                T.copy(mix, premix[row : row + 1, :])

    return main


def embedding(rows, vocab, width):
    @T.prim_func
    def main(
        table: T.Tensor((vocab, width), "bfloat16"),
        ids: T.Tensor((rows,), "int32"),
        out: T.Tensor((rows, width), "bfloat16"),
    ):
        with T.Kernel(T.ceildiv(rows, 2), is_npu=True) as (cid, vid):
            row = cid * 2 + vid
            value = T.alloc_ub((1, width), "bfloat16")
            if row < rows:
                token = ids[row]
                if token >= 0 and token < vocab:
                    T.copy(table[token : token + 1, :], value)
                    T.copy(value, out[row : row + 1, :])

    return main


def draft_embedding(rows, owners, width=5120):
    @T.prim_func
    def main(
        mapping: T.Tensor((rows, 8), "int32"),
        cache: T.Tensor((owners, width), "bfloat16"),
        mask: T.Tensor((width,), "bfloat16"),
        out: T.Tensor((rows, width), "bfloat16"),
    ):
        with T.Kernel(T.ceildiv(rows, 2), is_npu=True) as (cid, vid):
            row = cid * 2 + vid
            data = T.alloc_ub((1, width), "bfloat16")
            if row < rows:
                owner = mapping[row, 0]
                if mapping[row, 1] == 0:
                    T.copy(cache[owner : owner + 1, :], data)
                else:
                    T.copy(mask[:], data[0, :])
                T.copy(data, out[row : row + 1, :])

    return main


def merge(rows, capacity, ranks, vocab):
    @T.prim_func
    def main(
        gathered: T.Tensor((ranks * capacity * 2,), "float32"),
        output: T.Tensor((rows, 2), "float32"),
    ):
        with T.Kernel(T.ceildiv(rows, 2), is_npu=True) as (cid, vid):
            row = cid * 2 + vid
            selected = T.alloc_ub((1, 8), "float32")
            if row < rows:
                selected[0, 0] = gathered[row * 2]
                selected[0, 1] = gathered[row * 2 + 1]
                for rank in T.serial(1, ranks):
                    score = gathered[(rank * rows + row) * 2]
                    token = gathered[(rank * rows + row) * 2 + 1] + rank * vocab
                    if score > selected[0, 0] or (
                        score == selected[0, 0] and token < selected[0, 1]
                    ):
                        selected[0, 0] = score
                        selected[0, 1] = token
                T.copy(selected[:, :2], output[row : row + 1, :])

    return main


def commit(rows, owners, capacity, input_capacity, vocab, drafts=16, width=5120):
    @T.prim_func
    def main(
        mapping: T.Tensor((rows, 8), "int32"),
        output: T.Tensor((capacity, 2), "float32"),
        positions: T.Tensor((input_capacity,), "int32"),
        proposals: T.Tensor((owners, drafts), "int32"),
        state: T.Tensor((owners, 4), "int32"),
        ids: T.Tensor((input_capacity,), "int32"),
        history: T.Tensor((owners, 4), "int32"),
        table: T.Tensor((vocab, width), "bfloat16"),
        embeddings: T.Tensor((owners, width), "bfloat16"),
    ):
        with T.Kernel(T.ceildiv(rows, 2), is_npu=True) as (cid, vid):
            row = cid * 2 + vid
            result = T.alloc_ub((1, 8), "int32")
            previous = T.alloc_ub((1, 8), "int32")
            embedding = T.alloc_ub((1, width), "bfloat16")
            if row < rows:
                owner = mapping[row, 0]
                first = mapping[row, 1]
                count = mapping[row, 2]
                position_row = mapping[row, 4]
                accepted = T.alloc_ub((8,), "int32")
                accepted[0] = 0
                for i in T.serial(count - 1):
                    if accepted[0] == i and proposals[owner, i] == T.cast(
                        output[first + i, 1], "int32"
                    ):
                        accepted[0] = accepted[0] + 1
                result[0, 0] = T.cast(output[first + accepted[0], 1], "int32")
                result[0, 1] = positions[position_row] + accepted[0] + 1
                result[0, 2] = accepted[0]
                result[0, 3] = mapping[row, 6]
                for shift in T.serial(3):
                    index = position_row + accepted[0] - shift
                    start = mapping[row, 5]
                    previous[0, shift] = T.if_then_else(
                        index >= start,
                        ids[T.max(start, index)],
                        history[owner, T.max(0, T.min(2, start - 1 - index))],
                    )
                previous[0, 3] = mapping[row, 6]
                T.copy(result[:, :4], state[owner : owner + 1, :])
                T.copy(previous[:, :4], history[owner : owner + 1, :])
                winner = result[0, 0] // vocab
                if winner == mapping[row, 3]:
                    local = T.max(0, T.min(vocab - 1, result[0, 0] % vocab))
                    T.copy(table[local : local + 1, :], embedding)
                else:
                    T.tile.fill(embedding, 0.0)
                T.copy(embedding, embeddings[row : row + 1, :])

    return main


def scatter_embedding(rows, owners, width=5120):
    @T.prim_func
    def main(
        mapping: T.Tensor((rows, 8), "int32"),
        compact: T.Tensor((owners, width), "bfloat16"),
        cache: T.Tensor((owners, width), "bfloat16"),
    ):
        with T.Kernel(T.ceildiv(rows, 2), is_npu=True) as (cid, vid):
            row = cid * 2 + vid
            data = T.alloc_ub((1, width), "bfloat16")
            if row < rows:
                owner = mapping[row, 0]
                T.copy(compact[row : row + 1, :], data)
                T.copy(data, cache[owner : owner + 1, :])

    return main


def hashes(rows, owners, layers, heads, vocab, pad, barrier_a, barrier_b):
    width = heads * 3

    @T.prim_func
    def main(
        mapping: T.Tensor((rows, 8), "int32"),
        ids: T.Tensor((rows,), "int32"),
        history: T.Tensor((owners, 4), "int32"),
        token_map: T.Tensor((vocab,), "int32"),
        multipliers: T.Tensor((layers, 4), "uint64"),
        buckets: T.Tensor((layers, width, 2), "int64"),
        out: T.Tensor((rows, layers * width), "int64"),
        valid: T.Tensor((rows,), "int32"),
    ):
        with T.Kernel(T.ceildiv(rows, 2), is_npu=True) as (cid, vid):
            row = cid * 2 + vid
            window = T.alloc_ub((8,), "uint64")
            rolling = T.alloc_ub((8,), "uint64")
            flags = T.alloc_ub((8,), "int32")
            hashes = T.alloc_ub((1, width), "int64")
            if row < rows:
                owner = mapping[row, 0]
                offset = mapping[row, 1]
                flags[0] = T.if_then_else(ids[row] == barrier_a or ids[row] == barrier_b, 0, 1)
                flags[1] = flags[0]
                for shift in T.serial(4):
                    window[shift] = T.cast(token_map[pad], "uint64")
                    raw = T.if_then_else(
                        shift <= offset,
                        ids[T.max(0, row - shift)],
                        history[owner, T.max(0, shift - offset - 1)],
                    )
                    if raw < 0 or raw == barrier_a or raw == barrier_b:
                        flags[1] = 0
                    if flags[1] != 0:
                        window[shift] = T.cast(token_map[T.max(0, raw)], "uint64")
                for layer in T.serial(layers):
                    rolling[0] = window[0] * multipliers[layer, 0]
                    for order in T.serial(1, 4):
                        rolling[0] = T.bitwise_xor(
                            rolling[0], window[order] * multipliers[layer, order]
                        )
                        for head in T.serial(heads):
                            bucket = (order - 1) * heads + head
                            hashes[0, bucket] = 0
                            if flags[0] != 0:
                                # Scalar FloorMod lowers to C remainder on A2.
                                hashes[0, bucket] = T.floormod(
                                    T.cast(rolling[0], "int64"), buckets[layer, bucket, 0]
                                )
                                if hashes[0, bucket] < 0:
                                    hashes[0, bucket] = (
                                        hashes[0, bucket] + buckets[layer, bucket, 0]
                                    )
                                hashes[0, bucket] = hashes[0, bucket] + buckets[layer, bucket, 1]
                    T.copy(hashes, out[row : row + 1, layer * width : (layer + 1) * width])
                T.copy(flags[:1], valid[row : row + 1])

    return main


def save_draft(rows, owners, capacity, stride, drafts=16):
    @T.prim_func
    def main(
        mapping: T.Tensor((rows, 8), "int32"),
        indices: T.Tensor((capacity,), "int32"),
        proposals: T.Tensor((owners, drafts), "int32"),
    ):
        with T.Kernel(T.ceildiv(rows, 2), is_npu=True) as (cid, vid):
            row = cid * 2 + vid
            result = T.alloc_ub((1, drafts), "int32")
            if row < rows:
                owner = mapping[row, 0]
                for i in T.serial(drafts):
                    result[0, i] = 0
                    if i < mapping[row, 2]:
                        result[0, i] = indices[row * stride + i]
                T.copy(result, proposals[owner : owner + 1, :])

    return main
