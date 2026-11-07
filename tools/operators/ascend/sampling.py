"""Dynamic TP-sharded categorical sampling with counter-based Philox2x32-10.

One bounded template serves ordinary and verification rows. Controls contain
request/position keys, the global vocabulary origin and inverse temperature;
zero inverse temperature selects greedy. Only the active rows are accessed.
"""

import tilelang.language as T


@T.macro
def uniform(out, first, seed, stream):
    counter = T.alloc_ub(out.shape, "int32")
    other = T.alloc_ub(out.shape, "int32")
    key = T.alloc_ub(out.shape, "int32")
    a = T.alloc_ub(out.shape, "int32")
    b = T.alloc_ub(out.shape, "int32")
    carry = T.alloc_ub(out.shape, "int32")
    high = T.alloc_ub(out.shape, "int32")
    low = T.alloc_ub(out.shape, "int32")
    keys = T.alloc_ub(out.shape, "int32")
    counter_u = T.alloc_ub(out.shape, "uint32")
    T.reinterpretcast(counter_u, counter, "uint32_t")
    b_u = T.alloc_ub(out.shape, "uint32")
    T.reinterpretcast(b_u, b, "uint32_t")
    carry_u = T.alloc_ub(out.shape, "uint32")
    T.reinterpretcast(carry_u, carry, "uint32_t")
    low_u = T.alloc_ub(out.shape, "uint32")
    T.reinterpretcast(low_u, low, "uint32_t")
    a_u = T.alloc_ub(out.shape, "uint32")
    T.reinterpretcast(a_u, a, "uint32_t")
    key_u = T.alloc_ub(out.shape, "uint32")
    T.reinterpretcast(key_u, key, "uint32_t")
    counter_h = T.alloc_ub((1, out.shape[1] * 2), "uint16")
    T.reinterpretcast(counter_h, counter, "uint16_t")
    a_h = T.alloc_ub((1, out.shape[1] * 2), "uint16")
    T.reinterpretcast(a_h, a, "uint16_t")
    keys_h = T.alloc_ub((1, out.shape[1] * 2), "uint16")
    T.reinterpretcast(keys_h, keys, "uint16_t")
    key_h = T.alloc_ub((1, out.shape[1] * 2), "uint16")
    T.reinterpretcast(key_h, key, "uint16_t")
    b_h = T.alloc_ub((1, out.shape[1] * 2), "uint16")
    T.reinterpretcast(b_h, b, "uint16_t")
    high_h = T.alloc_ub((1, out.shape[1] * 2), "uint16")
    T.reinterpretcast(high_h, high, "uint16_t")
    other_h = T.alloc_ub((1, out.shape[1] * 2), "uint16")
    T.reinterpretcast(other_h, other, "uint16_t")
    T.tile.createvecindex(counter, T.int32(first))
    T.pipe_barrier("V")
    T.tile.fill(other, T.reinterpret("int32", stream))
    T.pipe_barrier("V")
    for round_index in T.unroll(10):
        T.tile.fill(keys, 65535)
        T.pipe_barrier("V")
        T.tile.bitwise_and(a_h, counter_h, keys_h)
        T.pipe_barrier("V")
        T.tile.bitwise_rshift(b_u, counter_u, 16)
        T.pipe_barrier("V")
        T.tile.fill(key, 53651)
        T.pipe_barrier("V")
        T.tile.mul(low, a, key)
        T.pipe_barrier("V")
        T.tile.fill(key, 53846)
        T.pipe_barrier("V")
        T.tile.mul(high, b, key)
        T.pipe_barrier("V")
        T.tile.mul(a, a, key)
        T.pipe_barrier("V")
        T.tile.fill(key, 53651)
        T.pipe_barrier("V")
        T.tile.mul(b, b, key)
        T.pipe_barrier("V")
        T.tile.bitwise_rshift(carry_u, low_u, 16)
        T.pipe_barrier("V")
        T.tile.bitwise_and(key_h, a_h, keys_h)
        T.pipe_barrier("V")
        T.tile.add(carry, carry, key)
        T.pipe_barrier("V")
        T.tile.bitwise_and(key_h, b_h, keys_h)
        T.pipe_barrier("V")
        T.tile.add(carry, carry, key)
        T.pipe_barrier("V")
        T.tile.bitwise_rshift(carry_u, carry_u, 16)
        T.pipe_barrier("V")
        T.tile.bitwise_rshift(a_u, a_u, 16)
        T.pipe_barrier("V")
        T.tile.bitwise_rshift(b_u, b_u, 16)
        T.pipe_barrier("V")
        T.tile.add(high, high, a)
        T.pipe_barrier("V")
        T.tile.add(high, high, b)
        T.pipe_barrier("V")
        T.tile.add(high, high, carry)
        T.pipe_barrier("V")
        T.tile.fill(key, -766062189)
        T.pipe_barrier("V")
        T.tile.mul(low, counter, key)
        T.pipe_barrier("V")
        # AscendC Xor is logical XOR; use integer bitwise And/Or/Not.
        T.tile.bitwise_and(key_h, high_h, other_h)
        T.pipe_barrier("V")
        T.tile.bitwise_or(high_h, high_h, other_h)
        T.pipe_barrier("V")
        T.tile.bitwise_not(key_h, key_h)
        T.pipe_barrier("V")
        T.tile.bitwise_and(high_h, high_h, key_h)
        T.pipe_barrier("V")
        T.tile.fill(
            keys, T.reinterpret("int32", seed + T.uint32(round_index) * T.uint32(2654435769))
        )
        T.pipe_barrier("V")
        T.tile.bitwise_and(key_h, high_h, keys_h)
        T.pipe_barrier("V")
        T.tile.bitwise_or(counter_h, high_h, keys_h)
        T.pipe_barrier("V")
        T.tile.bitwise_not(key_h, key_h)
        T.pipe_barrier("V")
        T.tile.bitwise_and(counter_h, counter_h, key_h)
        T.pipe_barrier("V")
        T.tile.fill(key, 0)
        T.pipe_barrier("V")
        T.tile.add(other, low, key)
        T.pipe_barrier("V")
    T.tile.bitwise_rshift(key_u, counter_u, 9)
    T.pipe_barrier("V")
    T.tile.cast(out, key, "CAST_NONE", out.shape[0] * out.shape[1])
    T.pipe_barrier("V")
    T.tile.add(out, out, 0.5)
    T.pipe_barrier("V")
    scale = T.alloc_ub((1, 1), "float32")
    scale[0, 0] = T.reinterpret("float32", T.uint32(872415232))  # exact 2**-23
    T.tile.mul(out, out, scale[0, 0])
    T.pipe_barrier("V")


def categorical(rows, vocab, tile=1024):
    parts = (vocab + tile - 1) // tile

    @T.prim_func
    def main(
        logits: T.Tensor((rows, vocab), "float32"),
        controls: T.Tensor((rows, 4), "int32"),
        active: T.Tensor((1,), "int32"),
        partial: T.Tensor((rows, parts, 2), "float32"),
    ):
        with T.Kernel(T.ceildiv(rows * parts, 2), is_npu=True) as (cid, vid):
            x = T.alloc_ub((1, tile), "float32")
            y = T.alloc_ub((1, tile), "float32")
            indices = T.alloc_ub((1, tile), "float32")
            mask = T.alloc_ub((tile // 8,), "uint8")
            maximum = T.alloc_ub((1, 1), "float32")
            minimum = T.alloc_ub((1, 1), "float32")
            pair = T.alloc_ub((1, 8), "float32")
            index = cid * 2 + vid
            row, part = index // parts, index % parts
            begin = part * tile
            live = T.min(tile, vocab - begin)
            if row < T.min(rows, active[0]):
                T.tile.fill(x, 0.0)
                T.copy(logits[row : row + 1, begin : begin + live], x[:, :live])
                # Detect invalid real logits before masking the padded tail.
                T.tile.compare(mask, x, x, "EQ")
                T.tile.fill(y, 1.0)
                T.tile.select(y, mask, y, 0.0, "VSEL_TENSOR_SCALAR_MODE")
                T.reduce_min(y, minimum, dim=1)
                T.tile.abs(y, x)
                T.reduce_max(y, maximum, dim=1)
                pair[0, 2] = T.if_then_else(
                    minimum[0, 0] == 0.0 or maximum[0, 0] == T.infinity("float32"),
                    1.0,
                    0.0,
                )
                inverse_temperature = T.reinterpret("float32", controls[row, 3])
                if inverse_temperature > 0.0:
                    uniform(
                        y,
                        controls[row, 2] + begin,
                        T.reinterpret("uint32", controls[row, 0]),
                        T.reinterpret("uint32", controls[row, 1]),
                    )
                    T.tile.ln(y, y)
                    T.tile.mul(y, y, -1.0)
                    T.tile.ln(y, y)
                    T.tile.mul(x, x, inverse_temperature)
                    T.tile.sub(x, x, y)
                T.tile.createvecindex(indices, T.float32(0.0))
                T.tile.compare(mask, indices, T.cast(live, "float32"), "LT")
                T.tile.select(x, mask, x, -T.infinity("float32"), "VSEL_TENSOR_SCALAR_MODE")
                T.reduce_max(x, maximum, dim=1)
                T.tile.add(indices, indices, T.cast(begin, "float32"))
                T.tile.compare(mask, x, maximum[0, 0], "EQ")
                T.tile.select(
                    indices, mask, indices, T.cast(vocab, "float32"), "VSEL_TENSOR_SCALAR_MODE"
                )
                T.reduce_min(indices, minimum, dim=1)
                pair[0, 0] = T.if_then_else(pair[0, 2] > 0.0, T.infinity("float32"), maximum[0, 0])
                pair[0, 1] = minimum[0, 0]
                T.copy(pair[:, :2], partial[row, part : part + 1, :])
                T.pipe_barrier("ALL")

    return main


def merge(rows, parts):
    @T.prim_func
    def main(
        partial: T.Tensor((rows, parts, 2), "float32"),
        active: T.Tensor((1,), "int32"),
        output: T.Tensor((rows, 2), "float32"),
    ):
        with T.Kernel(T.ceildiv(rows, 2), is_npu=True) as (cid, vid):
            pairs = T.alloc_ub((parts, 8), "float32")
            selected = T.alloc_ub((8,), "int32")
            pair = T.alloc_ub((1, 8), "float32")
            row = cid * 2 + vid
            if row < T.min(rows, active[0]):
                for part in T.serial(parts):
                    T.copy(partial[row, part : part + 1, :], pairs[part : part + 1, :2])
                selected[0] = 0
                for part in T.serial(1, parts):
                    if pairs[part, 0] > pairs[selected[0], 0] or (
                        pairs[part, 0] == pairs[selected[0], 0]
                        and pairs[part, 1] < pairs[selected[0], 1]
                    ):
                        selected[0] = part
                pair[0, 0] = pairs[selected[0], 0]
                pair[0, 1] = pairs[selected[0], 1]
                T.copy(pair[:, :2], output[row : row + 1, :])

    return main


def markov(rows, vocab, block_size=16, tile=2048):
    """Fuse shared target randomness into the existing sequential Markov head.

    Controls are in the same physical owner/position order as the draft logits.
    Each step sees only earlier sampled draft tokens through the Markov bias.
    Zero inverse temperature retains the original greedy arithmetic and tie rule.
    """
    parts = (vocab + tile - 1) // tile

    @T.prim_func
    def main(
        base: T.Tensor((rows * block_size, vocab), "float32"),
        bias: T.Tensor((rows, vocab), "float32"),
        proposals: T.Tensor((rows * block_size, vocab), "float32"),
        packed: T.Tensor((parts, rows, 2), "float32"),
        step_index: T.Tensor((1,), "int32"),
        controls: T.Tensor((rows * block_size, 4), "int32"),
    ):
        with T.Kernel(T.ceildiv(rows * parts, 2), is_npu=True) as (cid, vid):
            scores = T.alloc_ub((1, tile), "float32")
            bias_tile = T.alloc_ub((1, tile), "float32")
            noise = T.alloc_ub((1, 1024), "float32")
            chunk = T.alloc_ub((1, 1024), "float32")
            indices = T.alloc_ub((1, tile), "float32")
            mask = T.alloc_ub((tile // 8,), "uint8")
            maximum = T.alloc_ub((1, 1), "float32")
            minimum = T.alloc_ub((1, 1), "float32")
            pair = T.alloc_ub((1, 8), "float32")
            index = cid * 2 + vid
            row, part = index // parts, index % parts
            begin = part * tile
            live = T.min(tile, vocab - begin)
            if row < rows:
                step = step_index[0]
                position = row * block_size + step
                T.tile.fill(scores, -T.infinity("float32"))
                T.tile.fill(bias_tile, 0.0)
                T.copy(base[position : position + 1, begin : begin + live], scores[:, :live])
                T.copy(bias[row : row + 1, begin : begin + live], bias_tile[:, :live])
                T.tile.add(scores, scores, bias_tile)
                T.copy(scores[:, :live], proposals[position : position + 1, begin : begin + live])
                inverse = T.reinterpret("float32", controls[position, 3])
                if inverse > 0.0:
                    # Keep the established 2048-wide candidate geometry while
                    # bounding Philox scratch to the shared 1024-wide template.
                    for half in T.serial(tile // 1024):
                        uniform(
                            noise,
                            controls[position, 2] + begin + half * 1024,
                            T.reinterpret("uint32", controls[position, 0]),
                            T.reinterpret("uint32", controls[position, 1]),
                        )
                        T.tile.ln(noise, noise)
                        T.tile.mul(noise, noise, -1.0)
                        T.tile.ln(noise, noise)
                        T.copy(scores[:, half * 1024 : (half + 1) * 1024], chunk)
                        T.tile.mul(chunk, chunk, inverse)
                        T.tile.sub(chunk, chunk, noise)
                        T.copy(chunk, scores[:, half * 1024 : (half + 1) * 1024])
                T.tile.createvecindex(indices, T.float32(0.0))
                T.tile.compare(mask, indices, T.cast(live, "float32"), "LT")
                T.tile.select(
                    scores, mask, scores, -T.infinity("float32"), "VSEL_TENSOR_SCALAR_MODE"
                )
                T.reduce_max(scores, maximum, dim=1)
                T.tile.add(indices, indices, T.cast(begin, "float32"))
                T.tile.compare(mask, scores, maximum[0, 0], "EQ")
                T.tile.select(
                    indices, mask, indices, T.cast(vocab, "float32"), "VSEL_TENSOR_SCALAR_MODE"
                )
                T.reduce_min(indices, minimum, dim=1)
                pair[0, 0] = maximum[0, 0]
                pair[0, 1] = minimum[0, 0]
                T.copy(pair[:, :2], packed[part, row : row + 1, :])
                T.pipe_barrier("ALL")

    return main
