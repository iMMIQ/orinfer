"""DSv4.1 C2 pooling and request-private 32-row FP32 rings on A2.

Metadata per token: [absolute_position, ring_page, predecessor_input_row,
active, last_committed_position, commit, reserved, reserved]. Page zero is
reserved. Pool reads precede the separate ring commit kernel on one stream.
"""

import tilelang.language as T


def pool(tokens, pages, width=512):
    @T.prim_func
    def main(
        kv: T.Tensor((tokens, width), "float32"),
        scores: T.Tensor((tokens, width), "float32"),
        ring: T.Tensor((pages, 32, 2 * width), "float32"),
        metadata: T.Tensor((tokens, 8), "int32"),
        out: T.Tensor((tokens, width), "bfloat16"),
        errors: T.Tensor((tokens,), "int32"),
    ):
        with T.Kernel(T.ceildiv(tokens, 2), is_npu=True) as (cid, vid):
            meta = T.alloc_ub((8,), "int32")
            previous_meta = T.alloc_ub((8,), "int32")
            k0 = T.alloc_ub((width,), "float32")
            k1 = T.alloc_ub((width,), "float32")
            s0 = T.alloc_ub((width,), "float32")
            s1 = T.alloc_ub((width,), "float32")
            maximum = T.alloc_ub((width,), "float32")
            denominator = T.alloc_ub((width,), "float32")
            y = T.alloc_ub((width,), "bfloat16")
            flag = T.alloc_ub((8,), "int32")
            row = cid * 2 + vid
            if row < tokens:
                T.copy(metadata[row, :], meta)
                T.tile.fill(y, 0.0)
                flag[0] = 0
                position = meta[0]
                page = meta[1]
                previous = meta[2]
                if meta[3] != 0:
                    if (
                        position < 0
                        or page <= 0
                        or page >= pages
                        or previous < -1
                        or previous >= tokens
                    ):
                        flag[0] = 1
                    else:
                        if position % 2 == 1:
                            if previous >= 0:
                                T.copy(metadata[previous, :], previous_meta)
                                if (
                                    previous_meta[0] != position - 1
                                    or previous_meta[1] != page
                                    or previous_meta[3] == 0
                                ):
                                    flag[0] = 1
                            if flag[0] == 0:
                                T.copy(kv[row, :], k1)
                                T.copy(scores[row, :], s1)
                                if previous >= 0:
                                    T.copy(kv[previous, :], k0)
                                    T.copy(scores[previous, :], s0)
                                else:
                                    T.copy(ring[page, (position - 1) % 32, :width], k0)
                                    T.copy(ring[page, (position - 1) % 32, width:], s0)
                                T.tile.max(maximum, s0, s1)
                                T.tile.sub(s0, s0, maximum)
                                T.tile.sub(s1, s1, maximum)
                                T.tile.exp(s0, s0)
                                T.tile.exp(s1, s1)
                                T.tile.add(denominator, s0, s1)
                                T.tile.div(s0, s0, denominator)
                                T.tile.div(s1, s1, denominator)
                                T.tile.mul(k0, k0, s0)
                                T.tile.mul(k1, k1, s1)
                                T.tile.add(k0, k0, k1)
                                T.tile.cast(y, k0, "CAST_RINT", width)
                T.copy(y, out[row, :])
                T.copy(flag[:1], errors[row : row + 1])

    return main


def commit(tokens, pages, width=512):
    """Only the last 32 committed positions write; owners/positions must be unique."""

    @T.prim_func
    def main(
        kv: T.Tensor((tokens, width), "float32"),
        scores: T.Tensor((tokens, width), "float32"),
        metadata: T.Tensor((tokens, 8), "int32"),
        ring: T.Tensor((pages, 32, 2 * width), "float32"),
        errors: T.Tensor((tokens,), "int32"),
    ):
        with T.Kernel(T.ceildiv(tokens, 2), is_npu=True) as (cid, vid):
            meta = T.alloc_ub((8,), "int32")
            k = T.alloc_ub((width,), "float32")
            s = T.alloc_ub((width,), "float32")
            flag = T.alloc_ub((8,), "int32")
            row = cid * 2 + vid
            if row < tokens:
                T.copy(metadata[row, :], meta)
                position = meta[0]
                page = meta[1]
                last = meta[4]
                flag[0] = 0
                if meta[3] != 0 and meta[5] != 0:
                    if position < 0 or page <= 0 or page >= pages or last < position:
                        flag[0] = 1
                    else:
                        if last - position < 32:
                            T.copy(kv[row, :], k)
                            T.copy(scores[row, :], s)
                            T.copy(k, ring[page, position % 32, :width])
                            T.copy(s, ring[page, position % 32, width:])
                T.copy(flag[:1], errors[row : row + 1])

    return main
