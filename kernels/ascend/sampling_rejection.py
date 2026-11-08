"""TP-sharded exact rejection sampling around the shared categorical RNG.

Draft logits remain in a bounded owner/position pool. Only log-sum-exp and
selected-token statistics cross ranks; the vocabulary stays on device.
"""

import tilelang.language as T

from kernels.ascend.sampling import uniform


def statistics(rows, vocab, pool_rows, tile=1024):
    parts = (vocab + tile - 1) // tile

    @T.prim_func
    def main(
        logits: T.Tensor((rows, vocab), "float32"),
        draft: T.Tensor((pool_rows, vocab), "float32"),
        controls: T.Tensor((rows, 4), "int32"),
        proposals: T.Tensor((rows, 4), "int32"),
        active: T.Tensor((1,), "int32"),
        partial: T.Tensor((rows, parts, 4), "float32"),
    ):
        with T.Kernel(T.ceildiv(rows * parts, 2), is_npu=True) as (cid, vid):
            x = T.alloc_ub((1, tile), "float32")
            y = T.alloc_ub((1, tile), "float32")
            indices = T.alloc_ub((1, tile), "float32")
            bits = T.alloc_ub((tile // 8,), "uint8")
            maximum = T.alloc_ub((1, 1), "float32")
            total = T.alloc_ub((1, 1), "float32")
            packet = T.alloc_ub((1, 8), "float32")
            index = cid * 2 + vid
            row, part = index // parts, index % parts
            start = part * tile
            live = T.min(tile, vocab - start)
            if row < T.min(rows, active[0]):
                inverse = T.reinterpret("float32", controls[row, 3])
                inverse = T.if_then_else(inverse > 0.0, inverse, 1.0)
                for kind in T.unroll(2):
                    T.tile.fill(x, 0.0)
                    if kind == 0:
                        T.copy(logits[row : row + 1, start : start + live], x[:, :live])
                    elif proposals[row, 0] >= 0:
                        pool_row = proposals[row, 0]
                        T.copy(draft[pool_row : pool_row + 1, start : start + live], x[:, :live])
                    T.tile.compare(bits, x, x, "EQ")
                    T.tile.fill(y, 1.0)
                    T.tile.select(y, bits, y, 0.0, "VSEL_TENSOR_SCALAR_MODE")
                    T.reduce_min(y, total, dim=1)
                    packet[0, 6] = T.if_then_else(total[0, 0] == 0.0, 1.0, 0.0)
                    T.tile.abs(y, x)
                    T.reduce_max(y, maximum, dim=1)
                    packet[0, 6] = T.if_then_else(
                        maximum[0, 0] == T.infinity("float32"), 1.0, packet[0, 6]
                    )
                    T.tile.mul(x, x, inverse)
                    T.tile.createvecindex(indices, T.float32(0.0))
                    T.tile.compare(bits, indices, T.cast(live, "float32"), "LT")
                    T.tile.select(x, bits, x, -T.infinity("float32"), "VSEL_TENSOR_SCALAR_MODE")
                    T.reduce_max(x, maximum, dim=1)
                    packet[0, kind * 2] = T.if_then_else(
                        packet[0, 6] > 0.0, T.infinity("float32"), maximum[0, 0]
                    )
                    T.tile.sub(y, x, maximum[0, 0])
                    T.tile.exp(y, y)
                    T.reduce_sum(y, total, dim=1)
                    packet[0, kind * 2 + 1] = total[0, 0]
                T.copy(packet[:, :4], partial[row, part : part + 1, :])
                T.pipe_barrier("ALL")

    return main


def merge_statistics(rows, vocab, pool_rows):
    parts = (vocab + 1023) // 1024
    width = ((parts + 63) // 64) * 64

    @T.prim_func
    def main(
        partial: T.Tensor((rows, parts, 4), "float32"),
        logits: T.Tensor((rows, vocab), "float32"),
        draft: T.Tensor((pool_rows, vocab), "float32"),
        controls: T.Tensor((rows, 4), "int32"),
        proposals: T.Tensor((rows, 4), "int32"),
        active: T.Tensor((1,), "int32"),
        out: T.Tensor((rows, 8), "float32"),
    ):
        with T.Kernel(T.ceildiv(rows, 2), is_npu=True) as (cid, vid):
            values = T.alloc_ub((parts, 8), "float32")
            maxima = T.alloc_ub((1, width), "float32")
            masses = T.alloc_ub((1, width), "float32")
            maximum = T.alloc_ub((1, 1), "float32")
            total = T.alloc_ub((1, 1), "float32")
            packet = T.alloc_ub((1, 8), "float32")
            row = cid * 2 + vid
            if row < T.min(rows, active[0]):
                T.tile.fill(packet, 0.0)
                for part in T.serial(parts):
                    T.copy(partial[row, part : part + 1, :], values[part : part + 1, :4])
                for kind in T.unroll(2):
                    T.tile.fill(maxima, -T.infinity("float32"))
                    T.tile.fill(masses, 0.0)
                    for part in T.serial(parts):
                        maxima[0, part] = values[part, kind * 2]
                        masses[0, part] = values[part, kind * 2 + 1]
                    T.reduce_max(maxima, maximum, dim=1)
                    packet[0, kind * 2] = maximum[0, 0]
                    if maximum[0, 0] < T.infinity("float32"):
                        T.tile.sub(maxima, maxima, maximum[0, 0])
                        T.tile.exp(maxima, maxima)
                        T.tile.mul(maxima, maxima, masses)
                        T.reduce_sum(maxima, total, dim=1)
                        packet[0, kind * 2 + 1] = total[0, 0]
                packet[0, 4] = -T.infinity("float32")
                packet[0, 5] = -T.infinity("float32")
                token = proposals[row, 1] - controls[row, 2]
                if proposals[row, 0] >= 0 and token >= 0 and token < vocab:
                    inverse = T.reinterpret("float32", controls[row, 3])
                    packet[0, 4] = logits[row, token] * inverse
                    packet[0, 5] = draft[proposals[row, 0], token] * inverse
                T.copy(packet, out[row : row + 1, :])

    return main


def normalize(rows, ranks):
    width = max(8, ranks)

    @T.prim_func
    def main(
        gathered: T.Tensor((ranks, rows, 8), "float32"),
        proposals: T.Tensor((rows, 4), "int32"),
        active: T.Tensor((1,), "int32"),
        out: T.Tensor((rows, 4), "float32"),
    ):
        with T.Kernel(T.ceildiv(rows, 2), is_npu=True) as (cid, vid):
            values = T.alloc_ub((ranks, 8), "float32")
            maxima = T.alloc_ub((1, width), "float32")
            masses = T.alloc_ub((1, width), "float32")
            chosen = T.alloc_ub((1, width), "float32")
            maximum = T.alloc_ub((1, 1), "float32")
            total = T.alloc_ub((1, 1), "float32")
            logarithm = T.alloc_ub((1, 8), "float32")
            packet = T.alloc_ub((1, 8), "float32")
            row = cid * 2 + vid
            if row < T.min(rows, active[0]):
                T.tile.fill(packet, 0.0)
                for rank in T.serial(ranks):
                    T.copy(gathered[rank, row : row + 1, :], values[rank : rank + 1, :])
                for kind in T.unroll(2):
                    T.tile.fill(maxima, -T.infinity("float32"))
                    T.tile.fill(masses, 0.0)
                    T.tile.fill(chosen, -T.infinity("float32"))
                    for rank in T.serial(ranks):
                        maxima[0, rank] = values[rank, kind * 2]
                        masses[0, rank] = values[rank, kind * 2 + 1]
                        chosen[0, rank] = values[rank, 4 + kind]
                    T.reduce_max(maxima, maximum, dim=1)
                    if maximum[0, 0] == T.infinity("float32"):
                        packet[0, 3] = 1.0
                    else:
                        T.tile.sub(maxima, maxima, maximum[0, 0])
                        T.tile.exp(maxima, maxima)
                        T.tile.mul(maxima, maxima, masses)
                        T.reduce_sum(maxima, total, dim=1)
                        T.tile.fill(logarithm, total[0, 0])
                        T.tile.ln(logarithm, logarithm)
                        packet[0, kind] = maximum[0, 0] + logarithm[0, 0]
                        T.reduce_max(chosen, total, dim=1)
                        packet[0, 4 + kind] = total[0, 0]
                if proposals[row, 0] >= 0:
                    accept_logprob = (packet[0, 4] - packet[0, 0]) - (packet[0, 5] - packet[0, 1])
                    packet[0, 2] = T.if_then_else(
                        T.reinterpret("float32", proposals[row, 2]) < T.min(0.0, accept_logprob),
                        1.0,
                        0.0,
                    )
                T.copy(packet[:, :4], out[row : row + 1, :])

    return main


def select(rows, vocab, pool_rows, tile=1024):
    parts = (vocab + tile - 1) // tile

    @T.prim_func
    def main(
        logits: T.Tensor((rows, vocab), "float32"),
        draft: T.Tensor((pool_rows, vocab), "float32"),
        controls: T.Tensor((rows, 4), "int32"),
        proposals: T.Tensor((rows, 4), "int32"),
        normalizers: T.Tensor((rows, 4), "float32"),
        active: T.Tensor((1,), "int32"),
        partial: T.Tensor((rows, parts, 2), "float32"),
    ):
        with T.Kernel(T.ceildiv(rows * parts, 2), is_npu=True) as (cid, vid):
            p = T.alloc_ub((1, tile), "float32")
            q = T.alloc_ub((1, tile), "float32")
            noise = T.alloc_ub((1, tile), "float32")
            indices = T.alloc_ub((1, tile), "float32")
            bits = T.alloc_ub((tile // 8,), "uint8")
            maximum = T.alloc_ub((1, 1), "float32")
            minimum = T.alloc_ub((1, 1), "float32")
            packet = T.alloc_ub((1, 8), "float32")
            index = cid * 2 + vid
            row, part = index // parts, index % parts
            start = part * tile
            live = T.min(tile, vocab - start)
            if row < T.min(rows, active[0]):
                packet[0, 0] = T.reinterpret("float32", T.uint32(0xFF7FFFFF))
                packet[0, 1] = 0.0
                pool_row = proposals[row, 0]
                if normalizers[row, 3] > 0.0:
                    packet[0, 0] = T.infinity("float32")
                elif pool_row >= 0 and normalizers[row, 2] > 0.0:
                    token = proposals[row, 1] - controls[row, 2]
                    if token >= start and token < start + live:
                        packet[0, 0] = 0.0
                        packet[0, 1] = T.cast(token, "float32")
                else:
                    T.tile.fill(p, 0.0)
                    T.copy(logits[row : row + 1, start : start + live], p[:, :live])
                    inverse = T.reinterpret("float32", controls[row, 3])
                    if inverse > 0.0:
                        T.tile.mul(p, p, inverse)
                        if pool_row >= 0:
                            T.tile.fill(q, 0.0)
                            T.copy(
                                draft[pool_row : pool_row + 1, start : start + live], q[:, :live]
                            )
                            T.tile.mul(q, q, inverse)
                            T.tile.sub(p, p, normalizers[row, 0])
                            T.tile.sub(q, q, normalizers[row, 1])
                            # log(max(p-q,0)), avoiding probability underflow.
                            T.tile.sub(q, q, p)
                            T.tile.min(q, q, 0.0)
                            T.tile.exp(q, q)
                            T.tile.mul(q, q, -1.0)
                            T.tile.add(q, q, 1.0)
                            T.tile.ln(q, q)
                            T.tile.add(p, p, q)
                        uniform(
                            noise,
                            controls[row, 2] + start,
                            T.reinterpret("uint32", controls[row, 0]),
                            T.reinterpret("uint32", controls[row, 1]),
                        )
                        T.tile.ln(noise, noise)
                        T.tile.mul(noise, noise, -1.0)
                        T.tile.ln(noise, noise)
                        T.tile.sub(p, p, noise)
                    T.tile.createvecindex(indices, T.float32(0.0))
                    T.tile.compare(bits, indices, T.cast(live, "float32"), "LT")
                    T.tile.select(p, bits, p, -T.infinity("float32"), "VSEL_TENSOR_SCALAR_MODE")
                    T.reduce_max(p, maximum, dim=1)
                    if maximum[0, 0] != -T.infinity("float32"):
                        T.tile.add(indices, indices, T.cast(start, "float32"))
                        T.tile.compare(bits, p, maximum[0, 0], "EQ")
                        T.tile.select(
                            indices,
                            bits,
                            indices,
                            T.cast(vocab, "float32"),
                            "VSEL_TENSOR_SCALAR_MODE",
                        )
                        T.reduce_min(indices, minimum, dim=1)
                        packet[0, 0] = maximum[0, 0]
                        packet[0, 1] = minimum[0, 0]
                T.copy(packet[:, :2], partial[row, part : part + 1, :])
                T.pipe_barrier("ALL")

    return main
