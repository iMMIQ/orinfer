"""Masked categorical draws and bounded probability summaries on the same row ABI."""

import tilelang.language as T

from kernels.ascend.sampling import uniform

TOP = 20
WIDTH = 48


def local(rows, vocab, tile=1024):
    parts = (vocab + tile - 1) // tile
    mask_bytes = ((vocab + 255) // 256) * 32

    @T.prim_func
    def main(
        logits: T.Tensor((rows, vocab), "float32"),
        controls: T.Tensor((rows, 4), "int32"),
        active: T.Tensor((1,), "int32"),
        masks: T.Tensor((rows, mask_bytes), "uint8"),
        options: T.Tensor((rows, 2), "int32"),
        partial: T.Tensor((rows, parts, WIDTH), "float32"),
    ):
        with T.Kernel(T.ceildiv(rows * parts, 2), is_npu=True) as (cid, vid):
            x = T.alloc_ub((1, tile), "float32")
            y = T.alloc_ub((1, tile), "float32")
            scores = T.alloc_ub((1, tile), "float32")
            indices = T.alloc_ub((1, tile), "float32")
            bits = T.alloc_ub((tile // 8,), "uint8")
            maximum = T.alloc_ub((1, 1), "float32")
            minimum = T.alloc_ub((1, 1), "float32")
            packet = T.alloc_ub((1, WIDTH), "float32")
            index = cid * 2 + vid
            row, part = index // parts, index % parts
            begin = part * tile
            live = T.min(tile, vocab - begin)
            if row < T.min(rows, active[0]):
                T.tile.fill(packet, 0.0)
                T.tile.fill(x, 0.0)
                T.copy(logits[row : row + 1, begin : begin + live], x[:, :live])
                # Check original real logits before introducing masked -inf.
                T.tile.compare(bits, x, x, "EQ")
                T.tile.fill(y, 1.0)
                T.tile.select(y, bits, y, 0.0, "VSEL_TENSOR_SCALAR_MODE")
                T.reduce_min(y, minimum, dim=1)
                T.tile.abs(y, x)
                T.reduce_max(y, maximum, dim=1)
                packet[0, 45] = T.if_then_else(
                    minimum[0, 0] == 0.0 or maximum[0, 0] == T.infinity("float32"),
                    1.0,
                    0.0,
                )
                if options[row, 0] != 0:
                    T.copy(masks[row, begin // 8 : begin // 8 + tile // 8], bits)
                    T.tile.select(x, bits, x, -T.infinity("float32"), "VSEL_TENSOR_SCALAR_MODE")
                inverse = T.reinterpret("float32", controls[row, 3])
                if inverse > 0.0:
                    T.tile.mul(x, x, inverse)
                T.tile.createvecindex(indices, T.float32(0.0))
                T.tile.compare(bits, indices, T.cast(live, "float32"), "LT")
                T.tile.select(x, bits, x, -T.infinity("float32"), "VSEL_TENSOR_SCALAR_MODE")
                T.copy(x, scores)
                if inverse > 0.0:
                    uniform(
                        y,
                        controls[row, 2] + begin,
                        T.reinterpret("uint32", controls[row, 0]),
                        T.reinterpret("uint32", controls[row, 1]),
                    )
                    T.tile.ln(y, y)
                    T.tile.mul(y, y, -1.0)
                    T.tile.ln(y, y)
                    T.tile.sub(scores, scores, y)
                T.reduce_max(scores, maximum, dim=1)
                T.tile.add(indices, indices, T.cast(begin, "float32"))
                T.tile.compare(bits, scores, maximum[0, 0], "EQ")
                T.tile.select(indices, bits, indices, T.float32(vocab), "VSEL_TENSOR_SCALAR_MODE")
                T.reduce_min(indices, minimum, dim=1)
                packet[0, 0] = T.if_then_else(
                    packet[0, 45] > 0.0, T.infinity("float32"), maximum[0, 0]
                )
                packet[0, 1] = minimum[0, 0]
                if minimum[0, 0] < T.cast(begin + live, "float32"):
                    packet[0, 2] = x[0, T.cast(minimum[0, 0], "int32") - begin]
                packet[0, 3] = -T.infinity("float32")
                if options[row, 1] > 0:
                    T.reduce_max(x, maximum, dim=1)
                    packet[0, 3] = maximum[0, 0]
                    if maximum[0, 0] != -T.infinity("float32"):
                        T.tile.sub(y, x, maximum[0, 0])
                        T.tile.exp(y, y)
                        T.reduce_sum(y, minimum, dim=1)
                        packet[0, 4] = minimum[0, 0]
                    T.copy(x, y)
                    for k in T.serial(TOP):
                        if k < options[row, 1] - 1:
                            T.reduce_max(y, maximum, dim=1)
                            T.tile.createvecindex(indices, T.cast(begin, "float32"))
                            T.tile.compare(bits, y, maximum[0, 0], "EQ")
                            T.tile.select(
                                indices, bits, indices, T.float32(vocab), "VSEL_TENSOR_SCALAR_MODE"
                            )
                            T.reduce_min(indices, minimum, dim=1)
                            packet[0, 5 + 2 * k] = maximum[0, 0]
                            packet[0, 6 + 2 * k] = minimum[0, 0]
                            T.tile.createvecindex(indices, T.cast(begin, "float32"))
                            T.tile.compare(bits, indices, minimum[0, 0], "NE")
                            T.tile.select(
                                y, bits, y, -T.infinity("float32"), "VSEL_TENSOR_SCALAR_MODE"
                            )
                T.copy(packet, partial[row, part : part + 1, :])
                T.pipe_barrier("ALL")

    return main


def merge(rows, parts):
    count = ((parts * TOP + 63) // 64) * 64
    norm_count = ((parts + 63) // 64) * 64

    @T.prim_func
    def main(
        partial: T.Tensor((rows, parts, WIDTH), "float32"),
        active: T.Tensor((1,), "int32"),
        options: T.Tensor((rows, 2), "int32"),
        output: T.Tensor((rows, WIDTH), "float32"),
    ):
        with T.Kernel(T.ceildiv(rows, 2), is_npu=True) as (cid, vid):
            values = T.alloc_ub((parts, WIDTH), "float32")
            packet = T.alloc_ub((1, WIDTH), "float32")
            chosen = T.alloc_ub((8,), "int32")
            norm = T.alloc_ub((1, norm_count), "float32")
            mass = T.alloc_ub((1, norm_count), "float32")
            scores = T.alloc_ub((1, count), "float32")
            ids = T.alloc_ub((1, count), "float32")
            indices = T.alloc_ub((1, count), "float32")
            bits = T.alloc_ub((count // 8,), "uint8")
            maximum = T.alloc_ub((1, 1), "float32")
            minimum = T.alloc_ub((1, 1), "float32")
            row = cid * 2 + vid
            if row < T.min(rows, active[0]):
                T.copy(partial[row, :, :], values)
                chosen[0] = 0
                for p in T.serial(1, parts):
                    if values[p, 0] > values[chosen[0], 0] or (
                        values[p, 0] == values[chosen[0], 0] and values[p, 1] < values[chosen[0], 1]
                    ):
                        chosen[0] = p
                T.copy(values[chosen[0] : chosen[0] + 1, :], packet)
                if options[row, 1] > 0:
                    T.tile.fill(norm, -T.infinity("float32"))
                    T.tile.fill(mass, 0.0)
                    for p in T.serial(parts):
                        norm[0, p] = values[p, 3]
                        mass[0, p] = values[p, 4]
                    T.reduce_max(norm, maximum, dim=1)
                    packet[0, 3] = maximum[0, 0]
                    packet[0, 4] = 0.0
                    if maximum[0, 0] != -T.infinity("float32"):
                        T.tile.sub(norm, norm, maximum[0, 0])
                        T.tile.exp(norm, norm)
                        T.tile.mul(norm, norm, mass)
                        T.reduce_sum(norm, minimum, dim=1)
                        packet[0, 4] = minimum[0, 0]
                    T.tile.fill(scores, -T.infinity("float32"))
                    T.tile.fill(ids, T.float32(16777215))
                    for p in T.serial(parts):
                        for k in T.serial(TOP):
                            if k < options[row, 1] - 1:
                                scores[0, p * TOP + k] = values[p, 5 + 2 * k]
                                ids[0, p * TOP + k] = values[p, 6 + 2 * k]
                    for k in T.serial(TOP):
                        if k < options[row, 1] - 1:
                            T.reduce_max(scores, maximum, dim=1)
                            T.tile.compare(bits, scores, maximum[0, 0], "EQ")
                            T.tile.select(
                                indices, bits, ids, T.float32(16777215), "VSEL_TENSOR_SCALAR_MODE"
                            )
                            T.reduce_min(indices, minimum, dim=1)
                            packet[0, 5 + 2 * k] = maximum[0, 0]
                            packet[0, 6 + 2 * k] = minimum[0, 0]
                            T.tile.compare(bits, ids, minimum[0, 0], "NE")
                            T.tile.select(
                                scores,
                                bits,
                                scores,
                                -T.infinity("float32"),
                                "VSEL_TENSOR_SCALAR_MODE",
                            )
                T.copy(packet, output[row : row + 1, :])

    return main
