"""Prefill attention with shared contiguous KV banks and packed token/head tiles."""

import tilelang.language as T


def selected_mask(rows, length, selected=512):
    if length % 128:
        raise ValueError("Dense prefill attention requires a 128-aligned KV bank")

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
                for i in T.serial(selected):
                    key = ids[i]
                    if key >= 0 and key < length:
                        values[key] = 1
                T.set_flag("S", "MTE3", 6)
                T.wait_flag("S", "MTE3", 6)
                T.copy(values, mask[row, :])
                T.set_flag("MTE3", "V", 7)
                T.wait_flag("MTE3", "V", 7)

    return main


def qk(rows, capacity, ratio=2, compressed=True, heads=8, dimension=512):
    long_length = capacity // ratio
    if capacity % 256 or ratio not in (1, 2) or heads != 8 or dimension != 512:
        raise ValueError("Packed prefill QK requires C%256=0, ratio=1/2 and H8/D512")
    length = capacity + (long_length if compressed else 0)
    tiles = length // 128

    @T.prim_func
    def main(
        query: T.Tensor((rows * heads, dimension), "bfloat16"),
        swa: T.Tensor((capacity, dimension), "bfloat16"),
        long_kv: T.Tensor((long_length, dimension), "bfloat16"),
        positions: T.Tensor((rows,), "int32"),
        scores: T.Tensor((rows * heads, length), "float32"),
    ):
        with T.Kernel(T.ceildiv(rows * heads, 128) * tiles, is_npu=True) as (cid, vid):
            first, key = cid // tiles * 128, cid % tiles * 128
            with T.Scope("C"):
                a = T.alloc_L1((128, 128), "bfloat16")
                b = T.alloc_L1((128, 128), "bfloat16")
                c = T.alloc_L0C((128, 128), "float32")
                valid = T.min(128, rows * heads - first)
                end = positions[rows - 1] + 1
                live = T.if_then_else(key < capacity, end, capacity + end // ratio)
                if key < live:
                    for k in T.serial(dimension // 128):
                        T.copy(query[first : first + valid, k * 128 : (k + 1) * 128], a[:valid, :])
                        if key < capacity:
                            T.copy(swa[key : key + 128, k * 128 : (k + 1) * 128], b)
                        else:
                            T.copy(
                                long_kv[
                                    key - capacity : key - capacity + 128, k * 128 : (k + 1) * 128
                                ],
                                b,
                            )
                        T.gemm_v0(a, b, c, transpose_B=True, init=(k == 0))
                    T.copy(c[:valid, :], scores[first : first + valid, key : key + 128])

    return main


def softmax(rows, capacity, ratio=2, compressed=True, heads=8, dimension=512):
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


def pv(rows, capacity, ratio=2, compressed=True, heads=8, dimension=512):
    long_length = capacity // ratio
    length = capacity + (long_length if compressed else 0)
    if capacity % 256 or heads != 8 or dimension != 512:
        raise ValueError("Packed prefill PV requires C%256=0 and H8/D512")

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
                end = positions[rows - 1] + 1
                for k in T.serial(T.ceildiv(end, 128)):
                    T.copy(
                        probabilities[first : first + valid, k * 128 : (k + 1) * 128], a[:valid, :]
                    )
                    T.copy(swa[k * 128 : (k + 1) * 128, column : column + 64], b)
                    T.gemm_v0(a, b, c, init=(k == 0))
                if compressed:
                    for k in T.serial(T.ceildiv(end // ratio, 128)):
                        T.copy(
                            probabilities[
                                first : first + valid, capacity + k * 128 : capacity + (k + 1) * 128
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
