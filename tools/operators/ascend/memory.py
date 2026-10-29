"""A2 cache/embedding row movement, Engram dequantization and complex RoPE."""

import math

import tilelang.language as T


def partition_rows(rows, count, width, first, stride=8, dtype="int32", repeat_last=False):
    """Pack interleaved rows; pad data with zero or repeat the last source row.

    Repeated index rows preserve the causal indexer's monotone tile guard.
    Padded rows have zero selection lengths and never contribute to TopK.
    """
    if count < 1 or width < 1 or stride < 1 or not 0 <= first < stride:
        raise ValueError("Invalid interleaved row partition")
    block = math.gcd(width, 256)
    blocks = width // block

    @T.prim_func
    def main(
        source: T.Tensor((rows, width), dtype),
        out: T.Tensor((count, width), dtype),
    ):
        with T.Kernel(T.ceildiv(count, 2) * blocks, is_npu=True) as (cid, vid):
            data = T.alloc_ub((block,), dtype)
            zeros = T.alloc_ub((max(1, block // 2),), "int16")
            row = (cid // blocks) * 2 + vid
            column = (cid % blocks) * block
            index = first + row * stride
            if row < count:
                if repeat_last:
                    index = T.min(index, rows - 1)
                    T.copy(source[index, column : column + block], data)
                else:
                    if dtype == "int8":
                        T.reinterpretcast(data, zeros, "int8_t")
                        T.tile.fill(zeros, 0)
                    elif block < 8:
                        for lane in T.serial(block):
                            data[lane] = 0
                    else:
                        T.tile.fill(data, 0)
                    if index < rows:
                        T.copy(source[index, column : column + block], data)
                T.copy(data, out[row, column : column + block])
                # The INT8 view aliases the INT16 zero-fill buffer. Preserve
                # its lifetime across the bounded grid's next loop iteration.
                T.pipe_barrier("ALL")

    return main


def interleave_rows(rows, count, width, ranks=8, dtype="int32"):
    """Restore global row order after gathering equal interleaved partitions."""
    if min(count, width, ranks) < 1:
        raise ValueError("Invalid interleaved gather geometry")
    block = math.gcd(width, 256)
    blocks = width // block

    @T.prim_func
    def main(
        source: T.Tensor((ranks * count, width), dtype),
        out: T.Tensor((rows, width), dtype),
    ):
        with T.Kernel(T.ceildiv(rows, 2) * blocks, is_npu=True) as (cid, vid):
            data = T.alloc_ub((block,), dtype)
            row = cid // blocks * 2 + vid
            column = cid % blocks * block
            if row < rows:
                index = row % ranks * count + row // ranks
                T.copy(source[index, column : column + block], data)
                T.copy(data, out[row, column : column + block])

    return main


def zero_row_tail(rows, capacity, width, dtype="bfloat16"):
    """Define unused fixed-capacity collective rows without touching live rows."""
    if capacity < 2 or width < 1 or dtype not in ("bfloat16", "float32"):
        raise ValueError("Invalid collective tail geometry")
    block = math.gcd(width, 256)
    blocks = width // block

    @T.prim_func
    def main(
        positions: T.Tensor((rows,), "int32"),
        out: T.Tensor((capacity, width), dtype),
    ):
        with T.Kernel(T.max(1, T.ceildiv((capacity - rows) * blocks, 2)), is_npu=True) as (
            cid,
            vid,
        ):
            zeros = T.alloc_ub((block,), dtype)
            index = cid * 2 + vid
            row = rows + index // blocks
            column = index % blocks * block
            if row < capacity:
                T.tile.fill(zeros, 0)
                T.copy(zeros, out[row, column : column + block])

    return main


def last_row(rows, width):
    """Copy the actual last row from a capacity-backed dynamic activation."""
    block = math.gcd(width, 256)

    @T.prim_func
    def main(
        source: T.Tensor((rows, width), "bfloat16"),
        out: T.Tensor((1, width), "bfloat16"),
    ):
        with T.Kernel(T.ceildiv(width // block, 2), is_npu=True) as (cid, vid):
            value = T.alloc_ub((block,), "bfloat16")
            column = (cid * 2 + vid) * block
            if column < width:
                T.copy(source[rows - 1, column : column + block], value)
                T.copy(value, out[0, column : column + block])

    return main


def gather_quantized_rows(rows, table_rows, width):
    """Move one complete INT8 activation and its FP32 scale per owner."""
    if width % 32 or (isinstance(table_rows, int) and not 0 < table_rows <= 2147483647):
        raise ValueError("Quantized gather requires aligned width and positive table rows")

    @T.prim_func
    def main(
        table: T.Tensor((table_rows, width), "int8"),
        scales: T.Tensor((table_rows,), "float32"),
        indices: T.Tensor((rows,), "int32"),
        out: T.Tensor((rows, width), "int8"),
        out_scales: T.Tensor((rows,), "float32"),
        errors: T.Tensor((rows,), "int32"),
    ):
        with T.Kernel(T.ceildiv(rows, 2), is_npu=True) as (cid, vid):
            a = T.alloc_ub((width,), "int8")
            zeros = T.alloc_ub((width // 2,), "int16")
            scale = T.alloc_ub((8,), "float32")
            flag = T.alloc_ub((8,), "int32")
            row = cid * 2 + vid
            if row < rows:
                index = indices[row]
                T.reinterpretcast(a, zeros, "int8_t")
                T.tile.fill(zeros, 0)
                T.tile.fill(scale, 0.0)
                if index >= 0 and index < table_rows:
                    T.copy(table[index, :], a)
                    T.copy(scales[index : index + 1], scale[:1])
                T.copy(a, out[row, :])
                T.copy(scale[:1], out_scales[row : row + 1])
                flag[0] = T.if_then_else(index < -1 or index >= table_rows, 1, 0)
                T.copy(flag[:1], errors[row : row + 1])

    return main


def gather_columns(rows, width, ranks=8):
    """Convert rank-major AllGather output into row-major concatenated columns."""
    if width % ranks or (width // ranks) % 16:
        raise ValueError("Gathered columns must have aligned, equal rank partitions")
    local = width // ranks

    @T.prim_func
    def main(
        gathered: T.Tensor((ranks, rows, local), "bfloat16"),
        out: T.Tensor((rows, width), "bfloat16"),
    ):
        with T.Kernel(T.ceildiv(rows * ranks, 2), is_npu=True) as (cid, vid):
            data = T.alloc_ub((local,), "bfloat16")
            index = cid * 2 + vid
            row, rank = index // ranks, index % ranks
            if row < rows:
                T.copy(gathered[rank, row, :], data)
                T.copy(data, out[row, rank * local : (rank + 1) * local])

    return main


def gather_rows(rows, table_rows, width, dtype="bfloat16", index_dtype="int64"):
    if isinstance(table_rows, int) and table_rows > 2147483647:
        raise ValueError("A2 row table exceeds signed 32-bit addressing")
    block = math.gcd(width, 256)
    blocks = width // block

    @T.prim_func
    def main(
        table: T.Tensor((table_rows, width), dtype),
        indices: T.Tensor((rows,), index_dtype),
        out: T.Tensor((rows, width), dtype),
        errors: T.Tensor((rows,), "int32"),
    ):
        with T.Kernel(T.ceildiv(rows, 2) * blocks, is_npu=True) as (cid, vid):
            a = T.alloc_ub((block,), dtype)
            zeros = T.alloc_ub((max(1, block // 2),), "int16")
            flag = T.alloc_ub((8,), "int32")
            row = cid // blocks * 2 + vid
            column = cid % blocks * block
            if row < rows:
                index = indices[row]
                if dtype == "int8":
                    T.reinterpretcast(a, zeros, "int8_t")
                    T.tile.fill(zeros, 0)
                else:
                    T.tile.fill(a, 0.0)
                if index >= 0 and index < table_rows:
                    T.copy(table[T.cast(index, "int32"), column : column + block], a)
                T.copy(a, out[row, column : column + block])
                if column == 0:
                    flag[0] = T.if_then_else(index < -1 or index >= table_rows, 1, 0)
                    T.copy(flag[:1], errors[row : row + 1])

    return main


def scatter_rows(rows, table_rows, width, dtype="bfloat16", index_dtype="int64"):
    """Destination indices must be unique. -1 means no write; other OOBs flag an error."""
    if table_rows > 2147483647:
        raise ValueError("A2 row table exceeds signed 32-bit addressing")
    block = math.gcd(width, 256)
    blocks = width // block

    @T.prim_func
    def main(
        values: T.Tensor((rows, width), dtype),
        indices: T.Tensor((rows,), index_dtype),
        table: T.Tensor((table_rows, width), dtype),
        errors: T.Tensor((rows,), "int32"),
    ):
        with T.Kernel(T.ceildiv(rows, 2) * blocks, is_npu=True) as (cid, vid):
            a = T.alloc_ub((block,), dtype)
            flag = T.alloc_ub((8,), "int32")
            row = cid // blocks * 2 + vid
            column = cid % blocks * block
            if row < rows:
                index = indices[row]
                if index >= 0 and index < table_rows:
                    T.copy(values[row, column : column + block], a)
                    T.copy(a, table[T.cast(index, "int32"), column : column + block])
                if column == 0:
                    flag[0] = T.if_then_else(index < -1 or index >= table_rows, 1, 0)
                    T.copy(flag[:1], errors[row : row + 1])

    return main


def dequant_rows(rows, width, group=32):
    @T.prim_func
    def main(
        values: T.Tensor((rows, width), "int8"),
        scales: T.Tensor((rows, width // group), "float32"),
        out: T.Tensor((rows, width), "bfloat16"),
    ):
        with T.Kernel(T.ceildiv(rows, 2), is_npu=True) as (cid, vid):
            q = T.alloc_ub((width,), "int8")
            half_values = T.alloc_ub((width,), "float16")
            a = T.alloc_ub((width,), "float32")
            scale = T.alloc_ub((width // group,), "float32")
            y = T.alloc_ub((width,), "bfloat16")
            row = cid * 2 + vid
            if row < rows:
                T.copy(values[row, :], q)
                T.copy(scales[row, :], scale)
                T.tile.cast(half_values, q, "CAST_NONE", width)
                T.tile.cast(a, half_values, "CAST_NONE", width)
                for g in T.serial(width // group):
                    T.tile.mul(
                        a[g * group : (g + 1) * group], a[g * group : (g + 1) * group], scale[g]
                    )
                T.tile.cast(y, a, "CAST_RINT", width)
                T.copy(y, out[row, :])

    return main


def rope(rows, heads, rotary=64):
    """Adjacent complex pairs. Caller supplies the model's YaRN/theta cache."""
    half = rotary // 2

    @T.prim_func
    def main(
        x: T.Tensor((rows, heads, rotary), "bfloat16"),
        cosine: T.Tensor((rows, half), "float32"),
        sine: T.Tensor((rows, half), "float32"),
        out: T.Tensor((rows, heads, rotary), "bfloat16"),
    ):
        with T.Kernel(T.ceildiv(rows * heads, 2), is_npu=True) as (cid, vid):
            xb = T.alloc_ub((rotary,), "bfloat16")
            a = T.alloc_ub((rotary,), "float32")
            even = T.alloc_ub((half,), "float32")
            odd = T.alloc_ub((half,), "float32")
            co = T.alloc_ub((half,), "float32")
            si = T.alloc_ub((half,), "float32")
            product = T.alloc_ub((half,), "float32")
            result = T.alloc_ub((2, half), "float32")
            offsets = T.alloc_ub((rotary,), "uint32")
            y = T.alloc_ub((rotary,), "bfloat16")
            index = cid * 2 + vid
            row = index // heads
            head = index % heads
            if row < rows:
                T.copy(x[row, head, :], xb)
                T.copy(cosine[row, :], co)
                T.copy(sine[row, :], si)
                T.tile.cast(a, xb, "CAST_NONE", rotary)
                T.tile.gather_mask(even, a, "P0101")
                T.tile.gather_mask(odd, a, "P1010")
                T.tile.mul(result[0, :], even, co)
                T.tile.mul(product, odd, si)
                T.tile.sub(result[0, :], result[0, :], product)
                T.tile.mul(result[1, :], odd, co)
                T.tile.mul(product, even, si)
                T.tile.add(result[1, :], result[1, :], product)
                for i in T.Parallel(rotary):
                    offsets[i] = T.cast((i % 2 * half + i // 2) * 4, "uint32")
                T.tile.gather(a, result, offsets, 0)
                T.tile.cast(y, a, "CAST_RINT", rotary)
                T.copy(y, out[row, head, :])

    return main
