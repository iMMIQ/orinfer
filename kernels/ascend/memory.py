"""A2 cache/embedding row movement, Engram dequantization and complex RoPE."""

import math

import tilelang.language as T


def pad_blocks(rows, table_rows, block_size, width, dtype="bfloat16"):
    """Pack live owner prefixes into one fixed-stride Markov workspace.

    Endpoints describe the actual compact packet. Only live rows are written;
    the packet's active Markov chain never reads the remaining padding.
    """
    if min(table_rows, block_size, width) < 1 or dtype not in ("bfloat16", "float32", "int32"):
        raise ValueError("Invalid owner block geometry")

    @T.prim_func
    def main(
        source: T.Tensor((table_rows, width), dtype),
        endpoints: T.Tensor((rows,), "int32"),
        out: T.Tensor((rows * block_size, width), dtype),
    ):
        with T.Kernel(T.ceildiv(rows * block_size, 2), is_npu=True) as (cid, vid):
            data = T.alloc_ub((max(8, width),), dtype)
            index = cid * 2 + vid
            owner, position = index // block_size, index % block_size
            if owner < rows:
                end = endpoints[owner]
                start = T.if_then_else(owner > 0, endpoints[T.max(0, owner - 1)] + 1, 0)
                if start >= 0 and end >= start and end < table_rows:
                    if position <= end - start and position < block_size:
                        T.copy(source[start + position, :], data[:width])
                        T.copy(data[:width], out[index, :])

    return main


def partition_rows(
    rows,
    count,
    width,
    first,
    stride=8,
    dtype="int32",
    repeat_last=False,
    dynamic_count=False,
    all_partitions=False,
    tile_columns=256,
):
    """Pack interleaved rows; pad data with zero or repeat the last source row.

    Repeated index rows preserve the causal indexer's monotone tile guard.
    Padded rows have zero selection lengths and never contribute to TopK.
    """
    if count < 1 or width < 1 or stride < 1 or not 0 <= first < stride:
        raise ValueError("Invalid interleaved row partition")
    if all_partitions and first != 0:
        raise ValueError("All-partition packing starts at partition zero")
    if tile_columns < 1 or tile_columns > 8192 or tile_columns & (tile_columns - 1):
        raise ValueError("Movement column tile must be a power of two in 1..8192")
    if dynamic_count:
        count = T.ceildiv(rows, stride)
    total = count * (stride if all_partitions else 1)
    elem_bytes = {"int8": 1, "bfloat16": 2, "float16": 2, "int32": 4, "float32": 4}.get(dtype, 4)
    max_cols = max(tile_columns, 16384 // elem_bytes)
    block = width if width <= max_cols else math.gcd(width, max_cols)
    blocks = width // block

    @T.prim_func
    def main(
        source: T.Tensor((rows, width), dtype),
        out: T.Tensor((total, width), dtype),
    ):
        with T.Kernel(20, is_npu=True) as (cid, vid):
            data = T.alloc_ub((block,), dtype)
            zero_buf = T.alloc_ub((block,), dtype)
            zeros = T.alloc_ub((max(1, block // 2),), "int16")
            aiv = cid * 2 + vid
            items = total * blocks
            if aiv < items:
                if not repeat_last:
                    if dtype == "int8" and block >= 8:
                        T.reinterpretcast(zero_buf, zeros, "int8_t")
                        T.tile.fill(zeros, 0)
                    elif block < 8:
                        for lane in T.serial(block):
                            zero_buf[lane] = 0
                    else:
                        T.tile.fill(zero_buf, 0)
                    T.pipe_barrier("ALL")
                for step in T.serial(T.ceildiv(items - aiv, 40)):
                    flat = aiv + step * 40
                    row = flat // blocks
                    column = (flat % blocks) * block
                    index = (
                        row // count + (row % count) * stride
                        if all_partitions
                        else first + row * stride
                    )
                    if repeat_last:
                        src_row = T.min(index, rows - 1)
                        T.copy(source[src_row, column : column + block], data)
                        T.copy(data, out[row, column : column + block])
                    else:
                        if index < rows:
                            T.copy(source[index, column : column + block], data)
                            T.copy(data, out[row, column : column + block])
                        else:
                            T.copy(zero_buf, out[row, column : column + block])

    return main


def interleave_rows(
    rows, count, width, ranks=8, dtype="int32", dynamic_count=False, tile_columns=256
):
    """Restore global row order after gathering equal interleaved partitions."""
    if min(count, width, ranks) < 1:
        raise ValueError("Invalid interleaved gather geometry")
    if tile_columns < 1 or tile_columns > 8192 or tile_columns & (tile_columns - 1):
        raise ValueError("Movement column tile must be a power of two in 1..8192")
    if dynamic_count:
        count = T.ceildiv(rows, ranks)
    elem_bytes = {"int8": 1, "bfloat16": 2, "float16": 2, "int32": 4, "float32": 4}.get(dtype, 4)
    max_cols = max(tile_columns, 16384 // elem_bytes)
    block = width if width <= max_cols else math.gcd(width, max_cols)
    blocks = width // block

    @T.prim_func
    def main(
        source: T.Tensor((ranks * count, width), dtype),
        out: T.Tensor((rows, width), dtype),
    ):
        with T.Kernel(20, is_npu=True) as (cid, vid):
            data = T.alloc_ub((block,), dtype)
            aiv = cid * 2 + vid
            items = rows * blocks
            if aiv < items:
                for step in T.serial(T.ceildiv(items - aiv, 40)):
                    flat = aiv + step * 40
                    row = flat // blocks
                    column = (flat % blocks) * block
                    index = row % ranks * count + row // ranks
                    T.copy(source[index, column : column + block], data)
                    T.copy(data, out[row, column : column + block])

    return main


def zero_row_tail(rows, capacity, width, dtype="bfloat16", partitions=0, first=None):
    """Define unused fixed-capacity collective rows without touching live rows."""
    if capacity < 2 or width < 1 or dtype not in ("bfloat16", "float32"):
        raise ValueError("Invalid collective tail geometry")
    if first is not None:
        if partitions < 2 or not 0 <= first < partitions:
            raise ValueError("Invalid interleaved tail partition")
    elif partitions and (partitions < 1 or capacity % partitions):
        raise ValueError("Collective partition must divide its capacity")
    elem_bytes = 2 if dtype == "bfloat16" else 4
    max_cols = 16384 // elem_bytes
    block = width if width <= max_cols else math.gcd(width, max_cols)
    blocks = width // block
    start = T.ceildiv(rows - first, partitions) if first is not None else rows
    end = (
        T.ceildiv(rows, partitions)
        if first is not None
        else T.ceildiv(rows, partitions) * partitions
        if partitions
        else capacity
    )

    @T.prim_func
    def main(
        positions: T.Tensor((rows,), "int32"),
        out: T.Tensor((capacity, width), dtype),
    ):
        with T.Kernel(20, is_npu=True) as (cid, vid):
            zeros = T.alloc_ub((block,), dtype)
            aiv = cid * 2 + vid
            if start < end:
                items = (end - start) * blocks
                if aiv < items:
                    if block < 8:
                        for lane in T.serial(block):
                            zeros[lane] = 0
                    else:
                        T.tile.fill(zeros, 0)
                    T.pipe_barrier("ALL")
                    for step in T.serial(T.ceildiv(items - aiv, 40)):
                        index = aiv + step * 40
                        row = start + index // blocks
                        column = (index % blocks) * block
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


def gather_quantized_rows(rows, table_rows, width, table_stride=0):
    """Move one complete INT8 activation and its FP32 scale per owner."""
    if width % 32 or (isinstance(table_rows, int) and not 0 < table_rows <= 2147483647):
        raise ValueError("Quantized gather requires aligned width and positive table rows")
    stride = table_stride or width
    if stride < width or stride % 32:
        raise ValueError("Quantized gather requires an aligned physical row stride")

    @T.prim_func
    def main(
        table: T.Tensor((table_rows, stride), "int8"),
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
                    T.copy(table[index, :width], a)
                    T.copy(scales[index : index + 1], scale[:1])
                T.copy(a, out[row, :])
                T.copy(scale[:1], out_scales[row : row + 1])
                flag[0] = T.if_then_else(index < -1 or index >= table_rows, 1, 0)
                T.copy(flag[:1], errors[row : row + 1])

    return main


def gather_quantized_active(rows, table_rows, width, selected=6):
    """20-AIC persistent active-bounded INT8 gather for direct W4 MoE routing.

    Reads only the active routed prefix indicated by active_counts[rows // selected]
    and zeroes error flags across the remaining row span.
    """
    if width % 32 or (isinstance(table_rows, int) and not 0 < table_rows <= 2147483647):
        raise ValueError("Active quantized gather requires aligned width and positive table rows")

    @T.prim_func
    def main(
        table: T.Tensor((table_rows, width), "int8"),
        scales: T.Tensor((table_rows,), "float32"),
        indices: T.Tensor((rows,), "int32"),
        active_counts: T.Tensor((rows // selected + 1,), "int32"),
        out: T.Tensor((rows, width), "int8"),
        out_scales: T.Tensor((rows,), "float32"),
        errors: T.Tensor((rows,), "int32"),
    ):
        with T.Kernel(20, is_npu=True) as (cid, vid):
            x = T.alloc_ub((width,), "int8")
            sc = T.alloc_ub((8,), "float32")
            flag = T.alloc_ub((8,), "int32")
            aiv = cid * 2 + vid
            active = T.max(0, T.min(rows, active_counts[rows // selected]))
            if aiv < rows:
                for step in T.serial(T.ceildiv(rows - aiv, 40)):
                    row = aiv + step * 40
                    flag[0] = 0
                    if row < active:
                        idx = indices[row]
                        if idx >= 0 and idx < table_rows:
                            T.copy(table[idx, :width], x)
                            T.copy(scales[idx : idx + 1], sc[:1])
                            T.copy(x, out[row, :])
                            T.copy(sc[:1], out_scales[row : row + 1])
                        else:
                            flag[0] = T.if_then_else(idx == -1, 0, 1)
                    T.copy(flag[:1], errors[row : row + 1])
                    T.set_flag("MTE3", "V", 7)
                    T.wait_flag("MTE3", "V", 7)

    return main


def gather_columns(rows, width, ranks=8, dtype="bfloat16", output_width=0, paths=1):
    """Restore row-major columns, including overlapping aligned projection tails.

    Each rank produces an aligned physical column span. When output_width is
    smaller, the final spans overlap within the original matrix; copy only
    each rank's logical columns without padding or changing arithmetic.
    """
    element_bytes = {"bfloat16": 2, "float32": 4, "int32": 4}.get(dtype)
    output = output_width or width
    if (
        element_bytes is None
        or not 1 <= paths <= 8
        or ranks < 1
        or width % ranks
        or output % ranks
        or not width // ranks <= output <= width
        or (width // ranks * element_bytes) % 32
        or (output // ranks * element_bytes) % 32
    ):
        raise ValueError("Gathered columns must have aligned, equal rank partitions")
    local = width // ranks
    logical = output // ranks
    output_shape = (rows, output) if paths == 1 else (rows, paths, output)

    @T.prim_func
    def main(
        gathered: T.Tensor((ranks, rows, local), dtype),
        out: T.Tensor(output_shape, dtype),
    ):
        with T.Kernel(20, is_npu=True) as (cid, vid):
            data = T.alloc_ub((logical,), dtype)
            aiv = cid * 2 + vid
            items = rows * ranks
            if aiv < items:
                for step in T.serial(T.ceildiv(items - aiv, 40)):
                    index = aiv + step * 40
                    row, rank = index // ranks, index % ranks
                    first = T.max(0, rank * logical + local - output)
                    T.copy(gathered[rank, row, first : first + logical], data)
                    if paths == 1:
                        T.copy(data, out[row, rank * logical : (rank + 1) * logical])
                    else:
                        for path in T.serial(paths):
                            T.copy(data, out[row, path, rank * logical : (rank + 1) * logical])

    return main


def gather_rows(rows, table_rows, width, dtype="bfloat16", index_dtype="int64"):
    if isinstance(table_rows, int) and table_rows > 2147483647:
        raise ValueError("A2 row table exceeds signed 32-bit addressing")
    elem_bytes = {"int8": 1, "bfloat16": 2, "float16": 2, "int32": 4, "float32": 4}.get(dtype, 4)
    max_cols = 16384 // elem_bytes
    block = width if width <= max_cols else math.gcd(width, max_cols)
    blocks = width // block

    @T.prim_func
    def main(
        table: T.Tensor((table_rows, width), dtype),
        indices: T.Tensor((rows,), index_dtype),
        out: T.Tensor((rows, width), dtype),
        errors: T.Tensor((rows,), "int32"),
    ):
        with T.Kernel(20, is_npu=True) as (cid, vid):
            a = T.alloc_ub((block,), dtype)
            zero_buf = T.alloc_ub((block,), dtype)
            zeros = T.alloc_ub((max(1, block // 2),), "int16")
            flag = T.alloc_ub((8,), "int32")
            aiv = cid * 2 + vid
            items = rows * blocks
            if aiv < items:
                if dtype == "int8" and block >= 8:
                    T.reinterpretcast(zero_buf, zeros, "int8_t")
                    T.tile.fill(zeros, 0)
                elif block < 8:
                    for lane in T.serial(block):
                        zero_buf[lane] = 0
                else:
                    T.tile.fill(zero_buf, 0)
                T.pipe_barrier("ALL")
                for step in T.serial(T.ceildiv(items - aiv, 40)):
                    flat = aiv + step * 40
                    row = flat // blocks
                    column = (flat % blocks) * block
                    index = indices[row]
                    if index >= 0 and index < table_rows:
                        T.copy(table[T.cast(index, "int32"), column : column + block], a)
                        T.copy(a, out[row, column : column + block])
                    else:
                        T.copy(zero_buf, out[row, column : column + block])
                    if column == 0:
                        flag[0] = T.if_then_else(index < -1 or index >= table_rows, 1, 0)
                        T.copy(flag[:1], errors[row : row + 1])
                    T.set_flag("MTE3", "V", 7)
                    T.wait_flag("MTE3", "V", 7)

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
    groups = width // group
    tile_rows = 8 if width <= 512 and groups * 4 % 32 == 0 else 1

    @T.prim_func
    def main(
        values: T.Tensor((rows, width), "int8"),
        scales: T.Tensor((rows, groups), "float32"),
        out: T.Tensor((rows, width), "bfloat16"),
    ):
        with T.Kernel(20, is_npu=True) as (cid, vid):
            q = T.alloc_ub((tile_rows, width), "int8")
            half_values = T.alloc_ub((tile_rows * width,), "float16")
            a = T.alloc_ub((tile_rows * width,), "float32")
            q_flat = T.alloc_ub((tile_rows * width,), "int8")
            scale = T.alloc_ub((tile_rows, groups), "float32")
            s_flat = T.alloc_ub((tile_rows * groups,), "float32")
            y = T.alloc_ub((tile_rows, width), "bfloat16")
            y_flat = T.alloc_ub((tile_rows * width,), "bfloat16")
            T.reinterpretcast(q_flat, q, "int8_t")
            T.reinterpretcast(s_flat, scale, "float")
            T.reinterpretcast(y_flat, y, "bfloat16_t")
            aiv = cid * 2 + vid
            tiles = T.ceildiv(rows, tile_rows)
            if aiv < tiles:
                for step in T.serial(T.ceildiv(tiles - aiv, 40)):
                    first = (aiv + step * 40) * tile_rows
                    valid = T.min(tile_rows, rows - first)
                    T.copy(values[first : first + valid, :], q[:valid, :])
                    T.copy(scales[first : first + valid, :], scale[:valid, :])
                    T.pipe_barrier("ALL")
                    T.tile.cast(half_values, q_flat, "CAST_NONE", tile_rows * width)
                    T.tile.cast(a, half_values, "CAST_NONE", tile_rows * width)
                    for idx in T.serial(valid * groups):
                        T.tile.mul(
                            a[idx * group : (idx + 1) * group],
                            a[idx * group : (idx + 1) * group],
                            s_flat[idx],
                        )
                    T.tile.cast(y_flat, a, "CAST_RINT", tile_rows * width)
                    T.copy(y[:valid, :], out[first : first + valid, :])
                    T.set_flag("MTE3", "V", 7)
                    T.wait_flag("MTE3", "V", 7)

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


def copy_words(rows, chunk=2048):
    """Copy `rows` int32 words in `chunk`-word blocks (for batched pointer-table D2D spans)."""
    if chunk % 16 or chunk < 32:
        raise ValueError("Word copy chunk must be a 16-word multiple >= 32")
    half = chunk // 2

    @T.prim_func
    def main(
        src: T.Tensor((rows,), "int32"),
        dst: T.Tensor((rows,), "int32"),
    ):
        with T.Kernel(T.ceildiv(rows, chunk), is_npu=True) as (cid, vid):
            buf = T.alloc_ub((half,), "int32")
            first = cid * chunk + vid * half
            n = T.min(half, rows - first)
            if n > 0:
                T.copy(src[first : first + n], buf[:n])
                T.copy(buf[:n], dst[first : first + n])

    return main
