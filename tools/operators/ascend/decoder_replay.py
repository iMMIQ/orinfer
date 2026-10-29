"""Compaction at a decoder replay boundary; absolute coordinates stay intact."""

import tilelang.language as T


def compact_suffix(rows, width, keep, dtype="bfloat16"):
    if width < 1 or keep < 1 or dtype not in ("bfloat16", "float32", "int32"):
        raise ValueError("Invalid decoder replay compaction geometry")
    tile = 256 if width >= 256 else 32
    columns = T.ceildiv(width, tile)

    @T.prim_func
    def main(
        source: T.Tensor((rows, width), dtype),
        output: T.Tensor((keep, width), dtype),
    ):
        with T.Kernel(T.ceildiv(T.min(rows, keep) * columns, 2), is_npu=True) as (cid, vid):
            values = T.alloc_ub((tile,), dtype)
            index = cid * 2 + vid
            row, column = index // columns, index % columns * tile
            if row < T.min(rows, keep):
                count = T.min(tile, width - column)
                old = rows - T.min(rows, keep) + row
                T.copy(source[old, column : column + count], values[:count])
                T.copy(values[:count], output[row, column : column + count])
                T.pipe_barrier("ALL")

    return main


def replay_start(rows, keep):
    if keep < 1:
        raise ValueError("Decoder replay needs a positive suffix")

    @T.prim_func
    def main(
        positions: T.Tensor((rows,), "int32"),
        output: T.Tensor((1,), "int32"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            value = T.alloc_ub((8,), "int32")
            if vid == 0:
                value[0] = T.if_then_else(rows > keep, positions[T.max(0, rows - keep)], 0)
                T.copy(value[:1], output[:1])

    return main
