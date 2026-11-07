"""Route exact integer schedules by measured per-expert row reuse.

Both schedules retain the same global row offsets. Only tile ownership is
partitioned, so their outputs are disjoint without dispatching activations twice.
"""

import tilelang.language as T
from kernels.shared.rows import row_count


def validate(experts, hot_m, cold_m, cold_rows):
    if any(type(x) is not int or x <= 0 for x in (experts, hot_m, cold_m, cold_rows)):
        raise ValueError("Invalid expert partition geometry")


def offsets(experts, hot_m, cold_m, cold_rows):
    validate(experts, hot_m, cold_m, cold_rows)
    width = 1 << (experts - 1).bit_length()

    @T.prim_func
    def main(
        Counts: T.Tensor((experts,), T.int32),
        RowOffsets: T.Tensor((experts,), T.int32),
        TileOffsets: T.Tensor((experts,), T.int32),
        TileCount: T.Tensor((1,), T.int32),
        ColdOffsets: T.Tensor((experts,), T.int32),
        ColdCount: T.Tensor((1,), T.int32),
    ):
        with T.Kernel(1, threads=128):
            counts = T.alloc_fragment((width,), T.int32)
            rows = T.alloc_fragment((width,), T.int32)
            hot = T.alloc_fragment((width,), T.int32)
            cold = T.alloc_fragment((width,), T.int32)
            for e in T.Parallel(width):
                counts[e] = T.if_then_else(e < experts, Counts[e], 0)
                rows[e] = counts[e]
                hot[e] = T.if_then_else(counts[e] > cold_rows, T.ceildiv(counts[e], hot_m), 0)
                cold[e] = T.if_then_else(counts[e] <= cold_rows, T.ceildiv(counts[e], cold_m), 0)
            T.cumsum(rows, dim=0)
            T.cumsum(hot, dim=0)
            T.cumsum(cold, dim=0)
            for e in T.Parallel(experts):
                RowOffsets[e] = rows[e] - counts[e]
                TileOffsets[e] = hot[e] - T.if_then_else(
                    counts[e] > cold_rows, T.ceildiv(counts[e], hot_m), 0
                )
                ColdOffsets[e] = cold[e] - T.if_then_else(
                    counts[e] <= cold_rows, T.ceildiv(counts[e], cold_m), 0
                )
            for e in T.Parallel(width):
                if e == width - 1:
                    TileCount[0] = hot[e]
                    ColdCount[0] = cold[e]

    return main


def tiles(M, experts, top_k, hot_m, cold_m, cold_rows, dynamic_rows=False):
    validate(experts, hot_m, cold_m, cold_rows)
    if any(type(x) is not int or x <= 0 for x in (M, top_k)) or top_k > experts:
        raise ValueError("Invalid partition row capacity")
    hot_cap = (M * top_k + hot_m - 1) // hot_m + min(experts, M * top_k)
    cold_cap = (M * top_k + cold_m - 1) // cold_m + min(experts, M * top_k)
    m = row_count(M, dynamic_rows)

    @T.prim_func
    def main(
        Counts: T.Tensor((experts,), T.int32),
        TileOffsets: T.Tensor((experts,), T.int32),
        TileExpert: T.Tensor((hot_cap,), T.int32),
        TileRow: T.Tensor((hot_cap,), T.int32),
        ColdOffsets: T.Tensor((experts,), T.int32),
        ColdExpert: T.Tensor((cold_cap,), T.int32),
        ColdRow: T.Tensor((cold_cap,), T.int32),
    ):
        with T.Kernel(experts, threads=128) as e:
            for tile in T.Parallel(T.ceildiv(m, min(hot_m, cold_m))):
                if Counts[e] <= cold_rows:
                    if tile * cold_m < Counts[e]:
                        destination = ColdOffsets[e] + tile
                        ColdExpert[destination] = e
                        ColdRow[destination] = tile * cold_m
                else:
                    if tile * hot_m < Counts[e]:
                        destination = TileOffsets[e] + tile
                        TileExpert[destination] = e
                        TileRow[destination] = tile * hot_m

    return main
