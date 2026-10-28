"""Row-batched mHC FP32 normalization with the original eight-split boundary."""

import tilelang.language as T
from tools.operators.ascend.vector_math import rsqrt_fp32


def reduce(rows, width, splits=8, eps=1e-20, tile_rows=16):
    if splits != 8 or tile_rows not in (8, 16, 32):
        raise ValueError("Batched mHC supports eight splits and 8/16/32-row tiles")
    count = tile_rows * 32

    @T.prim_func
    def main(
        partials: T.Tensor((rows * splits, 32), "float32"),
        out: T.Tensor((rows, 32), "float32"),
    ):
        with T.Kernel(T.ceildiv(rows, tile_rows * 2), is_npu=True) as (cid, vid):
            p = T.alloc_ub((tile_rows * splits, 32), "float32")
            total = T.alloc_ub((tile_rows, 32), "float32")
            variance = T.alloc_ub((tile_rows, 32), "float32")
            inverse = T.alloc_ub((tile_rows, 32), "float32")
            correction = T.alloc_ub((tile_rows, 32), "float32")
            lane = T.alloc_ub((tile_rows, 32), "float32")
            floor_f = T.alloc_ub((tile_rows, 32), "float32")
            floor_i = T.alloc_ub((tile_rows, 32), "int32")
            offsets = T.alloc_ub((tile_rows, 32), "uint32")
            mask = T.alloc_ub((count // 8,), "uint8")
            first = (cid * 2 + vid) * tile_rows
            if first < rows:
                valid = T.min(tile_rows, rows - first)
                T.copy(
                    partials[first * splits : (first + valid) * splits, :], p[: valid * splits, :]
                )
                for row in T.serial(tile_rows):
                    T.reduce_sum(
                        p[row * splits : (row + 1) * splits, :], total[row : row + 1, :], dim=0
                    )
                T.tile.createvecindex(lane, 0.0)
                T.tile.mul(floor_f, lane, 0.03125)
                T.tile.cast(floor_i, floor_f, "CAST_FLOOR", count)
                T.tile.cast(floor_f, floor_i, "CAST_NONE", count)
                T.tile.mul(floor_f, floor_f, 32.0)
                T.tile.sub(lane, lane, floor_f)
                T.tile.add(floor_f, floor_f, 24.0)
                T.tile.mul(floor_f, floor_f, 4.0)
                T.tile.cast(floor_i, floor_f, "CAST_RINT", count)
                T.reinterpretcast(offsets, floor_i, "uint32_t")
                T.tile.gather(variance, total, offsets, 0)
                T.tile.div(variance, variance, float(4 * width))
                T.tile.add(variance, variance, eps)
                rsqrt_fp32(variance, inverse, correction)
                T.tile.mul(total, total, inverse)
                T.tile.compare(mask, lane, 24.0, "LT")
                T.tile.select(total, mask, total, 0.0, "VSEL_TENSOR_SCALAR_MODE")
                T.copy(total[:valid, :], out[first : first + valid, :])
                T.set_flag("MTE3", "V", 7)
                T.wait_flag("MTE3", "V", 7)

    return main
