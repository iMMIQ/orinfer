"""Shared mHC FP32 normalization with the original eight-split boundary."""

import tilelang.language as T
from kernels.ascend.vector_math import rsqrt_fp32


def normalization(tile_rows, width, splits=8, eps=1e-20):
    """Inline the same normalization into standalone and Sinkhorn producers."""
    if splits != 8:
        raise ValueError("mHC normalization requires eight splits")
    count = tile_rows * 32

    @T.macro
    def normalize(partials, total, first, valid):
        norm_p = T.alloc_ub((tile_rows * splits, 32), "float32")
        norm_variance = T.alloc_ub((tile_rows, 32), "float32")
        norm_inverse = T.alloc_ub((tile_rows, 32), "float32")
        norm_correction = T.alloc_ub((tile_rows, 32), "float32")
        norm_lane = T.alloc_ub((tile_rows, 32), "float32")
        norm_floor_f = T.alloc_ub((tile_rows, 32), "float32")
        norm_floor_i = T.alloc_ub((tile_rows, 32), "int32")
        norm_offsets = T.alloc_ub((tile_rows, 32), "uint32")
        norm_mask = T.alloc_ub((count // 8,), "uint8")
        T.copy(partials[first * splits : (first + valid) * splits, :], norm_p[: valid * splits, :])
        for row in T.serial(tile_rows):
            T.reduce_sum(
                norm_p[row * splits : (row + 1) * splits, :], total[row : row + 1, :], dim=0
            )
        T.tile.createvecindex(norm_lane, 0.0)
        T.tile.mul(norm_floor_f, norm_lane, 0.03125)
        T.tile.cast(norm_floor_i, norm_floor_f, "CAST_FLOOR", count)
        T.tile.cast(norm_floor_f, norm_floor_i, "CAST_NONE", count)
        T.tile.mul(norm_floor_f, norm_floor_f, 32.0)
        T.tile.sub(norm_lane, norm_lane, norm_floor_f)
        T.tile.add(norm_floor_f, norm_floor_f, 24.0)
        T.tile.mul(norm_floor_f, norm_floor_f, 4.0)
        T.tile.cast(norm_floor_i, norm_floor_f, "CAST_RINT", count)
        T.reinterpretcast(norm_offsets, norm_floor_i, "uint32_t")
        T.tile.gather(norm_variance, total, norm_offsets, 0)
        T.tile.div(norm_variance, norm_variance, float(4 * width))
        T.tile.add(norm_variance, norm_variance, eps)
        rsqrt_fp32(norm_variance, norm_inverse, norm_correction)
        T.tile.mul(total, total, norm_inverse)
        T.tile.compare(norm_mask, norm_lane, 24.0, "LT")
        T.tile.select(total, norm_mask, total, 0.0, "VSEL_TENSOR_SCALAR_MODE")

    return normalize


def reduce(rows, width, splits=8, eps=1e-20, tile_rows=16):
    if splits != 8 or tile_rows not in (8, 16, 32):
        raise ValueError("Batched mHC supports eight splits and 8/16/32-row tiles")
    normalize = normalization(tile_rows, width, splits, eps)

    @T.prim_func
    def main(
        partials: T.Tensor((rows * splits, 32), "float32"),
        out: T.Tensor((rows, 32), "float32"),
    ):
        with T.Kernel(T.ceildiv(rows, tile_rows * 2), is_npu=True) as (cid, vid):
            total = T.alloc_ub((tile_rows, 32), "float32")
            first = (cid * 2 + vid) * tile_rows
            if first < rows:
                valid = T.min(tile_rows, rows - first)
                normalize(partials, total, first, valid)
                T.copy(total[:valid, :], out[first : first + valid, :])
                T.set_flag("MTE3", "V", 7)
                T.wait_flag("MTE3", "V", 7)

    return main
