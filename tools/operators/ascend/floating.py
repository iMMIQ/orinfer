"""FP32 projection boundaries with exact BF16 operand decomposition on A2."""

import math
import tilelang.language as T


def split(rows, width, tile=256):
    if tile < 16 or tile > 8192 or tile % 16:
        raise ValueError("FP32 decomposition tile must be 16-aligned and at most 8192")
    block = math.gcd(width, tile)
    tiles = width // block

    @T.prim_func
    def main(
        source: T.Tensor((rows, width), "float32"),
        out: T.Tensor((3, rows, width), "bfloat16"),
    ):
        with T.Kernel(T.ceildiv(rows * tiles, 2), is_npu=True) as (cid, vid):
            x = T.alloc_ub((block,), "float32")
            restored = T.alloc_ub((block,), "float32")
            part = T.alloc_ub((block,), "bfloat16")
            index = cid * 2 + vid
            row, column = index // tiles, index % tiles * block
            if row < rows:
                T.copy(source[row, column : column + block], x)
                for plane in T.serial(3):
                    T.tile.cast(part, x, "CAST_RINT", block)
                    T.copy(part, out[plane, row, column : column + block])
                    T.set_flag("MTE3", "V", 7)
                    T.wait_flag("MTE3", "V", 7)
                    T.tile.cast(restored, part, "CAST_NONE", block)
                    T.tile.sub(x, x, restored)

    return main


def partials(
    rows,
    output,
    reduction,
    input_parts=1,
    weight_parts=3,
    splits=8,
    tile_m=16,
    block_k=128,
    variance=False,
    pipeline_stages=0,
    variance_rows=1,
    tile_n=32,
    variance_prefetch=False,
):
    if output % 32 or reduction % splits or reduction // splits % block_k:
        raise ValueError("Floating Cube requires aligned output and split reduction tiles")
    if input_parts not in (1, 3) or weight_parts not in (1, 3) or input_parts * weight_parts != 3:
        raise ValueError("One FP32 operand must have three BF16 parts")
    if variance and (input_parts != 1 or output != 32 or reduction % 256):
        raise ValueError("mHC variance requires one BF16 input and 32 output lanes")
    if tile_m not in (16, 32, 64, 128):
        raise ValueError("Floating Cube M tile must be 16, 32, 64 or 128")
    if tile_n not in (32, 64, 128) or output % tile_n or (variance and tile_n != 32):
        raise ValueError("Floating output tile must divide N and retain HC lanes")
    blocks = output // tile_n
    loops = reduction // splits // block_k
    if pipeline_stages not in (0, 2, 3) or (
        pipeline_stages and loops * input_parts * weight_parts < pipeline_stages
    ):
        raise ValueError("Floating pipeline requires enough K/part iterations for its stages")
    if max(1, pipeline_stages) * block_k * (tile_m + tile_n) * 2 > 524032:
        raise ValueError("Floating pipeline exceeds the A2 L1 allocation")
    variance_chunk = math.gcd(reduction // splits, 256)
    if variance_rows not in (1, 8, 16):
        raise ValueError("Variance row tile must be 1, 8 or 16")
    batch_rows = min(variance_rows, tile_m // 2)

    @T.prim_func
    def main(
        activation: T.Tensor((input_parts, rows, reduction), "bfloat16"),
        weight: T.Tensor((weight_parts, output, reduction), "bfloat16"),
        out: T.Tensor((rows, output // 32 * splits * 32), "float32"),
    ):
        with T.Kernel(T.ceildiv(rows, tile_m) * blocks * splits, is_npu=True) as (cid, vid):
            split_id = cid % splits
            tile = cid // splits % blocks
            first = cid // (splits * blocks) * tile_m
            with T.Scope("C"):
                a = T.alloc_L1((tile_m, block_k), "bfloat16")
                b = T.alloc_L1((tile_n, block_k), "bfloat16")
                c = T.alloc_L0C((tile_m, tile_n), "float32")
                valid = T.min(tile_m, rows - first)
                for step in T.Pipelined(
                    loops * input_parts * weight_parts, num_stages=pipeline_stages
                ):
                    k = step // (input_parts * weight_parts)
                    ap = step // weight_parts % input_parts
                    wp = step % weight_parts
                    column = (split_id * loops + k) * block_k
                    T.copy(
                        activation[ap, first : first + valid, column : column + block_k],
                        a[:valid, :],
                    )
                    T.copy(
                        weight[wp, tile * tile_n : (tile + 1) * tile_n, column : column + block_k],
                        b,
                    )
                    T.gemm_v0(a, b, c, transpose_B=True, init=(step == 0))
                T.copy(
                    c[:valid, :],
                    out[
                        first : first + valid,
                        (tile * splits + split_id) * tile_n : (tile * splits + split_id + 1)
                        * tile_n,
                    ],
                )
            if variance:
                with T.Scope("V"):
                    xb = T.alloc_ub((batch_rows, variance_chunk), "bfloat16")
                    bank = T.alloc_ub(
                        (batch_rows, reduction // splits if variance_prefetch else 16), "bfloat16"
                    )
                    x = T.alloc_ub((batch_rows, variance_chunk), "float32")
                    square = T.alloc_ub((batch_rows, variance_chunk), "float32")
                    partial = T.alloc_ub((batch_rows, 1), "float32")
                    total = T.alloc_ub((batch_rows, 1), "float32")
                    padded = T.alloc_ub((batch_rows, 8), "float32")
                    lane = T.alloc_ub((batch_rows * 8,), "float32")
                    row_f = T.alloc_ub((batch_rows * 8,), "float32")
                    row_i = T.alloc_ub((batch_rows * 8,), "int32")
                    mask = T.alloc_ub((batch_rows,), "uint8")
                    for part in T.serial(tile_m // (2 * batch_rows)):
                        row = first + vid * (tile_m // 2) + part * batch_rows
                        if row < rows:
                            valid_rows = T.min(batch_rows, rows - row)
                            T.tile.fill(total, 0.0)
                            if variance_prefetch:
                                T.copy(
                                    activation[
                                        0,
                                        row : row + valid_rows,
                                        split_id * (reduction // splits) : (split_id + 1)
                                        * (reduction // splits),
                                    ],
                                    bank,
                                )
                            for k in T.serial(reduction // splits // variance_chunk):
                                column = split_id * (reduction // splits) + k * variance_chunk
                                if variance_prefetch:
                                    T.copy(
                                        bank[:, k * variance_chunk : (k + 1) * variance_chunk], xb
                                    )
                                else:
                                    T.copy(
                                        activation[
                                            0,
                                            row : row + valid_rows,
                                            column : column + variance_chunk,
                                        ],
                                        xb,
                                    )
                                T.tile.cast(x, xb, "CAST_NONE", batch_rows * variance_chunk)
                                T.tile.mul(square, x, x)
                                T.reduce_sum(square, partial, dim=1)
                                T.tile.add(total, total, partial)
                            if batch_rows == 1:
                                T.copy(
                                    total[0, :1], out[row, split_id * 32 + 24 : split_id * 32 + 25]
                                )
                            else:
                                # Each row stores a complete aligned eight-lane
                                # suffix: variance followed by reserved zeros.
                                T.tile.broadcast(padded, total)
                                T.tile.createvecindex(lane, 0.0)
                                T.tile.mul(row_f, lane, 0.125)
                                T.tile.cast(row_i, row_f, "CAST_FLOOR", batch_rows * 8)
                                T.tile.cast(row_f, row_i, "CAST_NONE", batch_rows * 8)
                                T.tile.mul(row_f, row_f, 8.0)
                                T.tile.compare(mask, lane, row_f, "EQ")
                                T.tile.select(padded, mask, padded, 0.0, "VSEL_TENSOR_SCALAR_MODE")
                                T.copy(
                                    padded[:valid_rows, :],
                                    out[
                                        row : row + valid_rows,
                                        split_id * 32 + 24 : split_id * 32 + 32,
                                    ],
                                )
                            T.set_flag("MTE3", "V", 7)
                            T.wait_flag("MTE3", "V", 7)

    return main


def reduce_partials(rows, output, tile_n=32, splits=8):
    """Restore the common 32-column/eight-split reduction from wide Cube tiles."""
    if tile_n not in (32, 64, 128) or output % tile_n or splits != 8:
        raise ValueError("Invalid floating partial layout")
    tiles, groups = output // 32, tile_n // 32

    @T.prim_func
    def main(
        partials: T.Tensor((rows, output // tile_n, splits, tile_n), "float32"),
        out: T.Tensor((rows, output), "float32"),
    ):
        with T.Kernel(T.ceildiv(rows * tiles, 2), is_npu=True) as (cid, vid):
            p = T.alloc_ub((splits, 32), "float32")
            total = T.alloc_ub((1, 32), "float32")
            index = cid * 2 + vid
            tile, row = index % tiles, index // tiles
            if row < rows:
                column = tile % groups * 32
                T.copy(partials[row, tile // groups, :, column : column + 32], p)
                T.reduce_sum(p, total, dim=0)
                T.copy(total[0, :], out[row, tile * 32 : (tile + 1) * 32])

    return main
