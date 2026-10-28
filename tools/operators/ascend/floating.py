"""FP32 projection boundaries with exact BF16 operand decomposition on A2."""

import math
import tilelang.language as T


def split(rows, width):
    block = math.gcd(width, 256)
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
):
    if output % 32 or reduction % splits or reduction // splits % block_k:
        raise ValueError("Floating Cube requires aligned output and split reduction tiles")
    if input_parts not in (1, 3) or weight_parts not in (1, 3) or input_parts * weight_parts != 3:
        raise ValueError("One FP32 operand must have three BF16 parts")
    if variance and (input_parts != 1 or output != 32 or reduction % 256):
        raise ValueError("mHC variance requires one BF16 input and 32 output lanes")
    if tile_m not in (16, 32, 64):
        raise ValueError("Floating Cube M tile must be 16, 32 or 64")
    blocks = output // 32
    loops = reduction // splits // block_k
    variance_chunk = math.gcd(reduction // splits, 256)

    @T.prim_func
    def main(
        activation: T.Tensor((input_parts, rows, reduction), "bfloat16"),
        weight: T.Tensor((weight_parts, output, reduction), "bfloat16"),
        out: T.Tensor((rows, blocks * splits * 32), "float32"),
    ):
        with T.Kernel(T.ceildiv(rows, tile_m) * blocks * splits, is_npu=True) as (cid, vid):
            split_id = cid % splits
            tile = cid // splits % blocks
            first = cid // (splits * blocks) * tile_m
            with T.Scope("C"):
                a = T.alloc_L1((tile_m, block_k), "bfloat16")
                b = T.alloc_L1((32, block_k), "bfloat16")
                c = T.alloc_L0C((tile_m, 32), "float32")
                valid = T.min(tile_m, rows - first)
                for step in T.serial(loops * input_parts * weight_parts):
                    k = step // (input_parts * weight_parts)
                    ap = step // weight_parts % input_parts
                    wp = step % weight_parts
                    column = (split_id * loops + k) * block_k
                    T.copy(
                        activation[ap, first : first + valid, column : column + block_k],
                        a[:valid, :],
                    )
                    T.copy(
                        weight[wp, tile * 32 : (tile + 1) * 32, column : column + block_k],
                        b,
                    )
                    T.gemm_v0(a, b, c, transpose_B=True, init=(step == 0))
                T.copy(
                    c[:valid, :],
                    out[
                        first : first + valid,
                        (tile * splits + split_id) * 32 : (tile * splits + split_id + 1) * 32,
                    ],
                )
            if variance:
                with T.Scope("V"):
                    xb = T.alloc_ub((1, variance_chunk), "bfloat16")
                    x = T.alloc_ub((1, variance_chunk), "float32")
                    square = T.alloc_ub((1, variance_chunk), "float32")
                    partial = T.alloc_ub((1, 1), "float32")
                    total = T.alloc_ub((1, 1), "float32")
                    for lane in T.serial(tile_m // 2):
                        row = first + vid * (tile_m // 2) + lane
                        if row < rows:
                            T.tile.fill(total, 0.0)
                            for k in T.serial(reduction // splits // variance_chunk):
                                column = split_id * (reduction // splits) + k * variance_chunk
                                T.copy(
                                    activation[0, row : row + 1, column : column + variance_chunk],
                                    xb,
                                )
                                T.tile.cast(x, xb, "CAST_NONE", variance_chunk)
                                T.tile.mul(square, x, x)
                                T.reduce_sum(square, partial, dim=1)
                                T.tile.add(total, total, partial)
                            T.copy(total[0, :1], out[row, split_id * 32 + 24 : split_id * 32 + 25])

    return main
