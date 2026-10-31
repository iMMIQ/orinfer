"""A2 NT integer/BF16 projections with a common Cube/Vector epilogue contract."""

import math

import tilelang.language as T


def projection(
    rows,
    output,
    reduction,
    integer=False,
    block_k=128,
    tile_n=64,
    tile_m=16,
    pipeline_stages=0,
    unit_scale=False,
):
    if integer and unit_scale:
        raise ValueError("Direct BF16 output requires a BF16 projection with unit scales")
    alignment = 32 if integer else 128
    if (isinstance(rows, int) and rows < 1) or output % 32 or reduction % alignment:
        raise ValueError("A2 projection requires positive rows and aligned N/K")
    if tile_n not in (32, 64, 128, 256):
        raise ValueError("A2 projection N tile must be 32, 64, 128 or 256")
    if tile_m not in (16, 32, 64, 128, 256) or (tile_m == 256 and not integer):
        raise ValueError("A2 projection M tile must fit the INT8/BF16 L0A slot")
    block_m, block_n = tile_m, math.gcd(output, tile_n)
    vector_m = block_m // 2
    if block_n * 128 * (1 if integer else 2) > 32768:
        raise ValueError("A2 projection B tile exceeds the L0B ping-pong slot")
    if block_m * 128 * (1 if integer else 2) > 32768 or block_m * block_n * 4 > 131072:
        raise ValueError("A2 projection tile exceeds its L0A/L0C allocation")
    legal_k = 0 < block_k <= 512 and block_k % alignment == 0
    if not legal_k or reduction % block_k:
        raise ValueError("A2 projection K tile must be aligned, divide K and be <=512")
    if pipeline_stages not in (0, 2, 3) or (
        pipeline_stages and reduction // block_k < pipeline_stages
    ):
        raise ValueError("Projection pipeline requires enough K iterations for its stages")
    if max(1, pipeline_stages) * block_k * (block_m + block_n) * (1 if integer else 2) > 524032:
        raise ValueError("Projection pipeline exceeds the A2 L1 allocation")
    m_blocks = (rows + block_m - 1) // block_m
    n_blocks = output // block_n
    input_dtype = "int8" if integer else "bfloat16"
    accum_dtype = "int32" if integer else "float32"

    @T.prim_func
    def main(
        activation: T.Tensor((rows, reduction), input_dtype),
        weight: T.Tensor((output, reduction), input_dtype),
        token_scale: T.Tensor((rows,), "float32"),
        channel_scale: T.Tensor((output,), "float32"),
        out: T.Tensor((rows, output), "bfloat16"),
        workspace: T.Tensor((m_blocks * block_m, output), accum_dtype),
    ):
        with T.Kernel(m_blocks * n_blocks, is_npu=True) as (cid, vid):
            bm = cid // n_blocks
            bn = cid % n_blocks
            with T.Scope("C"):
                a = T.alloc_L1((block_m, block_k), input_dtype)
                w = T.alloc_L1((block_n, block_k), input_dtype)
                c = T.alloc_L0C((block_m, block_n), accum_dtype)
                valid_rows = T.min(block_m, rows - bm * block_m)
                for k in T.Pipelined(reduction // block_k, num_stages=pipeline_stages):
                    T.copy(
                        activation[
                            bm * block_m : bm * block_m + valid_rows,
                            k * block_k : (k + 1) * block_k,
                        ],
                        a,
                    )
                    T.copy(
                        weight[bn * block_n : (bn + 1) * block_n, k * block_k : (k + 1) * block_k],
                        w,
                    )
                    T.gemm_v0(a, w, c, transpose_B=True, init=(k == 0))
                if unit_scale:
                    T.copy(
                        c,
                        out[
                            bm * block_m : bm * block_m + valid_rows,
                            bn * block_n : (bn + 1) * block_n,
                        ],
                    )
                    # The bounded dynamic grid reuses L0C for its next tile.
                    T.pipe_barrier("FIX")
                else:
                    T.copy(
                        c,
                        workspace[
                            bm * block_m : (bm + 1) * block_m,
                            bn * block_n : (bn + 1) * block_n,
                        ],
                    )
                    T.set_cross_flag("FIX", 0)
                    T.wait_cross_flag(1)
            if not unit_scale:
                with T.Scope("V"):
                    raw = T.alloc_ub((vector_m, block_n), accum_dtype)
                    accumulator = T.alloc_ub((vector_m, block_n), "float32")
                    sa = T.alloc_ub((vector_m, 1), "float32")
                    sw = T.alloc_ub((1, block_n), "float32")
                    broadcast = T.alloc_ub((vector_m, block_n), "float32")
                    y = T.alloc_ub((vector_m, block_n), "bfloat16")
                    start = bm * block_m + vid * vector_m
                    T.wait_cross_flag(0)
                    if start < rows:
                        valid = T.min(vector_m, rows - start)
                        T.copy(
                            workspace[start : start + vector_m, bn * block_n : (bn + 1) * block_n],
                            raw,
                        )
                        if integer:
                            T.tile.cast(accumulator, raw, "CAST_RINT", vector_m * block_n)
                        else:
                            T.copy(raw, accumulator)
                        T.tile.fill(sa, 0.0)
                        T.copy(token_scale[start : start + valid], sa[:valid, :])
                        T.copy(channel_scale[bn * block_n : (bn + 1) * block_n], sw)
                        T.tile.broadcast(broadcast, sa)
                        T.tile.mul(accumulator, accumulator, broadcast)
                        T.tile.broadcast(broadcast, sw)
                        T.tile.mul(accumulator, accumulator, broadcast)
                        T.tile.cast(y, accumulator, "CAST_RINT", vector_m * block_n)
                        T.copy(
                            y[:valid, :],
                            out[start : start + valid, bn * block_n : (bn + 1) * block_n],
                        )
                    T.set_cross_flag("MTE3", 1)

    if unit_scale:
        return main
    return main.with_attr("orinfer.coupled_resources", True).with_attr(
        "orinfer.explicit_cross_sync", True
    )


def fp32_partials(rows, output, reduction, splits=8, activation_dtype="bfloat16"):
    """FP32-sensitive compressor projections; BF16 activations, FP32 weights."""
    if output % 32 or reduction % splits:
        raise ValueError("FP32 projection requires N%32=0 and K%splits=0")
    tiles = output // 32
    chunk = math.gcd(reduction // splits, 256)
    loops = reduction // splits // chunk

    @T.prim_func
    def main(
        activation: T.Tensor((rows, reduction), activation_dtype),
        weight: T.Tensor((output, reduction), "float32"),
        out: T.Tensor((rows, tiles, splits, 32), "float32"),
    ):
        with T.Kernel(T.ceildiv(rows * tiles * splits, 2), is_npu=True) as (cid, vid):
            xb = T.alloc_ub((1, chunk), activation_dtype)
            x = T.alloc_ub((1, chunk), "float32")
            w = T.alloc_ub((32, chunk), "float32")
            product = T.alloc_ub((32, chunk), "float32")
            partial = T.alloc_ub((32, 1), "float32")
            total = T.alloc_ub((32, 1), "float32")
            index = cid * 2 + vid
            split = index % splits
            tile = index // splits % tiles
            row = index // (splits * tiles)
            if row < rows:
                T.tile.fill(total, 0.0)
                for k in T.serial(loops):
                    start = (split * loops + k) * chunk
                    T.copy(activation[row : row + 1, start : start + chunk], xb)
                    T.copy(weight[tile * 32 : (tile + 1) * 32, start : start + chunk], w)
                    if activation_dtype == "float32":
                        T.copy(xb, x)
                    else:
                        T.tile.cast(x, xb, "CAST_NONE", chunk)
                    T.tile.broadcast(product, x)
                    T.tile.mul(product, product, w)
                    T.reduce_sum(product, partial, dim=1)
                    T.tile.add(total, total, partial)
                T.copy(total, out[row, tile, split, :])
                # Bounded dynamic grids reuse this UB tile for another row.
                # Complete the GM store before the following zero-fill.
                T.pipe_barrier("ALL")

    return main


def fp32_reduce(rows, output, splits=8):
    tiles = output // 32

    @T.prim_func
    def main(
        partials: T.Tensor((rows, tiles, splits, 32), "float32"),
        out: T.Tensor((rows, output), "float32"),
    ):
        with T.Kernel(T.ceildiv(rows * tiles, 2), is_npu=True) as (cid, vid):
            p = T.alloc_ub((splits, 32), "float32")
            total = T.alloc_ub((1, 32), "float32")
            index = cid * 2 + vid
            tile = index % tiles
            row = index // tiles
            if row < rows:
                T.copy(partials[row, tile, :, :], p)
                T.reduce_sum(p, total, dim=0)
                T.copy(total[0, :], out[row, tile * 32 : (tile + 1) * 32])
                T.pipe_barrier("ALL")

    return main
