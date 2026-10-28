"""Exact INT8 Cube scores for DSv4.1's full-coverage causal index view."""

import tilelang.language as T


def causal_logits(
    rows,
    length,
    heads=32,
    dimension=128,
    tile_s=64,
    vector_scales=True,
    tile_rows=1,
    vector_mask=False,
):
    if heads != 32 or dimension != 128 or tile_s not in (32, 64, 128):
        raise ValueError("Causal index Cube requires H32/D128 and aligned key tiles")
    if tile_rows not in (1, 2, 4, 8) or tile_rows * heads * tile_s * 4 > 131072:
        raise ValueError("Causal index query tile exceeds A2 L0C")
    blocks = (length + tile_s - 1) // tile_s
    padded = blocks * tile_s
    lanes = tile_s // 2
    factor = dimension**-0.5 * heads**-0.5

    @T.prim_func
    def main(
        query: T.Tensor((rows * heads, dimension), "int8"),
        query_scale: T.Tensor((rows, heads), "float16"),
        keys: T.Tensor((length, dimension), "int8"),
        key_scale: T.Tensor((length,), "float16"),
        weights: T.Tensor((rows, heads), "bfloat16"),
        indices: T.Tensor((rows, length), "int32"),
        out: T.Tensor((rows, length), "float32"),
        workspace: T.Tensor((rows * heads, padded), "int32"),
    ):
        with T.Kernel(T.ceildiv(rows, tile_rows) * blocks, is_npu=True) as (cid, vid):
            first, tile = cid // blocks * tile_rows, cid % blocks
            valid_rows = T.min(tile_rows, rows - first)
            start = tile * tile_s
            with T.Scope("C"):
                a = T.alloc_L1((tile_rows * heads, dimension), "int8")
                b = T.alloc_L1((tile_s, dimension), "int8")
                c = T.alloc_L0C((tile_rows * heads, tile_s), "int32")
                if indices[first + valid_rows - 1, start] >= 0:
                    valid = T.min(tile_s, length - start)
                    T.copy(
                        query[first * heads : (first + valid_rows) * heads, :],
                        a[: valid_rows * heads, :],
                    )
                    T.copy(keys[start : start + valid, :], b[:valid, :])
                    T.gemm_v0(a, b, c, transpose_B=True, init=True)
                    T.copy(
                        c[: valid_rows * heads, :],
                        workspace[
                            first * heads : (first + valid_rows) * heads, start : start + tile_s
                        ],
                    )
                    T.set_cross_flag("FIX", 0)
            with T.Scope("V"):
                raw = T.alloc_ub((heads, lanes), "int32")
                dots = T.alloc_ub((heads, lanes), "float32")
                qs_half = T.alloc_ub((heads, 1), "float16")
                qs = T.alloc_ub((heads, 1), "float32")
                ks_half = T.alloc_ub((lanes,), "float16")
                ks = T.alloc_ub((1, lanes), "float32")
                w_bf = T.alloc_ub((heads, 1), "bfloat16")
                w_half = T.alloc_ub((heads, 1), "float16")
                w = T.alloc_ub((heads, 1), "float32")
                partial = T.alloc_ub((1, lanes), "float32")
                total = T.alloc_ub((1, lanes), "float32")
                ids = T.alloc_ub((lanes,), "int32")
                mask = T.alloc_ub((lanes // 8,), "uint8")
                expanded = T.alloc_ub((heads, lanes), "float32")
                column = start + vid * lanes
                if indices[first + valid_rows - 1, start] >= 0:
                    T.wait_cross_flag(0)
                for row_offset in T.serial(tile_rows):
                    row = first + row_offset
                    if row < rows:
                        if indices[row, start] >= 0:
                            if column < length:
                                valid = T.min(lanes, length - column)
                                T.copy(
                                    workspace[
                                        row * heads : (row + 1) * heads, column : column + lanes
                                    ],
                                    raw,
                                )
                                T.tile.cast(dots, raw, "CAST_RINT", heads * lanes)
                                T.copy(query_scale[row, :], qs_half)
                                T.tile.cast(qs, qs_half, "CAST_NONE", heads)
                                T.copy(weights[row, :], w_bf)
                                T.tile.cast(w, w_bf, "CAST_NONE", heads)
                                T.tile.mul(w, w, factor)
                                T.tile.cast(w_half, w, "CAST_RINT", heads)
                                T.tile.cast(w, w_half, "CAST_NONE", heads)
                                T.copy(key_scale[column : column + valid], ks_half[:valid])
                                T.tile.cast(ks, ks_half, "CAST_NONE", lanes)
                                T.tile.fill(total, 0.0)
                                if vector_scales:
                                    T.tile.max(dots, dots, 0.0)
                                    T.tile.broadcast(expanded, qs)
                                    T.tile.mul(dots, dots, expanded)
                                    T.tile.broadcast(expanded, ks)
                                    T.tile.mul(dots, dots, expanded)
                                    T.tile.broadcast(expanded, w)
                                    T.tile.mul(dots, dots, expanded)
                                for head in T.serial(heads):
                                    T.copy(dots[head : head + 1, :], partial)
                                    if not vector_scales:
                                        T.tile.max(partial, partial, 0.0)
                                        T.tile.mul(partial, partial, qs[head, 0])
                                        T.tile.mul(partial, partial, ks)
                                        T.tile.mul(partial, partial, w[head, 0])
                                    T.tile.add(total, total, partial)
                                T.copy(indices[row, column : column + valid], ids[:valid])
                                if vector_mask:
                                    T.tile.cast(partial, ids, "CAST_NONE", lanes)
                                    T.tile.compare(mask, partial, 0.0, "GE")
                                    T.tile.select(
                                        total,
                                        mask,
                                        total,
                                        -T.infinity("float32"),
                                        "VSEL_TENSOR_SCALAR_MODE",
                                    )
                                else:
                                    for lane in T.serial(lanes):
                                        if ids[lane] < 0:
                                            total[0, lane] = -T.infinity("float32")
                                T.copy(total[0, :valid], out[row, column : column + valid])
                        else:
                            if column < length:
                                valid = T.min(lanes, length - column)
                                T.tile.fill(total, -T.infinity("float32"))
                                T.copy(total[0, :valid], out[row, column : column + valid])

    return main
