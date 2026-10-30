"""Bounded Cube QK/PV with FP32 masked softmax and BF16 probabilities."""

import tilelang.language as T


def _head_geometry(heads, head_tile):
    padded = (heads + 15) // 16 * 16
    if head_tile not in (16, 32, 64):
        raise ValueError("Cube head tile must be16,32or64")
    tile = min(padded, head_tile)
    if padded % tile:
        raise ValueError("Head tile must divide the padded head dimension")
    return padded, tile, padded // tile


def qk(rows, heads, selected, dimension=512, tile_s=64, head_tile=16):
    if not 0 < heads <= 64 or selected % tile_s or dimension % 128:
        raise ValueError("Cube attention requires H<=64, S%tile_s=0 and D%128=0")
    blocks = selected // tile_s
    padded_heads, block_heads, groups = _head_geometry(heads, head_tile)

    @T.prim_func
    def main(
        query: T.Tensor((rows, heads, dimension), "bfloat16"),
        kv: T.Tensor((rows, selected, dimension), "bfloat16"),
        scores: T.Tensor((rows, padded_heads, selected), "float32"),
    ):
        with T.Kernel(rows * groups * blocks, is_npu=True) as (cid, vid):
            row, tile = cid // (groups * blocks), cid % blocks
            first = cid // blocks % groups * block_heads
            live = T.min(block_heads, heads - first)
            with T.Scope("C"):
                a = T.alloc_L1((block_heads, 128), "bfloat16")
                b = T.alloc_L1((tile_s, 128), "bfloat16")
                c = T.alloc_L0C((block_heads, tile_s), "float32")
                for k in T.serial(dimension // 128):
                    T.copy(query[row, first : first + live, k * 128 : (k + 1) * 128], a[:live, :])
                    T.copy(kv[row, tile * tile_s : (tile + 1) * tile_s, k * 128 : (k + 1) * 128], b)
                    T.gemm_v0(a, b, c, transpose_B=True, init=(k == 0))
                T.copy(
                    c, scores[row, first : first + block_heads, tile * tile_s : (tile + 1) * tile_s]
                )

    return main


@T.macro
def softmax_values(
    x,
    mask,
    maximum,
    denominator,
    sink_value,
    p,
    selected,
    dimension,
    head_tile,
    expanded,
):
    """Full-row FP32 normalization, with the same scalar reciprocal per head."""
    T.tile.mul(x, x, dimension**-0.5)
    T.tile.select(x, mask, x, -T.infinity("float32"), "VSEL_TENSOR_SCALAR_MODE")
    T.reduce_max(x, maximum, dim=1)
    T.tile.max(maximum, maximum, sink_value)
    if head_tile == 1:
        T.tile.sub(x, x, maximum[0, 0])
    else:
        T.tile.broadcast(expanded, maximum)
        T.tile.sub(x, x, expanded)
    T.tile.exp(x, x)
    T.reduce_sum(x, denominator, dim=1)
    T.tile.sub(sink_value, sink_value, maximum)
    T.tile.exp(sink_value, sink_value)
    T.tile.add(denominator, denominator, sink_value)
    # Vector Div uses a different approximation from scalar reciprocal/Muls.
    # Keep the latter so batching heads preserves the BF16 probability bits.
    for lane in T.serial(head_tile):
        T.tile.div(x[lane : lane + 1, :], x[lane : lane + 1, :], denominator[lane, 0])
    T.tile.cast(p, x, "CAST_RINT", head_tile * selected)


@T.macro
def softmax_row(
    scores,
    indices,
    sink,
    probabilities,
    row,
    head,
    x,
    ids,
    valid,
    mask,
    maximum,
    denominator,
    sink_value,
    p,
    selected,
    dimension,
    prepare_mask=True,
):
    T.copy(scores[row, head, :], x)
    if prepare_mask:
        T.copy(indices[row, :], ids)
        T.tile.cast(valid, ids, "CAST_NONE", selected)
        T.tile.compare(mask, valid, -1.0, "GT")
    sink_value[0, 0] = sink[head]
    softmax_values(x, mask, maximum, denominator, sink_value, p, selected, dimension, 1, x)
    T.copy(p, probabilities[row, head, :])


@T.macro
def softmax_head_tile(
    scores,
    sink,
    probabilities,
    row,
    head,
    x,
    mask,
    maximum,
    denominator,
    sink_value,
    p,
    expanded,
    selected,
    dimension,
    head_tile,
):
    T.copy(scores[row, head : head + head_tile, :], x)
    T.copy(sink[head : head + head_tile], sink_value[:, 0])
    softmax_values(
        x, mask, maximum, denominator, sink_value, p, selected, dimension, head_tile, expanded
    )
    T.copy(p, probabilities[row, head : head + head_tile, :])


def softmax(rows, heads, selected, dimension=512):
    if not 0 < heads <= 64 or selected % 64:
        raise ValueError("Cube softmax requires H<=64 and S%64=0")
    padded_heads = (heads + 15) // 16 * 16

    @T.prim_func
    def main(
        scores: T.Tensor((rows, padded_heads, selected), "float32"),
        indices: T.Tensor((rows, selected), "int32"),
        sink: T.Tensor((heads,), "float32"),
        probabilities: T.Tensor((rows, padded_heads, selected), "bfloat16"),
    ):
        with T.Kernel(T.ceildiv(rows * heads, 2), is_npu=True) as (cid, vid):
            x = T.alloc_ub((1, selected), "float32")
            ids = T.alloc_ub((1, selected), "int32")
            valid = T.alloc_ub((1, selected), "float32")
            mask = T.alloc_ub((selected // 8,), "uint8")
            maximum = T.alloc_ub((1, 1), "float32")
            denominator = T.alloc_ub((1, 1), "float32")
            sink_value = T.alloc_ub((1, 1), "float32")
            p = T.alloc_ub((1, selected), "bfloat16")
            index = cid * 2 + vid
            row, head = index // heads, index % heads
            if row < rows:
                softmax_row(
                    scores,
                    indices,
                    sink,
                    probabilities,
                    row,
                    head,
                    x,
                    ids,
                    valid,
                    mask,
                    maximum,
                    denominator,
                    sink_value,
                    p,
                    selected,
                    dimension,
                )

    return main


def pv(rows, heads, selected, dimension=512, tile_d=32, head_tile=16):
    if not 0 < heads <= 64 or selected % 128 or dimension % tile_d:
        raise ValueError("Cube PV requires H<=64, S%128=0 and D%tile_d=0")
    blocks = dimension // tile_d
    padded_heads, block_heads, groups = _head_geometry(heads, head_tile)
    vector_heads = block_heads // 2

    @T.prim_func
    def main(
        probabilities: T.Tensor((rows, padded_heads, selected), "bfloat16"),
        kv: T.Tensor((rows, selected, dimension), "bfloat16"),
        output: T.Tensor((rows, heads, dimension), "bfloat16"),
        workspace: T.Tensor((rows, padded_heads, dimension), "float32"),
    ):
        with T.Kernel(rows * groups * blocks, is_npu=True) as (cid, vid):
            row, tile = cid // (groups * blocks), cid % blocks
            first = cid // blocks % groups * block_heads
            with T.Scope("C"):
                a = T.alloc_L1((block_heads, 128), "bfloat16")
                b = T.alloc_L1((128, tile_d), "bfloat16")
                c = T.alloc_L0C((block_heads, tile_d), "float32")
                for k in T.serial(selected // 128):
                    T.copy(
                        probabilities[row, first : first + block_heads, k * 128 : (k + 1) * 128], a
                    )
                    T.copy(kv[row, k * 128 : (k + 1) * 128, tile * tile_d : (tile + 1) * tile_d], b)
                    T.gemm_v0(a, b, c, init=(k == 0))
                T.copy(
                    c,
                    workspace[
                        row, first : first + block_heads, tile * tile_d : (tile + 1) * tile_d
                    ],
                )
                T.set_cross_flag("FIX", 0)
                T.wait_cross_flag(1)
            with T.Scope("V"):
                x = T.alloc_ub((vector_heads, tile_d), "float32")
                y = T.alloc_ub((vector_heads, tile_d), "bfloat16")
                start = first + vid * vector_heads
                T.wait_cross_flag(0)
                if start < heads:
                    valid = T.min(vector_heads, heads - start)
                    T.copy(
                        workspace[
                            row, start : start + vector_heads, tile * tile_d : (tile + 1) * tile_d
                        ],
                        x,
                    )
                    T.tile.cast(y, x, "CAST_RINT", vector_heads * tile_d)
                    T.copy(
                        y[:valid, :],
                        output[row, start : start + valid, tile * tile_d : (tile + 1) * tile_d],
                    )
                T.set_cross_flag("MTE3", 1)

    return main.with_attr("orinfer.coupled_resources", True).with_attr(
        "orinfer.explicit_cross_sync", True
    )
