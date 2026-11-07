"""Transport routed A8 rows without a contiguous activation unpack."""

import tilelang.language as T

from tools.operators.ascend.quantization import row_quantizer


def routed_layout(width, routes, ranks):
    if width < 32 or width % 32 or routes < 8 or routes % 8 or ranks < 1:
        raise ValueError("Routed transport requires aligned activation and route widths")
    words = width // 4
    pitch = ((words + 2 * routes + 1 + 7) // 8) * 8
    return words, pitch


def quantize_routed(rows, width, routes=8, ranks=8):
    words, pitch = routed_layout(width, routes, ranks)
    count = T.ceildiv(rows, ranks)
    quantize_row = row_quantizer(width)

    @T.prim_func
    def main(
        x: T.Tensor((count, width), "bfloat16"),
        indices: T.Tensor((count, routes), "int32"),
        weights: T.Tensor((count, routes), "float32"),
        out: T.Tensor((count, pitch), "int32"),
    ):
        with T.Kernel(T.ceildiv(count, 2), is_npu=True) as (cid, vid):
            packed = T.alloc_ub((pitch,), "int32")
            xb = T.alloc_ub((1, width), "bfloat16")
            y = T.alloc_ub((1, width), "int8")
            scale = T.alloc_ub((1, 1), "float32")
            half_scale = T.alloc_ub((1, 1), "float16")
            wb = T.alloc_ub((routes,), "float32")
            wi = T.alloc_ub((routes,), "int32")
            T.reinterpretcast(y, packed, "int8_t")
            T.reinterpretcast(wi, wb, "int32_t")
            row = cid * 2 + vid
            if row < count:
                T.tile.fill(packed, 0)
                T.pipe_barrier("ALL")
                T.copy(x[row : row + 1, :], xb)
                quantize_row(xb, y, scale, half_scale)
                T.pipe_barrier("ALL")
                T.copy(indices[row, :], packed[words : words + routes])
                T.copy(weights[row, :], wb)
                T.pipe_barrier("ALL")
                T.copy(wi, packed[words + routes : words + 2 * routes])
                packed[words + 2 * routes] = T.reinterpret("int32", scale[0, 0])
                T.pipe_barrier("ALL")
                T.copy(packed, out[row, :])
                T.pipe_barrier("ALL")

    return main.with_attr("orinfer.component_sources", ["tools/operators/ascend/quantization.py"])


def unpack_routed_metadata(rows, width, routes=8, ranks=8):
    words, pitch = routed_layout(width, routes, ranks)
    count = T.ceildiv(rows, ranks)

    @T.prim_func
    def main(
        source: T.Tensor((ranks * count, pitch), "int32"),
        scales: T.Tensor((rows,), "float32"),
        indices: T.Tensor((rows, routes), "int32"),
        weights: T.Tensor((rows, routes), "float32"),
    ):
        with T.Kernel(T.ceildiv(rows, 2), is_npu=True) as (cid, vid):
            ib = T.alloc_ub((routes,), "int32")
            wb = T.alloc_ub((routes,), "float32")
            wi = T.alloc_ub((routes,), "int32")
            sb = T.alloc_ub((8,), "float32")
            si = T.alloc_ub((8,), "int32")
            T.reinterpretcast(wb, wi, "float")
            T.reinterpretcast(sb, si, "float")
            row = cid * 2 + vid
            if row < rows:
                T.copy(source[row, words : words + routes], ib)
                T.copy(source[row, words + routes : words + 2 * routes], wi)
                si[0] = source[row, words + 2 * routes]
                T.pipe_barrier("ALL")
                T.copy(ib, indices[row, :])
                T.copy(wb, weights[row, :])
                T.copy(sb[:1], scales[row : row + 1])
                T.pipe_barrier("ALL")

    return main
