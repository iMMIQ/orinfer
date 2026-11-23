"""Restore routed expert rows and combine them with the shared expert on A2."""

import math

import tilelang.language as T


def combine(rows, expert_rows, width, selected=6, output_dtype="bfloat16", tile=256):
    """Restore a packet; dynamic expert rows retain the producer row expression."""
    if isinstance(expert_rows, int) and expert_rows <= 0:
        raise ValueError("Expert row capacity must be positive")
    if tile not in (256, 512, 1024, 1280, 2560, 5120):
        raise ValueError("Unsupported expert combine column tile")
    block = math.gcd(width, tile)
    blocks = width // block
    stride = (selected + 7) // 8 * 8

    @T.prim_func
    def main(
        experts: T.Tensor((expert_rows, width), "bfloat16"),
        shared: T.Tensor((rows, width), "bfloat16"),
        row_map: T.Tensor((rows, stride), "int32"),
        weights: T.Tensor((rows, stride), "float32"),
        out: T.Tensor((rows, width), output_dtype),
        errors: T.Tensor((rows,), "int32"),
    ):
        with T.Kernel(20, is_npu=True) as (cid, vid):
            xb = T.alloc_ub((block,), "bfloat16")
            sb = T.alloc_ub((block,), "bfloat16")
            x = T.alloc_ub((block,), "float32")
            total = T.alloc_ub((block,), "float32")
            s = T.alloc_ub((block,), "float32")
            ids = T.alloc_ub((stride,), "int32")
            w = T.alloc_ub((stride,), "float32")
            flag = T.alloc_ub((8,), "int32")
            y = T.alloc_ub((block,), output_dtype)
            aiv = cid * 2 + vid
            items = rows * blocks
            if aiv < items:
                for step in T.serial(T.ceildiv(items - aiv, 40)):
                    flat = aiv + step * 40
                    row = flat // blocks
                    column = (flat % blocks) * block
                    T.copy(shared[row, column : column + block], sb)
                    T.copy(row_map[row, :], ids)
                    T.copy(weights[row, :], w)
                    flag[0] = 0
                    flag[1] = 0
                    for route in T.serial(selected):
                        index = ids[route]
                        if index >= 0 and index < expert_rows:
                            T.copy(experts[index, column : column + block], xb)
                            T.tile.cast(x, xb, "CAST_NONE", block)
                            T.tile.mul(x, x, w[route])
                            if flag[1] == 0:
                                T.copy(x, total)
                                flag[1] = 1
                            else:
                                T.tile.add(total, total, x)
                        else:
                            if index < -1 or index >= expert_rows:
                                flag[0] = 1
                    if flag[1] == 0 and output_dtype == "bfloat16":
                        T.copy(sb, out[row, column : column + block])
                    else:
                        T.tile.cast(s, sb, "CAST_NONE", block)
                        if flag[1] == 0:
                            T.copy(s, total)
                        else:
                            T.tile.add(total, total, s)
                        if output_dtype == "float32":
                            T.copy(total, y)
                        else:
                            T.tile.cast(y, total, "CAST_RINT", block)
                        T.copy(y, out[row, column : column + block])
                    if column == 0:
                        T.copy(flag[:1], errors[row : row + 1])
                    T.set_flag("MTE3", "V", 7)
                    T.wait_flag("MTE3", "V", 7)

    return main
