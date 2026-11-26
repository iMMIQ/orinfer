"""Engram basis restoration and gate, with FP32 normalization on A2."""

import math

import tilelang.language as T

from kernels.ascend.vector_math import rsqrt_fp32


def cast_fp32(rows, width, tile=2048):
    """Persistent BF16->FP32 cast for unrotated Engram basis restoration."""
    if (4 * width) % tile or tile % 32:
        raise ValueError("Engram cast requires aligned tiles dividing 4 * width")
    tiles_per_row = (4 * width) // tile

    @T.prim_func
    def main(
        hidden: T.Tensor((rows * 4 * width,), "bfloat16"),
        out: T.Tensor((rows * 4 * width,), "float32"),
    ):
        with T.Kernel(20, is_npu=True) as (cid, vid):
            xb = T.alloc_ub((tile,), "bfloat16")
            xf = T.alloc_ub((tile,), "float32")
            aiv = cid * 2 + vid
            total = rows * tiles_per_row
            if aiv < total:
                for step in T.serial(T.ceildiv(total - aiv, 40)):
                    index = aiv + step * 40
                    start = index * tile
                    T.copy(hidden[start : start + tile], xb)
                    T.tile.cast(xf, xb, "CAST_NONE", tile)
                    T.copy(xf, out[start : start + tile])
                    T.set_flag("MTE3", "V", 7)
                    T.wait_flag("MTE3", "V", 7)

    return main


def gate(rows, width, eps=1e-20):
    block = math.gcd(width, 256)
    blocks = width // block

    @T.prim_func
    def main(
        hidden: T.Tensor((rows, 4, width), "bfloat16"),
        original: T.Tensor((rows, 4, width), "float32"),
        key: T.Tensor((rows, 4, width), "bfloat16"),
        value: T.Tensor((rows, width), "bfloat16"),
        channel_weight: T.Tensor((4, width), "float32"),
        token_mask: T.Tensor((rows,), "int8"),
        out: T.Tensor((rows, 4, width), "bfloat16"),
    ):
        with T.Kernel(T.ceildiv(rows * 4, 2), is_npu=True) as (cid, vid):
            o = T.alloc_ub((1, block), "float32")
            kb = T.alloc_ub((1, block), "bfloat16")
            k = T.alloc_ub((1, block), "float32")
            weight = T.alloc_ub((1, block), "float32")
            product = T.alloc_ub((1, block), "float32")
            partial = T.alloc_ub((1, 1), "float32")
            dot = T.alloc_ub((1, 1), "float32")
            o_norm = T.alloc_ub((1, 1), "float32")
            k_norm = T.alloc_ub((1, 1), "float32")
            o_inverse = T.alloc_ub((1, 1), "float32")
            k_inverse = T.alloc_ub((1, 1), "float32")
            correction = T.alloc_ub((1, 1), "float32")
            magnitude = T.alloc_ub((1, 1), "float32")
            gate_value = T.alloc_ub((1, 1), "float32")
            hb = T.alloc_ub((1, block), "bfloat16")
            vb = T.alloc_ub((1, block), "bfloat16")
            h = T.alloc_ub((1, block), "float32")
            v = T.alloc_ub((1, block), "float32")
            y = T.alloc_ub((1, block), "bfloat16")
            index = cid * 2 + vid
            row = index // 4
            path = index % 4
            if row < rows:
                T.tile.fill(dot, 0.0)
                T.tile.fill(o_norm, 0.0)
                T.tile.fill(k_norm, 0.0)
                for chunk in T.serial(blocks):
                    start = chunk * block
                    T.copy(original[row, path, start : start + block], o)
                    T.copy(key[row, path, start : start + block], kb)
                    T.copy(channel_weight[path, start : start + block], weight)
                    T.tile.cast(k, kb, "CAST_NONE", block)
                    T.tile.mul(product, o, weight)
                    T.tile.mul(product, product, k)
                    T.reduce_sum(product, partial, dim=1)
                    T.tile.add(dot, dot, partial)
                    T.tile.mul(product, o, o)
                    T.reduce_sum(product, partial, dim=1)
                    T.tile.add(o_norm, o_norm, partial)
                    T.tile.mul(product, k, k)
                    T.reduce_sum(product, partial, dim=1)
                    T.tile.add(k_norm, k_norm, partial)
                T.tile.div(o_norm, o_norm, float(width))
                T.tile.div(k_norm, k_norm, float(width))
                T.tile.add(o_norm, o_norm, eps)
                T.tile.add(k_norm, k_norm, eps)
                rsqrt_fp32(o_norm, o_inverse, correction)
                rsqrt_fp32(k_norm, k_inverse, correction)
                T.tile.mul(dot, dot, o_inverse)
                T.tile.mul(dot, dot, k_inverse)
                T.tile.mul(dot, dot, width**-0.5)
                T.tile.abs(magnitude, dot)
                T.tile.max(magnitude, magnitude, 1e-6)
                rsqrt_fp32(magnitude, o_inverse, correction)
                T.tile.mul(magnitude, magnitude, o_inverse)
                if T.bitwise_and(T.reinterpret("uint32", dot[0, 0]), T.uint32(2147483648)) != 0:
                    T.tile.mul(magnitude, magnitude, -1.0)
                T.tile.sigmoid(gate_value, magnitude)
                if token_mask[row] == 0:
                    T.tile.fill(gate_value, 0.0)
                for chunk in T.serial(blocks):
                    start = chunk * block
                    T.copy(hidden[row, path, start : start + block], hb)
                    T.copy(value[row, start : start + block], vb)
                    T.tile.cast(h, hb, "CAST_NONE", block)
                    T.tile.cast(v, vb, "CAST_NONE", block)
                    T.tile.mul(v, v, gate_value[0, 0])
                    T.tile.add(h, h, v)
                    T.tile.cast(y, h, "CAST_RINT", block)
                    T.copy(y, out[row, path, start : start + block])

    return main
