"""Engram basis restoration and gate, with FP32 normalization on A2."""

import math

import tilelang.language as T

from tools.operators.ascend.vector_math import rsqrt_fp32


def rotate_blocks(rows, width, block=128):
    if width % block or block % 32:
        raise ValueError("Engram rotation requires H%block=0 and block%32=0")
    groups = width // block
    tiles = block // 32

    @T.prim_func
    def main(
        hidden: T.Tensor((rows, 4, width), "bfloat16"),
        rotation: T.Tensor((block, block), "float32"),
        out: T.Tensor((rows, 4, width), "float32"),
    ):
        with T.Kernel(T.ceildiv(rows * 4 * groups * tiles, 2), is_npu=True) as (cid, vid):
            xb = T.alloc_ub((1, block), "bfloat16")
            x = T.alloc_ub((1, block), "float32")
            w = T.alloc_ub((32, block), "float32")
            product = T.alloc_ub((32, block), "float32")
            total = T.alloc_ub((32, 1), "float32")
            index = cid * 2 + vid
            tile = index % tiles
            group = index // tiles % groups
            path = index // (tiles * groups) % 4
            row = index // (tiles * groups * 4)
            if row < rows:
                T.copy(hidden[row, path, group * block : (group + 1) * block], xb)
                T.copy(rotation[tile * 32 : (tile + 1) * 32, :], w)
                T.tile.cast(x, xb, "CAST_NONE", block)
                T.tile.broadcast(product, x)
                T.tile.mul(product, product, w)
                T.reduce_sum(product, total, dim=1)
                T.copy(
                    total,
                    out[row, path, group * block + tile * 32 : group * block + (tile + 1) * 32],
                )

    return main


def hadamard32(rows, width, tile_groups=64):
    """Signed Walsh block32 rotation, validated against the bank offline.

    Multiply BF16 inputs by the bank's original FP32 magnitude before the
    butterfly. Output signs come from the first column of that same bank.
    """
    count = tile_groups * 32
    if tile_groups not in (16, 32, 64) or (4 * width) % (tile_groups * 32):
        raise ValueError("Hadamard32 requires complete aligned group tiles")
    tiles = 4 * width // count

    @T.prim_func
    def main(
        hidden: T.Tensor((rows * 4 * width,), "bfloat16"),
        rotation: T.Tensor((1024,), "float32"),
        out: T.Tensor((rows * 4 * width,), "float32"),
    ):
        with T.Kernel(T.ceildiv(rows * tiles, 2), is_npu=True) as (cid, vid):
            xb = T.alloc_ub((count,), "bfloat16")
            p = T.alloc_ub((count,), "float32")
            even = T.alloc_ub((count // 2,), "float32")
            odd = T.alloc_ub((count // 2,), "float32")
            left = T.alloc_ub((count // 2,), "float32")
            right = T.alloc_ub((count // 2,), "float32")
            w = T.alloc_ub((1024,), "float32")
            lane = T.alloc_ub((count,), "float32")
            group = T.alloc_ub((count,), "float32")
            floor_i = T.alloc_ub((count,), "int32")
            offsets = T.alloc_ub((count,), "uint32")
            sign_offsets = T.alloc_ub((count,), "uint32")
            signs = T.alloc_ub((count,), "float32")
            result = T.alloc_ub((count,), "float32")
            tile = cid * 2 + vid
            if tile < rows * tiles:
                T.copy(hidden[tile * count : (tile + 1) * count], xb)
                T.copy(rotation, w)
                T.tile.cast(p, xb, "CAST_NONE", count)
                T.tile.mul(p, p, w[0])
                for stage in T.serial(5):
                    T.tile.gather_mask(even, p, "P0101")
                    T.tile.gather_mask(odd, p, "P1010")
                    T.tile.add(left, even, odd)
                    T.tile.sub(right, even, odd)
                    T.copy(left, p[: count // 2])
                    T.copy(right, p[count // 2 :])
                T.tile.createvecindex(lane, 0.0)
                T.tile.mul(group, lane, 0.03125)
                T.tile.cast(floor_i, group, "CAST_FLOOR", count)
                T.tile.cast(group, floor_i, "CAST_NONE", count)
                T.tile.mul(result, group, 32.0)
                T.tile.sub(lane, lane, result)
                # group-major output from Walsh-major butterfly storage.
                T.tile.mul(result, lane, float(tile_groups))
                T.tile.add(result, result, group)
                T.tile.mul(result, result, 4.0)
                T.tile.cast(floor_i, result, "CAST_RINT", count)
                T.reinterpretcast(offsets, floor_i, "uint32_t")
                T.tile.gather(result, p, offsets, 0)
                T.tile.mul(lane, lane, 128.0)
                T.tile.cast(floor_i, lane, "CAST_RINT", count)
                T.reinterpretcast(sign_offsets, floor_i, "uint32_t")
                T.tile.gather(signs, w, sign_offsets, 0)
                T.tile.div(signs, signs, w[0])
                T.tile.mul(result, result, signs)
                T.copy(result, out[tile * count : (tile + 1) * count])
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
