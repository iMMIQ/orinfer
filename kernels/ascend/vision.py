"""Shared BF16 vision boundaries and bounded BF16 QK / FP16 PV attention."""

import math

import tilelang.language as T


def qkv_rope(rows, capacity, heads=16, dimension=64, padding_multiple=128):
    if dimension != 64 or capacity % padding_multiple or padding_multiple not in (128, 256, 512):
        raise ValueError("Vision QKV requires D64 and aligned physical padding")
    hidden = heads * dimension
    half = hidden // 2
    padded = T.ceildiv(rows, padding_multiple) * padding_multiple

    @T.prim_func
    def main(
        source: T.Tensor((rows, 3 * hidden), "float32"),
        bias: T.Tensor((3 * hidden,), "bfloat16"),
        phase: T.Tensor((rows, dimension), "float32"),
        qkv: T.Tensor((2, capacity, hidden), "bfloat16"),
        values: T.Tensor((capacity, hidden), "float16"),
    ):
        with T.Kernel(T.ceildiv(padded, 2), is_npu=True) as (cid, vid):
            raw = T.alloc_ub((hidden,), "float32")
            weight = T.alloc_ub((hidden,), "float32")
            rounded_index = T.alloc_ub((hidden,), "float32")
            cosine_base = T.alloc_ub((32,), "float32")
            sine_base = T.alloc_ub((32,), "float32")
            cosine = T.alloc_ub((half,), "float32")
            sine = T.alloc_ub((half,), "float32")
            left = T.alloc_ub((half,), "float32")
            right = T.alloc_ub((half,), "float32")
            temporary = T.alloc_ub((half,), "float32")
            rotated = T.alloc_ub((hidden,), "float32")
            left_indices = T.alloc_ub((half,), "uint32")
            left_indices_i = T.alloc_ub((half,), "int32")
            right_indices = T.alloc_ub((half,), "uint32")
            right_indices_i = T.alloc_ub((half,), "int32")
            phase_indices = T.alloc_ub((half,), "uint32")
            phase_indices_i = T.alloc_ub((half,), "int32")
            output_indices = T.alloc_ub((hidden,), "uint32")
            output_indices_i = T.alloc_ub((hidden,), "int32")
            rounded = T.alloc_ub((hidden,), "bfloat16")
            value_half = T.alloc_ub((hidden,), "float16")
            row = cid * 2 + vid
            if row < padded:
                if row < rows:
                    # Construct gather byte offsets with vector arithmetic. Per-element
                    # scalar index stores cost more than the rotary arithmetic itself.
                    T.tile.createvecindex(raw, 0.0)
                    T.tile.mul(weight, raw, 1.0 / 32.0)
                    T.tile.cast(output_indices_i, weight, "CAST_FLOOR", hidden)
                    T.tile.cast(weight, output_indices_i, "CAST_NONE", hidden)
                    T.tile.mul(rotated, raw, 1.0 / 64.0)
                    T.tile.cast(output_indices_i, rotated, "CAST_FLOOR", hidden)
                    T.tile.cast(rotated, output_indices_i, "CAST_NONE", hidden)
                    T.tile.mul(raw, weight, -32.0)
                    T.tile.createvecindex(rounded_index, 0.0)
                    T.tile.add(raw, raw, rounded_index)
                    T.tile.mul(rounded_index, rotated, 32.0)
                    T.tile.add(raw, raw, rounded_index)
                    T.tile.mul(rotated, rotated, -2.0)
                    T.tile.add(rotated, rotated, weight)
                    T.tile.mul(rotated, rotated, float(half))
                    T.tile.add(raw, raw, rotated)
                    T.tile.mul(raw, raw, 4.0)
                    T.tile.cast(output_indices_i, raw, "CAST_RINT", hidden)
                    T.tile.createvecindex(left, 0.0)
                    T.tile.mul(temporary, left, 1.0 / 32.0)
                    T.tile.cast(left_indices_i, temporary, "CAST_FLOOR", half)
                    T.tile.cast(temporary, left_indices_i, "CAST_NONE", half)
                    T.tile.mul(temporary, temporary, 32.0)
                    T.tile.add(right, left, temporary)
                    T.tile.mul(right, right, 4.0)
                    T.tile.cast(left_indices_i, right, "CAST_RINT", half)
                    T.tile.add(right, right, 128.0)
                    T.tile.cast(right_indices_i, right, "CAST_RINT", half)
                    T.tile.sub(left, left, temporary)
                    T.tile.mul(left, left, 4.0)
                    T.tile.cast(phase_indices_i, left, "CAST_RINT", half)
                    T.reinterpretcast(left_indices, left_indices_i, "uint32_t")
                    T.reinterpretcast(right_indices, right_indices_i, "uint32_t")
                    T.reinterpretcast(phase_indices, phase_indices_i, "uint32_t")
                    T.reinterpretcast(output_indices, output_indices_i, "uint32_t")
                    T.copy(phase[row, :32], cosine_base)
                    T.copy(phase[row, 32:], sine_base)
                    T.tile.gather(cosine, cosine_base, phase_indices, 0)
                    T.tile.gather(sine, sine_base, phase_indices, 0)
                    for part in T.serial(3):
                        column = part * hidden
                        T.copy(source[row, column : column + hidden], raw)
                        T.copy(bias[column : column + hidden], rounded)
                        T.tile.cast(weight, rounded, "CAST_NONE", hidden)
                        T.tile.add(raw, raw, weight)
                        T.tile.cast(rounded, raw, "CAST_RINT", hidden)
                        if part < 2:
                            T.tile.cast(raw, rounded, "CAST_NONE", hidden)
                            T.tile.gather(left, raw, left_indices, 0)
                            T.tile.gather(right, raw, right_indices, 0)
                            T.tile.mul(temporary, right, sine)
                            T.tile.mul(left, left, cosine)
                            T.tile.sub(rotated[:half], left, temporary)
                            T.tile.gather(left, raw, left_indices, 0)
                            T.tile.mul(temporary, left, sine)
                            T.tile.mul(right, right, cosine)
                            T.tile.add(rotated[half:], right, temporary)
                            T.tile.gather(raw, rotated, output_indices, 0)
                            T.tile.cast(rounded, raw, "CAST_RINT", hidden)
                            T.copy(rounded, qkv[part, row, :])
                        else:
                            T.tile.cast(raw, rounded, "CAST_NONE", hidden)
                            T.tile.cast(value_half, raw, "CAST_RINT", hidden)
                            T.copy(value_half, values[row, :])
                else:
                    T.tile.fill(rounded, 0.0)
                    for part in T.serial(2):
                        T.copy(rounded, qkv[part, row, :])
                    T.tile.fill(value_half, 0.0)
                    T.copy(value_half, values[row, :])

    return main


def merge(rows, capacity, width=1024, ratio=3, channels=128):
    if width % channels or channels % 32 or ratio != 3:
        raise ValueError("Vision spatial merge requires aligned channels and a 3x3 window")
    block = channels * ratio * ratio
    blocks = width // channels

    @T.prim_func
    def main(
        source: T.Tensor((capacity, width), "bfloat16"),
        grid: T.Tensor((8,), "int32"),
        output: T.Tensor((rows, width * ratio * ratio), "bfloat16"),
    ):
        with T.Kernel(T.ceildiv(rows * blocks, 2), is_npu=True) as (cid, vid):
            raw = T.alloc_ub((block,), "bfloat16")
            value = T.alloc_ub((block,), "float32")
            transposed = T.alloc_ub((block,), "float32")
            index = T.alloc_ub((block,), "uint32")
            packed = T.alloc_ub((block,), "bfloat16")
            tile = cid * 2 + vid
            row, first = tile // blocks, tile % blocks * channels
            if row < rows:
                columns = T.ceildiv(grid[1], ratio)
                top, left = row // columns * ratio, row % columns * ratio
                T.tile.fill(raw, 0.0)
                for part in T.serial(ratio * ratio):
                    y, x = top + part // ratio, left + part % ratio
                    if y < grid[0] and x < grid[1]:
                        T.copy(
                            source[y * grid[1] + x, first : first + channels],
                            raw[part * channels : (part + 1) * channels],
                        )
                T.tile.cast(value, raw, "CAST_NONE", block)
                for lane in T.Parallel(block):
                    index[lane] = (lane % (ratio * ratio) * channels + lane // (ratio * ratio)) * 4
                T.tile.gather(transposed, value, index, 0)
                T.tile.cast(packed, transposed, "CAST_RINT", block)
                T.copy(
                    packed, output[row, first * ratio * ratio : (first + channels) * ratio * ratio]
                )

    return main


def epilogue(rows, width, activation="linear", biased=True, residual=False, tile=256):
    if activation not in ("linear", "gelu", "silu_mul") or width % 32:
        raise ValueError("Invalid vision epilogue geometry")
    if activation == "silu_mul" and (biased or residual):
        raise ValueError("The vision gated FFN has no input bias or residual")
    block = math.gcd(width, tile)
    columns = width * (2 if activation == "silu_mul" else 1)
    blocks = width // block

    @T.prim_func
    def main(
        source: T.Tensor((rows, columns), "float32"),
        bias: T.Tensor((columns,), "bfloat16"),
        skip: T.Tensor((rows, width), "bfloat16"),
        output: T.Tensor((rows, width), "bfloat16"),
    ):
        with T.Kernel(T.ceildiv(rows * blocks, 2), is_npu=True) as (cid, vid):
            value = T.alloc_ub((block,), "float32")
            a = T.alloc_ub((block,), "float32")
            b = T.alloc_ub((block,), "float32")
            c = T.alloc_ub((block,), "float32")
            d = T.alloc_ub((block,), "float32")
            rounded = T.alloc_ub((block,), "bfloat16")
            mask = T.alloc_ub((((block + 255) // 256) * 32,), "uint8")
            index = cid * 2 + vid
            row, column = index // blocks, index % blocks * block
            if row < rows:
                T.copy(source[row, column : column + block], value)
                if biased:
                    T.copy(bias[column : column + block], rounded)
                    T.tile.cast(a, rounded, "CAST_NONE", block)
                    T.tile.add(value, value, a)
                # nn.Linear returns BF16 before either activation or residual.
                T.tile.cast(rounded, value, "CAST_RINT", block)
                T.tile.cast(value, rounded, "CAST_NONE", block)
                if activation == "silu_mul":
                    T.tile.mul(a, value, -1.0)
                    T.tile.exp(a, a)
                    T.tile.add(a, a, 1.0)
                    T.tile.div(value, value, a)
                    # The reference SiLU and multiplication have separate BF16 boundaries.
                    T.tile.cast(rounded, value, "CAST_RINT", block)
                    T.tile.cast(value, rounded, "CAST_NONE", block)
                    T.copy(source[row, width + column : width + column + block], a)
                    T.tile.cast(rounded, a, "CAST_RINT", block)
                    T.tile.cast(a, rounded, "CAST_NONE", block)
                    T.tile.mul(value, value, a)
                elif activation == "gelu":
                    # A&S 7.1.26 erf approximation, maximum absolute error 1.5e-7.
                    # Keep Linear's BF16 rounding before evaluating GELU in FP32.
                    T.tile.mul(a, value, 0.7071067811865476)
                    T.tile.abs(b, a)
                    T.tile.mul(c, b, 0.3275911)
                    T.tile.add(c, c, 1.0)
                    T.tile.fill(d, 1.0)
                    T.tile.div(c, d, c)
                    T.tile.mul(b, c, 1.061405429)
                    T.tile.add(b, b, -1.453152027)
                    T.tile.mul(b, b, c)
                    T.tile.add(b, b, 1.421413741)
                    T.tile.mul(b, b, c)
                    T.tile.add(b, b, -0.284496736)
                    T.tile.mul(b, b, c)
                    T.tile.add(b, b, 0.254829592)
                    T.tile.mul(b, b, c)
                    T.tile.mul(c, a, a)
                    T.tile.mul(c, c, -1.0)
                    T.tile.exp(c, c)
                    T.tile.mul(b, b, c)
                    T.tile.sub(b, d, b)
                    T.tile.mul(c, b, -1.0)
                    T.tile.compare(mask, a, 0.0, "LT")
                    T.tile.select(b, mask, c, b, "VSEL_TENSOR_TENSOR_MODE")
                    T.tile.add(b, b, 1.0)
                    T.tile.mul(value, value, b)
                    T.tile.mul(value, value, 0.5)
                if residual:
                    T.copy(skip[row, column : column + block], rounded)
                    T.tile.cast(a, rounded, "CAST_NONE", block)
                    T.tile.add(value, value, a)
                T.tile.cast(rounded, value, "CAST_RINT", block)
                T.copy(rounded, output[row, column : column + block])

    return main


def attention(
    rows,
    capacity,
    heads=16,
    dimension=64,
    tile_m=128,
    tile_k=256,
    workspace_slots=64,
):
    """Bidirectional per-image online softmax; scratch grows linearly with patches."""
    if dimension != 64 or tile_m not in (16, 32, 64, 128, 256) or tile_k not in (64, 128, 256, 512):
        raise ValueError("Vision attention requires D64 and bounded M/K tiles")
    if capacity % tile_k or not 1 <= workspace_slots <= 256:
        raise ValueError("Vision QKV capacity and persistent workspace slots must be bounded")
    queries = T.ceildiv(rows, tile_m)
    blocks = queries * heads
    half = tile_m // 2
    if tile_m * tile_k * 4 > 131072:
        raise ValueError("Vision score accumulator exceeds A2 L0C")
    qk_l0 = dimension
    # The shared non-transposed GEMM keeps 32 KiB B ping-pong slots even
    # when the whole K extent fits; retain its no-N-split limit as well.
    pv_l0 = min(tile_k, 32768 // tile_m, 16384 // dimension)

    # Scores are one key tile ahead of PV. P and product each retain one slot;
    # their completion handshake retires that slot before its next writer.
    buffers = workspace_slots * 2

    @T.macro
    def qk(qkv, query, key, dots, scores, head, part, destination):
        begin = part * tile_k
        T.copy(qkv[1, begin : begin + tile_k, head * dimension : (head + 1) * dimension], key)
        T.gemm_v0(query, key, dots, transpose_B=True, init=True, kL0Size=qk_l0)
        T.copy(dots, scores[destination, :, :])

    @T.macro
    def pv(values, prob, value, product, probabilities, products, head, part, source, destination):
        begin = part * tile_k
        T.copy(probabilities[source, :, :], prob)
        T.copy(values[begin : begin + tile_k, head * dimension : (head + 1) * dimension], value)
        T.gemm_v0(prob, value, product, init=True, kL0Size=pv_l0)
        T.copy(product, products[destination, :, :])

    @T.prim_func
    def main(
        qkv: T.Tensor((2, capacity, heads * dimension), "bfloat16"),
        values: T.Tensor((capacity, heads * dimension), "float16"),
        output: T.Tensor((rows, heads * dimension), "bfloat16"),
        scores: T.Tensor((buffers, tile_m, tile_k), "float32"),
        probabilities: T.Tensor((workspace_slots, tile_m, tile_k), "float16"),
        products: T.Tensor((workspace_slots, tile_m, dimension), "float32"),
    ):
        # A fixed physical grid gives every CTA one private workspace slot.
        # Keep the dynamic logical trip count inside the kernel. Re-wrapping a
        # min(dynamic_grid, slots) launch makes the pinned compiler recursively
        # expand interval bounds in vector row slices.
        with T.Kernel(workspace_slots, is_npu=True) as (physical, vid):
            for iteration in T.serial(T.ceildiv(blocks, workspace_slots)):
                cid = physical + iteration * workspace_slots
                if cid < blocks:
                    slot = cid % workspace_slots
                    head = cid % heads
                    first = cid // heads * tile_m
                    valid = T.min(tile_m, rows - first)
                    with T.Scope("C"):
                        query = T.alloc_L1((tile_m, dimension), "bfloat16")
                        key = T.alloc_L1((tile_k, dimension), "bfloat16")
                        value = T.alloc_L1((tile_k, dimension), "float16")
                        prob = T.alloc_L1((tile_m, tile_k), "float16")
                        dots = T.alloc_L0C((tile_m, tile_k), "float32")
                        # FIX completes each score copy before PV reuses L0C.
                        # An explicit alias keeps this lifetime independent of
                        # the pinned compiler's recurrent-buffer reuse planner.
                        product = T.decl_buffer(
                            (tile_m, dimension), "float32", data=dots.data, scope="wmma.accumulator"
                        )
                        T.copy(
                            qkv[
                                0, first : first + valid, head * dimension : (head + 1) * dimension
                            ],
                            query[:valid, :],
                        )
                        qk(qkv, query, key, dots, scores, head, 0, slot * 2)
                        T.set_cross_flag("FIX", 0)
                        for pair in T.serial(T.ceildiv(rows, tile_k * 2)):
                            for phase in T.unroll(2):
                                part = pair * 2 + phase
                                if part < T.ceildiv(rows, tile_k):
                                    if part + 1 < T.ceildiv(rows, tile_k):
                                        qk(
                                            qkv,
                                            query,
                                            key,
                                            dots,
                                            scores,
                                            head,
                                            part + 1,
                                            slot * 2 + (1 - phase),
                                        )
                                        T.set_cross_flag("FIX", (1 - phase) * 4)
                                    T.wait_cross_flag(1 + phase * 4)
                                    pv(
                                        values,
                                        prob,
                                        value,
                                        product,
                                        probabilities,
                                        products,
                                        head,
                                        part,
                                        slot,
                                        slot,
                                    )
                                    T.set_cross_flag("FIX", 2)
                                    T.wait_cross_flag(3)
                    with T.Scope("V"):
                        score = T.alloc_ub((half, tile_k), "float32")
                        maximum = T.alloc_ub((half, 1), "float32")
                        next_maximum = T.alloc_ub((half, 1), "float32")
                        denominator = T.alloc_ub((half, 1), "float32")
                        total = T.alloc_ub((half, 1), "float32")
                        alpha = T.alloc_ub((half, 1), "float32")
                        carried = T.alloc_ub((half, dimension), "float32")
                        product_v = T.alloc_ub((half, dimension), "float32")
                        expanded = T.alloc_ub((half, 8), "float32")
                        prob_b = T.alloc_ub((half, tile_k), "float16")
                        result_b = T.alloc_ub((half, dimension), "bfloat16")
                        lanes = T.alloc_ub((tile_k,), "float32")
                        mask = T.alloc_ub((((tile_k + 255) // 256) * 32,), "uint8")
                        T.tile.fill(maximum, -1e30)
                        T.tile.fill(denominator, 0.0)
                        T.tile.fill(carried, 0.0)
                        T.tile.createvecindex(lanes, 0.0)
                        for pair in T.serial(T.ceildiv(rows, tile_k * 2)):
                            for phase in T.unroll(2):
                                part = pair * 2 + phase
                                if part < T.ceildiv(rows, tile_k):
                                    group = phase * 4
                                    scratch = slot * 2 + phase
                                    T.wait_cross_flag(group)
                                    T.copy(scores[scratch, vid * half : (vid + 1) * half, :], score)
                                    T.tile.mul(score, score, dimension**-0.5)
                                    T.tile.compare(
                                        mask, lanes, T.cast(rows - part * tile_k, "float32"), "LT"
                                    )
                                    if rows - part * tile_k < tile_k:
                                        for row in T.serial(half):
                                            T.tile.select(
                                                score[row, :],
                                                mask,
                                                score[row, :],
                                                -1e30,
                                                "VSEL_TENSOR_SCALAR_MODE",
                                            )
                                    T.reduce_max(score, next_maximum, dim=1)
                                    T.tile.max(next_maximum, maximum, next_maximum)
                                    T.tile.sub(alpha, maximum, next_maximum)
                                    T.tile.exp(alpha, alpha)
                                    for column in T.unroll(tile_k // 64):
                                        T.tile.row_expand_sub_experiment(
                                            score[:, column * 64 : (column + 1) * 64],
                                            score[:, column * 64 : (column + 1) * 64],
                                            next_maximum,
                                            expanded,
                                        )
                                    T.tile.exp(score, score)
                                    T.reduce_sum(score, total, dim=1)
                                    # Normalize P before its FP16 boundary, then rebase
                                    # the already normalized prefix output by its mass.
                                    T.tile.mul(denominator, denominator, alpha)
                                    T.tile.add(total, denominator, total)
                                    T.tile.div(alpha, denominator, total)
                                    T.copy(total, denominator)
                                    for column in T.unroll(tile_k // 64):
                                        T.tile.row_expand_div_experiment(
                                            score[:, column * 64 : (column + 1) * 64],
                                            score[:, column * 64 : (column + 1) * 64],
                                            denominator,
                                            expanded,
                                        )
                                    T.tile.cast(prob_b, score, "CAST_RINT", half * tile_k)
                                    T.copy(
                                        prob_b,
                                        probabilities[slot, vid * half : (vid + 1) * half, :],
                                    )
                                    T.set_cross_flag("MTE3", 1 + group)
                                    T.wait_cross_flag(2)
                                    T.copy(
                                        products[slot, vid * half : (vid + 1) * half, :], product_v
                                    )
                                    T.set_cross_flag("MTE2", 3)
                                    T.tile.row_expand_mul_experiment(
                                        carried, carried, alpha, expanded
                                    )
                                    T.tile.add(carried, carried, product_v)
                                    T.copy(next_maximum, maximum)
                        T.tile.cast(result_b, carried, "CAST_RINT", half * dimension)
                        begin = first + vid * half
                        if begin < rows:
                            count = T.min(half, rows - begin)
                            T.copy(
                                result_b[:count, :],
                                output[
                                    begin : begin + count, head * dimension : (head + 1) * dimension
                                ],
                            )

    return (
        main.with_attr("orinfer.coupled_resources", True)
        .with_attr("orinfer.explicit_cross_sync", True)
        .with_attr("orinfer.linear_allocation", True)
    )
