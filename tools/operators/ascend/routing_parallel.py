"""Stable expert regrouping with parallel bounded chunks and a small prefix scan."""

import tilelang.language as T


def clear(rows, experts=384, selected=6):
    lanes = (selected + 7) // 8 * 8

    @T.prim_func
    def main(
        indices: T.Tensor((rows, lanes), "int32"),
        gather: T.Tensor((rows * selected,), "int32"),
        restore: T.Tensor((rows, lanes), "int32"),
        errors: T.Tensor((rows + 1,), "int32"),
    ):
        with T.Kernel(T.ceildiv(rows, 2), is_npu=True) as (cid, vid):
            ids = T.alloc_ub((lanes,), "int32")
            minus = T.alloc_ub((lanes,), "int32")
            flag = T.alloc_ub((8,), "int32")
            row = cid * 2 + vid
            if row < rows:
                T.copy(indices[row, :], ids)
                T.tile.fill(minus, -1)
                T.copy(minus, restore[row, :])
                T.copy(minus[:selected], gather[row * selected : (row + 1) * selected])
                T.set_flag("MTE2", "S", 6)
                T.wait_flag("MTE2", "S", 6)
                flag[0] = 0
                for lane in T.serial(selected):
                    if ids[lane] < 0 or ids[lane] >= experts:
                        flag[0] = 1
                T.set_flag("S", "MTE3", 6)
                T.wait_flag("S", "MTE3", 6)
                T.copy(flag[:1], errors[row : row + 1])
                T.set_flag("MTE3", "V", 7)
                T.wait_flag("MTE3", "V", 7)

    return main


def local(rows, base=0, experts=48, selected=6, chunk=32):
    if selected != 6 or chunk != 32:
        raise ValueError("Parallel regroup currently uses 32-token chunks and six routes")
    chunks = T.ceildiv(rows, chunk)
    width = chunk * 8

    @T.prim_func
    def main(
        indices: T.Tensor((rows * 8,), "int32"),
        positions: T.Tensor((experts, chunks, width), "int32"),
        counts: T.Tensor((experts, chunks), "int32"),
    ):
        with T.Kernel(T.ceildiv(experts * chunks, 2), is_npu=True) as (cid, vid):
            ids = T.alloc_ub((width,), "int32")
            values = T.alloc_ub((width,), "float32")
            lanes = T.alloc_ub((width,), "float32")
            floor_i = T.alloc_ub((width,), "int32")
            floor_f = T.alloc_ub((width,), "float32")
            mask = T.alloc_ub((width // 8,), "uint8")
            score = T.alloc_ub((width,), "float32")
            pairs = T.alloc_ub((width * 2,), "float32")
            ordered = T.alloc_ub((width,), "int32")
            valid = T.alloc_ub((1, width), "float32")
            total = T.alloc_ub((1, 1), "float32")
            count = T.alloc_ub((8,), "int32")
            task = cid * 2 + vid
            expert, part = task // chunks, task % chunks
            if expert < experts:
                available = T.min(chunk, rows - part * chunk)
                T.tile.fill(ids, -1)
                T.copy(
                    indices[part * chunk * 8 : (part * chunk + available) * 8], ids[: available * 8]
                )
                T.tile.cast(values, ids, "CAST_NONE", width)
                T.tile.createvecindex(lanes, 0.0)
                T.tile.mul(score, lanes, -1.0)
                T.tile.compare(mask, values, T.cast(base + expert, "float32"), "EQ")
                T.tile.select(score, mask, score, -T.infinity("float32"), "VSEL_TENSOR_SCALAR_MODE")
                # Ignore padded route lanes even if their indices look valid.
                T.tile.mul(floor_f, lanes, 0.125)
                T.tile.cast(floor_i, floor_f, "CAST_FLOOR", width)
                T.tile.cast(floor_f, floor_i, "CAST_NONE", width)
                T.tile.mul(floor_f, floor_f, 8.0)
                T.tile.sub(values, lanes, floor_f)
                T.tile.compare(mask, values, float(selected), "LT")
                T.tile.select(score, mask, score, -T.infinity("float32"), "VSEL_TENSOR_SCALAR_MODE")
                # GM->UB copies zero-pad their physical destination. Explicitly
                # exclude the final partial token chunk so padded zero IDs do
                # not become real routes to expert zero on rank 0.
                T.tile.compare(mask, lanes, T.cast(available * 8, "float32"), "LT")
                T.tile.select(score, mask, score, -T.infinity("float32"), "VSEL_TENSOR_SCALAR_MODE")
                T.tile.topk(pairs, score, width, width)
                T.tile.gather_mask(values, pairs, "P0101")
                T.tile.compare(mask, values, -T.infinity("float32"), "GT")
                T.tile.fill(valid, 1.0)
                T.tile.select(valid, mask, valid, 0.0, "VSEL_TENSOR_SCALAR_MODE")
                T.reduce_sum(valid, total, dim=1)
                T.tile.cast(count[:1], total, "CAST_RINT", 1)
                T.copy(count[:1], counts[expert, part : part + 1])
                T.tile.mul(values, values, -1.0)
                T.tile.add(values, values, T.cast(part * chunk * 8, "float32"))
                T.tile.select(values, mask, values, -1.0, "VSEL_TENSOR_SCALAR_MODE")
                T.tile.cast(ordered, values, "CAST_RINT", width)
                T.copy(ordered, positions[expert, part, :])
                T.set_flag("MTE3", "V", 7)
                T.wait_flag("MTE3", "V", 7)

    return main


def prefix(rows, experts=48, selected=6, chunk=32, pad_last=False, scan_rows=0):
    if scan_rows:
        return _vector_prefix(rows, experts, selected, chunk, pad_last, scan_rows)
    chunks = T.ceildiv(rows, chunk)

    @T.prim_func
    def main(
        counts: T.Tensor((experts, chunks), "int32"),
        offsets: T.Tensor((experts, chunks), "int32"),
        groups: T.Tensor((experts,), "int64"),
        errors: T.Tensor((rows + 1,), "int32"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            errors[rows] = 0
            for expert in T.serial(experts):
                groups[expert] = 0
                for part in T.serial(chunks):
                    offsets[expert, part] = errors[rows]
                    errors[rows] = errors[rows] + counts[expert, part]
                    groups[expert] = groups[expert] + T.cast(counts[expert, part], "int64")
            if pad_last:
                groups[experts - 1] = groups[experts - 1] + rows * selected - errors[rows]
            for row in T.serial(1, rows):
                if errors[row] != 0:
                    errors[0] = 1

    return main


def _vector_prefix(rows, experts, selected, chunk, pad_last, scan_rows):
    """Exact bounded scan; each nonnegative routing count is at most six per token."""
    if not 1 <= scan_rows <= 4096 or experts != 48 or selected != 6 or chunk != 32:
        raise ValueError("Vector regroup scan requires E48/top6/chunk32 and at most 4096 rows")
    chunks = T.ceildiv(rows, chunk)
    width = ((experts * ((scan_rows + chunk - 1) // chunk) + 63) // 64) * 64
    steps = (width - 1).bit_length()

    @T.prim_func
    def main(
        counts: T.Tensor((experts * chunks,), "int32"),
        offsets: T.Tensor((experts * chunks,), "int32"),
        groups: T.Tensor((experts,), "int64"),
        errors: T.Tensor((rows + 1,), "int32"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            values = T.alloc_ub((1, width), "float32")
            shifted = T.alloc_ub((1, width), "float32")
            lanes = T.alloc_ub((1, width), "float32")
            work = T.alloc_ub((width,), "int32")
            addresses = T.alloc_ub((width,), "uint32")
            mask = T.alloc_ub((width // 8,), "uint8")
            group_values = T.alloc_ub((experts,), "int64")
            summary = T.alloc_ub((8,), "int32")
            total = T.alloc_ub((8,), "int32")
            maximum = T.alloc_ub((1, 1), "float32")
            if vid == 0:
                T.tile.fill(work, 0)
                T.copy(counts[: experts * chunks], work[: experts * chunks])
                T.tile.cast(values, work, "CAST_NONE", width)
                # All intermediate sums are integers <= 6*4096, exactly
                # representable in FP32. Zero padding cannot create routes.
                for step in T.serial(steps):
                    distance = T.shift_left(1, step)
                    T.tile.createvecindex(lanes, 0.0)
                    T.tile.add(lanes, lanes, -T.cast(distance, "float32"))
                    T.tile.max(lanes, lanes, 0.0)
                    T.tile.mul(lanes, lanes, 4.0)
                    T.tile.cast(work, lanes, "CAST_RINT", width)
                    T.reinterpretcast(addresses, work, "uint32_t")
                    T.tile.gather(shifted, values, addresses, 0)
                    T.tile.createvecindex(lanes, 0.0)
                    T.tile.compare(mask, lanes, T.cast(distance, "float32"), "GE")
                    T.tile.select(shifted, mask, shifted, 0.0, "VSEL_TENSOR_SCALAR_MODE")
                    T.tile.add(values, values, shifted)
                T.set_flag("V", "S", 6)
                T.wait_flag("V", "S", 6)
                total[0] = T.cast(values[0, experts * chunks - 1], "int32")
                for expert in T.serial(experts):
                    group_values[expert] = T.cast(values[0, (expert + 1) * chunks - 1], "int64")
                    if expert > 0:
                        group_values[expert] = group_values[expert] - T.cast(
                            values[0, expert * chunks - 1], "int64"
                        )
                if pad_last:
                    group_values[experts - 1] = (
                        group_values[experts - 1] + rows * selected - total[0]
                    )
                T.set_flag("S", "MTE3", 6)
                T.wait_flag("S", "MTE3", 6)
                T.copy(group_values, groups)
                T.copy(total[:1], errors[rows : rows + 1])
                T.copy(counts[: experts * chunks], work[: experts * chunks])
                T.tile.cast(lanes, work, "CAST_NONE", width)
                T.tile.sub(values, values, lanes)
                T.tile.cast(work, values, "CAST_RINT", width)
                T.copy(work[: experts * chunks], offsets[: experts * chunks])
                T.set_flag("MTE3", "V", 7)
                T.wait_flag("MTE3", "V", 7)
                T.tile.fill(work, 0)
                T.copy(errors[:rows], work[:rows])
                T.set_flag("MTE2", "S", 6)
                T.wait_flag("MTE2", "S", 6)
                summary[0] = work[0]
                work[0] = 0
                T.set_flag("S", "V", 6)
                T.wait_flag("S", "V", 6)
                T.tile.cast(lanes, work, "CAST_NONE", width)
                T.tile.abs(lanes, lanes)
                T.reduce_max(lanes, maximum, dim=1)
                T.set_flag("V", "S", 6)
                T.wait_flag("V", "S", 6)
                if maximum[0, 0] != 0.0:
                    summary[0] = 1
                T.set_flag("S", "MTE3", 6)
                T.wait_flag("S", "MTE3", 6)
                T.copy(summary[:1], errors[:1])

    return main


def pack(rows, experts=48, selected=6, chunk=32):
    chunks = T.ceildiv(rows, chunk)
    width = chunk * 8

    @T.prim_func
    def main(
        positions: T.Tensor((experts, chunks, width), "int32"),
        counts: T.Tensor((experts, chunks), "int32"),
        offsets: T.Tensor((experts, chunks), "int32"),
        gather: T.Tensor((rows * selected,), "int32"),
    ):
        with T.Kernel(T.ceildiv(experts * chunks, 2), is_npu=True) as (cid, vid):
            ids = T.alloc_ub((width,), "int32")
            original = T.alloc_ub((width,), "float32")
            row_f = T.alloc_ub((width,), "float32")
            row_i = T.alloc_ub((width,), "int32")
            task = cid * 2 + vid
            expert, part = task // chunks, task % chunks
            if expert < experts:
                count, start = counts[expert, part], offsets[expert, part]
                if count > 0:
                    T.copy(positions[expert, part, :], ids)
                    T.tile.cast(original, ids, "CAST_NONE", width)
                    T.tile.mul(row_f, original, 0.125)
                    T.tile.cast(row_i, row_f, "CAST_FLOOR", width)
                    T.copy(row_i[:count], gather[start : start + count])
                    T.set_flag("MTE3", "V", 7)
                    T.wait_flag("MTE3", "V", 7)

    return main


def restore(rows, base=0, experts=48, selected=6, chunk=32):
    chunks = T.ceildiv(rows, chunk)
    width = chunk * 8

    @T.prim_func
    def main(
        indices: T.Tensor((rows * 8,), "int32"),
        offsets: T.Tensor((experts, chunks), "int32"),
        restore: T.Tensor((rows, 8), "int32"),
    ):
        with T.Kernel(T.ceildiv(rows, 2), is_npu=True) as (cid, vid):
            ids = T.alloc_ub((width,), "int32")
            values = T.alloc_ub((width,), "float32")
            lanes = T.alloc_ub((width,), "float32")
            floor_i = T.alloc_ub((width,), "int32")
            lane_mod = T.alloc_ub((width,), "float32")
            valid = T.alloc_ub((1, width), "float32")
            matching = T.alloc_ub((1, width), "float32")
            total = T.alloc_ub((1, 1), "float32")
            mask = T.alloc_ub((width // 8,), "uint8")
            result = T.alloc_ub((8,), "int32")
            row = cid * 2 + vid
            if row < rows:
                part = row // chunk
                available = T.min(chunk, rows - part * chunk)
                T.tile.fill(ids, -1)
                T.copy(
                    indices[part * chunk * 8 : (part * chunk + available) * 8], ids[: available * 8]
                )
                T.tile.cast(values, ids, "CAST_NONE", width)
                T.tile.createvecindex(lanes, 0.0)
                T.tile.mul(lane_mod, lanes, 0.125)
                T.tile.cast(floor_i, lane_mod, "CAST_FLOOR", width)
                T.tile.cast(lane_mod, floor_i, "CAST_NONE", width)
                T.tile.mul(lane_mod, lane_mod, 8.0)
                T.tile.sub(lane_mod, lanes, lane_mod)
                T.tile.compare(mask, lane_mod, float(selected), "LT")
                T.tile.fill(valid, 1.0)
                T.tile.select(valid, mask, valid, 0.0, "VSEL_TENSOR_SCALAR_MODE")
                for lane in T.serial(8):
                    result[lane] = -1
                for lane in T.serial(selected):
                    expert = indices[row * 8 + lane] - base
                    if expert >= 0 and expert < experts:
                        T.tile.compare(mask, values, T.cast(expert + base, "float32"), "EQ")
                        T.tile.select(matching, mask, valid, 0.0, "VSEL_TENSOR_SCALAR_MODE")
                        T.tile.compare(
                            mask, lanes, T.cast((row % chunk) * 8 + lane + 1, "float32"), "LT"
                        )
                        T.tile.select(matching, mask, matching, 0.0, "VSEL_TENSOR_SCALAR_MODE")
                        T.reduce_sum(matching, total, dim=1)
                        T.set_flag("V", "S", 6)
                        T.wait_flag("V", "S", 6)
                        result[lane] = offsets[expert, part] + T.cast(total[0, 0], "int32") - 1
                        T.set_flag("S", "V", 6)
                        T.wait_flag("S", "V", 6)
                T.set_flag("S", "MTE3", 6)
                T.wait_flag("S", "MTE3", 6)
                T.copy(result, restore[row, :])
                T.set_flag("MTE3", "S", 7)
                T.wait_flag("MTE3", "S", 7)

    return main
