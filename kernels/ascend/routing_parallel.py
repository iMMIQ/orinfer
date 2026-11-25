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


def _compact_routes(destination, positions, masks, width, selected, live):
    from tilelang import tvm

    return tvm.tir.call_intrin(
        "handle",
        tvm.ir.Op.get("tl.ascend_mma"),
        f"orinfer_compact_routes<{width}, {selected}>",
        destination.access_ptr("rw"),
        positions.access_ptr("r"),
        masks.access_ptr("rw"),
        False,
        live,
    )


def local(rows, base=0, experts=48, selected=6, chunk=32):
    """Compact per-chunk expert routes via hardware GatherMask.

    Groups 8 experts per AIV task to cut redundant `indices` GM reads and block
    launches 8x, and writes only the 32-byte-aligned live route prefix (skipping
    positions GM writes completely on empty count==0 pairs).
    """
    if not 1 <= selected <= 8 or chunk != 32:
        raise ValueError("Parallel regroup requires 1–8 routes and 32-token chunks")
    chunks = T.ceildiv(rows, chunk)
    width = chunk * 8
    group_size = 8
    expert_groups = (experts + group_size - 1) // group_size

    @T.prim_func
    def main(
        indices: T.Tensor((rows * 8,), "int32"),
        positions: T.Tensor((experts, chunks, width), "int32"),
        counts: T.Tensor((experts, chunks), "int32"),
    ):
        with T.Kernel(T.ceildiv(expert_groups * chunks, 2), is_npu=True) as (cid, vid):
            with T.Scope("V"):
                ids = T.alloc_ub((width,), "int32")
                values = T.alloc_ub((width,), "float32")
                lanes = T.alloc_ub((width,), "int32")
                masks = T.alloc_ub((width // 4,), "uint8")
                ordered = T.alloc_ub((width + 8,), "int32")
                T.reads(indices[0 : rows * 8])
                T.writes(positions[0:experts, 0:chunks, 0:width], counts[0:experts, 0:chunks])
                task = cid * 2 + vid
                group, part = task // chunks, task % chunks
                if group < expert_groups:
                    available = T.min(chunk, rows - part * chunk)
                    T.tile.fill(ids, -1)
                    T.pipe_barrier("V")
                    T.set_flag("V", "MTE2", 6)
                    T.wait_flag("V", "MTE2", 6)
                    T.copy(
                        indices[part * width : part * width + available * 8], ids[: available * 8]
                    )
                    T.set_flag("MTE2", "V", 6)
                    T.wait_flag("MTE2", "V", 6)
                    T.tile.cast(values, ids, "CAST_NONE", width)
                    T.tile.createvecindex(lanes, T.cast(part * width, "int32"))
                    T.pipe_barrier("V")
                    for offset in T.serial(group_size):
                        expert = group * group_size + offset
                        if expert < experts:
                            T.tile.compare(
                                masks[: width // 8],
                                values,
                                T.cast(base + expert, "float32"),
                                "EQ",
                            )
                            T.pipe_barrier("V")
                            T.evaluate(
                                _compact_routes(
                                    ordered, lanes, masks, width, selected, available * 8
                                )
                            )
                            T.set_flag("V", "MTE3", 6)
                            T.wait_flag("V", "MTE3", 6)
                            T.set_flag("S", "MTE3", 6)
                            T.wait_flag("S", "MTE3", 6)
                            T.copy(ordered[width : width + 1], counts[expert, part : part + 1])
                            cnt = ordered[width]
                            if cnt > 0:
                                aligned = T.min(width, T.ceildiv(cnt, 8) * 8)
                                T.copy(ordered[:aligned], positions[expert, part, :aligned])
                            T.set_flag("MTE3", "V", 7)
                            T.wait_flag("MTE3", "V", 7)

    return main.with_attr(
        "orinfer.component_sources", ["kernels/ascend/include/route_mask.h"]
    ).with_attr("orinfer.explicit_sync", True)


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


@T.macro
def _scan_shift_add(values, shifted, width, distance, n):
    if distance < width:
        if distance < n:
            T.tile.fill(shifted, 0.0)
            T.copy(values[0, 0 : width - distance], shifted[0, distance:width])
            T.tile.add(values, values, shifted)


def _vector_prefix(rows, experts, selected, chunk, pad_last, scan_rows):
    """Exact bounded scan over the model's local experts and selected routes."""
    if (
        not 1 <= scan_rows <= 4096
        or not 1 <= experts <= 128
        or not 1 <= selected <= 8
        or chunk != 32
    ):
        raise ValueError("Vector regroup scan requires E1–128/top1–8/chunk32 and at most 4096 rows")
    chunks = T.ceildiv(rows, chunk)
    width = ((experts * ((scan_rows + chunk - 1) // chunk) + 63) // 64) * 64

    @T.prim_func
    def main(
        counts: T.Tensor((experts * chunks,), "int32"),
        offsets: T.Tensor((experts * chunks,), "int32"),
        groups: T.Tensor((experts,), "int64"),
        errors: T.Tensor((rows + 1,), "int32"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            values = T.alloc_ub((1, width), "float32")
            orig = T.alloc_ub((1, width), "float32")
            shifted = T.alloc_ub((1, width), "float32")
            base_lanes = T.alloc_ub((1, width), "float32")
            lanes = T.alloc_ub((1, width), "float32")
            work = T.alloc_ub((width,), "int32")
            addresses = T.alloc_ub((width,), "uint32")
            mask = T.alloc_ub((width // 8,), "uint8")
            group_values = T.alloc_ub((experts,), "int64")
            summary = T.alloc_ub((8,), "int32")
            total = T.alloc_ub((8,), "int32")
            maximum = T.alloc_ub((1, 1), "float32")
            if vid == 0:
                n = experts * chunks
                T.tile.fill(work, 0)
                T.copy(counts[:n], work[:n])
                T.tile.cast(values, work, "CAST_NONE", width)
                T.copy(values, orig)
                T.tile.createvecindex(base_lanes, 0.0)
                # All intermediate sums are integers <= 6*4096, exactly
                # representable in FP32. Zero padding cannot create routes.
                # Unrolled compile-time distances skip dead steps on small
                # batches and replace >=8-element gathers with aligned UB copies.
                for step in T.serial(3):
                    distance = T.shift_left(1, step)
                    if distance < n:
                        T.tile.add(lanes, base_lanes, -T.cast(distance, "float32"))
                        T.tile.max(lanes, lanes, 0.0)
                        T.tile.mul(lanes, lanes, 4.0)
                        T.tile.cast(work, lanes, "CAST_RINT", width)
                        T.reinterpretcast(addresses, work, "uint32_t")
                        T.tile.gather(shifted, values, addresses, 0)
                        T.tile.compare(mask, base_lanes, T.cast(distance, "float32"), "GE")
                        T.tile.select(shifted, mask, shifted, 0.0, "VSEL_TENSOR_SCALAR_MODE")
                        T.tile.add(values, values, shifted)
                _scan_shift_add(values, shifted, width, 8, n)
                _scan_shift_add(values, shifted, width, 16, n)
                _scan_shift_add(values, shifted, width, 32, n)
                _scan_shift_add(values, shifted, width, 64, n)
                _scan_shift_add(values, shifted, width, 128, n)
                _scan_shift_add(values, shifted, width, 256, n)
                _scan_shift_add(values, shifted, width, 512, n)
                _scan_shift_add(values, shifted, width, 1024, n)
                _scan_shift_add(values, shifted, width, 2048, n)
                _scan_shift_add(values, shifted, width, 4096, n)
                _scan_shift_add(values, shifted, width, 8192, n)
                T.set_flag("V", "S", 6)
                T.wait_flag("V", "S", 6)
                total[0] = T.cast(values[0, n - 1], "int32")
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
                T.tile.sub(values, values, orig)
                T.tile.cast(work, values, "CAST_RINT", width)
                T.copy(work[:n], offsets[:n])
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
    group_size = 8
    expert_groups = (experts + group_size - 1) // group_size

    @T.prim_func
    def main(
        positions: T.Tensor((experts, chunks, width), "int32"),
        counts: T.Tensor((experts, chunks), "int32"),
        offsets: T.Tensor((experts, chunks), "int32"),
        gather: T.Tensor((rows * selected,), "int32"),
    ):
        with T.Kernel(T.ceildiv(expert_groups * chunks, 2), is_npu=True) as (cid, vid):
            ids = T.alloc_ub((width,), "int32")
            original = T.alloc_ub((width,), "float32")
            row_f = T.alloc_ub((width,), "float32")
            row_i = T.alloc_ub((width,), "int32")
            task = cid * 2 + vid
            group, part = task // chunks, task % chunks
            if group < expert_groups:
                T.tile.fill(ids, 0)
                for offset in T.serial(group_size):
                    expert = group * group_size + offset
                    if expert < experts:
                        count, start = counts[expert, part], offsets[expert, part]
                        if count > 0:
                            aligned = T.min(width, T.ceildiv(count, 8) * 8)
                            T.copy(positions[expert, part, :aligned], ids[:aligned])
                            T.tile.cast(original, ids, "CAST_NONE", width)
                            T.tile.mul(row_f, original, 0.125)
                            T.tile.cast(row_i, row_f, "CAST_FLOOR", width)
                            T.copy(row_i[:count], gather[start : start + count])
                            T.set_flag("MTE3", "V", 7)
                            T.wait_flag("MTE3", "V", 7)

    return main


def restore(rows, base=0, experts=48, selected=6, chunk=32, scan_rows=2048):
    """Chunk-level scalar cursor scan replacing 32x redundant per-row GM loads."""
    from kernels.ascend.kv_int8 import _contiguous_view

    if not 1 <= selected <= 8 or chunk != 32 or scan_rows < chunk:
        raise ValueError("Restore scan requires 1–8 routes and 32-token chunks")
    chunks = T.ceildiv(rows, chunk)
    capacity = (scan_rows + chunk - 1) // chunk
    width = chunk * 8
    expert_capacity = (experts + 7) // 8 * 8

    @T.prim_func
    def main(
        indices: T.Tensor((rows * 8,), "int32"),
        offsets: T.Tensor((experts, chunks), "int32"),
        restore: T.Tensor((rows, 8), "int32"),
    ):
        flat_offsets = _contiguous_view(offsets, (experts * chunks,))
        with T.Kernel(T.ceildiv(chunks, 2), is_npu=True) as (cid, vid):
            T.reads(indices[0 : rows * 8], offsets[0:experts, 0:chunks])
            T.writes(restore[0:rows, 0:8])
            with T.Scope("V"):
                ids = T.alloc_ub((width,), "int32")
                starts = T.alloc_ub((experts * capacity,), "int32")
                cursor = T.alloc_ub((expert_capacity,), "int32")
                result = T.alloc_ub((chunk, 8), "int32")
                part = cid * 2 + vid
                if part < chunks:
                    available = T.min(chunk, rows - part * chunk)
                    T.copy(
                        indices[part * width : part * width + available * 8], ids[: available * 8]
                    )
                    if chunks <= capacity:
                        T.copy(flat_offsets[:], starts[: experts * chunks])
                    T.tile.fill(result, -1)
                    T.pipe_barrier("V")
                    T.set_flag("MTE2", "S", 5)
                    T.wait_flag("MTE2", "S", 5)
                    T.set_flag("V", "S", 5)
                    T.wait_flag("V", "S", 5)
                    if chunks <= capacity:
                        for expert in T.serial(experts):
                            cursor[expert] = starts[expert * chunks + part]
                    else:
                        for expert in T.serial(experts):
                            cursor[expert] = offsets[expert, part]
                    for row in T.serial(available):
                        for lane in T.serial(selected):
                            expert = ids[row * 8 + lane] - base
                            if 0 <= expert and expert < experts:
                                result[row, lane] = cursor[expert]
                                cursor[expert] = cursor[expert] + 1
                    T.set_flag("S", "MTE3", 5)
                    T.wait_flag("S", "MTE3", 5)
                    T.copy(
                        result[:available, :], restore[part * chunk : part * chunk + available, :]
                    )
                    T.set_flag("MTE3", "V", 7)
                    T.wait_flag("MTE3", "V", 7)

    return main.with_attr("orinfer.explicit_sync", True)


def small_regroup(
    rows,
    base=0,
    experts=48,
    selected=6,
    total_experts=384,
    pad_last=False,
    max_rows=128,
):
    """Single-kernel bit-exact MoE regrouping for small decode/tail batches (rows <= 128).

    Replaces the 5-kernel clear/local/prefix/pack/restore chain with one UB pass
    eliminating 4 inter-kernel launches/barriers and intermediate GM traffic per layer.
    """
    if not 1 <= selected <= 8 or not 1 <= max_rows <= 256 or not 1 <= experts <= 128:
        raise ValueError("Small regroup requires top1-8, E1-128 and max_rows <= 256")
    lanes = (selected + 7) // 8 * 8
    exp_ub = (experts + 7) // 8 * 8

    @T.prim_func
    def main(
        indices: T.Tensor((rows, lanes), "int32"),
        gather: T.Tensor((rows * selected,), "int32"),
        restore: T.Tensor((rows, lanes), "int32"),
        groups: T.Tensor((experts,), "int64"),
        errors: T.Tensor((rows + 1,), "int32"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            T.reads(indices[0:rows, 0:lanes])
            T.writes(
                gather[0 : rows * selected],
                restore[0:rows, 0:lanes],
                groups[0:experts],
                errors[0 : rows + 1],
            )
            with T.Scope("V"):
                ids = T.alloc_ub((max_rows, lanes), "int32")
                gather_ub = T.alloc_ub((max_rows * selected,), "int32")
                restore_ub = T.alloc_ub((max_rows, lanes), "int32")
                counts = T.alloc_ub((exp_ub,), "int32")
                cursor = T.alloc_ub((exp_ub,), "int32")
                grp_ub = T.alloc_ub((exp_ub,), "int64")
                err_ub = T.alloc_ub((max_rows + 8,), "int32")
                state = T.alloc_ub((8,), "int32")
                if vid == 0:
                    T.copy(indices[:rows, :], ids[:rows, :])
                    T.tile.fill(gather_ub, -1)
                    T.tile.fill(restore_ub, -1)
                    T.tile.fill(counts, 0)
                    T.tile.fill(err_ub, 0)
                    T.tile.fill(state, 0)
                    T.pipe_barrier("V")
                    T.set_flag("MTE2", "S", 5)
                    T.wait_flag("MTE2", "S", 5)
                    T.set_flag("V", "S", 5)
                    T.wait_flag("V", "S", 5)
                    for r in T.serial(rows):
                        for l in T.serial(selected):
                            raw = ids[r, l]
                            if raw < 0 or raw >= total_experts:
                                state[0] = 1
                            e = raw - base
                            if 0 <= e and e < experts:
                                counts[e] = counts[e] + 1
                    for e in T.serial(experts):
                        c = counts[e]
                        cursor[e] = state[1]
                        grp_ub[e] = T.cast(c, "int64")
                        state[1] = state[1] + c
                    if pad_last:
                        grp_ub[experts - 1] = grp_ub[experts - 1] + T.cast(
                            rows * selected - state[1], "int64"
                        )
                    for r in T.serial(rows):
                        for l in T.serial(selected):
                            e = ids[r, l] - base
                            if 0 <= e and e < experts:
                                pos = cursor[e]
                                cursor[e] = pos + 1
                                gather_ub[pos] = r
                                restore_ub[r, l] = pos
                    err_ub[0] = state[0]
                    err_ub[rows] = state[1]
                    T.set_flag("S", "MTE3", 5)
                    T.wait_flag("S", "MTE3", 5)
                    T.copy(gather_ub[: rows * selected], gather[: rows * selected])
                    T.copy(restore_ub[:rows, :], restore[:rows, :])
                    T.copy(grp_ub[:experts], groups[:experts])
                    T.copy(err_ub[: rows + 1], errors[: rows + 1])
                    T.set_flag("MTE3", "V", 7)
                    T.wait_flag("MTE3", "V", 7)

    return main.with_attr("orinfer.explicit_sync", True)
