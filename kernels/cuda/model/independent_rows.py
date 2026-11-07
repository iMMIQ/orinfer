"""Batch independent M1 TileLang programs without changing their arithmetic.

The x grid gains a request dimension. Shared boundary tensors use contiguous
rows; private state and scratch are looked up in a stable caller-owned address
table. Null rows are inactive. The original CTA geometry, reductions, casts,
barriers and tensor-core operations are preserved.
"""

import math

from tilelang import tvm
from tvm import tirx as tir

POINTER_SOURCE = r"""
__device__ __forceinline__ void* orin_row_pointer(unsigned long long address) {
    return reinterpret_cast<void*>(address);
}
__device__ __forceinline__ void* orin_row_offset(const void* base, unsigned long long bytes) {
    return const_cast<char*>(static_cast<const char*>(base)) + bytes;
}
"""


def indirect_parameters(function, columns, operands, offsets=None):
    """Replace private pointer parameters, preserving the original launch and scalars.

    The caller selects one owner by slicing the fixed pointer table. Shared
    row pointers keep their original byte views; this wrapper adds no row axis.
    """
    offsets = offsets or {}
    pointer_param = tir.Var("Pointers_handle", "handle")
    pointers = tir.decl_buffer((len(columns),), "uint64", name="Pointers")
    params, buffers, identities = [pointer_param], {pointer_param: pointers}, [None]
    aliases, data, bindings, flat = {}, {}, [], {}
    column_params = []
    for param in function.params:
        original = function.buffer_map.get(param)
        operand = (
            None
            if original is None
            else (original.name if original.name in operands else "live_" + original.name.lower())
        )
        if original is None or operand not in operands:
            params.append(param)
            identities.append(original.name if original is not None else param.name)
            if original is not None:
                buffers[param] = original
            continue
        pointer = tir.Var(
            original.name + "_indirect", tvm.ir.PointerType(tvm.ir.PrimType(original.dtype))
        )
        if operands[operand] not in columns:
            raise ValueError("Unknown indirect operand column")
        column = tir.Var("column_" + operand, "int32")
        column_params.append(column)
        value = tir.call_extern(
            "handle",
            "orin_row_pointer",
            tir.BufferLoad(pointers, [column]) + offsets.get(operand, 0),
        )
        alias = tir.decl_buffer(
            original.shape,
            original.dtype,
            name=original.name,
            data=pointer,
            strides=original.strides,
            elem_offset=original.elem_offset,
            data_alignment=original.data_alignment,
            offset_factor=original.offset_factor,
        )
        aliases[original] = alias
        data[original.data] = pointer
        bindings.extend([tir.Bind(pointer, value), tir.DeclBuffer(alias)])
        flat[original.data] = tir.decl_buffer(
            (math.prod(original.shape),), original.dtype, name=original.name + "_flat", data=pointer
        )

    def indirect_buffer(buffer):
        if buffer not in aliases and buffer.data in data:
            aliases[buffer] = tir.decl_buffer(
                buffer.shape,
                buffer.dtype,
                name=buffer.name,
                data=data[buffer.data],
                strides=buffer.strides,
                elem_offset=buffer.elem_offset,
                data_alignment=buffer.data_alignment,
                offset_factor=buffer.offset_factor,
            )
        return aliases.get(buffer, buffer)

    def region(value):
        return tir.BufferRegion(indirect_buffer(value.buffer), value.region)

    def replace(node):
        if (
            isinstance(node, tir.Call)
            and getattr(node.op, "name", "") == "tirx.tvm_access_ptr"
            and node.args[1] in flat
        ):
            return tir.Call(
                node.dtype,
                tvm.ir.Op.get("tl.access_ptr"),
                [tir.BufferLoad(flat[node.args[1]], [node.args[2]]), node.args[3], node.args[4]],
            )
        if isinstance(node, tir.BufferLoad) and node.buffer.data in data:
            return tir.BufferLoad(indirect_buffer(node.buffer), node.indices, node.predicate)
        if isinstance(node, tir.BufferStore) and node.buffer.data in data:
            return tir.BufferStore(
                indirect_buffer(node.buffer), node.value, node.indices, node.predicate
            )
        if isinstance(node, tir.DeclBuffer) and node.buffer.data in data:
            return tir.DeclBuffer(indirect_buffer(node.buffer))
        return None

    def root(node):
        if isinstance(node, tir.SBlock):
            annotations = dict(node.annotations)
            body = node.body
            if node.name_hint == "tilelang_root":
                imported = str(annotations.get("pragma_import_c", ""))
                annotations["pragma_import_c"] = (
                    imported if "void* orin_row_pointer(" in imported else imported + POINTER_SOURCE
                )
                body = tir.SeqStmt(bindings + [body])
            return tir.SBlock(
                node.iter_vars,
                [region(value) for value in node.reads],
                [region(value) for value in node.writes],
                node.name_hint,
                body,
                node.init,
                node.alloc_buffers,
                [
                    tir.MatchBufferRegion(indirect_buffer(value.buffer), region(value.source))
                    for value in node.match_buffers
                ],
                annotations,
            )
        return None

    body = tir.stmt_functor.ir_transform(function.body, None, replace)
    body = tir.stmt_functor.substitute(body, data)
    body = tir.stmt_functor.ir_transform(body, None, root)
    params.extend(column_params)
    identities.extend(param.name for param in column_params)
    return tir.PrimFunc(params, body, function.ret_type, buffers, function.attrs), identities


def independent_rows(
    function,
    rows,
    roles,
    columns,
    row_stride=None,
    dynamic=False,
    offsets=None,
    batch_parameters=(),
    scalar_columns=None,
    row_offset=None,
    row_capacity=None,
):
    """roles maps input buffer names to ('weight'|'row'|'private', identity).

    columns is the architecture's private-buffer column map. Returns a PrimFunc
    and its input names in parameter order (None denotes the pointer table).
    offsets optionally selects each input's byte range within its private arena.
    batch_parameters binds named scheduling scalars to the live owner count,
    without changing the single-owner tensor shapes or address strides.
    Only used inputs survive, so unused prefix snapshots need no allocation.
    """
    if not 2 <= rows <= 128:
        raise ValueError("Independent decode rows must be within 2..128")
    used = set()
    offsets = offsets or {}
    scalar_columns = scalar_columns or {}
    grids = []

    def inspect(node):
        if isinstance(node, (tir.BufferLoad, tir.BufferStore)):
            used.add(node.buffer.data)
        if isinstance(node, tir.Var):
            used.add(node)
        if isinstance(node, tir.For) and node.thread_binding is not None:
            if node.thread_binding.thread_tag.startswith("blockIdx."):
                grids.append(node)

    tir.stmt_functor.post_order_visit(function.body, inspect)
    x_grids = [g for g in grids if g.thread_binding.thread_tag == "blockIdx.x"]
    if len(x_grids) != 1:
        raise ValueError("Expected one independent CTA x grid")
    grid = x_grids[0]
    count = tir.Var("m", "int32") if dynamic else rows
    analyzer = tvm.arith.Analyzer()
    for param in function.params:
        if param.name in scalar_columns:
            analyzer.update(param, tvm.arith.ConstIntBound(1, scalar_columns[param.name][1]))
        elif param.name in batch_parameters:
            analyzer.update(param, tvm.arith.ConstIntBound(1, rows))
    extents = {}
    for g in grids:
        bound = analyzer.const_int_bound(g.extent).max_value
        if bound <= 0 or bound >= 2**31:
            raise ValueError("Unbounded independent CTA geometry")
        extents[g.loop_var] = int(bound)
    grid_width = extents[grid.loop_var]
    lane = tir.floordiv(grid.loop_var, grid_width)
    pointer_param = tir.Var("Pointers_handle", "handle")
    stride = row_stride or len(columns)
    pointers = tir.decl_buffer((128, len(columns)), "uint64", name="Pointers", strides=(stride, 1))
    params, buffers, identities = [pointer_param], {pointer_param: pointers}, [None]
    replacement, data, lets, flat = {}, {}, [], {}
    scalar_lets = []
    private_columns = []
    scalar_replacement = {}

    def control(identity, label):
        pointer = tir.Var(label + "_ptr", tvm.ir.PointerType(tvm.ir.PrimType("int32")))
        alias = tir.decl_buffer((1,), "int32", name=label, data=pointer)
        scalar_lets.extend(
            [
                tir.Bind(
                    pointer,
                    tir.call_extern(
                        "handle",
                        "orin_row_pointer",
                        tir.BufferLoad(pointers, [lane, columns[identity]]),
                    ),
                ),
                tir.DeclBuffer(alias),
            ]
        )
        private_columns.append(columns[identity])
        return tir.BufferLoad(alias, [0])

    row_index = control(row_offset, "RowOffset") if row_offset else lane
    for param in function.params:
        if param not in function.buffer_map:
            if param.name in scalar_columns:
                scalar_lets.append(
                    tir.Bind(param, control(scalar_columns[param.name][0], param.name + "_control"))
                )
                continue
            if param.name in batch_parameters:
                scalar_replacement[param] = count
                continue
            if param in used:
                params.append(param)
                identities.append(param.name)
            continue
        original = function.buffer_map[param]
        if original.data not in used:
            continue
        mode, identity = roles[original.name]
        if mode == "weight":
            params.append(param)
            buffers[param] = original
            identities.append(original.name)
            continue
        pointer = tir.Var(
            original.name + "_row", tvm.ir.PointerType(tvm.ir.PrimType(original.dtype))
        )
        if mode == "private":
            column = columns[identity]
            private_columns.append(column)
            address = tir.BufferLoad(pointers, [lane, column])
            value = tir.call_extern(
                "handle", "orin_row_pointer", address + offsets.get(original.name, 0)
            )
        elif mode == "row":
            if row_offset:
                substitutions = {p: 1 for p in function.params if p.name in scalar_columns}
                width = int(
                    analyzer.simplify(
                        tir.stmt_functor.substitute(math.prod(original.shape), substitutions)
                    )
                )
            else:
                width = math.prod(int(n) for n in original.shape)
            expanded = tir.decl_buffer(
                ((row_capacity or rows) * width,), original.dtype, name=original.name
            )
            params.append(param)
            buffers[param] = expanded
            identities.append(original.name)
            value = tir.call_extern(
                "handle",
                "orin_row_offset",
                expanded.data,
                tir.Cast("uint64", row_index * width * tvm.DataType(original.dtype).bits // 8),
            )
        else:
            raise ValueError("Unknown independent-row input mode")
        alias = tir.decl_buffer(
            original.shape,
            original.dtype,
            name=original.name,
            data=pointer,
            strides=original.strides,
        )
        replacement[original] = alias
        data[original.data] = pointer
        flat[original.data] = tir.decl_buffer(
            (math.prod(original.shape),),
            original.dtype,
            name=original.name + "_flat",
            data=pointer,
        )
        lets.append((pointer, value))
    if {p.name for p in scalar_replacement} != set(batch_parameters):
        raise ValueError("Missing independent-row scheduling scalar")
    if not private_columns:
        raise ValueError("Independent rows require private state for padding")

    def replace_buffer(node):
        if isinstance(node, tir.Call) and getattr(node.op, "name", "") == "tirx.tvm_access_ptr":
            if node.args[1] in flat:
                # Preserve cp.async while exposing the alias's concrete extent
                # to TileLang's safe-memory pass. Raw data vars alone cannot
                # identify address-table-backed allocations.
                return tir.Call(
                    node.dtype,
                    tvm.ir.Op.get("tl.access_ptr"),
                    [
                        tir.BufferLoad(flat[node.args[1]], [node.args[2]]),
                        node.args[3],
                        node.args[4],
                    ],
                )
        if isinstance(node, tir.BufferLoad) and node.buffer in replacement:
            return tir.BufferLoad(replacement[node.buffer], node.indices, node.predicate)
        if isinstance(node, tir.BufferStore) and node.buffer in replacement:
            return tir.BufferStore(
                replacement[node.buffer], node.value, node.indices, node.predicate
            )
        return None

    def rewrite(node):
        if isinstance(node, tir.SBlock) and node.name_hint == "tilelang_root":
            body = node.body
            for g in grids:
                live = tir.stmt_functor.substitute(g.extent, scalar_replacement)
                index = tir.floormod(g.loop_var, grid_width) if g is grid else g.loop_var
                if scalar_columns:
                    body = tir.IfThenElse(index < live, body, None)
            body = tir.SeqStmt(
                scalar_lets
                + [tir.Bind(pointer, value) for pointer, value in lets]
                + [tir.DeclBuffer(alias) for alias in replacement.values()]
                + [body]
            )
            body = tir.IfThenElse(
                tir.BufferLoad(pointers, [lane, private_columns[0]]) != 0, body, None
            )
            annotations = dict(node.annotations)
            annotations["pragma_import_c"] = (
                str(annotations.get("pragma_import_c", "")) + POINTER_SOURCE
            )
            return tir.SBlock(
                node.iter_vars,
                node.reads,
                node.writes,
                node.name_hint,
                body,
                node.init,
                node.alloc_buffers,
                node.match_buffers,
                annotations,
            )
        if isinstance(node, tir.For) and node.loop_var.same_as(grid.loop_var):
            return tir.For(
                node.loop_var,
                node.min,
                grid_width * count,
                node.kind,
                node.body,
                node.thread_binding,
                node.annotations,
            )
        if (
            isinstance(node, tir.For)
            and node.thread_binding is not None
            and node.thread_binding.thread_tag.startswith("blockIdx.")
            and scalar_columns
        ):
            return tir.For(
                node.loop_var,
                node.min,
                extents[node.loop_var],
                node.kind,
                node.body,
                node.thread_binding,
                node.annotations,
            )
        return None

    body = tir.stmt_functor.ir_transform(function.body, None, replace_buffer)
    body = tir.stmt_functor.substitute(
        body,
        {grid.loop_var: tir.floormod(grid.loop_var, grid_width), **data, **scalar_replacement},
    )
    # Restore the loop's binding variable: substitution rewrites its uses only.
    body = tir.stmt_functor.ir_transform(body, None, rewrite)
    if dynamic:
        params.append(count)
        identities.append("__rows__")
    return tir.PrimFunc(params, body, function.ret_type, buffers, function.attrs), identities
