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


def independent_rows(function, rows, roles, columns, row_stride=None, dynamic=False, offsets=None):
    """roles maps input buffer names to ('weight'|'row'|'private', identity).

    columns is the architecture's private-buffer column map. Returns a PrimFunc
    and its input names in parameter order (None denotes the pointer table).
    offsets optionally selects each input's byte range within its private arena.
    Only used inputs survive, so unused prefix snapshots need no allocation.
    """
    if not 2 <= rows <= 128:
        raise ValueError("Independent decode rows must be within 2..128")
    used = set()
    offsets = offsets or {}
    grids = []

    def inspect(node):
        if isinstance(node, (tir.BufferLoad, tir.BufferStore)):
            used.add(node.buffer.data)
        if isinstance(node, tir.Var):
            used.add(node)
        if isinstance(node, tir.For) and node.thread_binding is not None:
            if node.thread_binding.thread_tag == "blockIdx.x":
                grids.append(node)

    tir.stmt_functor.post_order_visit(function.body, inspect)
    if len(grids) != 1 or not isinstance(grids[0].extent, tir.IntImm):
        raise ValueError("Expected one statically sized independent CTA x grid")
    grid = grids[0]
    count = tir.Var("m", "int32") if dynamic else rows
    lane = tir.floordiv(grid.loop_var, grid.extent)
    pointer_param = tir.Var("Pointers_handle", "handle")
    stride = row_stride or len(columns)
    pointers = tir.decl_buffer((128, len(columns)), "uint64", name="Pointers", strides=(stride, 1))
    params, buffers, identities = [pointer_param], {pointer_param: pointers}, [None]
    replacement, data, lets, flat = {}, {}, [], {}
    private_columns = []
    for param in function.params:
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
            width = math.prod(int(n) for n in original.shape)
            expanded = tir.decl_buffer((rows * width,), original.dtype, name=original.name)
            params.append(param)
            buffers[param] = expanded
            identities.append(original.name)
            value = tir.call_extern(
                "handle",
                "orin_row_offset",
                expanded.data,
                tir.Cast("uint64", lane * width * tvm.DataType(original.dtype).bits // 8),
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
            (math.prod(int(n) for n in original.shape),),
            original.dtype,
            name=original.name + "_flat",
            data=pointer,
        )
        lets.append((pointer, value))
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
            body = tir.SeqStmt(
                [tir.Bind(pointer, value) for pointer, value in lets]
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
                grid.extent * count,
                node.kind,
                node.body,
                node.thread_binding,
                node.annotations,
            )
        return None

    body = tir.stmt_functor.ir_transform(function.body, None, replace_buffer)
    body = tir.stmt_functor.substitute(
        body, {grid.loop_var: tir.floormod(grid.loop_var, grid.extent), **data}
    )
    # Restore the loop's binding variable: substitution rewrites its uses only.
    body = tir.stmt_functor.ir_transform(body, None, rewrite)
    if dynamic:
        params.append(count)
        identities.append("__rows__")
    return tir.PrimFunc(params, body, function.ret_type, buffers, function.attrs), identities


def history_commit(columns, source_column, destination_column, row_stride=None):
    """One stream-ordered copy launch for every request's convolution history."""
    import tilelang.language as T
    from tools.operators.common import orin_jit

    stride = row_stride or columns

    @orin_jit
    def build():
        @T.prim_func
        def kernel(Pointers: T.Tensor((128, stride), "uint64"), m: T.int32):
            with T.Kernel(120, m, threads=256) as (chunk, lane):
                T.import_source(POINTER_SOURCE)
                if Pointers[lane, source_column] != 0:
                    source = T.bind(
                        T.call_extern("handle", "orin_row_pointer", Pointers[lane, source_column]),
                        var=T.ptr("float16"),
                    )
                    destination = T.bind(
                        T.call_extern(
                            "handle", "orin_row_pointer", Pointers[lane, destination_column]
                        ),
                        var=T.ptr("float16"),
                    )
                    X = T.decl_buffer((30720,), "float16", data=source)
                    Y = T.decl_buffer((30720,), "float16", data=destination)
                    for i in T.Parallel(256):
                        Y[chunk * 256 + i] = X[chunk * 256 + i]

        return kernel

    return build()
