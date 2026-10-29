"""Indirect owner bindings around the unchanged A2 primitive arithmetic.

The host builds compact (owner, physical CTA) work items. Pointer-table column
zero stores the owner's real query length; remaining columns hold validated
operand addresses. One shared active word disables state access during capture.
"""


def indirect(function, columns, owner_capacity, token_capacity, grid_limit=64, table_columns=None):
    import math
    from tilelang import tvm
    from tools.model.row_contract import expression

    tir = tvm.tir
    old_rows = {}
    grids = []
    for tensor in function.buffer_map.values():
        for dim in tensor.shape:
            tir.stmt_functor.post_order_visit(
                dim,
                lambda node: (
                    old_rows.setdefault(node, 1)
                    if isinstance(node, tir.Var) and node.name == "m"
                    else None
                ),
            )

    def inspect(node):
        if (
            isinstance(node, tir.AttrStmt)
            and node.attr_key == "thread_extent"
            and node.node.thread_tag == "blockIdx.x"
        ):
            grids.append(node)

    tir.stmt_functor.post_order_visit(function.body, inspect)
    if len(old_rows) != 1 or len(grids) != 1:
        raise ValueError("Owner wrapper requires one row scalar and one CTA grid")
    old_m = next(iter(old_rows))
    grid = grids[0]
    grid_expression = expression(str(tvm.arith.Analyzer().simplify(grid.value)))
    m = tir.Var("m", "int32")
    q = tir.Var("owner_rows", "int32")
    owner = tir.Var("owner_index", "int32")
    cid = grid.node.var
    work = tir.decl_buffer((m * 2, 2), "int32", name="OwnerWork")
    pointers = tir.decl_buffer(
        (
            owner_capacity,
            (table_columns if table_columns is not None else max(columns.values()) + 1) + 1,
        ),
        "uint64",
        name="OwnerPointers",
    )
    active = tir.decl_buffer((1,), "int32", name="PacketActive")
    work_param, pointer_param, active_param = [
        tir.Var(n, "handle")
        for n in ("OwnerWork_handle", "OwnerPointers_handle", "PacketActive_handle")
    ]
    params = [work_param, pointer_param, active_param]
    buffers = {work_param: work, pointer_param: pointers, active_param: active}
    aliases, data, flat, lets, identities = {}, {}, {}, [], [None, None, None]
    for param in function.params:
        if param not in function.buffer_map:
            continue
        original = function.buffer_map[param]
        if original.name not in columns:
            params.append(param)
            buffers[param] = original
            identities.append(original.name)
            continue
        pointer = tir.Var(
            original.name + "_owner", tvm.ir.PointerType(tvm.ir.PrimType(original.dtype))
        )
        shape = [tir.stmt_functor.substitute(d, {old_m: q}) for d in original.shape]
        alias = tir.decl_buffer(shape, original.dtype, name=original.name, data=pointer)
        aliases[original] = alias
        data[original.data] = pointer
        flat[original.data] = tir.decl_buffer(
            (tvm.arith.Analyzer().simplify(math.prod(shape)),),
            original.dtype,
            name=original.name + "_flat",
            data=pointer,
        )
        value = tir.reinterpret(
            "handle", tir.BufferLoad(pointers, [owner, columns[original.name] + 1])
        )
        lets.append((pointer, value))

    def replacement(node):
        if (
            isinstance(node, tir.Call)
            and getattr(node.op, "name", "") == "tir.tvm_access_ptr"
            and node.args[1] in flat
        ):
            return tir.Call(
                node.dtype,
                tvm.ir.Op.get("tl.access_ptr"),
                [tir.BufferLoad(flat[node.args[1]], [node.args[2]]), node.args[3], node.args[4]],
            )
        if isinstance(node, tir.BufferLoad) and node.buffer in aliases:
            return tir.BufferLoad(aliases[node.buffer], node.indices)
        if isinstance(node, tir.BufferStore) and node.buffer in aliases:
            return tir.BufferStore(aliases[node.buffer], node.value, node.indices)
        return None

    body = tir.stmt_functor.ir_transform(grid.body, None, replacement)
    iteration = tir.Var("owner_tile", "int32")
    physical = tir.BufferLoad(work, [cid, 1])
    logical = physical + iteration * grid_limit
    extent = tir.stmt_functor.substitute(grid.value, {old_m: q})
    body = tir.stmt_functor.substitute(body, {old_m: q, grid.node.var: logical, **data})

    def inside_threads(value):
        if isinstance(value, tir.AttrStmt) and value.attr_key == "thread_extent":
            return tir.AttrStmt(value.node, value.attr_key, value.value, inside_threads(value.body))
        result = tir.For(
            iteration,
            0,
            tir.floordiv(extent - physical + grid_limit - 1, grid_limit),
            tir.ForKind.SERIAL,
            value,
        )
        for pointer, address in reversed(lets):
            result = tir.LetStmt(pointer, address, result)
        result = tir.LetStmt(q, tir.Cast("int32", tir.BufferLoad(pointers, [owner, 0])), result)
        result = tir.LetStmt(owner, tir.BufferLoad(work, [cid, 0]), result)
        return tir.IfThenElse(
            tir.And(tir.BufferLoad(active, [0]) != 0, tir.BufferLoad(work, [cid, 0]) >= 0),
            result,
            None,
        )

    wrapped = tir.AttrStmt(grid.node, grid.attr_key, m * 2, inside_threads(body))

    def replace_grid(node):
        return wrapped if node.same_as(grid) else None

    root = tir.stmt_functor.ir_transform(function.body, None, replace_grid, ["tir.AttrStmt"])
    # Buffer aliases carry owner-local strides. The remaining global buffer
    # parameters are immutable weights, independent of the live query length.
    result = tir.PrimFunc(params, root, function.ret_type, buffers, function.attrs)
    return result, identities, grid_expression


def lower_aliases(source, aliases):
    """Materialize TIR owner pointers as AscendC GlobalTensor views."""
    import re

    opening = re.search(r'extern "C" __global__ __aicore__ void \w+\([^)]*\)\s*\{', source)
    if opening is None:
        raise ValueError("Owner aliases require an A2 device entry")
    bindings = list(re.finditer(r"\w+\.SetGlobalBuffer\([^;]+;", source[opening.end() :]))
    if not bindings:
        raise ValueError("Owner aliases have no pointer-table binding")
    entry = opening.end() + bindings[-1].end()
    source = (
        source[:entry]
        + "\n AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(OwnerPointers);\n"
        + source[entry:]
    )
    types = {
        "bfloat16": "bfloat16_t",
        "float16": "half",
        "float32": "float",
        "int8": "int8_t",
        "int32": "int32_t",
        "int64": "int64_t",
        "uint64": "uint64_t",
    }
    for name, dtype in aliases.items():
        pattern = rf"void\*\s+{re.escape(name)}\s*=\s*([^;]+);"
        ctype = types[dtype]
        source, replaced = re.subn(
            pattern,
            lambda match: (
                f"AscendC::GlobalTensor<{ctype}> {name}; {name}.SetGlobalBuffer(reinterpret_cast<__gm__ {ctype}*>(reinterpret_cast<uint64_t>({match[1]})));"
            ),
            source,
        )
        if replaced == 0 and re.search(rf"\b{re.escape(name)}\b", source) is None:
            # Compile-time branches can remove an unused compressed KV input.
            continue
        if replaced != 1:
            raise ValueError("Missing or repeated owner alias: " + name)
    return source
