"""Symbolic token rows for offline AOT; allocations remain caller-owned."""

import tilelang.language as T


def row_count(capacity, dynamic=False):
    if type(dynamic) is not bool:
        raise ValueError("Invalid symbolic row policy")
    if not dynamic:
        return capacity
    if type(capacity) is not int or capacity < 1:
        raise ValueError("Invalid symbolic row capacity")
    if hasattr(T, "dynamic"):
        return T.dynamic("m")
    # TileLang-Ascend's pinned frontend uses the same free shape scalar,
    # before the CUDA frontend introduced T.dynamic.
    from tilelang import tvm

    return tvm.tir.Var("m", "int32")


def explicit_rows(function):
    """Keep m in the exported host ABI, including affine-only tensor shapes."""
    from tilelang import tvm

    tir = getattr(tvm, "tirx", None) or tvm.tir

    variables = set()

    def visit(node):
        if isinstance(node, tir.Var) and node.name == "m":
            variables.add(node)

    for parameter in function.params:
        if parameter.name == "m" and parameter.dtype != "handle":
            return function
    for buffer in function.buffer_map.values():
        for expression in [*buffer.shape, *buffer.strides]:
            tir.stmt_functor.post_order_visit(expression, visit)
    if not variables:
        # Local lane/index variables may also be named m. Only free scalars
        # can become a host argument; tensor geometry is authoritative first.
        for variable in tir.analysis.undefined_vars(function.body, list(function.params)):
            visit(variable)
    if len(variables) != 1:
        raise ValueError("Expected one symbolic row scalar")
    (rows,) = variables
    if any(p.same_as(rows) for p in function.params):
        return function
    return tir.PrimFunc(
        list(function.params) + [rows],
        function.body,
        function.ret_type,
        function.buffer_map,
        function.attrs,
    )


def explicit_scalar(function, scalar):
    """Expose a metadata scalar without changing tensor shapes or row strides."""
    from tvm import tirx as tir

    if any(p.same_as(scalar) for p in function.params):
        return function
    return tir.PrimFunc(
        list(function.params) + [scalar],
        function.body,
        function.ret_type,
        function.buffer_map,
        function.attrs,
    )


def static_rows(function, count=1):
    """Specialize a row factory before adding independent request CTAs.

    Only tensor-shape row scalars are bound; positions and KV lengths remain
    request-private device inputs. No arithmetic or reduction is rewritten.
    """
    from tvm import tirx as tir

    variables = set()

    def visit(node):
        if isinstance(node, tir.Var) and node.name in ("m", "rows", "tokens", "batch"):
            variables.add(node)

    for buffer in function.buffer_map.values():
        for expression in [*buffer.shape, *buffer.strides]:
            tir.stmt_functor.post_order_visit(expression, visit)
    if not variables:
        return function
    if len(variables) != 1 or type(count) is not int or count < 1:
        raise ValueError("Expected one positive row specialization")
    (rows,) = variables
    function = explicit_scalar(function, rows)
    return function.specialize({rows: count})
