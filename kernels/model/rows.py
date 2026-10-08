"""Symbolic token rows for offline AOT; allocations remain caller-owned."""
import tilelang.language as T


def row_count(capacity, dynamic=False):
    if type(dynamic) is not bool:
        raise ValueError('Invalid symbolic row policy')
    if not dynamic:
        return capacity
    if type(capacity) is not int or capacity < 1:
        raise ValueError('Invalid symbolic row capacity')
    return T.symbolic('m')


def explicit_rows(function):
    """Keep m in the exported host ABI, including affine-only tensor shapes."""
    from tvm import tirx as tir
    variables = set()

    def visit(node):
        if isinstance(node, tir.Var) and node.name == 'm':
            variables.add(node)

    tir.stmt_functor.post_order_visit(function.body, visit)
    if len(variables) != 1:
        raise ValueError('Expected one symbolic row scalar')
    rows, = variables
    if any(p.same_as(rows) for p in function.params):
        return function
    return tir.PrimFunc(list(function.params) + [rows], function.body,
                        function.ret_type, function.buffer_map, function.attrs)
