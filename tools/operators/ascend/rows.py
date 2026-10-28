"""Bound logical A2 grids and compose explicit Cube/Vector template stages."""


def split_resources(function):
    """Keep arithmetic/allocations; stream order replaces paired-core flags."""
    from tilelang import tvm

    tir = tvm.tir
    scopes = set()

    def inspect(node):
        if isinstance(node, tir.AttrStmt) and node.attr_key == "resource_scope":
            scopes.add(int(node.value))

    tir.stmt_functor.post_order_visit(function.body, inspect)
    if not {0, 1}.issubset(scopes):
        return [function]
    stages = []
    for selected in (0, 1):

        def select(node):
            if (
                isinstance(node, tir.AttrStmt)
                and node.attr_key == "resource_scope"
                and int(node.value) != selected
            ):
                return tir.Evaluate(0)
            if (
                isinstance(node, tir.Evaluate)
                and isinstance(node.value, tir.Call)
                and node.value.op.name
                in (
                    "tl.ascend_set_cross_flag",
                    "tl.ascend_wait_cross_flag",
                )
            ):
                return tir.Evaluate(0)
            return None

        body = tir.stmt_functor.ir_transform(
            function.body, select, None, ["tir.AttrStmt", "tir.Evaluate"]
        )
        stages.append(function.with_body(body))
    return stages


def bounded_grid(function, limit=64):
    from tilelang import tvm

    tir = tvm.tir
    if type(limit) is not int or not 1 <= limit <= 32767:
        raise ValueError("Invalid physical A2 grid limit")

    def lower(node):
        if not (
            isinstance(node, tir.AttrStmt)
            and node.attr_key == "thread_extent"
            and node.node.thread_tag == "blockIdx.x"
        ):
            return None
        physical = node.node.var
        iteration = tir.Var("row_tile", "int32")
        extent = node.value
        logical = physical + iteration * limit
        count = tir.floordiv(extent - physical + limit - 1, limit)

        def wrap(body):
            if isinstance(body, tir.AttrStmt) and body.attr_key == "thread_extent":
                return tir.AttrStmt(body.node, body.attr_key, body.value, wrap(body.body))
            return tir.For(
                iteration,
                0,
                count,
                tir.ForKind.SERIAL,
                tir.stmt_functor.substitute(body, {physical: logical}),
            )

        return tir.AttrStmt(node.node, node.attr_key, tir.Min(extent, limit), wrap(node.body))

    return function.with_body(
        tir.stmt_functor.ir_transform(function.body, None, lower, ["tir.AttrStmt"])
    )
