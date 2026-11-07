"""Physical operand geometry from the actual Ascend primitive, before publication."""


def physical_shapes(function, capacity):
    from tilelang import tvm

    analyzer = tvm.arith.Analyzer()
    result = []
    for parameter in function.params:
        if parameter not in function.buffer_map:
            continue
        tensor = function.buffer_map[parameter]
        shape = []
        for dimension in tensor.shape:
            variables = {}

            def visit(node):
                if isinstance(node, tvm.tir.Var):
                    if node.name != "m":
                        raise ValueError("Unexpected free A2 shape variable")
                    variables[node] = capacity

            tvm.tir.stmt_functor.post_order_visit(dimension, visit)
            value = int(analyzer.simplify(tvm.tir.stmt_functor.substitute(dimension, variables)))
            if value < 1:
                raise ValueError("A2 operand requires positive physical dimensions")
            shape.append(value)
        result.append(shape)
    return result
