"""Build symbolic launch contracts from the verified exported host ABI."""

import ast

from tools.operators.abi import evaluate


def resolve(value, rows, depth=0):
    """Evaluate the SDK's checked u32 row expressions during offline validation."""
    if type(rows) is not int or not 1 <= rows <= 0xFFFFFFFF or depth > 32:
        raise ValueError("Invalid row expression extent or depth")
    op = value.get("op")
    if op == "rows" and set(value) == {"op"}:
        return rows
    if op == "constant" and set(value) == {"op", "value"}:
        result = value["value"]
    elif op in ("add", "subtract", "multiply", "divide", "remainder", "minimum") and set(value) == {
        "op",
        "lhs",
        "rhs",
    }:
        left = resolve(value["lhs"], rows, depth + 1)
        right = resolve(value["rhs"], rows, depth + 1)
        if op in ("divide", "remainder") and right == 0:
            raise ValueError("Row expression divides by zero")
        result = {
            "add": lambda: left + right,
            "subtract": lambda: left - right,
            "multiply": lambda: left * right,
            "divide": lambda: left // right,
            "remainder": lambda: left % right,
            "minimum": lambda: min(left, right),
        }[op]()
    else:
        raise ValueError("Unknown row expression contract")
    if type(result) is not int or not 0 <= result <= 0xFFFFFFFF:
        raise ValueError("Row expression arithmetic exceeds u32")
    return result


def expression(source, row_multiplier=1, constants=None):
    def visit(node):
        if isinstance(node, ast.Name) and constants and node.id in constants:
            return dict(op="constant", value=constants[node.id])
        if (
            isinstance(node, ast.Constant)
            and type(node.value) is int
            and 0 <= node.value <= 0xFFFFFFFF
        ):
            return dict(op="constant", value=node.value)
        if isinstance(node, ast.Name) and node.id in ("M", "m", "rows", "batch"):
            rows = dict(op="rows")
            return (
                rows
                if row_multiplier == 1
                else dict(op="multiply", lhs=rows, rhs=dict(op="constant", value=row_multiplier))
            )
        operators = {
            ast.Add: "add",
            ast.Sub: "subtract",
            ast.Mult: "multiply",
            ast.FloorDiv: "divide",
            ast.Mod: "remainder",
        }
        if isinstance(node, ast.BinOp) and type(node.op) in operators:
            return dict(op=operators[type(node.op)], lhs=visit(node.left), rhs=visit(node.right))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "min"
            and len(node.args) == 2
            and not node.keywords
        ):
            return dict(op="minimum", lhs=visit(node.args[0]), rhs=visit(node.args[1]))
        raise ValueError(f"Unsupported row expression: {source}")

    return visit(ast.parse(source, mode="eval").body)


def contract(kernel, host, capacity=128, row_multiplier=1, constants=None):
    if type(row_multiplier) is not int or row_multiplier < 1:
        raise ValueError("Invalid launch row multiplier")
    rows = capacity * row_multiplier
    constants = constants or {}
    if set(constants) & {"M", "m", "rows", "batch"} or any(
        type(v) is not int or v < 0 or v > 0x7FFFFFFF for v in constants.values()
    ):
        raise ValueError("Invalid immutable launch scalars")
    dimensions = dict(M=rows, m=rows, rows=rows, batch=rows, **constants)
    launch = host["launch_expressions"]
    if kernel["grid"] != [evaluate(launch["gridDim" + axis], dimensions) for axis in "XYZ"]:
        raise ValueError("Host grid differs from capacity binding")
    if kernel["block"] != [evaluate(launch["blockDim" + axis], dimensions) for axis in "XYZ"]:
        raise ValueError("Host block differs from capacity binding")
    if kernel["shared_memory_bytes"] != evaluate(launch["sharedMemBytes"], dimensions):
        raise ValueError("Host shared memory differs from capacity binding")
    if len(kernel["args"]) != len(host["ordered_arguments"]):
        raise ValueError("Host argument count differs")
    args = []
    for index, (bound, exported) in enumerate(zip(kernel["args"], host["ordered_arguments"])):
        if exported["ctype"] in ("ctypes.c_int32", "c_int32"):
            if bound != dict(kind="i32", value=evaluate(exported["value"], dimensions)):
                raise ValueError("Host scalar differs from capacity binding")
            args.append(
                dict(index=index, value=expression(exported["value"], row_multiplier, constants))
            )
        elif exported["ctype"] not in ("ctypes.c_void_p", "c_void_p") or bound["kind"] not in (
            "buffer",
            "buffer_slice",
        ):
            raise ValueError("Unsupported dynamic host ABI")
    grid = [expression(launch["gridDim" + axis], row_multiplier, constants) for axis in "XYZ"]
    return dict(name=kernel["name"], capacity=capacity, grid=grid, arguments=args)
