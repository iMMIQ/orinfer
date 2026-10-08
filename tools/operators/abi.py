"""CPU-only extraction of actual generated NVRTC launch ABI, without eval."""
import ast
import re


def validate_parameter_count(source, host):
    """Reject incomplete generated wrappers before they reach cuLaunchKernel.

    Symbolic affine tensor dimensions can leave a CUDA scalar out of a
    compiler-generated host wrapper. The emitted CUDA prototype is independent
    evidence of parameter count; never pad missing arguments speculatively.
    """
    prototype = re.search(r'\b' + re.escape(host['symbol']) +
                          r'\s*\(([^)]*)\)\s*(?:;|\{)', source)
    if prototype is None:
        raise ValueError('Missing CUDA kernel prototype')
    parameters = [p.strip() for p in prototype[1].split(',') if p.strip() not in ('', 'void')]
    if len(parameters) != len(host['ordered_arguments']):
        raise ValueError('CUDA and generated host parameter counts differ')


def _items(node):
    return node.elts if isinstance(node, (ast.Tuple, ast.List)) else [node]


def parse_host(source):
    tree = ast.parse(source)
    call = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "call")
    launch = {}
    values = types = None
    result = []
    # Generated call consists of sequential assignments and status-check ifs.
    for node in call.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                if target.id == "config":
                    launch = {}
                if target.id == "arg_values":
                    values = [ast.unparse(item) for item in _items(node.value)]
                if target.id == "arg_types":
                    types = [ast.unparse(item) for item in _items(node.value)]
            elif isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                if target.value.id == "config":
                    launch[target.attr] = ast.unparse(node.value)
        calls = [item for item in ast.walk(node.value)
                 if isinstance(item, ast.Call) and isinstance(item.func, ast.Name)
                 and item.func.id in {"cuLaunchKernelEx", "cuLaunchKernel"}]
        for item in calls:
            if item.func.id != "cuLaunchKernelEx":
                raise ValueError("Unexpected launch API; add explicit parser support")
            kernel = item.args[1]
            if not (isinstance(kernel, ast.Subscript)
                    and isinstance(kernel.value, ast.Name) and kernel.value.id == "kernels"
                    and isinstance(kernel.slice, ast.Constant)):
                raise ValueError("Unable to identify actual kernel symbol")
            if values is None or types is None or len(values) != len(types):
                raise ValueError("Missing/mismatched generated argument ABI")
            result.append({"symbol": kernel.slice.value, "launch_expressions": dict(launch),
                           "ordered_arguments": [{"value": value, "ctype": kind}
                                                 for value, kind in zip(values, types)]})
    if not result:
        raise ValueError("No generated CUDA launches found")
    return result


def evaluate(expression, variables):
    """Bind known scalar dimensions using arithmetic AST only, never Python eval."""
    def walk(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.Name):
            return variables[node.id]
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                return -walk(node.operand)
            if isinstance(node.op, ast.UAdd):
                return walk(node.operand)
        if isinstance(node, ast.BinOp):
            left, right = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.FloorDiv):
                return left // right
            if isinstance(node.op, ast.Mod):
                return left % right
        raise ValueError("Unsupported dimension expression")
    return walk(ast.parse(expression, mode="eval").body)
