"""Lift a published TileLang CUDA body into a TileLang operand-table wrapper.

The imported body retains its instructions and CTA geometry. Only its entry
point becomes a device helper; pointer indirection uses independent_rows.
"""

import re


def primitive(source, kernel, host, buffers):
    import tilelang.language as T
    from tvm import tirx as tir

    match = re.search(r"\b" + re.escape(kernel["symbol"]) + r"\s*\(([^)]*)\)", source)
    if match is None:
        raise ValueError("Missing retained TileLang kernel prototype")
    declarations = match[1].split(",") if match[1].strip() else []
    if len(declarations) != len(kernel["args"]) or len(declarations) != len(
        host["ordered_arguments"]
    ):
        raise ValueError("Retained CUDA, host and native parameter counts differ")
    params, buffer_map, values, scalars, casts, helper_parameters = [], {}, [], {}, [], []
    for i, (decl, actual, argument) in enumerate(
        zip(declarations, kernel["args"], host["ordered_arguments"])
    ):
        parsed = re.fullmatch(r"\s*(.*?)\b([A-Za-z_]\w*)\s*", decl)
        if parsed is None:
            raise ValueError("Unsupported retained CUDA parameter")
        dtype, cuda_name = parsed.groups()
        dtype = dtype.replace("__restrict__", "").strip()
        value = argument["value"]
        if argument["ctype"] == "ctypes.c_void_p":
            if (
                actual["kind"] not in ("buffer", "buffer_slice")
                or not value.endswith(".data_ptr()")
                or "*" not in dtype
            ):
                raise ValueError("Retained pointer ABI differs")
            role = value.removesuffix(".data_ptr()")
            handle = tir.Var(role + "_handle", "handle")
            native_dtype = buffers[actual["name"]]["dtype"]
            dtype_name = {
                "i8": "int8",
                "i32": "int32",
                "u8": "uint8",
                "u16": "uint16",
                "u32": "uint32",
                "u64": "uint64",
                "i64": "int64",
                "f16": "float16",
                "bf16": "bfloat16",
                "f32": "float32",
                "f64": "float64",
            }[native_dtype]
            buffer = tir.decl_buffer((1,), dtype_name, name=role)
            params.append(handle)
            buffer_map[handle] = buffer
            values.append(
                tir.call_intrin(
                    "handle",
                    "tirx.tvm_access_ptr",
                    tir.type_annotation(buffer.dtype),
                    buffer.data,
                    0,
                    1,
                    1 if "const" in dtype.split() else 3,
                )
            )
            helper_parameters.append(f"{'const ' if 'const' in dtype.split() else ''}void* p{i}")
            casts.append(f"({dtype})p{i}")
        elif argument["ctype"] == "ctypes.c_int32" and actual["kind"] == "i32" and dtype == "int":
            if not re.fullmatch(r"[A-Za-z_]\w*", value):
                raise ValueError("Retained scalar must have an explicit host identity")
            scalar = tir.Var(value, "int32")
            params.append(scalar)
            values.append(scalar)
            scalars[value] = scalar
            helper_parameters.append(f"int p{i}")
            casts.append(f"p{i}")
        else:
            raise ValueError("Unsupported retained host parameter")
    symbol = kernel["symbol"]
    body_name = "orin_retained_body"
    header = (
        r'extern\s+"C"\s+__global__\s+void\s+(?:__launch_bounds__\([^)]*\)\s*)?'
        + re.escape(symbol)
        + r"\s*\("
    )
    imported, count = re.subn(header, "__device__ __forceinline__ void " + body_name + "(", source)
    if count != 2:
        raise ValueError("Expected one retained CUDA declaration and definition")
    helper_parameters.append("void* retained_shared")
    imported += (
        "\n__device__ __forceinline__ void orin_retained_call("
        + ", ".join(helper_parameters)
        + ") {\n  "
        + body_name
        + "("
        + ", ".join(casts)
        + ");\n}\n"
    )
    shared_bytes = eval(
        host["launch_expressions"]["sharedMemBytes"], {"__builtins__": {}, "min": tir.min}, scalars
    )
    shared = (
        None
        if isinstance(shared_bytes, int) and shared_bytes == 0
        else tir.decl_buffer((shared_bytes,), "uint8", name="RetainedShared", scope="shared.dyn")
    )
    values.append(
        tir.reinterpret("handle", tir.const(0, "uint64"))
        if shared is None
        else tir.call_intrin(
            "handle", "tirx.tvm_access_ptr", tir.type_annotation("uint8"), shared.data, 0, 1, 3
        )
    )
    call = tir.Evaluate(tir.call_extern("void", "orin_retained_call", *values))

    @T.prim_func
    def shell(Placeholder: T.Tensor((1,), T.uint8)):
        with T.Kernel(1, 1, 1, threads=(32, 1, 1)):
            Placeholder[0] = 0

    launch = host["launch_expressions"]
    dimensions = {}
    for axis in "XYZ":
        for prefix, binding in [("gridDim", "blockIdx."), ("blockDim", "threadIdx.")]:
            expression = launch[prefix + axis]
            if not re.fullmatch(r"[A-Za-z_0-9+*/%() ,\-]+", expression):
                raise ValueError("Unsupported retained launch expression")
            dimensions[binding + axis.lower()] = eval(
                expression, {"__builtins__": {}, "min": tir.min}, scalars
            )

    def replace(node):
        if isinstance(node, tir.BufferStore):
            return call
        if isinstance(node, tir.For) and node.thread_binding is not None:
            return tir.For(
                node.loop_var,
                node.min,
                dimensions[node.thread_binding.thread_tag],
                node.kind,
                node.body,
                node.thread_binding,
                node.annotations,
            )
        if isinstance(node, tir.AttrStmt) and node.attr_key == "thread_extent":
            return tir.AttrStmt(node.node, node.attr_key, dimensions[node.node.var.name], node.body)
        if isinstance(node, tir.SBlock):
            annotations = dict(node.annotations)
            if node.name_hint == "tilelang_root":
                annotations["pragma_import_c"] = imported
                annotations["tl.launch_threads"] = kernel["block"]
            return tir.SBlock(
                node.iter_vars,
                [],
                [],
                node.name_hint,
                node.body,
                node.init,
                ([shared] if shared is not None else [])
                if node.name_hint == "tilelang_root"
                else node.alloc_buffers,
                node.match_buffers,
                annotations,
            )
        return None

    body = tir.stmt_functor.ir_transform(shell.body, None, replace)
    attrs = dict(shell.attrs)
    attrs["global_symbol"] = "kernel"
    return tir.PrimFunc(params, body, shell.ret_type, buffer_map).with_attrs(attrs)
