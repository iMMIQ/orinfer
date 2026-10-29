"""Offline A2 AOT compilation with a checked, pointer-table C ABI for Rust."""

import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tempfile
import threading

import tilelang
from tilelang.jit.adapter.libgen import LibraryGenerator

from tools.build.files import atomic_json, file_hash, source_hash
from tools.operators.ascend.vector_launch import lower_independent_cube, lower_independent_vector
from tools.operators.ascend.rows import bounded_grid, split_resources
from tools.operators.ascend.scalar_cache import lower_scalar_cache

ROOT = Path(__file__).resolve().parents[3]
SOURCES = [
    p
    for p in Path(__file__).parent.glob("*.py")
    if p.name != "validate.py" and not p.name.startswith("test_")
]
LOWER_LOCK = threading.RLock()


def lower_a2(function, configs):
    # The pinned compiler probes torch.npu.is_available() when choosing the
    # storage rewrite. In a device-free compiler container that enables an
    # unsafe generic reuse pass, despite the explicit Ascend target. Select the
    # NPU rewrite directly; GPU visibility must not alter the generated program.
    with LOWER_LOCK:
        rewrite = tilelang.transform.AscendStorageRewrite
        tilelang.transform.AscendStorageRewrite = lambda **_: rewrite(is_npu=True)
        try:
            with tilelang.tvm.transform.PassContext(opt_level=3, config=configs):
                return tilelang.engine.lower(function, target="ascendc", platform="A2")
        finally:
            tilelang.transform.AscendStorageRewrite = rewrite


def compile_kernel(
    function, name, operands, parameters, cache, *, row_capacity=None, grid_limit=64
):
    """Checked pointer ABI, optionally with one explicit bounded row scalar."""
    if row_capacity is not None:
        if type(row_capacity) is not int or not 2 <= row_capacity <= 4096:
            raise ValueError("Invalid dynamic row capacity")
        parameters = dict(parameters, _row_capacity=row_capacity, _grid_limit=grid_limit)
    # The advertised tensors describe physical capacity. Check that the actual
    # compiler geometry agrees, including padded scratch and expert row factors.
    tir = tilelang.tvm.tir
    scalars = [p for p in function.params if p not in function.buffer_map]
    # This Ascend frontend appends free shape variables to its host ABI itself.
    # Appending CUDA's explicit shape parameter would declare m twice.
    for tensor in function.buffer_map.values():
        for dimension in tensor.shape:

            def visit(node):
                if isinstance(node, tir.Var) and not any(node.same_as(v) for v in scalars):
                    scalars.append(node)

            tir.stmt_functor.post_order_visit(dimension, visit)
    if row_capacity is None and scalars:
        raise ValueError("Static AOT has implicit scalar arguments")
    if row_capacity is not None and (len(scalars) != 1 or scalars[0].name != "m"):
        raise ValueError("Dynamic AOT requires exactly the shared m row scalar")
    if row_capacity is not None and not any(
        dimension.same_as(scalars[0])
        for tensor in function.buffer_map.values()
        for dimension in tensor.shape
    ):
        # The old frontend only discovers direct shape variables. Expert-only
        # operators have shapes such as 6*m; expose m explicitly in that case.
        function = tir.PrimFunc(
            list(function.params) + [scalars[0]],
            function.body,
            function.ret_type,
            function.buffer_map,
            function.attrs,
        )
    tensors = [function.buffer_map[p] for p in function.params if p in function.buffer_map]
    if len(tensors) != len(operands):
        raise ValueError("AOT tensor count differs from compiler ABI")
    if any(p["access"] not in ("read", "write", "read_write", "scratch") for p in operands):
        raise ValueError("AOT operand has an invalid access mode")
    if not any(p["access"] in ("write", "read_write") for p in operands) and not (
        name.startswith("private_rows.") and parameters.get("_owner_writes")
    ):
        raise ValueError("AOT kernel requires a declared output")
    analyzer = tilelang.tvm.arith.Analyzer()
    operands = [dict(spec) for spec in operands]
    for tensor, spec in zip(tensors, operands, strict=True):
        shape = [
            int(analyzer.simplify(tir.stmt_functor.substitute(d, {scalars[0]: row_capacity})))
            if scalars
            else int(d)
            for d in tensor.shape
        ]
        dtype = {
            "bf16": "bfloat16",
            "f16": "float16",
            "f32": "float32",
            "i8": "int8",
            "i32": "int32",
            "i64": "int64",
            "u64": "uint64",
        }.get(spec["dtype"], spec["dtype"])
        if str(tensor.dtype) != dtype or math.prod(shape) != math.prod(spec["shape"]):
            raise ValueError(f"AOT physical tensor contract differs: {spec['name']} {shape} {spec}")
        # Model bindings may flatten contiguous dimensions. Publish the actual
        # compiler tensor geometry while retaining the checked physical span.
        spec["shape"] = shape
    coupled = bool(function.attrs and function.attrs.get("orinfer.coupled_resources", False))
    stages = split_resources(function) if row_capacity is not None and not coupled else [function]
    explicit_sync = bool(function.attrs and function.attrs.get("orinfer.explicit_sync", False))
    if explicit_sync and len(stages) != 1:
        raise ValueError("Explicit pipeline synchronization requires a single native stage")
    if row_capacity is not None:
        stages = [bounded_grid(stage, grid_limit) for stage in stages]
    compiler = {
        "tilelang": tilelang.__version__,
        "bisheng": subprocess.check_output(["bisheng", "--version"], text=True).splitlines()[0],
        "cann_home": os.environ["ASCEND_HOME_PATH"],
        "storage_rewrite": "explicit-npu-v1",
        "vector_launch": "paired-aiv-v2",
        "scalar_cache": "entry-invalidate-exit-clean-v1",
        "cube_launch": "independent-aic-v1",
    }
    if explicit_sync:
        compiler["pipeline_sync"] = "explicit-v1"
    if coupled:
        compiler["resource_schedule"] = "coupled-cube-vector-v1"
    module = Path(__file__).with_name(name.split(".")[0] + ".py")
    # Production names identify the source module. Include its inlined math
    # and the shared compiler/file helpers; unrelated operators have independent
    # cache identities. Legacy validation names retain the conservative closure.
    sources = (
        [
            module,
            Path(__file__),
            Path(__file__).with_name("vector_math.py"),
            Path(__file__).with_name("vector_launch.py"),
            Path(__file__).with_name("rows.py"),
            Path(__file__).with_name("scalar_cache.py"),
        ]
        if "." in name and module in SOURCES
        else SOURCES
    )
    identity = {
        "source_sha256": source_hash(
            ROOT, sources + [ROOT / "tools/build/files.py", ROOT / "kernels/model/rows.py"]
        ),
        "compiler": compiler,
        "target": {
            "host_arch": platform.machine(),
            "backend": "ascend",
            "device_arch": "ascend910b",
        },
        "op": name,
        "operands": operands,
        "parameters": parameters,
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    output = cache / key
    if output.exists():
        manifest = json.loads((output / "kernel.json").read_text())
        if any(manifest[k] != v for k, v in identity.items()):
            raise ValueError("AOT cache identity differs")
        if file_hash(output / manifest["library"]["file"]) != manifest["library"]["sha256"]:
            raise ValueError("AOT cache library hash differs")
        return output / "kernel.json"
    configs = {
        tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: not explicit_sync,
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
        tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    }
    if len(stages) > 1:
        configs[tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC] = False
        configs[tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE] = False
    sources, calls, used = [], [], set()
    by_pointer = {f"{p['name']}_handle": i for i, p in enumerate(operands)}
    for index, stage_function in enumerate(stages):
        artifact = lower_a2(stage_function, configs)
        source = lower_independent_vector(
            artifact.kernel_source, grid_limit=grid_limit if row_capacity is not None else None
        )
        source = lower_independent_cube(source)
        if name.startswith("private_rows."):
            from tools.operators.ascend.private_rows import lower_aliases

            source = lower_aliases(source, parameters["_owner_aliases"])
        source = lower_scalar_cache(source)
        prototype = re.search(r'extern "C" void call\(([^)]*)\)', source)
        if prototype is None:
            raise ValueError("TileLang did not emit an AOT call wrapper")
        actual = [p.strip() for p in prototype[1].split(",")]
        if actual[-1] != "aclrtStream stream":
            raise ValueError("AOT call has no final stream argument")
        arguments, row_arguments = [], 0
        for argument in actual[:-1]:
            pointer = re.fullmatch(r"uint8_t\* (\w+)", argument)
            if pointer and pointer[1] in by_pointer:
                slot = by_pointer[pointer[1]]
                used.add(slot)
                arguments.append(f"static_cast<uint8_t*>(args[{slot}])")
            elif row_capacity is not None and argument in ("int64_t m", "int32_t m", "int m"):
                row_arguments += 1
                arguments.append("static_cast<int64_t>(rows)")
            else:
                raise ValueError(f"Unexpected generated AOT argument: {argument}")
        if row_arguments != int(row_capacity is not None):
            raise ValueError("AOT row scalar differs from its declared ABI")
        kernels = re.findall(r'extern "C" __global__ __aicore__ void (\w+)\(', source)
        if len(kernels) != 1:
            raise ValueError("Expected one device kernel per AOT stage")
        # Device traces identify the operator and exact compiled template.
        # The pointer-table host ABI retains its versioned public symbols.
        operator = re.sub(r"[^a-zA-Z0-9_]", "_", name)
        source = source.replace(kernels[0], f"orinfer_{operator}_{key[:12]}_s{index}")
        if len(stages) > 1:
            source = re.sub(r"\bcall\b", f"orinfer_stage{index}_call", source)
        entry = f"orinfer_stage{index}_call" if len(stages) > 1 else "call"
        calls.append(f"{entry}({', '.join(arguments)}, stream);")
        sources.append(source)
    if used != set(range(len(operands))):
        raise ValueError("AOT pipeline contains an unused declared operand")
    count = len(operands)
    headers = set()

    def header(match):
        line = match[0]
        if line in headers:
            return ""
        headers.add(line)
        return line

    source = re.sub(r"(?m)^#include[^\n]*\n", header, "\n".join(sources))
    version = 2 if row_capacity is not None else 1
    row_parameter = ", uint64_t rows" if row_capacity is not None else ""
    row_check = (
        f"if (rows == 0 || rows > {row_capacity}) return ACL_ERROR_INVALID_PARAM;"
        if row_capacity is not None
        else ""
    )
    calls = "\n    ".join(
        call
        + "\n    { int32_t status = aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL); if (status) return status; }"
        for call in calls
    )
    source += f"""
extern "C" const char* orinfer_ascend_kernel_spec_v1() {{
    return {json.dumps(json.dumps(identity, sort_keys=True))};
}}
extern "C" int32_t orinfer_ascend_kernel_v{version}(void* const* args, uint64_t count{row_parameter}, aclrtStream stream) {{
    if (!args || count != {count} || !stream) return ACL_ERROR_INVALID_PARAM;
    {row_check}
    for (uint64_t i = 0; i < count; ++i) if (!args[i]) return ACL_ERROR_INVALID_PARAM;
    {calls}
    return aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL);
}}
"""
    generator = LibraryGenerator(target="ascendc", platform="A2")
    generator.update_lib_code(source)
    generator.compile_lib(timeout=120)
    with tempfile.TemporaryDirectory(prefix="aot-", dir=cache) as temporary:
        stage = Path(temporary) / "package"
        stage.mkdir()
        shutil.copyfile(generator.get_lib_path(), stage / "kernel.so")
        (stage / "kernel.cpp").write_text(source)
        manifest = {
            "schema_version": 1,
            **identity,
            "library": {"file": "kernel.so", "sha256": file_hash(stage / "kernel.so")},
        }
        atomic_json(stage / "kernel.json", manifest)
        stage.rename(output)
    return output / "kernel.json"


def operand(name, dtype, shape, access="read"):
    return {"name": name, "dtype": dtype, "shape": list(shape), "access": access}
