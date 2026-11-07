"""Shared host-ABI lowering and AOT emission for the native model builder."""

import copy
import re

from orinfer_build.packaging.publication import AssetReader, link_or_copy
from orinfer_build.common.files import file_hash
from orinfer_build.contracts.row_contract import contract
from orinfer_build.contracts.abi import evaluate, parse_host, validate_parameter_count


def private_roles(source, columns, row_stride=None):
    """Recover certified pointer-table columns/byte views from exported wrappers."""
    loads = dict(re.findall(r"\b(\w+) = Pointers\[([^;]+)\];", source))
    roles, offsets = {}, {}
    for role, variable, offset in re.findall(
        r"void\* (\w+)_row = orin_row_pointer\(\(?([A-Za-z_][A-Za-z_0-9]*)"
        r"(?: \+ \(uint64_t\)(\d+))?\)?\);",
        source,
    ):
        expression = loads[variable]
        index = re.search(r"\+ \(int64_t\)(\d+)\)$", expression)
        if index is None and " + " in expression:
            raise ValueError("Unknown private-state column expression")
        if row_stride is not None and set(
            map(int, re.findall(r"\* \(int64_t\)(\d+)", expression))
        ) != {row_stride}:
            raise ValueError("Private-state row stride differs from model layout")
        column = int(index[1]) if index else 0
        if column >= len(columns):
            raise ValueError("Invalid private-state column")
        roles[role] = ("private", columns[column])
        offsets[role] = int(offset or 0)
    return roles, offsets


def pointer_bindings(kernel, host, *, allow_scalars=False):
    """Retain the original offsets; exported argument order is authoritative."""
    if len(kernel["args"]) != len(host["ordered_arguments"]):
        raise ValueError("Original host ABI differs from the native package")
    pointers = {}
    for actual, argument in zip(kernel["args"], host["ordered_arguments"]):
        value = argument["value"]
        if allow_scalars and argument["ctype"] == "ctypes.c_int32" and actual["kind"] == "i32":
            continue
        if argument["ctype"] != "ctypes.c_void_p" or not value.endswith(".data_ptr()"):
            raise ValueError("Small static projection must use pointer-only ABI")
        if actual["kind"] not in ("buffer", "buffer_slice"):
            raise ValueError("Invalid projection pointer")
        pointers[value.removesuffix(".data_ptr()")] = copy.deepcopy(actual)
    return pointers


def replace_binding(original, host, assets, pointers, *, dimensions=None):
    launch = host["launch_expressions"]
    args = []
    for argument in host["ordered_arguments"]:
        value = argument["value"]
        if dimensions is not None and argument["ctype"] == "ctypes.c_int32":
            args.append(dict(kind="i32", value=evaluate(value, dimensions)))
            continue
        if argument["ctype"] != "ctypes.c_void_p" or not value.endswith(".data_ptr()"):
            raise ValueError("Unexpected rebuilt projection ABI")
        args.append(copy.deepcopy(pointers[value.removesuffix(".data_ptr()")]))
    return dict(
        name=original["name"],
        **assets,
        symbol=host["symbol"],
        grid=[evaluate(launch["gridDim" + axis], dimensions or {}) for axis in "XYZ"],
        block=[evaluate(launch["blockDim" + axis], dimensions or {}) for axis in "XYZ"],
        shared_memory_bytes=evaluate(launch["sharedMemBytes"], dimensions or {}),
        cooperative=False,
        args=args,
    )


class Compiler:
    def __init__(self, origin, root, output, buffers):
        self.reader, self.root, self.output, self.buffers = (
            AssetReader(origin),
            root,
            output,
            buffers,
        )
        self.hosts, self.compiled = {}, {}

    def read(self, asset):
        reader = (
            self.reader if (self.reader.base / asset["file"]).exists() else AssetReader(self.root)
        )
        return reader.read(asset)

    def host(self, kernel):
        identity = kernel["host_abi"]["sha256"]
        if identity not in self.hosts:
            self.hosts[identity] = parse_host(self.read(kernel["host_abi"]).decode())[0]
        return self.hosts[identity]

    def pointers(self, kernel):
        return pointer_bindings(kernel, self.host(kernel), allow_scalars=True)

    def emit(
        self,
        old,
        name,
        bindings,
        rows,
        *,
        build=None,
        key=None,
        multiplier=1,
        symbolic=True,
        constants=None,
        kernel=None,
        artifact=None,
    ):
        if build is None and kernel is None and artifact is None:
            host, assets = self.host(old), {f: old[f] for f in ("module", "source", "host_abi")}
            source = self.read(old["source"]).decode()
        else:
            from kernels.shared.rows import explicit_rows
            import tilelang
            from orinfer_build.compiler.cuda.common import export_kernel

            if key not in self.compiled:
                print("COMPILE", key, flush=True)
                passes = {}
                if key.startswith("causal-gdn-"):
                    passes = {
                        "tl.disable_thread_storage_sync": True,
                        "tl.disable_shared_memory_reuse": True,
                    }
                if key.startswith("expert-"):
                    from orinfer_build.tuning.expert_tuning import (
                        EXACT_EXPERT_PASSES as EXACT_PASSES,
                    )

                    passes = EXACT_PASSES
                if artifact is None and kernel is None:
                    fn = explicit_rows(build()) if symbolic else build()
                    kernel = tilelang.compile(
                        fn,
                        out_idx=[],
                        execution_backend="nvrtc",
                        target={"kind": "cuda", "arch": "sm_87"},
                        pass_configs=passes,
                    )
                folder = artifact or self.output / "aot" / key
                if artifact is None:
                    export_kernel(kernel, folder)
                assets = {}
                for field, filename in [
                    ("module", "kernel.cubin"),
                    ("source", "kernel.cu"),
                    ("host_abi", "host.txt"),
                ]:
                    target = self.root / "shared-rows" / key / filename
                    target.parent.mkdir(parents=True, exist_ok=True)
                    link_or_copy(folder / filename, target)
                    assets[field] = dict(
                        file=str(target.relative_to(self.root)), sha256=file_hash(target)
                    )
                host = parse_host((folder / "host.txt").read_text())[0]
                source = (folder / "kernel.cu").read_text()
                validate_parameter_count(source, host)
                self.compiled[key] = host, assets, source
            host, assets, source = self.compiled[key]
        self.hosts[assets["host_abi"]["sha256"]] = host
        result = replace_binding(
            dict(old, name=name),
            host,
            assets,
            bindings,
            dimensions=dict(
                m=rows * multiplier,
                rows=rows * multiplier,
                batch=rows * multiplier,
                **(constants or {}),
            ),
        )
        prototype = re.search(r"\b" + re.escape(result["symbol"]) + r"\s*\(([^)]*)\)", source)
        if prototype is None or len(prototype[1].split(",")) != len(result["args"]):
            raise ValueError("Missing or inconsistent shared-row CUDA argument access")
        result["readonly_arguments"] = [
            "*" not in value or bool(re.search(r"\bconst\b", value))
            for value in prototype[1].split(",")
        ]
        launch = contract(result, host, rows, multiplier, constants)
        return result, launch
