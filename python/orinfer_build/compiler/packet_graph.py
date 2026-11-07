"""Preserve model arithmetic while moving private operands into a fixed table."""

from orinfer_build.common.paths import ROOT, work_path


import hashlib
import math
from pathlib import Path

from orinfer_build.common.files import source_hash, write_json

TABLE = "QwenMtpPacketPointers"


def build(b):
    if not b.mtp and not hasattr(b, "packet_spec"):
        return
    from kernels.cuda.model.independent_rows import indirect_parameters
    from kernels.cuda.model.packet_io import copy_rows
    from orinfer_build.compiler.compile import compile_function, instantiate
    from orinfer_build.models.qwen35.parallel import LANES

    root = ROOT
    b.aot.identity["mtp_packet_source"] = source_hash(
        root,
        [
            Path(__file__),
            root / "kernels/cuda/model/independent_rows.py",
            root / "kernels/cuda/model/packet_io.py",
            root / "python/orinfer_build/compiler/compile.py",
            root / "kernels/cuda/model/retained.py",
            root / "python/orinfer_build/compiler/native_packets.py",
        ],
    )
    columns = sorted(name for name, scope in b.weights.scopes.items() if scope == "sequence")
    indices = {name: i for i, name in enumerate(columns)}
    b.weights.fixed(TABLE, "u64", [LANES * len(columns)])
    originals = {k["name"]: k for k in b.aot.kernels}
    raw = {}
    aliases = {}
    scalar_arguments = {}
    resource_changes = {}
    for name, recipe in list(b.aot.recipes.items()):
        if not getattr(b, "generic_packets", False) and (
            not name.startswith("qwen_rows/") or any(s in name for s in ("/vision/", "/audio/"))
        ):
            continue
        key, make, _, rows, dynamic, divisor, multiplier = recipe
        kernel = originals[name]
        pointers = b.aot.compiler.pointers(kernel)
        private = {role: arg["name"] for role, arg in pointers.items() if arg["name"] in indices}
        if not private:
            continue
        offsets = {role: pointers[role].get("offset", 0) for role in private}
        # Geometry and private column placement determine code; tensor values do not.
        identity = hashlib.sha256(
            repr((key, len(columns), sorted(private), sorted(offsets.items()))).encode()
        ).hexdigest()
        alias = "qwen_rows/packet_kernel/" + hashlib.sha256(name.encode()).hexdigest()

        def wrapped(key=key, make=make, private=private, offsets=offsets):
            if key not in raw:
                raw[key] = make()
            original = raw[key]
            fn, _ = indirect_parameters(original.prim_func, indices, private, offsets)
            return compile_function(
                fn, pass_configs=original.pass_configs, compile_flags=original.compile_flags
            )

        bindings = {role: arg for role, arg in pointers.items() if role not in private}
        bindings["Pointers"] = dict(kind="buffer", name=TABLE)
        constants = {"column_" + role: indices[target] for role, target in private.items()}
        if getattr(b, "generic_packets", False):
            constants.update(
                (host["value"], arg["value"])
                for arg, host in zip(
                    kernel["args"], b.aot.compiler.host(kernel)["ordered_arguments"]
                )
                if arg["kind"] == "i32" and host["value"] not in ("m", "M", "rows", "batch")
            )
        b.aot.emit(
            alias,
            "mtp-packet-" + identity,
            wrapped,
            bindings,
            rows,
            dynamic=dynamic,
            row_divisor=divisor,
            row_multiplier=multiplier,
            constants=constants,
        )
        replacement = b.aot.kernels[-1]
        if getattr(b, "generic_packets", False):
            replacement["cooperative"] = kernel["cooperative"]
        if any(replacement[field] != kernel[field] for field in ("grid", "block", "cooperative")):
            raise ValueError(
                f"MTP indirection changes the original launch geometry: {name}: "
                + repr(
                    {
                        field: (kernel[field], replacement[field])
                        for field in ("grid", "block", "cooperative")
                        if kernel[field] != replacement[field]
                    }
                )
            )
        if replacement["shared_memory_bytes"] != kernel["shared_memory_bytes"]:
            resource_changes[name] = dict(
                original=kernel["shared_memory_bytes"], wrapped=replacement["shared_memory_bytes"]
            )
        original_host = b.aot.compiler.host(kernel)
        replacement_host = b.aot.compiler.host(replacement)
        scalar_positions = {
            arg["value"]: i
            for i, arg in enumerate(replacement_host["ordered_arguments"])
            if arg["ctype"] in ("ctypes.c_int32", "c_int32")
        }
        scalar_arguments[name] = []
        for i, arg in enumerate(original_host["ordered_arguments"]):
            if arg["ctype"] in ("ctypes.c_int32", "c_int32"):
                if arg["value"] not in scalar_positions:
                    raise ValueError(f"MTP indirection changes scalar ABI: {name}")
                scalar_arguments[name].append((i, scalar_positions[arg["value"]]))
        if getattr(b, "generic_packets", False) and name in b.original_dynamic:
            import copy

            original_launch = copy.deepcopy(b.original_dynamic[name])
            original_launch["name"] = alias
            remap = dict(scalar_arguments[name])
            for arg in original_launch["arguments"]:
                arg["index"] = remap[arg["index"]]
            b.aot.dynamic = [d for d in b.aot.dynamic if d["name"] != alias]
            b.aot.dynamic.append(original_launch)
        aliases[name] = alias

    copy_specs = b.packet_copies
    copies = []
    for source, destination, width in copy_specs:
        private = {
            role: name
            for role, name in [("Source", source), ("Destination", destination)]
            if name in indices
        }
        if not private:
            continue
        fn, inputs = indirect_parameters(instantiate(copy_rows, width), indices, private)
        bindings = {
            role: dict(kind="buffer", name=name)
            for role, name in [("Source", source), ("Destination", destination)]
            if role in inputs
        }
        bindings["Pointers"] = dict(kind="buffer", name=TABLE)
        name = f"qwen_rows/packet_copy/{source}/{destination}"
        identity = hashlib.sha256(str(fn).encode()).hexdigest()
        sizes = [
            math.prod(b.weights.buffers[n]["shape"])
            * {
                "u8": 1,
                "i8": 1,
                "u16": 2,
                "f16": 2,
                "bf16": 2,
                "u32": 4,
                "i32": 4,
                "f32": 4,
                "u64": 8,
                "i64": 8,
                "f64": 8,
            }[b.weights.buffers[n]["dtype"]]
            for n in (source, destination)
        ]
        capacity = max(2, min(size // width for size in sizes))
        b.aot.emit(
            name,
            "mtp-copy-" + identity,
            lambda fn=fn: compile_function(fn),
            bindings,
            capacity,
            constants={"column_" + role: indices[target] for role, target in private.items()},
        )
        copies.append(dict(source=source, destination=destination, kernel=name, row_bytes=width))
    write_json(b.aot.output / "mtp-packet-resources.json", resource_changes)
    table = dict(
        buffer=TABLE,
        columns=columns,
        kernels=aliases,
        scalar_arguments=scalar_arguments,
        copies=copies,
    )
    if hasattr(b, "packet_spec"):
        b.packet_spec = table
    else:
        b.mtp["packet_table"] = table


def inspect(model):
    """Find owner-bound operands in the registered native construction."""
    from orinfer_build.compiler.native_packets import Packets, profiles

    copy_sizes = {}
    private_names = set()
    with Packets(model, indirect=False) as packets:
        for operation, segments in profiles(packets.manifest):
            try:
                commands, _ = packets.plan(segments, operation)
            except ValueError:
                continue
            for command in commands:
                if command["sequence"] is None:
                    continue
                if command["kind"] == 0:
                    private_names.add(command["name"])
                elif command["kind"] == 1:
                    pair = (command["name"], command["destination"])
                    if any(
                        name in packets.data["buffer_scopes"]
                        and packets.data["buffer_scopes"][name] == "sequence"
                        for name in pair
                    ):
                        copy_sizes.setdefault(pair, []).append(command["bytes"])
                else:
                    raise ValueError("Private packet zero needs a declared TileLang implementation")
    return private_names, copy_sizes


def compile_package(data, package, manifest, origin, operator, output, cache, layout):
    """Compile the common table after a model has selected its arithmetic."""
    from contextlib import nullcontext
    from functools import reduce
    import copy
    import types
    from kernels.cuda.model.retained import primitive as retained
    from orinfer_build.models.qwen35.rebuild_mtp import ExistingWeights
    from orinfer_build.models.qwen35.compiler import Aot
    from orinfer_build.compiler.host_compiler import Compiler

    private_names, copy_sizes = layout
    if not private_names:
        raise ValueError("Native packet inspection found no private construction")
    b = types.SimpleNamespace(generic_packets=True, packet_spec={})
    b.weights = ExistingWeights(manifest, data["buffer_scopes"])
    b.weights.buffers.update(
        {buf["name"]: copy.deepcopy(buf) for buf in data["metadata"]["buffers"]}
    )
    b.mtp = copy.deepcopy(data["metadata"].get("mtp"))
    b.aot = Aot(operator, output, cache, b.weights.buffers, compile_lock=nullcontext)
    b.aot.compiler = Compiler(origin, operator, output, b.weights.buffers)
    old_table = data["metadata"].pop("packet_table", None)
    if b.mtp:
        old_table = b.mtp.pop("packet_table", None) or old_table
    old_names = set((old_table or {}).get("kernels", {}).values())
    old_names.update(c["kernel"] for c in (old_table or {}).get("copies", []))
    if "plan_kernel_contracts" in package:
        package["plan_kernel_contracts"] = [
            name for name in package["plan_kernel_contracts"] if name not in old_names
        ]
    package["kernels"] = [k for k in package["kernels"] if k["name"] not in old_names]
    package["dynamic_batch_kernels"] = [
        d for d in package["dynamic_batch_kernels"] if d["name"] not in old_names
    ]
    if b.mtp:
        data["metadata"]["mtp"] = b.mtp
    b.aot.kernels = copy.deepcopy(package["kernels"])
    b.aot.dynamic = copy.deepcopy(package["dynamic_batch_kernels"])
    dynamic = {k["name"]: k for k in package["dynamic_batch_kernels"]}
    b.original_dynamic = dynamic
    columns = {n for n, s in b.weights.scopes.items() if s == "sequence"}

    for kernel in package["kernels"]:
        if kernel["name"].startswith(("qwen_rows/packet_kernel/", "qwen_rows/packet_copy/")):
            continue
        if "/vision/" in kernel["name"] or "/audio/" in kernel["name"]:
            continue
        pointers = b.aot.compiler.pointers(kernel)
        if not any(v["name"] in columns for v in pointers.values()):
            continue
        key = "retained-" + kernel["source"]["sha256"] + "-" + kernel["host_abi"]["sha256"]

        def make(kernel=kernel):
            function = retained(
                b.aot.compiler.read(kernel["source"]).decode(),
                kernel,
                b.aot.compiler.host(kernel),
                b.weights.buffers,
            )
            return types.SimpleNamespace(
                prim_func=function,
                compile_flags=[],
                pass_configs={
                    "tl.disable_thread_storage_sync": True,
                    "tl.disable_shared_memory_reuse": True,
                },
            )

        b.aot.recipes[kernel["name"]] = (
            key,
            make,
            {},
            next(
                (
                    arg["value"]
                    for arg, host in zip(
                        kernel["args"], b.aot.compiler.host(kernel)["ordered_arguments"]
                    )
                    if arg["kind"] == "i32" and host["value"] in ("m", "M", "rows", "batch")
                ),
                1,
            ),
            kernel["name"] in dynamic,
            1,
            1,
        )
    b.packet_copies = [
        (source, destination, reduce(math.gcd, sizes))
        for (source, destination), sizes in sorted(copy_sizes.items())
    ]
    build(b)
    missing = private_names - b.packet_spec["kernels"].keys()
    # A private invocation may contain only shared operands; the native
    # binder checks and clears those owner annotations itself.
    for name in missing:
        kernel = next(k for k in package["kernels"] if k["name"] == name)
        if any(a.get("name") in columns for a in kernel["args"]):
            raise ValueError(f"Unwrapped private packet kernel: {name}")
    data["metadata"]["packet_table"] = b.packet_spec
    data["metadata"]["buffers"] = list(b.weights.buffers.values())
    data["buffer_scopes"] = b.weights.scopes
    package["kernels"], package["dynamic_batch_kernels"] = b.aot.kernels, b.aot.dynamic
    package["buffer_contracts"] = [
        {k: v for k, v in buf.items() if k != "data"} for buf in b.weights.buffers.values()
    ]
    package["toolchain"]["packet_graph_source"] = b.aot.identity["mtp_packet_source"]
    return b.packet_spec


def rebuild(args):
    """Add a common operand table while preserving each model's published body."""
    from orinfer_build.packaging.publication import atomic_model, clone_model, load_model
    from orinfer_build.packaging.compact import compact_package

    manifest = {}
    data, origin, package = load_model(args.model, plan=manifest)
    layout = inspect(args.model)
    with atomic_model(args.target) as stage:
        operator = clone_model(args.model, stage, origin)
        table = compile_package(
            data, package, manifest, origin, operator, args.output, args.cache, layout
        )
        data["execution_package"] = compact_package(operator, package, stage / "cache/packages")
        write_json(stage / "cache/model.json", data)
    write_json(
        args.output / "result.json",
        dict(
            model=str(args.target),
            complete=True,
            execution_package=data["execution_package"],
            private_kernels=len(layout[0]),
            wrapped_kernels=len(table["kernels"]),
            copy_pairs=len(layout[1]),
        ),
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=work_path("cache", "build", "packet-graph"))
    rebuild(parser.parse_args())


if __name__ == "__main__":
    main()
