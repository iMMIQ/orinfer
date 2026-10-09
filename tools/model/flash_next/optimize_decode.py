"""Rebuild small Flash Next verification tiles without changing weight payloads.

Reads native safetensors weights and standalone/bundled execution assets. The
new immutable package remains bundled, with the same registered execution ABI.
"""

import argparse
import copy
import hashlib
from pathlib import Path
import re
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.model.compact import compact_package
from tools.model.publication import (
    AssetReader,
    atomic_model,
    clone_model,
    file_hash,
    load_model,
    source_path,
    write_json,
)
from tools.model.flash_next.kernel_policy import expert_tile_config, router_tile
from tools.operators.abi import evaluate, parse_host, validate_parameter_count


def pointer_bindings(kernel, host):
    """Retain the original offsets; exported argument order is authoritative."""
    if len(kernel["args"]) != len(host["ordered_arguments"]):
        raise ValueError("Original host ABI differs from the native package")
    pointers = {}
    for actual, argument in zip(kernel["args"], host["ordered_arguments"]):
        value = argument["value"]
        if argument["ctype"] != "ctypes.c_void_p" or not value.endswith(".data_ptr()"):
            raise ValueError("Small static projection must use pointer-only ABI")
        if actual["kind"] not in ("buffer", "buffer_slice"):
            raise ValueError("Invalid projection pointer")
        pointers[value.removesuffix(".data_ptr()")] = copy.deepcopy(actual)
    return pointers


def replace_binding(original, host, assets, pointers):
    launch = host["launch_expressions"]
    args = []
    for argument in host["ordered_arguments"]:
        value = argument["value"]
        if argument["ctype"] != "ctypes.c_void_p" or not value.endswith(".data_ptr()"):
            raise ValueError("Unexpected rebuilt projection ABI")
        args.append(copy.deepcopy(pointers[value.removesuffix(".data_ptr()")]))
    return dict(
        name=original["name"],
        **assets,
        symbol=host["symbol"],
        grid=[evaluate(launch["gridDim" + axis], {}) for axis in "XYZ"],
        block=[evaluate(launch["blockDim" + axis], {}) for axis in "XYZ"],
        shared_memory_bytes=evaluate(launch["sharedMemBytes"], {}),
        cooperative=False,
        args=args,
    )


def upgrade(model, destination, report):
    import json
    import numpy as np
    from safetensors import safe_open
    from kernels.model.flash_next import dense_projection
    from kernels.model.integer_vq import integer_vq_grouped
    from tools.operators.common import configure, export_kernel
    from tools.quantization.vq import e8p_sign_table

    configure()
    data, origin, package = load_model(model)
    if data["architecture"] != "flash_next" or data["compute_policy"] != "int8_quality":
        raise ValueError("Expected a Flash Next INT8 quality package")
    buffers = {b["name"]: b for b in data["metadata"]["buffers"]}
    text = json.loads((model / "config.json").read_text())["text_config"]
    experts, hidden = text["num_experts"], text["hidden_size"]
    if experts != 512 or hidden != 2560:
        raise ValueError("Unsupported Flash Next geometry")
    operator = clone_model(model, destination, origin)
    reader = AssetReader(origin)
    index = json.loads((model / "cache/weights/model.safetensors.index.json").read_text())
    exports, host_cache, tables = {}, {}, {}

    def host(kernel):
        identity = kernel["host_abi"]["sha256"]
        if identity not in host_cache:
            (parsed,) = parse_host(reader.read(kernel["host_abi"]).decode())
            host_cache[identity] = parsed
        return host_cache[identity]

    def compile_kernel(key, factory):
        if key not in exports:
            folder = operator / "decode-aot" / key
            export_kernel(factory(), folder)
            (parsed,) = parse_host((folder / "host.txt").read_text())
            validate_parameter_count((folder / "kernel.cu").read_text(), parsed)
            assets = {
                field: dict(
                    file=str((folder / filename).relative_to(operator)),
                    sha256=file_hash(folder / filename),
                )
                for field, filename in (
                    ("module", "kernel.cubin"),
                    ("source", "kernel.cu"),
                    ("host_abi", "host.txt"),
                )
            }
            exports[key] = parsed, assets
        return exports[key]

    def short_table(original):
        name = original["name"]
        if name not in tables:
            b = buffers[name]
            tensor = b["data"]["tensor"]
            path = source_path((model / "cache/weights").resolve(), index["weight_map"][tensor])
            with safe_open(path, framework="np") as weights:
                raw = weights.get_tensor(tensor)
            if (
                b["shape"] != [1, 256, 2]
                or hashlib.sha256(raw.tobytes()).hexdigest() != b["data"]["sha256"]
            ):
                raise ValueError("Invalid E8P basis identity")
            expanded = e8p_sign_table(raw.view(np.int8).reshape(256, 8))
            digest = hashlib.sha256(expanded.tobytes()).hexdigest()
            candidates = [
                n
                for n, v in buffers.items()
                if (v.get("data") or {}).get("sha256") == digest
                and v["dtype"] == "u32"
                and v["shape"] == [1, 512, 4]
            ]
            if not candidates:
                raise ValueError("The exact E8P sign table is missing from model weights")
            tables[name] = dict(kind="buffer", name=candidates[0])
        return tables[name]

    updated = dict(experts=0, routers=0)
    for i, kernel in enumerate(package["kernels"]):
        program = kernel["name"].split("/")[0]
        match = re.fullmatch(r"(?:verify|flash_batch|mtp_warm)_m([1-8])", program)
        rows = int(match[1]) if match else 1 if program == "decode" else None
        if rows is None:
            continue
        weights = [
            buffers[a["name"]]
            for a in kernel["args"]
            if a.get("name") in buffers and buffers[a["name"]].get("data")
        ]
        packed = next((w for w in weights if w["name"].endswith("_exps.weight_packed")), None)
        router = next((w for w in weights if w["name"].endswith("ffn_gate_inp.weight_value")), None)
        if packed is not None and rows >= 2:
            e, groups, n, words = packed["shape"]
            if e != experts or words != 16 or n not in (1280, hidden):
                raise ValueError("Unexpected expert packing")
            k = groups * 128
            tiles = (rows * 10 + 15) // 16 + min(experts, rows * 10)
            if kernel["grid"][0] != tiles:
                raise ValueError("Unexpected static routed tile capacity")
            pointers = pointer_bindings(kernel, host(kernel))
            plan = expert_tile_config(rows)
            if not pointers["Book"]["name"].endswith("short_table"):
                pointers["Book"] = short_table(pointers["Book"])
            rebuilt, assets = compile_kernel(
                f"expert-m{rows}-n{n}-k{k}",
                lambda rows=rows, tiles=tiles, n=n, k=k, plan=plan: integer_vq_grouped(
                    rows * 10, experts, tiles, n, k, kind="e8p", shared_table=True, **plan
                ),
            )
            package["kernels"][i] = replace_binding(kernel, rebuilt, assets, pointers)
            updated["experts"] += 1
        elif router is not None:
            if router["dtype"] != "bf16" or router["shape"] != [experts, hidden]:
                raise ValueError("Unexpected router weight")
            pointers = pointer_bindings(kernel, host(kernel))
            bn = router_tile(rows, experts, hidden, experts, hidden)
            rebuilt, assets = compile_kernel(
                f"router-m{rows}",
                lambda rows=rows, bn=bn: dense_projection(
                    rows, experts, hidden, "bfloat16", "float32", block_n=bn
                ),
            )
            package["kernels"][i] = replace_binding(kernel, rebuilt, assets, pointers)
            updated["routers"] += 1
    if not all(updated.values()):
        raise ValueError("Expected verification experts and router bindings")
    digest = compact_package(operator, package, destination / "cache/packages")
    shutil.rmtree(operator)
    data["execution_package"] = digest
    write_json(destination / "cache/model.json", data)
    write_json(
        report / "upgrade.json",
        dict(
            execution_package=digest,
            replaced=updated,
            compiled_variants=len(exports),
            weight_bytes=data["metadata"]["weight_bytes"],
            weight_payloads_unchanged=True,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine", type=Path)
    args = parser.parse_args()
    with atomic_model(args.model_output, args.engine) as staging:
        upgrade(args.model.resolve(strict=True), staging, args.output)
    print("DECODE PACKAGE READY", args.model_output, flush=True)


if __name__ == "__main__":
    main()
