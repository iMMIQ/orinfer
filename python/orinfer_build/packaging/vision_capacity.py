"""Raise image-feature capacity with matching overlay kernels and native bindings.

Weight payloads and text row capacities stay fixed. Runtime sequence allocation
uses each request's actual visual-token count, within the new package ceiling.
"""

import argparse
import copy
from pathlib import Path
import shutil

from orinfer_build.common.files import write_json
from orinfer_build.packaging.publication import atomic_model, clone_model, load_model


def capacity_metadata(data, package, capacity):
    metadata = data["metadata"]
    vision = metadata.get("vision")
    if not vision or data["architecture"] not in ("qwen3_5", "flash_next"):
        raise ValueError("Expected a native Qwen image model")
    if (
        type(capacity) is not int
        or not vision["max_features"] < capacity <= metadata["max_context"]
    ):
        raise ValueError("New feature capacity must exceed the current limit and fit the context")
    name = vision["features"]
    if data["buffer_scopes"].get(name) != "sequence":
        raise ValueError("Image features must be request-owned")
    old_shape = [vision["max_features"], vision["hidden"]]
    # A package and its model descriptor must agree before publication.
    for buffers in (metadata["buffers"], package["buffer_contracts"]):
        buffer = next(b for b in buffers if b["name"] == name)
        if buffer["dtype"] != "f16" or buffer["shape"] != old_shape:
            raise ValueError("Unexpected image-feature buffer contract")
    vision["max_features"] = capacity
    for buffers in (metadata["buffers"], package["buffer_contracts"]):
        next(b for b in buffers if b["name"] == name)["shape"] = [capacity, vision["hidden"]]


def rebuild_overlays(data, package, manifest, compiler, capacity):
    from kernels.cuda.model import decode_packet
    from kernels.shared.rows import static_rows
    from kernels.cuda.vision.features import overlay
    from orinfer_build.compiler.compile import instantiate
    from orinfer_build.compiler.private_rows import wrap_private
    from orinfer_build.models.qwen35.packet import COLUMNS

    # Work on independent objects until every binding has compiled successfully.
    updated, updated_package = copy.deepcopy(data), copy.deepcopy(package)
    capacity_metadata(updated, updated_package, capacity)
    metadata = updated["metadata"]
    vision = metadata["vision"]
    dynamics = {k["name"]: k for k in package["dynamic_batch_kernels"]}
    changed = []
    replacements = {}
    launches = {}
    for old in package["kernels"]:
        name = old["name"]
        direct = any(a.get("name") == vision["features"] for a in old["args"])
        packet = name in ("qwen_rows/packet/overlay", "flash_decode/packet/inputs")
        if not direct and not packet:
            continue
        pointers = compiler.pointers(old)
        dynamic = dynamics.get(name)
        rows = dynamic["capacity"] if dynamic else old["grid"][0]
        # Legacy verification entries may publish a constant launch template
        # without a row scalar. Preserve that ABI as well as truly dynamic ones.
        symbolic = any(arg["kind"] == "i32" for arg in old["args"])
        if set(pointers) == {"Embedding", "Features", "Index", "Position"}:

            def build(rows=rows, symbolic=symbolic):
                return instantiate(
                    overlay,
                    rows,
                    vision["hidden"],
                    metadata["max_context"],
                    capacity,
                    dynamic_rows=symbolic,
                )
        elif name == "qwen_rows/packet/overlay":
            function = static_rows(
                instantiate(overlay, 1, vision["hidden"], metadata["max_context"], capacity)
            )
            modes = {
                "Embedding": ("row", pointers["Embedding"]["name"]),
                "Features": ("private", "Features"),
                "Index": ("private", "FeatureIndex"),
                "Position": ("private", "Step"),
            }
            originals = dict(
                Embedding=pointers["Embedding"],
                Features=dict(kind="buffer", name=vision["features"]),
                Index=dict(kind="buffer", name=vision["feature_index"]),
                Position=dict(kind="buffer", name=metadata["position"]),
            )
            table = pointers["Pointers"]
            function, pointers = wrap_private(
                function,
                rows,
                modes,
                COLUMNS,
                originals,
                table["name"],
                table_offset=table.get("offset", 0),
            )

            def build(function=function):
                return function

            symbolic = False
        elif name == "flash_decode/packet/inputs" and set(pointers) == {
            "Pointers",
            "Embedding",
            "Ple",
        }:
            layout = manifest["batch_layout"]
            stride = len(layout["layers"]) * len(layout["state_columns"])

            def build():
                return instantiate(
                    decode_packet.inputs, stride, metadata["max_context"], capacity, True
                )

            symbolic = False
        else:
            raise ValueError(f"Unsupported visual-feature consumer {name}")
        key = "vision-capacity-" + name.replace("/", "-") + f"-{capacity}"
        item, launch = compiler.emit(
            old, name, pointers, rows, build=build, key=key, symbolic=symbolic
        )
        # The published row/grid/scalar contract remains identical.
        if any(
            item[field] != old[field]
            for field in ("grid", "block", "shared_memory_bytes", "cooperative", "args")
        ):
            raise ValueError(f"Visual capacity changed row/ABI contract: {name}")
        if dynamic is not None and launch != dynamic:
            raise ValueError(f"Visual capacity changed dynamic contract: {name}")
        replacements[name] = item
        if dynamic:
            launches[name] = launch
        changed.append(name)
    if (
        not changed
        or data["architecture"] == "qwen3_5"
        and "qwen_rows/packet/overlay" not in changed
    ):
        raise ValueError("Missing visual overlay bindings")
    updated_package["kernels"] = [replacements.get(k["name"], k) for k in package["kernels"]]
    updated_package["dynamic_batch_kernels"] = [
        launches.get(k["name"], k) for k in package["dynamic_batch_kernels"]
    ]
    data.clear()
    data.update(updated)
    package.clear()
    package.update(updated_package)
    return changed


def main():
    from orinfer_build.packaging.compact import compact_package
    from orinfer_build.compiler.host_compiler import Compiler
    from orinfer_build.compiler.cuda.common import configure

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "model-output", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--features", type=int, help="Defaults to model context capacity")
    args = parser.parse_args()
    configure()
    manifest = {}
    data, origin, package = load_model(args.model, plan=manifest)
    capacity = data["metadata"]["max_context"] if args.features is None else args.features
    with atomic_model(args.model_output) as staging:
        root = clone_model(args.model, staging, origin)
        compiler = Compiler(
            origin, root, args.output, {b["name"]: b for b in data["metadata"]["buffers"]}
        )
        changed = rebuild_overlays(data, package, manifest, compiler, capacity)
        digest = compact_package(root, package, staging / "cache/packages")
        shutil.rmtree(root)
        data["execution_package"] = digest
        write_json(staging / "cache/model.json", data)
        write_json(
            args.output / "capacity.json",
            dict(features=capacity, bindings=changed, execution_package=digest),
        )
    print("COMPLETE", digest, flush=True)


if __name__ == "__main__":
    main()
