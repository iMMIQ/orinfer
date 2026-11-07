"""Rebuild a native Flash Next execution package with resumable real-weight tuning.

Input is the supported schema-1 model and its pinned architecture contract, not
arbitrary bare weights. Stateful/float/expert portfolios retain that certified
contract; independent integer projections are constructed from typed operands.
"""

from orinfer_build.common.paths import ROOT, work_path


import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(ROOT))

from orinfer_build.compiler.build_cache import BuildCache, identity  # noqa: E402
from orinfer_build.models.flash_next.construction import (
    expert_matrices,
    fixed_extents,
    prefill_names,
    projection,
    select_projections,
    bounded_measurements,
)  # noqa: E402
from orinfer_build.compiler.host_compiler import Compiler  # noqa: E402
from orinfer_build.packaging.publication import (
    ROOT,
    atomic_model,
    clone_model,
    load_model,
    source_path,
)  # noqa: E402
from orinfer_build.common.files import file_hash, write_json  # noqa: E402


def inventory(data, package, manifest, compiler):
    if data["architecture"] != "flash_next" or data["schema_version"] != 1:
        raise ValueError("Expected the current native Flash Next schema-1 model")
    if not manifest.get("row_execution"):
        raise ValueError("The native model must provide its shared row construction contract")
    fixed = fixed_extents(manifest)
    dynamic = {k["name"]: k for k in package["dynamic_batch_kernels"]}
    groups = {}
    for kernel in package["kernels"]:
        result = projection(
            kernel,
            compiler.host(kernel),
            compiler.pointers(kernel),
            compiler.buffers,
            dynamic,
            fixed,
        )
        if result is None:
            continue
        spec, weights = result
        groups.setdefault(spec, []).append((kernel, weights))
    if not groups:
        raise ValueError("No supported integer construction contracts found")
    return groups, dynamic


def source_identity():
    """Pin reachable production code and the Rust model contract, excluding other tools/tests."""
    from orinfer_build.common.sources import component_identity

    paths = [
        ROOT / p
        for p in (
            "Cargo.lock",
            "Cargo.toml",
            "rust-toolchain.toml",
            "tools/environments/cuda/uv.lock",
        )
    ]
    paths.extend((ROOT / "configs").glob("*.json"))
    for crate in ("orinfer-models", "orinfer-model-sdk", "orinfer-device"):
        base = ROOT / "crates" / crate
        paths.append(base / "Cargo.toml")
        paths.extend(
            p
            for p in (base / "src").rglob("*.rs")
            if "tests" not in p.parts and not p.name.startswith("test_")
        )
    return component_identity(["orinfer_build.models.flash_next.build"], extra=paths)


def weight_contract(weights, buffers):
    return {
        role: {k: buffers[operand["name"]][k] for k in ("dtype", "shape", "layout", "data")}
        for role, operand in sorted(weights.items())
    }


def read_weights(model, operands, buffers):
    import torch
    from safetensors import safe_open

    base = (model / "cache/weights").resolve(strict=True)
    index = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
    result = {}
    for role, operand in operands.items():
        buffer = buffers[operand["name"]]
        tensor = buffer["data"]["tensor"]
        with safe_open(source_path(base, index[tensor]), framework="pt", device="cpu") as stream:
            value = stream.get_tensor(tensor)
        raw = value.view(torch.uint8).numpy().tobytes()
        if hashlib.sha256(raw).hexdigest() != buffer["data"]["sha256"]:
            raise ValueError(f"Real tuning weight identity mismatch: {tensor}")
        if list(value.shape) != buffer["shape"]:
            raise ValueError(f"Real tuning weight shape mismatch: {tensor}")
        expected = {
            "i8": torch.int8,
            "u16": torch.uint16,
            "u32": torch.uint32,
            "f16": torch.float16,
        }
        if value.dtype != expected[buffer["dtype"]]:
            raise ValueError(f"Real tuning weight dtype mismatch: {tensor}")
        result[role] = value.cuda()
    return result


def toolchain():
    import tilelang
    import torch

    return dict(
        tilelang=tilelang.__version__,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(),
        capability=list(torch.cuda.get_device_capability()),
    )


def rebuild(args):
    started = time.monotonic()
    manifest = {}
    data, origin, package = load_model(args.model, plan=manifest)
    buffers = {b["name"]: b for b in data["metadata"]["buffers"]}
    inspector = Compiler(origin, origin, args.output, buffers)
    groups, dynamic = inventory(data, package, manifest, inspector)
    bounded = bounded_measurements(manifest)
    prefill = prefill_names(manifest)
    prefill_groups = {
        s
        for s, items in groups.items()
        if s.dynamic and any(k["name"] in prefill for k, _ in items)
    }
    if args.prefill_only:
        groups = {s: items for s, items in groups.items() if s in prefill_groups}
        if not groups:
            raise ValueError("No dynamic prefill projection contracts found")
    groups = select_projections(groups, kinds=args.projection_kinds, shapes=args.projection_shape)
    partial = args.prefill_only or args.projection_kinds or args.projection_shape
    experts = expert_matrices(package, inspector)
    if partial:
        experts = {}
    report = dict(
        source_package=data["execution_package"],
        source=source_identity(),
        model=file_hash(args.model / "cache/model.json"),
        operators=[dict(**s.spec(), bindings=len(items)) for s, items in sorted(groups.items())],
        preserved_bindings=len(package["kernels"]) - sum(map(len, groups.values())),
        expert_geometries=[list(key) for key in sorted(experts)],
        prefill_only=args.prefill_only,
        projection_kinds=args.projection_kinds,
        projection_shapes=args.projection_shape,
        prefill_tuning_rows=args.prefill_tuning_rows,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "inventory.json", report)
    if args.plan_only:
        print(json.dumps(report, indent=2), flush=True)
        return

    from orinfer_build.compiler.packet_graph import inspect as inspect_packets, compile_package

    packet_layout = inspect_packets(args.model)

    from orinfer_build.tuning.projection_tuning import ProjectionTuner
    from orinfer_build.compiler.cuda.common import export_kernel
    import torch

    contract = dict(**report, toolchain=toolchain(), seed=20261002)
    cache = BuildCache(args.cache)
    # A repeated command validates the completed output without rebuilding it.
    result_file = args.output / "results.json"
    if args.model_output.exists():
        output_data, _, _ = load_model(args.model_output)
        # The receipt is committed atomically with the model. Recover even if
        # interrupted after directory publication and before the external log.
        receipt = json.loads((args.model_output / "cache/build.json").read_text())
        if (
            receipt.get("contract") != identity(contract)
            or receipt.get("execution_package") != output_data["execution_package"]
        ):
            raise ValueError("Existing destination does not match this completed build")
        if not result_file.exists():
            write_json(
                result_file,
                dict(
                    complete=True,
                    contract=contract,
                    execution_package=output_data["execution_package"],
                    recovered_publication=True,
                ),
            )
        print("COMPLETE cached publication", output_data["execution_package"], flush=True)
        return

    tuning = ProjectionTuner(args.output, args.cache / "tuning")
    records = []
    certificates = []
    # Validate both legal expert tiles at actual N/K/E, without independently
    # changing the certified coupled route + up + down selection table.
    from orinfer_build.tuning.projection_tuning import make_prim, EXACT_PASSES, TARGET
    import tilelang

    for (count, n, k), operands in sorted(experts.items()):
        spec = dict(
            kind="expert",
            rows=4096,
            outputs=n,
            inputs=k,
            dtype="float16",
            dynamic=True,
            experts=count,
        )
        wc = weight_contract(operands, buffers)
        for bm, st in ((8, 2), (8, 4), (16, 2), (16, 4)):
            expert_contract = dict(
                source=contract["source"],
                toolchain=contract["toolchain"],
                seed=contract["seed"],
                spec=spec,
                weights=wc,
                block_m=bm,
                stages=st,
            )

            def certify(folder):
                actual = read_weights(args.model, operands, buffers)
                config = dict(block_m=bm, stages=st)
                candidate = tilelang.compile(
                    make_prim(spec, config),
                    out_idx=[],
                    execution_backend="nvrtc",
                    target=TARGET,
                    pass_configs=EXACT_PASSES,
                )
                for live in (1, 17, 512, 4096):
                    ProjectionTuner._validate(candidate, spec, config, live, actual)
                export_kernel(candidate, folder)
                write_json(
                    folder / "validation.json", dict(config=config, checked_rows=[1, 17, 512, 4096])
                )
                del actual
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            _, hit = cache.get(expert_contract, certify)
            certificates.append(dict(spec=spec, block_m=bm, stages=st, cache_hit=hit))
            print(
                "EXPERT CHECKED", count, n, k, bm, st, "cache" if hit else "validated", flush=True
            )
    with atomic_model(args.model_output) as staging:
        root = clone_model(args.model, staging, origin)
        compiler = Compiler(origin, root, args.output, buffers)
        replacements, launches = {}, {}
        for number, (spec, items) in enumerate(sorted(groups.items()), 1):
            # Geometry is shared between layers; a deterministic real operand
            # set supplies values, scales and a checked identity to the tuner.
            representative, operands = min(items, key=lambda item: item[0]["name"])
            wc = weight_contract(operands, buffers)
            measured = (
                sorted({min(r, spec.rows) for r in args.prefill_tuning_rows})
                if spec in prefill_groups
                else None
            )

            def compile_extent(measurements):
                entry_contract = dict(
                    source=contract["source"],
                    toolchain=contract["toolchain"],
                    seed=contract["seed"],
                    spec=spec.spec(),
                    weights=wc,
                    measure_rows=measurements,
                )

                def construct(folder):
                    actual = read_weights(args.model, operands, buffers)
                    kernel, config = tuning.compile(
                        **spec.spec(),
                        weights=actual,
                        weight_identity=identity(wc),
                        measure_rows=measurements,
                    )
                    export_kernel(kernel, folder)
                    write_json(
                        folder / "validation.json",
                        dict(config=config, tuning=list(tuning.results.values())[-1]),
                    )
                    del actual
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

                artifact, hit = cache.get(entry_contract, construct)
                return artifact, hit, identity(entry_contract)

            tick = time.monotonic()
            artifact, hit, key = compile_extent(measured)
            extents = {tuple(measured or []): (artifact, hit, key)}
            for old, _ in items:
                samples = (
                    bounded.get(old["name"])
                    if spec.dynamic and spec not in prefill_groups
                    else None
                )
                sample_key = tuple(samples or measured or [])
                if sample_key not in extents:
                    extents[sample_key] = compile_extent(list(sample_key))
                selected, _, selected_key = extents[sample_key]
                capacity = dynamic.get(old["name"], {}).get("capacity", spec.rows)
                item, launch = compiler.emit(
                    old,
                    old["name"],
                    compiler.pointers(old),
                    capacity,
                    key=selected_key,
                    artifact=selected,
                )
                replacements[old["name"]] = item
                if old["name"] in dynamic:
                    launches[old["name"]] = launch
            records.append(
                dict(
                    spec=spec.spec(),
                    key=key,
                    cache_hit=hit,
                    seconds=time.monotonic() - tick,
                    representative=representative["name"],
                    bounded_choices=[
                        dict(rows=list(rows), key=value[2])
                        for rows, value in extents.items()
                        if rows != tuple(measured or [])
                    ],
                )
            )
            write_json(
                args.output / "progress.json",
                dict(complete=False, finished=number, total=len(groups), operators=records),
            )
            print("BUILT", number, "/", len(groups), spec, "cache" if hit else "tuned", flush=True)
        if args.hc_combine_norm:
            from orinfer_build.models.flash_next.hc_fusion import publish as publish_hc_combine_norm

            replacements.update(
                publish_hc_combine_norm(package, manifest, compiler, args.output / "hc")
            )
        package["kernels"] = [replacements.get(k["name"], k) for k in package["kernels"]]
        package["dynamic_batch_kernels"] = [
            launches.get(k["name"], k) for k in package["dynamic_batch_kernels"]
        ]
        from orinfer_build.tuning.row_convergence import coalesce_adjacent_selections
        from orinfer_build.packaging.compact import compact_package

        from orinfer_build.models.sampling import append as append_sampling

        append_sampling(data, package, root, args.output / "sampling")
        if not partial:
            from orinfer_build.tuning.auxiliary_tuning import publish as publish_auxiliary

            publish_auxiliary(data, package, manifest, compiler, args.output / "auxiliary")
            if (
                data["metadata"].get("vision")
                and data["metadata"]["vision"]["max_features"] < data["metadata"]["max_context"]
            ):
                from orinfer_build.packaging.vision_capacity import rebuild_overlays

                rebuild_overlays(data, package, manifest, compiler, data["metadata"]["max_context"])
        coalesce_adjacent_selections(package)
        compile_package(
            data,
            package,
            manifest,
            root,
            root,
            args.output / "packets",
            args.cache / "packets",
            packet_layout,
        )
        digest = compact_package(root, package, staging / "cache/packages")
        shutil.rmtree(root)
        data["execution_package"] = digest
        write_json(staging / "cache/model.json", data)
        write_json(
            staging / "cache/build.json",
            dict(contract=identity(contract), execution_package=digest),
        )
    result = dict(
        complete=True,
        contract=contract,
        execution_package=digest,
        seconds=time.monotonic() - started,
        operators=records,
        expert_certificates=certificates,
    )
    write_json(result_file, result)
    write_json(
        args.output / "progress.json", dict(complete=True, finished=len(groups), total=len(groups))
    )
    print("COMPLETE", digest, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "model-output", "output"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--cache", type=Path, default=work_path("cache", "build", "flash-next"))
    p.add_argument("--plan-only", action="store_true")
    p.add_argument(
        "--prefill-only",
        action="store_true",
        help="Rebuild only declared prefill projections; preserve other certified bindings",
    )
    p.add_argument("--prefill-tuning-rows", type=int, nargs="+", default=[512, 2048, 4096])
    p.add_argument(
        "--hc-combine-norm",
        action="store_true",
        help="Replace decode HC combine bindings with the bit-identical fused combine-norm kernel",
    )
    p.add_argument(
        "--projection-kinds",
        nargs="+",
        choices=["q4", "w8", "swiglu"],
        help="Rebuild selected projection kinds; preserve other projections and auxiliary bindings",
    )
    p.add_argument(
        "--projection-shape",
        nargs=2,
        type=int,
        action="append",
        metavar=("N", "K"),
        help="Restrict rebuilds to these output/input dimensions; repeat for multiple geometries",
    )
    args = p.parse_args()
    if args.model.resolve() == args.model_output.resolve():
        p.error("model-output must differ from the immutable input")
    if any(r < 1 for r in args.prefill_tuning_rows):
        p.error("prefill-tuning-rows must be positive")
    if any(n < 1 or k < 1 for n, k in args.projection_shape or []):
        p.error("projection-shape dimensions must be positive")
    rebuild(args)


if __name__ == "__main__":
    main()
