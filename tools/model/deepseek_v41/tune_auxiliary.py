"""Tune A2 mHC and FP32 projection tiles with exact output and Graph checks."""

import argparse
import ctypes
import json
from pathlib import Path

from tools.build.files import atomic_json, file_hash
from tools.model.row_selection import aggregate_winner
from tools.model.tuning import compile_candidates, measure, read_record, record_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=7)
    parser.add_argument("--pipeline-only", action="store_true")
    parser.add_argument("--combine-only", action="store_true")
    parser.add_argument("--split-only", action="store_true")
    parser.add_argument("--convert-only", action="store_true")
    parser.add_argument("--hc-reuse-only", action="store_true")
    parser.add_argument("--sinkhorn-only", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--rows", type=int, nargs="+", default=[17, 128, 512, 2048])
    args = parser.parse_args()
    if (
        sum(
            (
                args.pipeline_only,
                args.combine_only,
                args.split_only,
                args.convert_only,
                args.hc_reuse_only,
                args.sinkhorn_only,
            )
        )
        > 1
    ):
        parser.error("Choose one auxiliary operator filter")
    import torch
    import torch_npu  # noqa: F401
    from kernels.model.rows import row_count
    from tools.model.deepseek_v41.floating import bf16_parts
    from tools.model.deepseek_v41.tune import file_hash_bytes, npu_benchmark, read_weight
    from tools.operators.ascend import dense, floating, model_ops, moe, vector
    from tools.operators.ascend.build import compile_kernel

    torch.set_num_threads(4)
    torch.manual_seed(20261002)
    if not args.compile_only:
        torch.npu.set_device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    model = json.loads(args.plan.read_text())
    base = args.plan.parent
    index = json.loads((args.model_dir / model["index"]["file"]).read_text())["weight_map"]
    definitions = {b["name"]: b for b in model["ranks"][0]["buffers"]}
    replacements = {}
    reduction_steps = {}
    reducer_manifests = set()
    paired_reducers = {}
    for packet in model["ranks"][0]["packets"]:
        for step in packet["body"] + packet["head"]:
            if step["op"] == "kernel":
                definition = json.loads((base / step["manifest"]).read_text())
                if definition["op"] in ("dense.fp32_reduce", "floating.reduce_partials"):
                    reduction_steps[step["buffers"][0]] = (step["manifest"], definition)
                    reducer_manifests.add(step["manifest"])
    seen = set()
    for packet in model["ranks"][0]["packets"]:
        for step in packet["body"] + packet["head"]:
            if step["op"] != "kernel" or step["manifest"] in seen:
                continue
            seen.add(step["manifest"])
            manifest = json.loads((base / step["manifest"]).read_text())
            kind = manifest["op"]
            if kind not in (
                "vector.mhc_post_pre",
                "floating.partials",
                "moe.combine",
                "floating.split",
                "model_ops.convert",
                "vector.mhc_split_sinkhorn",
            ):
                continue
            if (kind == "floating.split") != args.split_only:
                continue
            if (kind == "model_ops.convert") != args.convert_only:
                continue
            if (kind == "vector.mhc_split_sinkhorn") != args.sinkhorn_only:
                continue
            if kind == "model_ops.convert" and manifest["parameters"]["width"] < 16:
                continue
            if args.combine_only and kind != "moe.combine":
                continue
            if args.pipeline_only and kind != "floating.partials":
                continue
            if args.hc_reuse_only and kind != "vector.mhc_post_pre":
                continue
            params = dict(manifest["parameters"])
            if "_row_capacity" not in params:
                continue
            capacity = params.pop("_row_capacity")
            grid_limit = params.pop("_grid_limit", 64)
            if kind == "vector.mhc_split_sinkhorn":
                configs = [dict(tile_rows=r) for r in (1, 2, 4, 8, 16, 32)]
                factory = vector.mhc_split_sinkhorn
            elif kind in ("floating.split", "model_ops.convert"):
                configs = [dict(tile=t) for t in (256, 512, 1024, 1280, 2560, 5120)]
                factory = floating.split if kind == "floating.split" else model_ops.convert
            elif kind == "moe.combine":
                configs = [dict(tile=t) for t in (256, 512, 1024, 1280, 2560, 5120)]
                factory = moe.combine
            elif kind == "vector.mhc_post_pre":
                configs = [dict(tile=t) for t in (256, 512, 1024, 1280, 2560)]
                factory = vector.mhc_post_pre
                if args.hc_reuse_only:
                    configs = [dict(tile=params.get("tile", 256), reuse_residual=False)] + [
                        dict(tile=t, reuse_residual=True) for t in (1280, 2560, 5120)
                    ]
            else:
                configs = [dict(tile_m=t, tile_n=32) for t in (16, 32, 64)]
                factory = floating.partials
                if args.pipeline_only:
                    configs = [
                        dict(
                            pipeline_stages=s,
                            variance_rows=v,
                            tile_m=m,
                            tile_n=n,
                            variance_prefetch=prefetch,
                        )
                        for s in (0, 2, 3)
                        for v in ((1, 8, 16) if params.get("variance") else (1,))
                        for m in sorted({params.get("tile_m", 16), 128})
                        for n in ((32,) if params.get("variance") else (32, 64, 128))
                        for prefetch in (False,)
                    ]
                    baseline_config = dict(
                        pipeline_stages=0,
                        variance_rows=1,
                        tile_m=params.get("tile_m", 16),
                        tile_n=32,
                        variance_prefetch=False,
                    )
                    configs = [baseline_config] + [c for c in configs if c != baseline_config]

            def build(config):
                parameters = dict(params, **config)
                factory_args = dict(parameters, rows=row_count(capacity, True))
                if kind == "moe.combine":
                    factor, tail = divmod(params["expert_rows"], capacity)
                    if tail or factor < 1:
                        raise ValueError("Expert rows must have the packet row multiplier")
                    factory_args["expert_rows"] = factory_args["rows"] * factor
                function = factory(**factory_args)
                return compile_kernel(
                    function,
                    kind,
                    manifest["operands"],
                    parameters,
                    base / "cache",
                    row_capacity=capacity,
                    grid_limit=grid_limit,
                )

            baseline = build(configs[0])
            descriptor = json.loads(baseline.read_text())
            identity = dict(
                source=descriptor["source_sha256"],
                compiler=descriptor["compiler"],
                device=None if args.compile_only else torch.npu.get_device_name(args.device),
                parameters=params,
                configs=configs,
                samples=sorted({r for r in args.rows if 1 <= r <= capacity}),
                tuner_sha256=file_hash(Path(__file__)),
                seed=20261002,
                reduction_source=file_hash(Path(dense.__file__)),
            )
            weight = None
            if kind == "floating.partials":
                definition = definitions[step["buffers"][1]]
                if "prepared_weight" in definition:
                    weight = torch.from_file(
                        str(base / definition["prepared_weight"]),
                        shared=False,
                        size=definition["bytes"] // 2,
                        dtype=torch.bfloat16,
                    ).reshape(params["weight_parts"], params["output"], params["reduction"])
                else:
                    weight = read_weight(args.model_dir, index, definition["weight"])[None]
                identity["weight_sha256"] = file_hash_bytes(weight)

            def valid(record):
                paths_to_check = [record["manifest"]]
                if record.get("reducer_manifest"):
                    paths_to_check.append(record["reducer_manifest"])
                for relative in paths_to_check:
                    path = base / relative
                    spec = json.loads(path.read_text())
                    if spec["compiler"] != descriptor["compiler"] or spec["library"][
                        "sha256"
                    ] != file_hash(path.parent / "kernel.so"):
                        return False
                    if (
                        relative == record["manifest"]
                        and spec["source_sha256"] != descriptor["source_sha256"]
                    ):
                        return False
                return True

            if not identity["samples"]:
                raise ValueError("Missing auxiliary measurement rows")
            record = record_path(args.output, identity)
            cached = read_record(record, identity, valid) if not args.compile_only else None
            if cached:
                replacements[step["manifest"]] = cached["manifest"]
                if cached.get("reducer_manifest"):
                    replacements[reduction_steps[step["buffers"][2]][0]] = cached[
                        "reducer_manifest"
                    ]
                continue
            paths, legal, rejected = compile_candidates(configs, build, tolerate_failures=True)
            if legal[0] != configs[0]:
                raise ValueError("Missing baseline auxiliary kernel")
            if args.compile_only:
                print(dict(op=kind, compiled=len(paths), rejected=rejected), flush=True)
                continue
            libraries = [ctypes.CDLL(str(path.parent / "kernel.so")) for path in paths]
            entries = [lib.orinfer_ascend_kernel_v2 for lib in libraries]
            for entry in entries:
                entry.argtypes = [
                    ctypes.POINTER(ctypes.c_void_p),
                    ctypes.c_uint64,
                    ctypes.c_uint64,
                    ctypes.c_void_p,
                ]
                entry.restype = ctypes.c_int32
            tensors = []
            outputs = []
            for spec in manifest["operands"]:
                shape = list(spec["shape"])
                dtype = {
                    "bf16": torch.bfloat16,
                    "f16": torch.float16,
                    "f32": torch.float32,
                    "i32": torch.int32,
                }[spec["dtype"]]
                if spec["access"] == "write":
                    if kind == "floating.split":
                        shape = [3 * (capacity + 1) * params["width"]]
                    else:
                        shape[0] += 1
                    outputs.append(len(tensors))
                tensors.append(
                    torch.zeros(shape, dtype=dtype, device="npu")
                    if dtype == torch.int32
                    else torch.randn(shape, dtype=dtype, device="npu")
                )
            if weight is not None:
                tensors[1].copy_(weight)
            expected = [torch.empty_like(tensors[i]) for i in outputs]
            reference_tensors = list(tensors)
            for slot, out in zip(outputs, expected, strict=True):
                reference_tensors[slot] = out
            pointers = (ctypes.c_void_p * len(tensors))(*[t.data_ptr() for t in tensors])
            reference = (ctypes.c_void_p * len(tensors))(*[t.data_ptr() for t in reference_tensors])
            original = torch.randn(capacity, params["reduction"]) if weight is not None else None
            measurements = []
            invalid = set()
            for rows in sorted(set(identity["samples"] + [1, 2, 17, 31, 129, capacity])):
                if rows > capacity:
                    continue
                if kind == "moe.combine":
                    tensors[2][:rows].random_(-1, rows * 6)
                    tensors[2][0, 0] = rows * 6
                    tensors[2][0, 1] = -2
                if original is not None:
                    values = (
                        torch.from_numpy(bf16_parts(original[:rows].numpy())).view(torch.bfloat16)
                        if params["input_parts"] == 3
                        else original[:rows].bfloat16()[None]
                    )
                    tensors[0].view(-1)[: values.numel()].copy_(values.reshape(-1))

                def launch(which, args=pointers):
                    status = entries[which](
                        args, len(tensors), rows, torch.npu.current_stream()._as_parameter_
                    )
                    if status:
                        raise RuntimeError(f"Auxiliary kernel failed: {status}")

                def baseline_run():
                    for out in expected:
                        out.fill_(17)
                    launch(0, reference)

                baseline_run()
                times = {}
                for i in range(len(entries)):
                    if i in invalid:
                        continue
                    for slot in outputs:
                        tensors[slot].fill_(17)

                    def run():
                        launch(i)

                    def change():
                        tensors[0].neg_()
                        baseline_run()

                    def check():
                        torch.npu.synchronize()
                        for slot, ref in zip(outputs, expected, strict=True):
                            count = 3 * rows * params["width"] if kind == "floating.split" else rows
                            actual = tensors[slot][:count]
                            if kind == "floating.partials":
                                columns = legal[i].get("tile_n", params.get("tile_n", 32))
                                actual = (
                                    actual.reshape(
                                        rows, params["output"] // columns, 8, columns // 32, 32
                                    )
                                    .permute(0, 1, 3, 2, 4)
                                    .reshape(rows, -1)
                                )
                            torch.testing.assert_close(
                                actual,
                                ref[:rows].reshape(rows, -1)
                                if kind == "floating.partials"
                                else ref[:count],
                                atol=0,
                                rtol=0,
                            )
                            assert bool((tensors[slot][count:] == 17).all()), (
                                "Auxiliary kernel wrote tail"
                            )
                            if kind == "floating.split":
                                reference_parts = (
                                    torch.from_numpy(bf16_parts(tensors[0][:rows].cpu().numpy()))
                                    .view(torch.bfloat16)
                                    .reshape(-1)
                                )
                                torch.testing.assert_close(
                                    actual.cpu(), reference_parts, atol=0, rtol=0
                                )
                            if kind == "model_ops.convert":
                                converted = (
                                    tensors[0][:rows].cpu().float() * params.get("scale", 1.0)
                                ).to(actual.dtype)
                                torch.testing.assert_close(actual.cpu(), converted, atol=0, rtol=0)

                    try:
                        run()
                        check()
                        timing = measure(
                            run,
                            budget_ms=4,
                            max_repetitions=32,
                            change=change,
                            check=check,
                            benchmark=npu_benchmark,
                        )
                    except AssertionError as error:
                        if i == 0:
                            raise
                        invalid.add(i)
                        rejected.append(dict(config=legal[i], rows=rows, reason=str(error)))
                        continue
                    times[str(i)] = timing
                if rows in identity["samples"]:
                    measurements.append(dict(rows=rows, times=times))
                print(dict(op=kind, rows=rows, times=times), flush=True)
            for sample in measurements:
                sample["times"] = {
                    k: v for k, v in sample["times"].items() if int(k) not in invalid
                }
            selected = int(aggregate_winner(measurements))
            chosen = str(paths[selected].relative_to(base))
            reducer_manifest = None
            if kind == "floating.partials" and not params.get("variance"):
                _, reduce_spec = reduction_steps[step["buffers"][2]]
                columns = legal[selected].get("tile_n", 32)
                reduce_params = dict(rows=capacity, output=params["output"], splits=8)
                reference_path = compile_kernel(
                    dense.fp32_reduce(row_count(capacity, True), params["output"]),
                    "dense.fp32_reduce",
                    reduce_spec["operands"],
                    reduce_params,
                    base / "cache",
                    row_capacity=capacity,
                    grid_limit=grid_limit,
                )
                reduced_path = (
                    reference_path
                    if columns == 32
                    else compile_kernel(
                        floating.reduce_partials(
                            row_count(capacity, True), params["output"], tile_n=columns
                        ),
                        "floating.reduce_partials",
                        reduce_spec["operands"],
                        dict(reduce_params, tile_n=columns),
                        base / "cache",
                        row_capacity=capacity,
                        grid_limit=grid_limit,
                    )
                )
                reducer_manifest = str(reduced_path.relative_to(base))
                reduction_libraries = [
                    ctypes.CDLL(str(p.parent / "kernel.so")) for p in (reference_path, reduced_path)
                ]
                reduction_entries = [lib.orinfer_ascend_kernel_v2 for lib in reduction_libraries]
                reduced = [
                    torch.empty(capacity + 1, params["output"], device="npu") for _ in range(2)
                ]
                reduction_pointers = [
                    (ctypes.c_void_p * 2)(source.data_ptr(), destination.data_ptr())
                    for source, destination in zip((expected[0], tensors[2]), reduced, strict=True)
                ]
                for entry in reduction_entries:
                    entry.argtypes = [
                        ctypes.POINTER(ctypes.c_void_p),
                        ctypes.c_uint64,
                        ctypes.c_uint64,
                        ctypes.c_void_p,
                    ]
                    entry.restype = ctypes.c_int32

                def reduce_run(which):
                    status = reduction_entries[which](
                        reduction_pointers[which],
                        2,
                        rows,
                        torch.npu.current_stream()._as_parameter_,
                    )
                    if status:
                        raise RuntimeError(f"Floating reduction failed: {status}")

                for rows in sorted({1, 2, 17, 31, 129, capacity}):
                    if rows > capacity:
                        continue
                    if original is not None:
                        values = (
                            torch.from_numpy(bf16_parts(original[:rows].numpy())).view(
                                torch.bfloat16
                            )
                            if params["input_parts"] == 3
                            else original[:rows].bfloat16()[None]
                        )
                        tensors[0].view(-1)[: values.numel()].copy_(values.reshape(-1))
                    for target in reduced:
                        target.fill_(17)

                    def reference_run():
                        baseline_run()
                        reduce_run(0)

                    def selected_run():
                        launch(selected)
                        reduce_run(1)

                    def reduction_change():
                        tensors[0].neg_()
                        reference_run()

                    def reduction_check():
                        torch.npu.synchronize()
                        torch.testing.assert_close(
                            reduced[1][:rows], reduced[0][:rows], atol=0, rtol=0
                        )
                        assert bool((reduced[1][rows:] == 17).all()), (
                            "Floating reduction wrote tail"
                        )

                    reference_run()
                    selected_run()
                    reduction_check()
                    measure(
                        selected_run,
                        budget_ms=2,
                        max_repetitions=4,
                        change=reduction_change,
                        check=reduction_check,
                        benchmark=npu_benchmark,
                    )
                paired_reducers[step["manifest"]] = reducer_manifest
            result = dict(
                identity=identity,
                measurements=measurements,
                rejected=rejected,
                config=legal[selected],
                manifest=chosen,
                reducer_manifest=reducer_manifest,
            )
            atomic_json(record, result)
            replacements[step["manifest"]] = chosen
            print(dict(op=kind, **result), flush=True)
    if args.compile_only:
        return
    for rank in model["ranks"]:
        for packet in rank["packets"]:
            layouts = {}
            for step in packet["body"] + packet["head"]:
                if step["op"] != "kernel":
                    continue
                original_manifest = step["manifest"]
                if original_manifest in paired_reducers:
                    layouts[step["buffers"][2]] = paired_reducers[original_manifest]
                if original_manifest in reducer_manifests and step["buffers"][0] in layouts:
                    step["manifest"] = layouts[step["buffers"][0]]
                else:
                    step["manifest"] = replacements.get(original_manifest, original_manifest)
    atomic_json(args.plan.with_name(args.plan.stem + "-aux-tuned.json"), model)


if __name__ == "__main__":
    main()
