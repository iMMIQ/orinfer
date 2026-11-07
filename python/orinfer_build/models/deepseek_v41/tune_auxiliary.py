"""Tune A2 mHC and FP32 projection tiles with exact output and Graph checks."""

import argparse
import ctypes
from functools import partial
import json
from pathlib import Path

from orinfer_build.common.files import atomic_json, file_hash
from orinfer_build.contracts.row_selection import aggregate_winner, guarded_winner
from orinfer_build.tuning.tuning import compile_candidates, measure, read_record, record_path


def head_configs(parameters):
    """Keep the released head tile as reference and preserve its output layout."""
    baseline = dict(
        tile_m=parameters.get("tile_m", 16),
        tile_n=parameters.get("tile_n", 32),
        pipeline_stages=parameters.get("pipeline_stages", 0),
    )
    configs = [baseline]
    for m, s in ((baseline["tile_m"], 3), (128, 0), (128, 3)):
        candidate = dict(baseline, tile_m=m, pipeline_stages=s)
        if candidate not in configs:
            configs.append(candidate)
    return configs


def canonical_partials(values, rows, output, tile_n):
    """Compare logical columns/splits, excluding unpublished vocabulary padding."""
    blocks = (output + tile_n - 1) // tile_n
    return (
        values[:rows]
        .reshape(rows, blocks, 8, tile_n)
        .swapaxes(2, 3)
        .reshape(rows, blocks * tile_n, 8)[:, :output, :]
    )


def head_row_policy(records, contracts, maximum_regression=0.1):
    """Choose a fallback and one measured larger-row interval with matching ABI."""

    def signature(spec):
        return json.dumps(
            {
                **{
                    k: spec.get(k)
                    for k in ("op", "source_sha256", "compiler", "target", "operands")
                },
                "parameters": {
                    k: v
                    for k, v in spec["parameters"].items()
                    if k not in ("tile_m", "pipeline_stages", "_grid_limit")
                },
            },
            sort_keys=True,
        )

    names = sorted(contracts)
    policies = [dict(default=name, variant=None, minimum=1) for name in names]
    for fallback in names:
        for variant in names:
            if fallback == variant or signature(contracts[fallback]) != signature(
                contracts[variant]
            ):
                continue
            policies += [
                dict(default=fallback, variant=variant, minimum=row)
                for row in sorted({r["rows"] for r in records})
                if row > 1
            ]
    cases = [
        dict(
            rows=r["rows"],
            times={
                str(i): r["times"][
                    p["variant"]
                    if p["variant"] is not None and r["rows"] >= p["minimum"]
                    else p["default"]
                ]
                for i, p in enumerate(policies)
            },
        )
        for r in records
    ]
    reference = next(
        str(i) for i, p in enumerate(policies) if p["default"] == "0" and p["variant"] is None
    )
    selected = guarded_winner(cases, reference=reference, maximum_regression=maximum_regression)
    return policies[int(selected)]


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
    parser.add_argument("--reuse-operand-only", action="store_true")
    parser.add_argument("--quantize-only", action="store_true")
    parser.add_argument("--mask-only", action="store_true")
    parser.add_argument("--head-only", action="store_true")
    parser.add_argument("--maximum-regression", type=float, default=0.1)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--rows", type=int, nargs="+", default=[17, 128, 512, 2048])
    parser.add_argument(
        "--calls-per-replay",
        type=int,
        default=16,
        help="Independent operator calls per Graph; amortize host submission cost",
    )
    args = parser.parse_args()
    if not 1 <= args.calls_per_replay <= 64:
        parser.error("Graph replay call count must be in 1..64")
    if not 0 <= args.maximum_regression <= 1:
        parser.error("Maximum regression must be in 0..1")
    if (
        sum(
            (
                args.pipeline_only,
                args.combine_only,
                args.split_only,
                args.convert_only,
                args.hc_reuse_only,
                args.sinkhorn_only,
                args.reuse_operand_only,
                args.quantize_only,
                args.mask_only,
                args.head_only,
            )
        )
        > 1
    ):
        parser.error("Choose one auxiliary operator filter")
    import torch
    import torch_npu  # noqa: F401
    from kernels.shared.rows import row_count
    from orinfer_build.models.deepseek_v41.floating import bf16_parts
    from orinfer_build.tuning.ascend import npu_benchmark
    from orinfer_build.models.deepseek_v41.tune import file_hash_bytes, read_bank
    from kernels.ascend import attention_prefill, dense, floating, model_ops, moe, vector
    from orinfer_build.compiler.ascend.build import compile_kernel

    torch.set_num_threads(4)
    torch.manual_seed(20261002)
    if not args.compile_only:
        torch.npu.set_device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    model = json.loads(args.plan.read_text())
    base = args.plan.parent
    head_geometry = None
    if args.head_only:
        config = json.loads((args.model_dir / model["config"]["file"]).read_text())
        config = config.get("text_config", config)
        head_geometry = (config["vocab_size"] // len(model["ranks"]), config["hidden_size"])
    index = json.loads((args.model_dir / model["index"]["file"]).read_text())["weight_map"]
    definitions = {b["name"]: b for b in model["ranks"][0]["buffers"]}
    replacements = {}
    reduction_steps = {}
    reducer_manifests = set()
    paired_reducers = {}
    head_selections = {}
    for packet in model["ranks"][0]["packets"]:
        for step in packet["body"] + packet["head"]:
            if step["op"] == "kernel":
                definition = json.loads((base / step["manifest"]).read_text())
                if definition["op"] in ("dense.fp32_reduce", "floating.reduce_partials"):
                    reduction_steps[step["buffers"][0]] = (step["manifest"], definition)
                    reducer_manifests.add(step["manifest"])
    seen = set()
    matched_heads = 0
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
                "vector.quantize",
                "attention_prefill.selected_mask",
            ):
                continue
            if (kind == "floating.split") != args.split_only:
                continue
            if (kind == "model_ops.convert") != args.convert_only:
                continue
            if (kind == "vector.mhc_split_sinkhorn") != args.sinkhorn_only:
                continue
            if (kind == "vector.quantize") != args.quantize_only:
                continue
            if (kind == "attention_prefill.selected_mask") != args.mask_only:
                continue
            if kind == "vector.quantize" and manifest["parameters"]["width"] > 256:
                continue
            if kind == "model_ops.convert" and manifest["parameters"]["width"] < 16:
                continue
            if args.combine_only and kind != "moe.combine":
                continue
            if args.pipeline_only and kind != "floating.partials":
                continue
            if args.reuse_operand_only and kind != "floating.partials":
                continue
            if args.hc_reuse_only and kind != "vector.mhc_post_pre":
                continue
            if args.head_only and (
                kind != "floating.partials"
                or (manifest["parameters"]["output"], manifest["parameters"]["reduction"])
                != head_geometry
            ):
                continue
            if args.head_only:
                matched_heads += 1
                if step["manifest"] in model.get("row_selections", {}):
                    raise ValueError(
                        "Head tuning requires a frozen reference without head row choices"
                    )
            params = dict(manifest["parameters"])
            if "_row_capacity" not in params:
                if args.quantize_only and params.get("fp16_query_scale"):
                    path = compile_kernel(
                        vector.quantize(**params),
                        kind,
                        manifest["operands"],
                        params,
                        base / "cache",
                    )
                    replacements[step["manifest"]] = str(path.relative_to(base))
                    print(dict(op=kind, static_rows=params["rows"], manifest=str(path)), flush=True)
                continue
            capacity = params.pop("_row_capacity")
            grid_limit = params.pop("_grid_limit", 64)
            row_factor = 1
            if args.head_only:
                configs = head_configs(params)
                factory = floating.partials
            elif kind == "attention_prefill.selected_mask":
                baseline_config = dict(
                    explicit_sync=params.get("explicit_sync", False),
                    store_unroll=params.get("store_unroll", 1),
                )
                configs = [baseline_config] + [
                    dict(explicit_sync=explicit, store_unroll=unroll)
                    for explicit, unroll in (
                        (False, 1),
                        (True, 1),
                        (True, 8),
                        (True, 16),
                        (True, 32),
                    )
                    if dict(explicit_sync=explicit, store_unroll=unroll) != baseline_config
                ]
                factory = attention_prefill.selected_mask
            elif kind == "vector.quantize":
                row_factor, remainder = divmod(params["rows"], capacity)
                if remainder or row_factor < 1:
                    raise ValueError("Quantization rows must have the packet row multiplier")
                baseline_config = dict(tile_rows=params.get("tile_rows", 1))
                configs = [baseline_config] + [
                    dict(tile_rows=r)
                    for r in (8, 16, 32)
                    if r * params["width"] <= 4096 and r != baseline_config["tile_rows"]
                ]
                factory = vector.quantize
            elif kind == "vector.mhc_split_sinkhorn":
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
                if args.reuse_operand_only:
                    baseline_config = dict(
                        reuse_operand=params.get("reuse_operand", False),
                        pipeline_stages=params.get("pipeline_stages", 0),
                    )
                    configs = [baseline_config] + [
                        dict(reuse_operand=True, pipeline_stages=s)
                        for s in (0, 2, 3)
                        if dict(reuse_operand=True, pipeline_stages=s) != baseline_config
                    ]
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
                if args.head_only and config == configs[0]:
                    return base / step["manifest"]
                parameters = dict(params, **config)
                factory_args = dict(parameters, rows=row_count(capacity, True))
                if row_factor != 1:
                    factory_args["rows"] = factory_args["rows"] * row_factor
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
            head_compiled, head_contracts, head_reducer = None, {}, None
            if args.head_only:
                head_compiled = compile_candidates(configs, build, tolerate_failures=True)
                head_contracts = {
                    str(path.relative_to(base)): json.loads(path.read_text())
                    for path in head_compiled[0]
                }
                head_reducer = reduction_steps[step["buffers"][2]]
            identity = dict(
                source=descriptor["source_sha256"],
                compiler=descriptor["compiler"],
                device=None if args.compile_only else torch.npu.get_device_name(args.device),
                parameters=params,
                configs=configs,
                samples=sorted(
                    {
                        r
                        for r in ([*args.rows, capacity] if args.head_only else args.rows)
                        if 1 <= r <= capacity
                    }
                ),
                calls_per_replay=args.calls_per_replay,
                maximum_regression=args.maximum_regression if args.head_only else None,
                tuner_sha256=file_hash(Path(__file__)),
                seed=20261002,
                reduction_source=file_hash(Path(dense.__file__)),
            )
            if args.head_only:
                # A frozen package may use an older compatible compiler. Keep
                # both provenances instead of measuring a rebuilt reference.
                identity.update(head_candidates=head_contracts, head_reducer=list(head_reducer))
            weight = None
            if kind == "floating.partials":
                definition = definitions[step["buffers"][1]]
                weight = read_bank(
                    base,
                    args.model_dir,
                    index,
                    definition,
                    "bf16",
                    manifest["operands"][1]["shape"],
                )
                identity["weight_sha256"] = file_hash_bytes(weight)

            def valid(record):
                if args.head_only:
                    relative = record["manifest"]
                    if (
                        relative not in head_contracts
                        or record.get("reducer_manifest") != head_reducer[0]
                    ):
                        return False
                    names = [relative] + [
                        name for v in record.get("variants", []) for name in v["kernels"]
                    ]
                    if any(name not in head_contracts for name in names):
                        return False
                    for name, expected in [(name, head_contracts[name]) for name in names] + [
                        head_reducer
                    ]:
                        path = base / name
                        actual = json.loads(path.read_text())
                        if actual != expected or actual["library"]["sha256"] != file_hash(
                            path.parent / actual["library"]["file"]
                        ):
                            return False
                    return True
                expected_compiler = dict(descriptor["compiler"])
                if kind == "attention_prefill.selected_mask":
                    expected_compiler.pop("pipeline_sync", None)
                    if record["config"]["explicit_sync"]:
                        expected_compiler["pipeline_sync"] = "explicit-v1"
                paths_to_check = [record["manifest"]]
                if record.get("reducer_manifest"):
                    paths_to_check.append(record["reducer_manifest"])
                for relative in paths_to_check:
                    path = base / relative
                    spec = json.loads(path.read_text())
                    if spec["compiler"] != expected_compiler or spec["library"][
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
                if cached.get("variants"):
                    head_selections[cached["manifest"]] = cached["variants"]
                if cached.get("reducer_manifest"):
                    replacements[reduction_steps[step["buffers"][2]][0]] = cached[
                        "reducer_manifest"
                    ]
                continue
            paths, legal, rejected = head_compiled or compile_candidates(
                configs, build, tolerate_failures=True
            )
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
                    "i8": torch.int8,
                }[spec["dtype"]]
                if spec["access"] == "write":
                    if kind == "floating.split":
                        shape = [3 * (capacity + 1) * params["width"]]
                    else:
                        shape[0] += 1
                    outputs.append(len(tensors))
                tensors.append(
                    torch.zeros(shape, dtype=dtype, device="npu")
                    if dtype in (torch.int32, torch.int8)
                    else torch.randn(shape, dtype=dtype, device="npu")
                )
            if weight is not None:
                tensors[1].copy_(weight)
            if kind == "attention_prefill.selected_mask":
                tensors[0].random_(-2, params["length"] + 2)
            expected = [torch.empty_like(tensors[i]) for i in outputs]
            reference_tensors = list(tensors)
            for slot, out in zip(outputs, expected, strict=True):
                reference_tensors[slot] = out
            pointers = (ctypes.c_void_p * len(tensors))(*[t.data_ptr() for t in tensors])
            reference = (ctypes.c_void_p * len(tensors))(*[t.data_ptr() for t in reference_tensors])
            original = torch.randn(capacity, params["reduction"]) if weight is not None else None
            measurements = []
            invalid = set()
            validation_rows = [1, capacity] if args.head_only else [1, 2, 17, 31, 129, capacity]
            for rows in sorted(set(identity["samples"] + validation_rows)):
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
                            count = (
                                3 * rows * params["width"]
                                if kind == "floating.split"
                                else rows * row_factor
                            )
                            actual = tensors[slot][:count]
                            if kind == "floating.partials":
                                columns = legal[i].get("tile_n", params.get("tile_n", 32))
                                actual = canonical_partials(actual, rows, params["output"], columns)
                                reference_columns = params.get("tile_n", 32)
                                reference_value = canonical_partials(
                                    ref, rows, params["output"], reference_columns
                                )
                            torch.testing.assert_close(
                                actual,
                                reference_value if kind == "floating.partials" else ref[:count],
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
                            budget_ms=4 / args.calls_per_replay,
                            max_repetitions=32,
                            change=change,
                            check=check,
                            benchmark=partial(
                                npu_benchmark, calls_per_replay=args.calls_per_replay
                            ),
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
            policy = (
                head_row_policy(
                    measurements,
                    {
                        str(i): head_contracts[str(path.relative_to(base))]
                        for i, path in enumerate(paths)
                        if i not in invalid
                    },
                    args.maximum_regression,
                )
                if args.head_only
                else None
            )
            selected = int(policy["default"] if policy else aggregate_winner(measurements))
            chosen = str(paths[selected].relative_to(base))
            variants = []
            if policy and policy["variant"] is not None:
                variants = [
                    dict(
                        min_rows=policy["minimum"],
                        max_rows=capacity,
                        multiple_of=1,
                        kernels=[str(paths[int(policy["variant"])].relative_to(base))],
                    )
                ]
                head_selections[chosen] = variants
            reducer_manifest = None
            if kind == "floating.partials" and not params.get("variance"):
                _, reduce_spec = reduction_steps[step["buffers"][2]]
                reference_columns = params.get("tile_n", 32)
                columns = legal[selected].get("tile_n", reference_columns)
                reduce_params = dict(rows=capacity, output=params["output"], splits=8)
                reference_path = (
                    (base / head_reducer[0])
                    if args.head_only
                    else compile_kernel(
                        dense.fp32_reduce(row_count(capacity, True), params["output"])
                        if reference_columns == 32
                        else floating.reduce_partials(
                            row_count(capacity, True), params["output"], tile_n=reference_columns
                        ),
                        "dense.fp32_reduce"
                        if reference_columns == 32
                        else "floating.reduce_partials",
                        reduce_spec["operands"],
                        reduce_params
                        if reference_columns == 32
                        else dict(reduce_params, tile_n=reference_columns),
                        base / "cache",
                        row_capacity=capacity,
                        grid_limit=grid_limit,
                    )
                )
                reduced_path = (
                    reference_path
                    if columns == reference_columns
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

                reduce_rows = (
                    {1, 33, min(160, capacity), capacity}
                    if args.head_only
                    else {1, 2, 17, 31, 129, capacity}
                )
                for rows in sorted(reduce_rows):
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
                        implementation = (
                            int(policy["variant"])
                            if policy
                            and policy["variant"] is not None
                            and rows >= policy["minimum"]
                            else selected
                        )
                        launch(implementation)
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
                variants=variants,
            )
            if not valid(result):
                raise ValueError("Selected auxiliary kernel identity differs")
            atomic_json(record, result)
            replacements[step["manifest"]] = chosen
            print(dict(op=kind, **result), flush=True)
    if args.head_only and not matched_heads:
        raise ValueError("No published floating output heads match the model geometry")
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
    model.setdefault("row_selections", {}).update(head_selections)
    atomic_json(args.plan.with_name(args.plan.stem + "-aux-tuned.json"), model)


if __name__ == "__main__":
    main()
