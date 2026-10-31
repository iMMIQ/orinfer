"""Select shared A2 projection configurations across real token-row lengths.

Reads the native build contract and real checkpoint weights. Candidate kernels
share the projection factory, bounded row ABI and capacity-backed operands.
Only the selected library replaces each geometry in the published model plan.
"""

import argparse
import ctypes
from copy import deepcopy
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tools.build.files import atomic_json, file_hash
from tools.model.deepseek_v41.converge import unit_scale_manifests
from tools.model.row_selection import bounded_choices, guarded_winner, samples
from tools.model.tuning import compile_candidates, measure, read_record, record_path


def projection_configs(parameters, pipeline_only=False, wide_integer=False, joint=False):
    baseline = {
        key: parameters.get(key, default)
        for key, default in (
            ("tile_m", 16),
            ("tile_n", 64),
            ("block_k", 128),
            ("pipeline_stages", 0),
        )
    }
    if joint:
        configs = [
            dict(baseline, tile_m=m, tile_n=n, block_k=k, pipeline_stages=s)
            for m, n, k, s in (
                (32, 128, 128, 2),
                (32, 128, 512, 2),
                (32, 64, 512, 2),
                (64, 64, 512, 2),
                (64, 64, 128, 2),
                (64, 128, 128, 0),
                (64, 128, 128, 2),
                (64, 128, 128, 3),
                (128, 64, 128, 2),
                (128, 128, 128, 2),
                (64, 128, 256, 2),
                (64, 128, 512, 2),
            )
            if parameters["reduction"] % k == 0 and (s == 0 or parameters["reduction"] // k >= s)
        ]
    elif wide_integer:
        configs = [
            dict(baseline, tile_m=128, tile_n=256, pipeline_stages=s)
            for s in (0, 2, 3)
            if s == 0 or parameters["reduction"] // baseline["block_k"] >= s
        ]
    elif pipeline_only:
        configs = [
            dict(baseline, pipeline_stages=s, tile_m=m)
            for s in (0, 2, 3)
            for m in sorted({baseline["tile_m"], 128} | ({256} if parameters["integer"] else set()))
            if s == 0 or parameters["reduction"] // baseline["block_k"] >= s
        ]
    else:
        configs = [
            dict(baseline, tile_m=m, tile_n=n, block_k=k)
            for m in (16, 32, 64)
            for n in (64, 128)
            for k in sorted({128, 256, 512, baseline["block_k"]})
            if parameters["reduction"] % k == 0
        ]
    return [baseline] + [c for c in configs if c != baseline]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--package-output", type=Path, help="Publish only selected native assets")
    p.add_argument("--rows", type=int, nargs="+")
    p.add_argument("--maximum-regression", type=float, default=0.05)
    p.add_argument("--device", type=int, default=7)
    p.add_argument("--compile-workers", type=int, default=4)
    p.add_argument(
        "--geometry",
        type=int,
        nargs=2,
        action="append",
        metavar=("N", "K"),
        help="Limit tuning to these measured output/reduction geometries",
    )
    p.add_argument(
        "--local-only",
        action="store_true",
        help="Tune only dynamic projection kernels with an explicit local row binding",
    )
    p.add_argument("--decode-only", action="store_true", help="Tune the shared batch decode family")
    search = p.add_mutually_exclusive_group()
    search.add_argument(
        "--grid-only", action="store_true", help="Measure same-source 64/128/256 CTA bounds"
    )
    search.add_argument("--pipeline-only", action="store_true")
    search.add_argument(
        "--bf16-output",
        action="store_true",
        help="Compare shared BF16 Vector and direct Fixpipe output after verifying unit scales",
    )
    search.add_argument("--joint", action="store_true", help="Joint M/N/K/pipeline candidates")
    search.add_argument(
        "--wide-integer",
        action="store_true",
        help="Compare the current INT8 schedule with M128/N256 and legal pipeline stages",
    )
    args = p.parse_args()
    if not 1 <= args.compile_workers <= 8:
        p.error("compile workers must be in 1..8")
    if not math.isfinite(args.maximum_regression) or not 0 <= args.maximum_regression <= 1:
        p.error("maximum regression must be finite, within 0..1")
    args.output.mkdir(parents=True, exist_ok=True)
    import torch
    import torch_npu  # noqa: F401
    from kernels.model.rows import row_count
    from tools.operators.ascend.build import LOWER_LOCK, compile_kernel
    from tools.operators.ascend.dense import projection
    from tools.operators.ascend.geometry import physical_shapes

    torch.set_num_threads(4)
    torch.manual_seed(20261002)
    torch.npu.set_device(args.device)
    base = args.plan.parent
    model = json.loads(args.plan.read_text())
    selections = model.setdefault("row_selections", {})
    index = json.loads((args.model_dir / model["index"]["file"]).read_text())["weight_map"]
    definitions = {b["name"]: b for b in model["ranks"][0]["buffers"]}
    unit_scales = unit_scale_manifests(model, base) if args.bf16_output else set()
    groups = {}
    for packet in model["ranks"][0]["packets"]:
        if args.decode_only and not packet.get("batch_decode", False):
            continue
        for step in packet["body"] + packet["head"]:
            if step["op"] != "kernel" or step["manifest"] in groups:
                continue
            manifest = json.loads((base / step["manifest"]).read_text())
            if manifest["op"] == "dense.projection" and "_row_capacity" in manifest["parameters"]:
                if args.bf16_output and step["manifest"] not in unit_scales:
                    continue
                if (
                    args.geometry
                    and [manifest["parameters"]["output"], manifest["parameters"]["reduction"]]
                    not in args.geometry
                ):
                    continue
                if (
                    args.local_only
                    and step.get("fixed_rows") != manifest["parameters"]["_row_capacity"]
                    and "row_expression" not in step
                ):
                    continue
                groups[step["manifest"]] = (
                    manifest,
                    definitions[step["buffers"][1]],
                    definitions[step["buffers"][3]],
                )
    if not groups:
        raise ValueError("No dynamic projection kernels match the selected geometry and scope")
    replacements = {}
    for old, (manifest, weight, scaling) in groups.items():
        parameters = dict(manifest["parameters"])
        capacity = parameters.pop("_row_capacity")
        grid_limit = parameters.pop("_grid_limit", 64)
        if args.wide_integer and (not parameters["integer"] or parameters["output"] % 256):
            continue
        measured = (
            sorted({r for r in args.rows if 1 <= r <= capacity})
            if args.rows
            else sorted(set(samples(capacity)) | set(range(1, min(capacity, 128) + 1)))
        )
        if not measured:
            raise ValueError("Missing bounded prefill measurements")
        bank = read_bank(
            base,
            args.model_dir,
            index,
            weight,
            manifest["operands"][1]["dtype"],
            manifest["operands"][1]["shape"],
        )
        channel = (
            read_bank(base, args.model_dir, index, scaling, "f32", [parameters["output"]])
            .float()
            .reshape(-1)
            if parameters["integer"]
            else None
        )
        configs = projection_configs(parameters, args.pipeline_only, args.wide_integer, args.joint)
        if args.bf16_output:
            current = parameters.get("unit_scale", False)
            configs = [dict(configs[0], unit_scale=unit) for unit in (current, not current)]
        if args.grid_only:
            configs = [
                dict(configs[0], grid_limit=g) for g in dict.fromkeys((grid_limit, 64, 128, 256))
            ]
        baseline = configs[0]

        def compile_one(config):
            compiled_parameters = dict(parameters, **config)
            selected_grid = compiled_parameters.pop("grid_limit", grid_limit)
            params = dict(compiled_parameters)
            params["rows"] = row_count(capacity, True)
            with LOWER_LOCK:
                function = projection(**params)
                shapes = physical_shapes(function, capacity)
            operands = deepcopy(manifest["operands"])
            for operand, shape in zip(operands, shapes, strict=True):
                if operand["access"] == "scratch":
                    if math.prod(shape) > math.prod(operand["shape"]):
                        raise ValueError("Candidate exceeds the shared scratch allocation")
                    operand["shape"] = shape
            return compile_kernel(
                function,
                manifest["op"],
                operands,
                compiled_parameters,
                base / "cache",
                row_capacity=capacity,
                grid_limit=selected_grid,
            )

        current = json.loads(compile_one(baseline).read_text())
        identity = dict(
            source=current["source_sha256"],
            compiler=current["compiler"],
            device=torch.npu.get_device_name(args.device),
            parameters=parameters,
            configs=configs,
            samples=measured,
            maximum_regression=args.maximum_regression,
            seed=20261002,
            weight=weight,
            weight_sha256=file_hash_bytes(bank),
            scale_sha256=file_hash_bytes(channel) if channel is not None else None,
            scale=scaling,
            tuner_sha256=file_hash(Path(__file__)),
            convergence_sha256=file_hash(Path(__file__).with_name("converge.py")),
            geometry_sha256=file_hash(Path(__file__).parents[2] / "operators/ascend/geometry.py"),
            selection_sha256=file_hash(Path(__file__).parents[1] / "row_selection.py"),
        )
        record = record_path(args.output, identity)

        def valid_record(record):
            names = {record["manifest"]} | {p for v in record["variants"] for p in v["kernels"]}
            for name in names:
                selected = base / name
                descriptor = json.loads(selected.read_text())
                if not (
                    descriptor["source_sha256"] == current["source_sha256"]
                    and descriptor["compiler"] == current["compiler"]
                    and descriptor["library"]["sha256"] == file_hash(selected.parent / "kernel.so")
                ):
                    return False
            return True

        cached = read_record(record, identity, valid_record)
        if cached is not None:
            replacements[old] = cached["manifest"]
            selections.pop(old, None)
            if cached["variants"]:
                selections[cached["manifest"]] = cached["variants"]
            continue

        with ThreadPoolExecutor(max_workers=args.compile_workers) as pool:
            futures = {json.dumps(c, sort_keys=True): pool.submit(compile_one, c) for c in configs}
            paths, legal, rejected = compile_candidates(
                configs,
                lambda c: futures[json.dumps(c, sort_keys=True)].result(),
                tolerate_failures=True,
            )
        if not legal or legal[0] != baseline:
            raise ValueError("The validated baseline did not compile")
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
        n, k = parameters["output"], parameters["reduction"]
        integer = parameters["integer"]
        activation = (
            torch.randint(-127, 128, (capacity, k), dtype=torch.int8)
            if integer
            else torch.randn(capacity, k).bfloat16()
        ).npu()
        device_weight = bank.npu()
        token_scale = (torch.rand(capacity) * 0.01 if integer else torch.ones(capacity)).npu()
        channel_scale = (channel if integer else torch.ones(n)).npu()
        out = torch.empty((capacity + 64, n), dtype=torch.bfloat16, device="npu")
        scratch_rows = manifest["operands"][5]["shape"][0]
        scratch = torch.empty(
            (scratch_rows + 64, n),
            dtype=torch.int32 if integer else torch.float32,
            device="npu",
        )
        tensors = [activation, device_weight, token_scale, channel_scale, out, scratch]
        pointers = (ctypes.c_void_p * len(tensors))(*[t.data_ptr() for t in tensors])
        measurements = []
        invalid = set()
        for rows in measured:

            def launch(index):
                status = entries[index](
                    pointers, len(tensors), rows, torch.npu.current_stream()._as_parameter_
                )
                if status:
                    raise RuntimeError(f"A2 candidate launch failed: {status}")

            out.fill_(17)
            launch(0)
            torch.npu.synchronize()
            expected = out[:rows].clone()
            times = {}
            for i in range(len(entries)):
                if i in invalid:
                    continue
                out.fill_(17)
                scratch.fill_(17)

                def run():
                    launch(i)

                def check():
                    torch.npu.synchronize()
                    torch.testing.assert_close(
                        out[:rows],
                        expected,
                        atol=0 if integer else 0.003,
                        rtol=0 if integer else 0.008,
                    )
                    assert bool((out[rows:] == 17).all()), "A2 projection wrote the tail"
                    assert bool((scratch[scratch_rows:] == 17).all()), (
                        "A2 projection exceeded scratch"
                    )

                def change():
                    activation.neg_()
                    expected.neg_()

                try:
                    run()
                    check()
                    timing = measure(
                        run,
                        budget_ms=2,
                        max_repetitions=32,
                        change=change,
                        check=check,
                        benchmark=npu_benchmark,
                    )
                except AssertionError as error:
                    if i == 0:
                        raise
                    invalid.add(i)
                    rejected.append(
                        dict(config=legal[i], rows=rows, stage="numeric", reason=str(error))
                    )
                    print(
                        json.dumps(dict(rejected=legal[i], rows=rows, reason=str(error))),
                        flush=True,
                    )
                    continue
                times[str(i)] = timing
            measurements.append(dict(rows=rows, times=times))
        for sample in measurements:
            sample["times"] = {k: v for k, v in sample["times"].items() if int(k) not in invalid}
        selected = int(guarded_winner(measurements, maximum_regression=args.maximum_regression))
        chosen = str(paths[selected].relative_to(base))
        variants = [
            dict(
                min_rows=v["min_rows"],
                max_rows=v["max_rows"],
                multiple_of=1,
                kernels=[str(paths[int(v["implementation"])].relative_to(base))],
            )
            for v in bounded_choices(measurements, str(selected), min(capacity, 128))
        ]
        result = dict(
            identity=identity,
            measurements=measurements,
            rejected=rejected,
            config=legal[selected],
            manifest=chosen,
            variants=variants,
        )
        atomic_json(record, result)
        replacements[old] = chosen
        selections.pop(old, None)
        if variants:
            selections[chosen] = variants
        print(json.dumps(dict(geometry=[n, k, integer], **result)), flush=True)
        tensors = activation = device_weight = token_scale = channel_scale = out = scratch = (
            bank
        ) = None
        torch.npu.synchronize()
        torch.npu.empty_cache()
    for rank in model["ranks"]:
        for packet in rank["packets"]:
            for step in packet["body"] + packet["head"]:
                if step["op"] == "kernel":
                    for key in ("manifest", "uncompacted_manifest"):
                        if step.get(key) is not None:
                            step[key] = replacements.get(step[key], step[key])
    from tools.model.deepseek_v41.converge import compact_buffers, publish_package, validate_rows

    model = compact_buffers(model)
    validate_rows(model, base)
    atomic_json(args.plan.with_name(args.plan.stem + "-tuned.json"), model)
    if args.package_output:
        count = publish_package(model, base, args.package_output)
        print(json.dumps(dict(package=str(args.package_output), assets=count)), flush=True)


def read_weight(root, index, weight):
    import torch
    from safetensors import safe_open

    def read(name, rows=None, columns=None):
        with safe_open(root / index[name], framework="pt", device="cpu") as archive:
            return archive.get_slice(name)[rows or slice(None), columns or slice(None)].contiguous()

    if weight["kind"] == "matrix":
        return read(
            weight["name"],
            slice(weight["first_row"], weight["first_row"] + weight["rows"]),
            slice(weight["first_column"], weight["first_column"] + weight["columns"]),
        )
    if weight["kind"] != "shared":
        raise ValueError("Unsupported projection weight representation")
    prefix, rank, plane = weight["prefix"], weight["rank"], weight["plane"]
    if plane == "down":
        return read(prefix + ".w2.weight", columns=slice(rank * 288, (rank + 1) * 288))
    suffix = ".weight_scale" if plane == "scale" else ".weight"
    return torch.cat(
        [
            read(prefix + part + suffix, slice(rank * 288, (rank + 1) * 288))
            for part in (".w1", ".w3")
        ]
    )


def read_bank(base, root, index, definition, dtype, shape):
    """Read a logical NT projection bank from either build or prepared plans."""
    import math
    import torch

    if "weight" in definition:
        value = read_weight(root, index, definition["weight"])
    else:
        source = definition.get("prepared_weight") or definition.get("data", {}).get("file")
        if source is None:
            raise ValueError("Projection bank has no offline weight source")
        kind = {"i8": torch.int8, "bf16": torch.bfloat16, "f32": torch.float32}[dtype]
        path = base / source
        elements = math.prod(shape)
        if path.stat().st_size != elements * torch.empty((), dtype=kind).element_size():
            raise ValueError("Prepared projection bank differs from its logical tensor span")
        value = torch.from_file(str(path), shared=False, size=elements, dtype=kind)
    return value.reshape(shape).contiguous()


def file_hash_bytes(tensor):
    import hashlib

    return hashlib.sha256(
        tensor.view(-1).view(__import__("torch").uint8).numpy().tobytes()
    ).hexdigest()


def npu_benchmark(run, repetitions, warmup):
    import statistics
    import torch

    for _ in range(warmup):
        run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        run()
    for _ in range(2):
        graph.replay()
    torch.npu.synchronize()
    values = []
    for _ in range(repetitions):
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end))
    return dict(median_ms=statistics.median(values)), graph


if __name__ == "__main__":
    main()
