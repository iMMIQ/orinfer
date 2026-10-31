"""Select one shared contiguous-copy tile per wide BF16 row-movement geometry."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import ctypes
import json
from pathlib import Path

from tools.build.files import atomic_json, file_hash
from tools.model.deepseek_v41.assembly import publish_package, validate_rows
from tools.model.row_selection import guarded_winner
from tools.model.tuning import compile_candidates, measure, read_record, record_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=7)
    args = parser.parse_args()
    import torch
    import torch_npu  # noqa: F401
    from tools.model.deepseek_v41.build import compile_task
    from tools.model.deepseek_v41.tune import npu_benchmark

    torch.set_num_threads(4)
    torch.manual_seed(20261002)
    torch.npu.set_device(args.device)
    if args.output.exists():
        base = args.output / "source"
        model = json.loads((base / "model.json").read_text())
        if model != json.loads(args.plan.read_text()):
            raise ValueError("Movement tuning source plan differs")
    else:
        base = args.output / "source"
        model = json.loads(args.plan.read_text())
        publish_package(model, args.plan.parent, base)
    descriptions = {}
    for rank in model["ranks"]:
        for packet in rank["packets"]:
            for step in packet["body"] + packet["head"]:
                if step["op"] != "kernel" or step["manifest"] in descriptions:
                    continue
                spec = json.loads((base / step["manifest"]).read_text())
                if (
                    spec["op"] in ("memory.partition_rows", "memory.interleave_rows")
                    and spec["parameters"].get("dynamic_count")
                    and spec["parameters"].get("dtype") == "bfloat16"
                    and spec["parameters"]["width"] >= 512
                ):
                    descriptions[step["manifest"]] = spec
    replacements, reports = {}, []
    for old, spec in descriptions.items():
        parameters = spec["parameters"]
        capacity = parameters["_row_capacity"]
        measured = sorted(
            {r for r in (1, 32, 511, 512, capacity - 1, capacity) if 1 <= r <= capacity}
        )
        baseline = parameters.get("tile_columns", 256)
        configs = list(dict.fromkeys([baseline, 1024, 4096]))

        def compile_one(tile):
            arguments = {k: v for k, v in parameters.items() if not k.startswith("_")}
            arguments["tile_columns"] = tile
            task = dict(
                module="tools.operators.ascend.memory",
                name=spec["op"].split(".")[1],
                arguments=arguments,
                specs=deepcopy(spec["operands"]),
                row_capacity=capacity,
                dynamic_arguments=dict(rows=1),
                grid_limit=parameters["_grid_limit"],
                placeholder=str(tile),
            )
            return base / compile_task((task, str(base)))[1]

        current_path = compile_one(baseline)
        current = json.loads(current_path.read_text())
        identity = dict(
            source=current["source_sha256"],
            compiler=current["compiler"],
            parameters=parameters,
            configs=configs,
            rows=measured,
            device=torch.npu.get_device_name(args.device),
            tuner_sha256=file_hash(Path(__file__)),
            measurement_sha256=file_hash(Path(__file__).parents[1] / "tuning.py"),
            benchmark_sha256=file_hash(Path(__file__).with_name("tune.py")),
            selector_sha256=file_hash(Path(__file__).parents[1] / "row_selection.py"),
            maximum_regression=0.1,
            calls_per_replay=16,
        )
        record = record_path(args.output / "measurements", identity)
        record.parent.mkdir(parents=True, exist_ok=True)

        def valid(cached):
            path = base / cached["manifest"]
            desc = json.loads(path.read_text())
            return (
                desc["op"] == spec["op"]
                and desc["source_sha256"] == current["source_sha256"]
                and desc["compiler"] == current["compiler"]
                and desc["parameters"] == dict(parameters, tile_columns=cached["tile_columns"])
                and desc["library"]["sha256"] == file_hash(path.parent / desc["library"]["file"])
            )

        cached = read_record(record, identity, valid)
        if cached is not None:
            replacements[old] = cached["manifest"]
            reports.append(cached)
            continue
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {tile: pool.submit(compile_one, tile) for tile in configs}
            paths, legal, rejected = compile_candidates(
                configs, lambda tile: futures[tile].result()
            )
        libraries = [ctypes.CDLL(str((path.parent / "kernel.so").resolve())) for path in paths]
        entries = [lib.orinfer_ascend_kernel_v2 for lib in libraries]
        for entry in entries:
            entry.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_uint64,
                ctypes.c_uint64,
                ctypes.c_void_p,
            ]
            entry.restype = ctypes.c_int32
        width = parameters["width"]
        dtype = getattr(torch, parameters["dtype"])
        source = torch.randint(-100, 100, (capacity + 8, width), device="npu").to(dtype)
        out = torch.empty_like(source)
        pointers = (ctypes.c_void_p * 2)(source.data_ptr(), out.data_ptr())
        measurements = []
        for rows in measured:
            ranks = parameters.get("stride", parameters.get("ranks", 8))
            count = (rows + ranks - 1) // ranks
            if spec["op"] == "memory.interleave_rows":
                total = rows
                index = torch.arange(rows, device="npu")
                index = index % ranks * count + index // ranks
                valid_rows = None
            else:
                total = count * (ranks if parameters.get("all_partitions") else 1)
                index = torch.arange(total, device="npu")
                index = (
                    index // count + index % count * ranks
                    if parameters.get("all_partitions")
                    else parameters["first"] + index * ranks
                )
                valid_rows = index < rows
                index = index.clamp(max=rows - 1)

            def expected():
                value = source.index_select(0, index)
                if valid_rows is not None and not parameters.get("repeat_last"):
                    value[~valid_rows] = 0
                return value

            times = {}
            for i, entry in enumerate(entries):
                out.fill_(17)

                def run():
                    if entry(pointers, 2, rows, torch.npu.current_stream()._as_parameter_):
                        raise RuntimeError("Movement launch failed")

                def check():
                    torch.npu.synchronize()
                    torch.testing.assert_close(out[:total], expected(), atol=0, rtol=0)
                    assert bool((out[total:] == 17).all()), "Movement wrote beyond actual rows"

                def change():
                    source.neg_()

                def repeated():
                    for _ in range(16):
                        run()

                run()
                check()
                times[str(i)] = (
                    measure(
                        repeated,
                        budget_ms=8,
                        max_repetitions=64,
                        change=change,
                        check=check,
                        benchmark=npu_benchmark,
                    )
                    / 16
                )
            measurements.append(dict(rows=rows, times=times))
        selected = int(guarded_winner(measurements, maximum_regression=0.1))
        report = dict(
            identity=identity,
            manifest=str(paths[selected].relative_to(base)),
            tile_columns=legal[selected],
            measurements=measurements,
            rejected=rejected,
            timing_scope="NPU-event Graph replay, milliseconds per movement invocation",
        )
        atomic_json(record, report)
        replacements[old] = report["manifest"]
        reports.append(report)
        print(
            json.dumps(
                dict(
                    op=spec["op"],
                    parameters=parameters,
                    selected=legal[selected],
                    measurements=measurements,
                )
            ),
            flush=True,
        )
    for rank in model["ranks"]:
        for packet in rank["packets"]:
            for step in packet["body"] + packet["head"]:
                if step.get("manifest") in replacements:
                    step["manifest"] = replacements[step["manifest"]]
    validate_rows(model, base)
    assets = publish_package(model, base, args.output / "package")
    atomic_json(args.output / "selection.json", dict(assets=assets, selections=reports))
    print(args.output / "package/model.json", flush=True)


if __name__ == "__main__":
    main()
