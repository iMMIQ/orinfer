"""Select shared-head prefill softmax buckets with exact probability/Graph checks."""

import argparse
import ctypes
import json
from pathlib import Path

from tools.build.files import atomic_json, file_hash
from tools.model.deepseek_v41.tune import npu_benchmark
from tools.model.row_selection import aggregate_winner
from tools.model.tuning import measure, read_record, record_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", nargs="+", type=int, default=[17, 128, 512, 2048])
    parser.add_argument("--head-tiles", nargs=3, type=int, default=[4, 2, 1])
    parser.add_argument("--device", type=int, default=7)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    if any(h not in (1, 2, 4, 8) for h in args.head_tiles):
        parser.error("Head tiles must be 1, 2, 4 or 8")
    import torch
    import torch_npu
    from kernels.model.rows import row_count
    from tools.operators.ascend.attention_prefill import softmax
    from tools.operators.ascend.build import compile_kernel

    torch.set_num_threads(4)
    torch.manual_seed(20261002)
    if not args.compile_only:
        torch.npu.set_device(args.device)
    base = args.plan.parent
    model = json.loads(args.plan.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    groups = {}
    for packet in model["ranks"][0]["packets"]:
        for step in packet["body"] + packet["head"]:
            if step["op"] != "kernel" or step["manifest"] in groups:
                continue
            descriptor = json.loads((base / step["manifest"]).read_text())
            params = descriptor["parameters"]
            if (
                descriptor["op"] == "attention_prefill.softmax"
                and "_row_capacity" in params
                and params.get("compact")
                and params.get("live_buckets")
                and params.get("bucket", -1) == -1
            ):
                groups[step["manifest"]] = descriptor
    if not groups:
        raise ValueError("No untuned compact prefill softmax in this plan")
    replacements = {}
    for old, descriptor in groups.items():
        params = dict(descriptor["parameters"])
        capacity = params.pop("_row_capacity")
        grid_limit = params.pop("_grid_limit", 64)
        samples = sorted({r for r in args.rows if 1 <= r <= capacity})
        if not samples:
            raise ValueError("Missing bounded prefill softmax measurements")
        context = params["capacity"]
        buckets = min(3, (context + 2047) // 2048) if params["compressed"] else 1
        configs = [[{}], [dict(bucket=b, tile_heads=args.head_tiles[b]) for b in range(buckets)]]
        paths = [
            [
                compile_kernel(
                    softmax(**dict(params, **config, rows=row_count(capacity, True))),
                    descriptor["op"],
                    descriptor["operands"],
                    dict(params, **config),
                    base / "cache",
                    row_capacity=capacity,
                    grid_limit=grid_limit,
                )
                for config in group
            ]
            for group in configs
        ]
        print(dict(parameters=params, compiled=[[str(p) for p in g] for g in paths]), flush=True)
        if args.compile_only:
            continue
        current = json.loads(paths[0][0].read_text())
        identity = dict(
            source=current["source_sha256"],
            compiler=current["compiler"],
            device=torch.npu.get_device_name(args.device),
            parameters=params,
            configs=configs,
            samples=samples,
            measurement_base=0,
            validation_bases=[0, 2040, 4080, 6000, "context-rows"],
            seed=20261002,
            torch=torch.__version__,
            torch_npu=torch_npu.__version__,
            helpers={
                str(p): file_hash(p)
                for p in (
                    Path(__file__),
                    Path(__file__).with_name("tune.py"),
                    Path("tools/model/tuning.py"),
                    Path("tools/model/row_selection.py"),
                )
            },
        )

        def valid(record):
            for name in record["manifests"]:
                path = base / name
                spec = json.loads(path.read_text())
                if (
                    spec["source_sha256"] != current["source_sha256"]
                    or spec["compiler"] != current["compiler"]
                    or spec["library"]["sha256"] != file_hash(path.parent / "kernel.so")
                ):
                    return False
            return True

        record = record_path(args.output, identity)
        cached = read_record(record, identity, valid)
        if cached:
            replacements[old] = cached["manifests"]
            continue
        heads = params.get("heads", 8)
        long_length = context // params.get("ratio", 2)
        score_length = context + (long_length if params["compressed"] else 0)
        probability_length = min(384, context) + (long_length if params["compressed"] else 0)
        inputs = [
            torch.randn(capacity * heads, score_length, device="npu") * 4,
            torch.empty(capacity, dtype=torch.int32, device="npu"),
            torch.empty(capacity, long_length, dtype=torch.int32, device="npu"),
            torch.randn(heads, device="npu"),
        ]
        outputs = [
            torch.empty(
                (capacity + 1) * heads, probability_length, dtype=torch.bfloat16, device="npu"
            )
            for _ in paths
        ]
        libraries, calls = [], []
        for group, output in zip(paths, outputs, strict=True):
            entries = []
            for path in group:
                library = ctypes.CDLL(str(path.parent / "kernel.so"))
                libraries.append(library)
                entry = library.orinfer_ascend_kernel_v2
                entry.argtypes = [
                    ctypes.POINTER(ctypes.c_void_p),
                    ctypes.c_uint64,
                    ctypes.c_uint64,
                    ctypes.c_void_p,
                ]
                entry.restype = ctypes.c_int32
                pointers = (ctypes.c_void_p * 5)(*[t.data_ptr() for t in inputs + [output]])
                entries.append((entry, pointers))
            calls.append(entries)

        def positions(rows, first):
            values = torch.arange(rows, dtype=torch.int32) + first
            inputs[1][:rows].copy_(values)
            chosen = torch.arange(long_length)[None] < (values[:, None] + 1) // params["ratio"]
            chosen &= torch.rand(rows, long_length) > 0.3
            inputs[2][:rows].copy_(chosen.int())

        def launch(which):
            for entry, pointers in calls[which]:
                status = entry(pointers, 5, rows, torch.npu.current_stream()._as_parameter_)
                if status:
                    raise RuntimeError(f"Prefill softmax candidate failed: {status}")

        def check():
            torch.npu.synchronize()
            torch.testing.assert_close(
                outputs[1][: rows * heads], outputs[0][: rows * heads], atol=0, rtol=0
            )
            for output in outputs:
                assert bool((output[rows * heads :] == 17).all()), "Softmax wrote the tail"

        measurements = []
        for rows in sorted(set(samples + [1, 2, 17, 31, 129, capacity])):
            if rows > capacity or rows > context:
                continue
            positions(rows, 0)
            for output in outputs:
                output.fill_(17)
            launch(0)
            launch(1)
            check()
            graphs, times = [], {}
            for which in range(2):

                def change():
                    inputs[0].neg_()
                    launch(1 - which)

                times[str(which)] = measure(
                    lambda: launch(which),
                    budget_ms=4,
                    max_repetitions=32,
                    change=change,
                    check=check,
                    benchmark=npu_benchmark,
                )
                _, graph = npu_benchmark(lambda: launch(which), 1, 0)
                graphs.append(graph)
            if rows in samples:
                measurements.append(dict(rows=rows, times=times))
            for first in sorted({0, 2040, 4080, 6000, context - rows}):
                if first < 0 or first + rows > context:
                    continue
                positions(rows, first)
                inputs[0].neg_()
                inputs[3].add_(0.125)
                for graph in graphs:
                    graph.replay()
                check()
            print(dict(parameters=params, rows=rows, times=times, exact=True), flush=True)
        selected = int(aggregate_winner(measurements))
        manifests = [str(p.relative_to(base)) for p in paths[selected]]
        result = dict(
            identity=identity, measurements=measurements, selected=selected, manifests=manifests
        )
        atomic_json(record, result)
        replacements[old] = manifests
        torch.npu.synchronize()
        graphs = graph = None
        calls = entries = pointers = None
        libraries = library = entry = None
        inputs = outputs = output = None
        torch.npu.empty_cache()
    if not args.compile_only:
        for rank in model["ranks"]:
            for packet in rank["packets"]:
                for section in ("body", "head"):
                    expanded = []
                    for step in packet[section]:
                        if step["op"] == "kernel" and step["manifest"] in replacements:
                            expanded.extend(
                                dict(step, manifest=manifest)
                                for manifest in replacements[step["manifest"]]
                            )
                        else:
                            expanded.append(step)
                    packet[section] = expanded
        atomic_json(args.plan.with_name(args.plan.stem + "-softmax-tuned.json"), model)


if __name__ == "__main__":
    main()
