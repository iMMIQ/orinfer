"""Select causal index query/key tiles with exact FP32 order and Graph checks."""

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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=7)
    parser.add_argument("--rows", type=int, nargs="+", default=[128, 512, 2048])
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--mask-only", action="store_true")
    parser.add_argument("--vector-only", action="store_true")
    args = parser.parse_args()
    if args.mask_only and args.vector_only:
        parser.error("Choose one indexer operator filter")
    args.output.mkdir(parents=True, exist_ok=True)
    import torch
    import torch_npu  # noqa: F401
    from kernels.model.rows import row_count
    from tools.model.deepseek_v41.tune import npu_benchmark
    from tools.operators.ascend.build import compile_kernel
    from tools.operators.ascend.indexer_cube import causal_logits

    torch.set_num_threads(4)
    torch.manual_seed(20261002)
    if not args.compile_only:
        torch.npu.set_device(args.device)
    model = json.loads(args.plan.read_text())
    base = args.plan.parent
    replacements, seen = {}, set()
    for packet in model["ranks"][0]["packets"]:
        for step in packet["body"] + packet["head"]:
            if step["op"] != "kernel" or step["manifest"] in seen:
                continue
            seen.add(step["manifest"])
            manifest = json.loads((base / step["manifest"]).read_text())
            if manifest["op"] != "indexer_cube.causal_logits":
                continue
            params = dict(manifest["parameters"])
            capacity = params.pop("_row_capacity")
            limit = params.pop("_grid_limit", 64)
            length = params["length"]
            # The same physical workspace accommodates every selected key tile.
            if length % 128:
                raise ValueError("Indexer tuning requires a 128-aligned key bank")
            samples = sorted({r for r in args.rows if 1 <= r <= capacity})
            if not samples:
                raise ValueError("Missing indexer measurement rows")
            baseline = dict(vector_scales=False, tile_rows=1, tile_s=params.get("tile_s", 64))
            configs = [baseline] + [
                dict(vector_scales=True, tile_rows=m, tile_s=n)
                for m in (1, 2, 4, 8)
                for n in (64, 128)
            ]
            if args.mask_only:
                baseline = dict(
                    vector_scales=params.get("vector_scales", True),
                    tile_rows=params.get("tile_rows", 1),
                    tile_s=params.get("tile_s", 64),
                    vector_mask=False,
                )
                configs = [baseline, dict(baseline, vector_mask=True)]
            if args.vector_only:
                baseline = dict(
                    direct_sum=params.get("direct_sum", False),
                    prefetch_inputs=params.get("prefetch_inputs", False),
                )
                configs = [baseline] + [
                    dict(direct_sum=direct, prefetch_inputs=prefetch)
                    for direct, prefetch in (
                        (False, False),
                        (True, False),
                        (False, True),
                        (True, True),
                    )
                    if dict(direct_sum=direct, prefetch_inputs=prefetch) != baseline
                ]

            def build(config):
                parameters = dict(params, **config)
                return compile_kernel(
                    causal_logits(**dict(parameters, rows=row_count(capacity, True))),
                    manifest["op"],
                    manifest["operands"],
                    parameters,
                    base / "cache",
                    row_capacity=capacity,
                    grid_limit=limit,
                )

            baseline_path = build(baseline)
            descriptor = json.loads(baseline_path.read_text())
            identity = dict(
                source=descriptor["source_sha256"],
                compiler=descriptor["compiler"],
                device=None if args.compile_only else torch.npu.get_device_name(args.device),
                parameters=params,
                configs=configs,
                samples=samples,
                tuner_sha256=file_hash(Path(__file__)),
                seed=20261002,
            )

            def valid(record):
                path = base / record["manifest"]
                spec = json.loads(path.read_text())
                return (
                    spec["source_sha256"] == descriptor["source_sha256"]
                    and spec["compiler"] == descriptor["compiler"]
                    and spec["library"]["sha256"] == file_hash(path.parent / "kernel.so")
                )

            record = record_path(args.output, identity)
            cached = read_record(record, identity, valid) if not args.compile_only else None
            if cached:
                replacements[step["manifest"]] = cached["manifest"]
                continue
            paths, legal, rejected = compile_candidates(configs, build, tolerate_failures=True)
            if legal[0] != baseline:
                raise ValueError("Missing scalar-order indexer reference")
            if args.compile_only:
                print(dict(length=length, compiled=len(paths), rejected=rejected), flush=True)
                continue
            libraries = [ctypes.CDLL(str(p.parent / "kernel.so")) for p in paths]
            entries = [lib.orinfer_ascend_kernel_v2 for lib in libraries]
            for entry in entries:
                entry.argtypes = [
                    ctypes.POINTER(ctypes.c_void_p),
                    ctypes.c_uint64,
                    ctypes.c_uint64,
                    ctypes.c_void_p,
                ]
                entry.restype = ctypes.c_int32
            tensors = [
                torch.randint(-127, 128, (capacity, 32, 128), dtype=torch.int8, device="npu"),
                (torch.rand(capacity, 32, device="npu") * 0.01).half(),
                torch.randint(-127, 128, (length, 128), dtype=torch.int8, device="npu"),
                (torch.rand(length, device="npu") * 0.01).half(),
                torch.randn(capacity, 32, device="npu").bfloat16(),
                torch.empty(capacity, length, dtype=torch.int32, device="npu"),
                torch.empty(capacity + 1, length, device="npu"),
                torch.empty(capacity * 32, length, dtype=torch.int32, device="npu"),
            ]
            expected = torch.empty_like(tensors[6])
            pointers = (ctypes.c_void_p * 8)(*[t.data_ptr() for t in tensors])
            reference = (ctypes.c_void_p * 8)(
                *[expected.data_ptr() if i == 6 else t.data_ptr() for i, t in enumerate(tensors)]
            )
            measurements, invalid = [], set()
            checks = sorted(set(samples + [1, 2, 17, 31, 129, capacity]))
            ratio = model["capacity"] // length
            for rows in checks:
                if rows > capacity:
                    continue
                for position in (0, model["capacity"] - rows):
                    lengths = (torch.arange(rows) + position + 1) // ratio
                    ids = torch.arange(length).expand(rows, -1)
                    ids = torch.where(ids < lengths[:, None], ids, -1).int()
                    tensors[5][:rows].copy_(ids)

                    def launch(which, arguments=pointers):
                        status = entries[which](
                            arguments, 8, rows, torch.npu.current_stream()._as_parameter_
                        )
                        if status:
                            raise RuntimeError(f"Indexer launch failed: {status}")

                    def baseline_run():
                        expected.fill_(17)
                        launch(0, reference)

                    baseline_run()
                    times = {}
                    for i in range(len(entries)):
                        if i in invalid:
                            continue
                        tensors[6].fill_(17)

                        def run():
                            launch(i)

                        def change():
                            tensors[0].neg_()
                            baseline_run()

                        def check():
                            torch.npu.synchronize()
                            torch.testing.assert_close(
                                tensors[6][:rows], expected[:rows], atol=0, rtol=0
                            )
                            assert bool((tensors[6][rows:] == 17).all()), "Indexer wrote row tail"

                        try:
                            run()
                            check()
                            timing = measure(
                                run,
                                budget_ms=4,
                                max_repetitions=16,
                                change=change,
                                check=check,
                                benchmark=npu_benchmark,
                            )
                        except AssertionError as error:
                            if i == 0:
                                raise
                            invalid.add(i)
                            rejected.append(
                                dict(
                                    config=legal[i], rows=rows, position=position, reason=str(error)
                                )
                            )
                            continue
                        times[str(i)] = timing
                    if rows in samples and position == 0:
                        measurements.append(dict(rows=rows, times=times))
                    print(
                        dict(length=length, rows=rows, position=position, times=times), flush=True
                    )
            for sample in measurements:
                sample["times"] = {
                    k: v for k, v in sample["times"].items() if int(k) not in invalid
                }
            selected = int(aggregate_winner(measurements))
            chosen = str(paths[selected].relative_to(base))
            result = dict(
                identity=identity,
                measurements=measurements,
                rejected=rejected,
                config=legal[selected],
                manifest=chosen,
            )
            atomic_json(record, result)
            replacements[step["manifest"]] = chosen
            print(dict(length=length, selected=legal[selected]), flush=True)
    if args.compile_only:
        return
    for rank in model["ranks"]:
        for packet in rank["packets"]:
            for step in packet["body"] + packet["head"]:
                if step["op"] == "kernel":
                    step["manifest"] = replacements.get(step["manifest"], step["manifest"])
    atomic_json(args.plan.with_name(args.plan.stem + "-indexer-tuned.json"), model)


if __name__ == "__main__":
    main()
