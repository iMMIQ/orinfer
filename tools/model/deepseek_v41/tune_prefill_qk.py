"""Select live KV traversal for large prefill with exact scores and Graph checks."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import ctypes
import json
import multiprocessing
from pathlib import Path

from tools.build.files import atomic_json, file_hash
from tools.model.row_selection import aggregate_winner
from tools.model.tuning import measure, read_record, record_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=7)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--compile-workers", type=int, default=6)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    from tools.model.deepseek_v41.build import compile_task

    model = json.loads(args.plan.read_text())
    base = args.plan.parent
    tasks, geometries = [], {}
    for packet in model["ranks"][0]["packets"]:
        if packet.get("min_rows", 0) < 512:
            continue
        for step in packet["body"]:
            if step["op"] != "kernel" or step["manifest"] in geometries:
                continue
            original = json.loads((base / step["manifest"]).read_text())
            if original["op"] != "attention_prefill.qk":
                continue
            params = {k: v for k, v in original["parameters"].items() if not k.startswith("_")}
            capacity = original["parameters"]["_row_capacity"]
            configs = [
                dict(
                    live_tiles=False,
                    key_splits=1,
                    grid_limit=original["parameters"].get("_grid_limit", 64),
                )
            ]
            geometry = dict(
                original=original, configs=configs, paths=[step["manifest"]], placeholders=[]
            )
            geometries[step["manifest"]] = geometry
            choices = [(False, 1, configs[0]["grid_limit"])] + [
                (True, splits, limit) for splits in (1, 2, 4) for limit in (24, 48, 64)
            ]
            for live, splits, limit in choices:
                config = dict(live_tiles=live, key_splits=splits, grid_limit=limit)
                placeholder = f"@live_qk{len(tasks)}"
                configs.append(config)
                geometry["placeholders"].append(placeholder)
                tasks.append(
                    dict(
                        module="tools.operators.ascend.attention_prefill",
                        name="qk",
                        arguments=dict(params, live_tiles=live, key_splits=splits),
                        specs=original["operands"],
                        row_capacity=capacity,
                        grid_limit=limit,
                        dynamic_arguments={"rows": 1},
                        placeholder=placeholder,
                    )
                )
    if not geometries:
        raise ValueError("Missing large-prefill QK template")
    compiled = {}
    with ProcessPoolExecutor(
        max_workers=args.compile_workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        for placeholder, path in pool.map(compile_task, [(task, str(base)) for task in tasks]):
            compiled[placeholder] = path
            print(dict(compiled=len(compiled), total=len(tasks)), flush=True)
    for geometry in geometries.values():
        geometry["paths"].extend(compiled[p] for p in geometry.pop("placeholders"))
    atomic_json(args.output / "candidates.json", geometries)
    if args.compile_only:
        return
    import torch
    import torch_npu  # noqa: F401
    from tools.model.deepseek_v41.tune import npu_benchmark

    torch.set_num_threads(4)
    torch.manual_seed(20261002)
    torch.npu.set_device(args.device)
    replacements = {}
    for old_path, geometry in geometries.items():
        manifest = geometry["original"]
        params = manifest["parameters"]
        capacity, context = params["_row_capacity"], params["capacity"]
        ratio, compressed = params.get("ratio", 2), params.get("compressed", True)
        length = context + (context // ratio if compressed else 0)
        paths = geometry["paths"]
        identity = dict(
            configs=geometry["configs"],
            descriptors=[json.loads((base / p).read_text()) for p in paths],
            checks=[1, 17, 512, 513, 2047, capacity],
            samples=[512, capacity],
            device=torch.npu.get_device_name(args.device),
            tuner_sha256=file_hash(Path(__file__)),
            seed=20261002,
        )
        record = record_path(args.output, identity)

        def valid(result):
            chosen = base / result["manifest"]
            return (
                chosen.is_file()
                and result["manifest"] in paths
                and file_hash(chosen.parent / "kernel.so")
                == json.loads(chosen.read_text())["library"]["sha256"]
            )

        cached = read_record(record, identity, valid)
        if cached is not None:
            replacements[old_path] = cached["manifest"]
            print(dict(cached=True, parameters=params, config=cached["config"]), flush=True)
            continue
        query = torch.randn(capacity * 8, 512).bfloat16().npu()
        swa = (torch.randn(context, 512) * 0.2).bfloat16().npu()
        long = (torch.randn(context // ratio, 512) * 0.2).bfloat16().npu()
        positions = torch.empty(capacity, dtype=torch.int32, device="npu")
        output = torch.empty((capacity + 1) * 8, length, device="npu")
        expected = torch.empty_like(output)
        pointers = (ctypes.c_void_p * 5)(
            *[t.data_ptr() for t in (query, swa, long, positions, output)]
        )
        reference = (ctypes.c_void_p * 5)(
            query.data_ptr(),
            swa.data_ptr(),
            long.data_ptr(),
            positions.data_ptr(),
            expected.data_ptr(),
        )
        libraries, entries = [], []
        for path in paths:
            library = ctypes.CDLL(str((base / path).parent / "kernel.so"))
            libraries.append(library)
            entry = library.orinfer_ascend_kernel_v2
            entry.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_uint64,
                ctypes.c_uint64,
                ctypes.c_void_p,
            ]
            entry.restype = ctypes.c_int32
            entries.append(entry)
        records, rejected, invalid = [], [], set()
        for rows in sorted({1, 17, 512, 513, 2047, capacity}):
            if rows > capacity:
                continue
            for position in (0, context - rows):
                positions[:rows].copy_(torch.arange(rows, dtype=torch.int32) + position)

                def launch(which, arguments=pointers):
                    status = entries[which](
                        arguments, 5, rows, torch.npu.current_stream()._as_parameter_
                    )
                    if status:
                        raise RuntimeError(f"Prefill QK launch {status}")

                def baseline_run():
                    expected.fill_(17)
                    launch(0, reference)

                baseline_run()
                # Anchor one written score row in independent FP32 arithmetic.
                if rows == capacity:
                    torch.npu.synchronize()
                    bank = torch.cat((swa.cpu(), long.cpu())) if compressed else swa.cpu()
                    begin = max(0, (position - 127) // 128) * 128
                    end = min(context, ((position + 16 + 127) // 128) * 128)
                    cols = torch.arange(begin, end)
                    wanted = query[0].cpu().float() @ bank[cols].float().T
                    torch.testing.assert_close(
                        expected[0, begin:end].cpu(), wanted, atol=5e-5, rtol=5e-5
                    )
                    if compressed:
                        stop = min(context // ratio, ((position + 16) // ratio + 127) // 128 * 128)
                        wanted = query[0].cpu().float() @ long[:stop].cpu().float().T
                        torch.testing.assert_close(
                            expected[0, context : context + stop].cpu(),
                            wanted,
                            atol=5e-5,
                            rtol=5e-5,
                        )
                times = {}
                for i in range(len(entries)):
                    if i in invalid:
                        continue
                    output.fill_(17)

                    def run():
                        launch(i)

                    position_state = [position]

                    def change():
                        query.neg_()
                        position_state[0] = context - rows if position_state[0] == 0 else 0
                        positions[:rows].copy_(
                            torch.arange(rows, dtype=torch.int32) + position_state[0]
                        )
                        output.fill_(17)
                        baseline_run()

                    def check():
                        torch.npu.synchronize()
                        torch.testing.assert_close(output, expected, atol=0, rtol=0)

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
                                config=geometry["configs"][i],
                                rows=rows,
                                position=position,
                                reason=str(error),
                            )
                        )
                        continue
                    times[str(i)] = timing
                if rows in (512, capacity) and position == 0:
                    records.append(dict(rows=rows, times=times))
                print(
                    dict(
                        compressed=compressed,
                        ratio=ratio,
                        rows=rows,
                        position=position,
                        times=times,
                    ),
                    flush=True,
                )
        for row in records:
            row["times"] = {k: v for k, v in row["times"].items() if int(k) not in invalid}
        winner = int(aggregate_winner(records))
        result = dict(
            identity=identity,
            measurements=records,
            rejected=rejected,
            config=geometry["configs"][winner],
            manifest=paths[winner],
        )
        atomic_json(record, result)
        replacements[old_path] = paths[winner]
        torch.npu.empty_cache()
        print(dict(selected=geometry["configs"][winner], manifest=paths[winner]), flush=True)
    for rank in model["ranks"]:
        for packet in rank["packets"]:
            if packet.get("min_rows", 0) < 512:
                continue
            for step in packet["body"]:
                if step["op"] == "kernel":
                    step["manifest"] = replacements.get(step["manifest"], step["manifest"])
    output = base / (args.plan.stem + "-qk-tuned.json")
    atomic_json(output, model)
    print(output, flush=True)


if __name__ == "__main__":
    main()
