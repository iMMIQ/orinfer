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
    args = parser.parse_args()
    import torch
    import torch_npu  # noqa: F401
    from kernels.model.rows import row_count
    from tools.model.deepseek_v41.floating import bf16_parts
    from tools.model.deepseek_v41.tune import file_hash_bytes, npu_benchmark, read_weight
    from tools.operators.ascend import floating, vector
    from tools.operators.ascend.build import compile_kernel

    torch.set_num_threads(4)
    torch.manual_seed(20261002)
    torch.npu.set_device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    model = json.loads(args.plan.read_text())
    base = args.plan.parent
    index = json.loads((args.model_dir / model["index"]["file"]).read_text())["weight_map"]
    definitions = {b["name"]: b for b in model["ranks"][0]["buffers"]}
    replacements = {}
    seen = set()
    for packet in model["ranks"][0]["packets"]:
        for step in packet["body"] + packet["head"]:
            if step["op"] != "kernel" or step["manifest"] in seen:
                continue
            seen.add(step["manifest"])
            manifest = json.loads((base / step["manifest"]).read_text())
            kind = manifest["op"]
            if kind not in ("vector.mhc_post_pre", "floating.partials"):
                continue
            params = dict(manifest["parameters"])
            if "_row_capacity" not in params:
                continue
            capacity = params.pop("_row_capacity")
            grid_limit = params.pop("_grid_limit", 64)
            if kind == "vector.mhc_post_pre":
                configs = [dict(tile=t) for t in (256, 512, 1024, 1280, 2560)]
                factory = vector.mhc_post_pre
            else:
                configs = [dict(tile_m=t) for t in (16, 32, 64)]
                factory = floating.partials

            def build(config):
                parameters = dict(params, **config)
                function = factory(**dict(parameters, rows=row_count(capacity, True)))
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
                device=torch.npu.get_device_name(args.device),
                parameters=params,
                configs=configs,
                samples=[r for r in (17, 128, 512, 2048) if r <= capacity],
                tuner_sha256=file_hash(Path(__file__)),
                seed=20261002,
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
                path = base / record["manifest"]
                spec = json.loads(path.read_text())
                return (
                    spec["source_sha256"] == descriptor["source_sha256"]
                    and spec["compiler"] == descriptor["compiler"]
                    and spec["library"]["sha256"] == file_hash(path.parent / "kernel.so")
                )

            record = record_path(args.output, identity)
            cached = read_record(record, identity, valid)
            if cached:
                replacements[step["manifest"]] = cached["manifest"]
                continue
            paths, legal, rejected = compile_candidates(configs, build, tolerate_failures=True)
            if legal[0] != configs[0]:
                raise ValueError("Missing baseline auxiliary kernel")
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
                dtype = {"bf16": torch.bfloat16, "f32": torch.float32}[spec["dtype"]]
                if spec["access"] == "write":
                    shape[0] += 1
                    outputs.append(len(tensors))
                tensors.append(torch.randn(shape, dtype=dtype, device="npu"))
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
            for rows in sorted(set(identity["samples"] + [1, 2, 31, 129, capacity])):
                if rows > capacity:
                    continue
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
                            torch.testing.assert_close(
                                tensors[slot][:rows], ref[:rows], atol=0, rtol=0
                            )
                            assert bool((tensors[slot][rows:] == 17).all()), (
                                "Auxiliary kernel wrote tail"
                            )

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
            result = dict(
                identity=identity,
                measurements=measurements,
                rejected=rejected,
                config=legal[selected],
                manifest=chosen,
            )
            atomic_json(record, result)
            replacements[step["manifest"]] = chosen
            print(dict(op=kind, **result), flush=True)
    for rank in model["ranks"]:
        for packet in rank["packets"]:
            for step in packet["body"] + packet["head"]:
                if step["op"] == "kernel":
                    step["manifest"] = replacements.get(step["manifest"], step["manifest"])
    atomic_json(args.plan.with_name(args.plan.stem + "-aux-tuned.json"), model)


if __name__ == "__main__":
    main()
