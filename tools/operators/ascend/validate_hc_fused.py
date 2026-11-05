"""Compare fused mHC with published normalization/Sinkhorn, including Graph replay."""

import argparse
import ctypes
import json
from pathlib import Path

from tools.build.files import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference", type=Path, required=True, help="Published execution directory"
    )
    parser.add_argument("--fused", type=Path, required=True, help="Fused kernel manifest")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    import torch
    import torch_npu  # noqa: F401

    model = json.loads((args.reference / "model.json").read_text())
    fused = json.loads(args.fused.read_text())
    capacity = fused["parameters"]["_row_capacity"]
    paths = {}
    for packet in model["ranks"][0]["packets"]:
        for step in packet["body"]:
            if step.get("op") == "kernel":
                path = args.reference / step["manifest"]
                spec = json.loads(path.read_text())
                params = spec["parameters"]
                if params.get("_row_capacity", params.get("rows", 0)) < capacity:
                    continue
                if spec["op"] == "hc_prefill.reduce" and (
                    params.get("splits", 8) == 8 and params["width"] == fused["parameters"]["width"]
                ):
                    paths.setdefault(spec["op"], path)
                if spec["op"] == "vector.mhc_split_sinkhorn" and (
                    params.get("tile_rows", 1) == 32
                    and params.get("iterations", 20) == fused["parameters"]["iterations"]
                    and params.get("eps", 1e-6) == fused["parameters"]["eps"]
                ):
                    paths.setdefault(spec["op"], path)
    paths["fused"] = args.fused
    libraries = []

    def entry(path):
        spec = json.loads(path.read_text())
        library = ctypes.CDLL(str((path.parent / spec["library"]["file"]).resolve()))
        libraries.append(library)
        call = library.orinfer_ascend_kernel_v2
        call.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        call.restype = ctypes.c_int32
        return call

    calls = {key: entry(path) for key, path in paths.items()}
    torch.set_num_threads(4)
    torch.manual_seed(20261002)
    torch.npu.set_device(args.device)
    partial = torch.randn(capacity, 8, 32)
    partial[:, :, 24:] = torch.rand(capacity, 8, 8) * 2500 + 500
    partial = partial.npu()
    scale = torch.tensor([0.6, 0.7, 0.3, 0, 0, 0, 0, 0], device="npu")
    base = (torch.randn(32) * 0.1).npu()
    mix = torch.empty(capacity, 32, device="npu")
    reference = [torch.empty(capacity + 1, width, device="npu") for width in (8, 8, 16)]
    outputs = [torch.empty_like(value) for value in reference]

    def run(kind, rows):
        tensors = (
            [partial, mix]
            if kind == "hc_prefill.reduce"
            else [mix, scale, base, *reference]
            if kind == "vector.mhc_split_sinkhorn"
            else [partial, scale, base, *outputs]
        )
        pointers = (ctypes.c_void_p * len(tensors))(*(value.data_ptr() for value in tensors))
        status = calls[kind](
            pointers, len(tensors), rows, torch.npu.current_stream()._as_parameter_
        )
        if status:
            raise RuntimeError(f"mHC {kind} failed: {status}")

    def original(rows):
        run("hc_prefill.reduce", rows)
        run("vector.mhc_split_sinkhorn", rows)

    def check(rows):
        torch.npu.synchronize()
        for actual, expected in zip(outputs, reference, strict=True):
            torch.testing.assert_close(actual[:rows], expected[:rows], rtol=0, atol=0)
            assert bool((actual[rows:] == 17).all()), "Fused mHC overwrote the tail"

    reports = []
    for rows in sorted({min(capacity, n) for n in (1, 17, 127, 128, 192, 193, capacity)}):
        for value in outputs:
            value.fill_(17)
        original(rows)
        run("fused", rows)
        check(rows)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            run("fused", rows)
        partial[:, :, :24].mul_(0.9)
        scale.add_(0.003)
        original(rows)
        graph.replay()
        check(rows)
        result = dict(rows=rows, bitwise_equal=True, tail_preserved=True, changed_graph=True)
        if args.benchmark:
            from functools import partial as bind
            from tools.model.deepseek_v41.tune import npu_benchmark
            from tools.model.tuning import measure

            benchmark = bind(npu_benchmark, calls_per_replay=16)
            timings = {
                name: measure(
                    callback,
                    budget_ms=2,
                    max_repetitions=50,
                    benchmark=benchmark,
                )
                for name, callback in (
                    ("original_ms", lambda: original(rows)),
                    ("fused_ms", lambda: run("fused", rows)),
                )
            }
            result.update(timings, speedup=timings["original_ms"] / timings["fused_ms"])
        reports.append(result)
        print(reports[-1], flush=True)
        del graph
    atomic_json(args.output / "validation.json", reports)


if __name__ == "__main__":
    main()
