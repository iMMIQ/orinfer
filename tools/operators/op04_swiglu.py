"""Offline correctness/tuning/export for op04; Torch is reference-only."""

import argparse
import gc
import json
import re
import time
from pathlib import Path

import torch

from common import benchmark, configure, environment, error, export_kernel, identity, write_json
from kernels.operators.op04_swiglu import HIDDEN, launch, swiglu


ROWS = [1, 2, 3, 4, 5, 7, 8, 511, 512, 513, 2048, 8192]


def reference(x):
    g, u = x[:, :HIDDEN].float(), x[:, HIDDEN:].float()
    return (torch.nn.functional.silu(g) * u).half()


def physical(x, layout):
    if layout == "split":
        return x
    return torch.stack((x[:, :HIDDEN], x[:, HIDDEN:]), dim=-1).reshape(x.shape).contiguous()


def measure(kernel, x, y, repetitions):
    return benchmark(
        lambda: launch(kernel, x, y, stream=torch.cuda.current_stream().cuda_stream),
        repetitions=repetitions,
    )


def check(actual, expected):
    result = error(actual, expected)
    assert result["finite"] and result["relative_l2"] <= 0.001, result
    return result


def abi(kernel, exported, block, threads, layout):
    host = (exported / "host.txt").read_text()
    cuda = (exported / "kernel.cu").read_text()
    declaration = re.search(r'extern "C" __global__ void (\w+)\(([^;]+)\);', cuda)
    assert declaration, "Export must contain the actual CUDA entry declaration"
    arguments = [arg.strip() for arg in declaration.group(2).split(",")]
    return {
        "operator": "op04_swiglu",
        "sm": 87,
        "dynamic_dimension": "rows:int32",
        "entry_symbol": declaration.group(1),
        "ordered_arguments": [
            {
                "index": i,
                "name": arg.rsplit(" ", 1)[-1],
                "cuda_declaration": arg,
                "driver_type": "device_ptr:u64" if "*" in arg else "int32",
            }
            for i, arg in enumerate(arguments)
        ],
        "input": {"shape": ["M", 34816], "dtype": "float16", "layout": layout},
        "output": {"shape": ["M", 17408], "dtype": "float16", "layout": "row-major"},
        "launch": {
            "grid": [f"ceildiv(M*17408,{block})", 1, 1],
            "block": [threads, 1, 1],
            "shared_memory_bytes": 0,
            "cooperative": False,
        },
        "host_wrapper": host,
        "cuda_entry_declarations": re.findall(r"__global__[^\n]+", cuda),
        "argument_order_source": "host.txt generated launch declaration; not inferred from PrimFunc",
        "workspace_bytes": 0,
        "resident_weight_bytes": 0,
        "rounding": "FP32 gate/up, stable FP32 sigmoid, FP32 SiLU and multiply; final explicit FP16 cast",
        "aliasing": "X/Y disjoint, stable addresses during graph replay; caller supplies stream",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    configure()
    report = {
        "environment": environment(),
        "tuning": [],
        "cases": [],
        "semantics": "FP32 SiLU(gate)*up -> FP16; split gate first/up second",
        "source": identity(
            Path(__file__).resolve().parents[2] / "kernels/operators/op04_swiglu.py"
        ),
        "workspace_bytes": 0,
        "resident_weight_bytes": 0,
        "failures": [],
        "full_chain": "single standalone SwiGLU launch, no A8, packing excluded",
    }
    candidates = [(256, 128), (512, 128), (1024, 128), (1024, 256), (2048, 256)]
    if args.quick:
        candidates = [(1024, 256)]
    kernels = {}
    for layout in ("split", "interleaved"):
        for block, threads in candidates:
            started = time.perf_counter()
            kernel = swiglu(block, threads, layout)
            kernels[layout, block, threads] = kernel
            prepare = time.perf_counter() - started
            trial = {
                "layout": layout,
                "block": block,
                "threads": threads,
                "prepare_s": prepare,
                "timings": {},
            }
            for m in (1, 512):
                x = torch.randn((m, 2 * HIDDEN), device="cuda", dtype=torch.float16)
                p = physical(x, layout)
                y = torch.empty((m, HIDDEN), device="cuda", dtype=torch.float16)
                started = time.perf_counter()
                launch(kernel, p, y, stream=torch.cuda.current_stream().cuda_stream)
                torch.cuda.synchronize()
                trial.setdefault("first_use_ms", {})[str(m)] = 1000 * (
                    time.perf_counter() - started
                )
                trial.setdefault("error", {})[str(m)] = check(y, reference(x))
                timing, _ = measure(kernel, p, y, 60 if m == 1 else 20)
                trial["timings"][str(m)] = timing
                del x, p, y
            report["tuning"].append(trial)
            write_json(out / "progress.json", report)
    # Fixed reproducible selection, decode/prefill may choose distinct tiles.
    selected = {}
    for layout in ("split", "interleaved"):
        for mode, m in (("decode", 1), ("prefill", 512)):
            trials = [t for t in report["tuning"] if t["layout"] == layout]
            winner = min(trials, key=lambda t: t["timings"][str(m)]["median_ms"])
            selected[layout, mode] = (winner["block"], winner["threads"])
            export_dir = out / f"{layout}-{mode}"
            kernel = kernels[(layout,) + selected[layout, mode]]
            exported = export_kernel(kernel, export_dir)
            manifest = abi(kernel, export_dir, *selected[layout, mode], layout)
            manifest["artifacts"] = exported
            manifest["toolchain"] = report["environment"]
            write_json(export_dir / "abi.json", manifest)
    report["selected"] = {
        f"{k[0]}-{k[1]}": {"block": v[0], "threads": v[1]} for k, v in selected.items()
    }
    for m in ROWS:
        x = torch.randn((m, 2 * HIDDEN), device="cuda", dtype=torch.float16)
        original = x.clone()
        expected = reference(x)
        for layout in ("split", "interleaved"):
            mode = "decode" if m <= 8 else "prefill"
            kernel = kernels[(layout,) + selected[layout, mode]]
            p = physical(x, layout)
            y = torch.empty((m, HIDDEN), device="cuda", dtype=torch.float16)
            launch(kernel, p, y, stream=torch.cuda.current_stream().cuda_stream)
            torch.cuda.synchronize()
            result = {"M": m, "layout": layout, "error": check(y, expected)}
            timing, graph = measure(kernel, p, y, 60 if m <= 8 else (10 if m <= 513 else 3))
            result["timing"] = timing
            # Replay must observe a changed input, clear poisoned output, restore original.
            changed = original + 0.375
            p.copy_(physical(changed, layout))
            y.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            result["graph_changed_input"] = check(y, reference(changed))
            p.copy_(physical(original, layout))
            y.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            result["graph_restored_input"] = check(y, expected)
            result["input_bytes"] = p.numel() * p.element_size()
            result["output_bytes"] = y.numel() * y.element_size()
            result["effective_bandwidth_GB_s"] = (
                result["input_bytes"] + result["output_bytes"]
            ) / (timing["median_ms"] * 1e6)
            result["validation_process_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            report["cases"].append(result)
            write_json(out / "progress.json", report)
            del p, y, graph, changed
            gc.collect()
        del x, original, expected
    # Extreme finite outputs, zero/sign and final FP16 ties; no intentional half overflow.
    g = torch.tensor(
        [-65504, -1000, -100, -20, -1, -0.0, 0.0, 1, 20, 100, 1000, 65504],
        device="cuda",
        dtype=torch.float16,
    )
    u = torch.tensor(
        [1.0, -1.0, 100.0, 10.0, -3.0, 1.0, -1.0, -3.0, 10.0, 100.0, -1.0, 0.5],
        device="cuda",
        dtype=torch.float16,
    )
    x = torch.zeros((1, 2 * HIDDEN), device="cuda", dtype=torch.float16)
    x[0, :HIDDEN] = g.repeat((HIDDEN + len(g) - 1) // len(g))[:HIDDEN]
    x[0, HIDDEN:] = u.repeat((HIDDEN + len(u) - 1) // len(u))[:HIDDEN]
    report["extreme_cases"] = []
    for layout in ("split", "interleaved"):
        kernel = kernels[(layout,) + selected[layout, "decode"]]
        p, y = physical(x, layout), torch.empty((1, HIDDEN), device="cuda", dtype=torch.float16)
        launch(kernel, p, y, stream=torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        report["extreme_cases"].append(
            {
                "layout": layout,
                "error": check(y, reference(x)),
                "first12_output": y[0, :12].float().tolist(),
            }
        )
    report["budget"] = {
        "decode_M1_ms": 0.008,
        "prefill512_ms": 0.100,
        "prefill2048_ms": 0.4,
        "prefill8192_ms": 1.6,
        "status": "compare split selected measurements only; interleaved requires upstream layout",
    }
    report["status"] = "passed"
    write_json(out / "results.json", report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "selected": report["selected"],
                "cases": [
                    {
                        "M": c["M"],
                        "layout": c["layout"],
                        "ms": c["timing"]["median_ms"],
                        "relative_l2": c["error"]["relative_l2"],
                    }
                    for c in report["cases"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
