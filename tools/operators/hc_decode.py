"""HC tile tuning: keep BF16 operands, FP32 accumulation and FP16 boundaries."""

import argparse
from pathlib import Path
import torch
from kernels.model.hyperconnection import hc_projection
from tools.operators.common import configure, benchmark, error, write_json, export_kernel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    configure()
    report = {"complete": False, "cases": []}
    for n, k in [(320, 10240), (10240, 320), (4, 10240)]:
        x = (torch.randn((1, k), device="cuda") * 0.2).half()
        weight = (torch.randn((n, k), device="cuda") * 0.03).bfloat16()
        expected = torch.empty((1, n), device="cuda", dtype=torch.float16)
        actual = torch.empty_like(expected)
        ref = hc_projection(1, n, k, dtype="float16")
        rt, _ = benchmark(lambda: ref(x, weight, expected), repetitions=20)
        case = {"N": n, "K": k, "reference": rt, "candidates": []}
        for bn in (16, 32, 64):
            ref(x, weight, expected)
            kernel = hc_projection(1, n, k, dtype="float16", block_n=bn)
            timing, graph = benchmark(lambda: kernel(x, weight, actual), repetitions=20)
            assert torch.equal(actual, expected), error(actual, expected)
            x.mul_(0.9)
            graph.replay()
            ref(x, weight, expected)
            torch.cuda.synchronize()
            assert torch.equal(actual, expected), error(actual, expected)
            case["candidates"].append({"block_n": bn, "exact": True, "timing": timing})
            export_kernel(kernel, a.output / f"hc-{n}-{k}-{bn}")
            print("HC", n, k, bn, "old", rt["median_ms"], "new", timing["median_ms"], flush=True)
        report["cases"].append(case)
        write_json(a.output / "results.json", report)
    report["complete"] = True
    write_json(a.output / "results.json", report)


if __name__ == "__main__":
    main()
