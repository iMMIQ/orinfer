"""Validate larger warm projections against exact 16-row execution.

Uses the prepared model's real packed W4 tensors, FP32 split-K outputs,
non-aligned row tails and changed-input CUDA graph replay.
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from kernels.model.w4_small_m import w4_small_m
from kernels.operators.op23_final_norm import final_norm
from tools.operators.common import configure, write_json


def run(model, output):
    configure()
    weights = model / "cache/weights"
    index = json.loads((weights / "model.safetensors.index.json").read_text())["weight_map"]

    def tensor(name):
        with safe_open(str(weights / index[name]), framework="pt", device="cpu") as f:
            value = f.get_tensor(name)
        if name.endswith("_P"):
            value = value.view(torch.uint32)
        return value.cuda()

    results = []
    for name, n, k, split, tile in [
        ("MtpFC", 5120, 10240, 8, 128),
        ("MtpIn", 14336, 5120, 1, 128),
        ("MtpOut", 5120, 6144, 8, 128),
        ("MtpGateUp", 34816, 5120, 1, 128),
        ("MtpDown", 5120, 17408, 8, 64),
    ]:
        arrays = [tensor(name + suffix) for suffix in ["_P", "_S", "_Z"]]
        dtype = "float32" if split > 1 else "float16"
        base = w4_small_m(16, n, k, split, dtype, TILE_N=tile).torch_function
        for rows in [17, 64, 128, 512]:
            a = torch.randn(rows, k, device="cuda", dtype=torch.float16)
            y = torch.empty(split, rows, n, device="cuda", dtype=getattr(torch, dtype))
            kernel = w4_small_m(rows, n, k, split, dtype, TILE_N=tile).torch_function

            def call():
                kernel(a, *arrays, y)

            def check():
                for start in range(0, rows, 16):
                    count = min(16, rows - start)
                    inputs = torch.zeros(16, k, device="cuda", dtype=torch.float16)
                    inputs[:count].copy_(a[start : start + count])
                    expected = torch.empty(split, 16, n, device="cuda", dtype=y.dtype)
                    base(inputs, *arrays, expected)
                    assert torch.equal(y[:, start : start + count], expected[:, :count]), (
                        name,
                        rows,
                        start,
                    )

            call()
            check()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                call()
            a.mul_(0.5)
            graph.replay()
            check()
            torch.cuda.synchronize()
            results.append(
                dict(weight=name, rows=rows, bit_exact=True, changed_input_graph_replay=True)
            )
            print(name, rows, "passed", flush=True)
            write_json(output / "result.json", dict(status="running", projections=results))
        arrays = None
        base = None
        kernel = None
        a = None
        del y
        del graph
    for rows in [64, 128, 512]:
        x = torch.randn(rows, 5120, device="cuda", dtype=torch.float16)
        r = torch.randn(rows, 5120, device="cuda", dtype=torch.float32)
        w = torch.randn(5120, device="cuda", dtype=torch.float16)
        i = torch.tensor([rows - 1], device="cuda", dtype=torch.int32)
        y = torch.empty(1, 5120, device="cuda", dtype=torch.float16)
        expected = torch.empty_like(y)
        final_norm(rows, 1).torch_function(x, r, i, w, expected)
        final_norm(rows, 1, last_row=True).torch_function(x, r, i, w, y)
        assert torch.equal(y, expected)
    write_json(
        output / "result.json",
        dict(status="passed", projections=results, last_row_norm_bit_exact=True, seed=20261002),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.model, a.output)
