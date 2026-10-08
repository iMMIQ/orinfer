"""Compile/export a production norm and verify changed-input CUDA graph replay in a fresh cache."""

import argparse
from pathlib import Path
import torch
import tilelang
from kernels.model.greedy import greedy_merge
from kernels.model.rows import explicit_rows
from kernels.operators.op02_residual_norm import residual_norm
from tools.operators.abi import parse_host, validate_parameter_count
from tools.operators.common import configure, export_kernel, write_json


def dynamic_greedy(output):
    """One cubin must honor actual rows, affine shapes, tails and graph inputs."""
    fn = explicit_rows(greedy_merge.__wrapped__(2049, rows=8, dynamic_rows=True))
    kernel = tilelang.compile(
        fn, out_idx=[], execution_backend="nvrtc", target={"kind": "cuda", "arch": "sm_87"}
    )
    export_kernel(kernel, output)
    (host,) = parse_host((output / "host.txt").read_text())
    validate_parameter_count((output / "kernel.cu").read_text(), host)
    assert host["ordered_arguments"][-1] == {"ctype": "ctypes.c_int32", "value": "m"}
    for rows in (3, 7):
        values = torch.tensor([1.0, 2.0, 2.0], device="cuda").repeat(rows)
        indices = torch.tensor([0, 1024, 2048], device="cuda", dtype=torch.int32).repeat(rows)
        invalid = torch.zeros(rows * 3, device="cuda", dtype=torch.int32)
        result = torch.full((rows * 2 + 2,), -99, device="cuda", dtype=torch.int32)

        def run():
            kernel.adapter.func(
                values,
                indices,
                invalid,
                result[: rows * 2],
                rows,
                stream=torch.cuda.current_stream().cuda_stream,
            )

        run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        for selected, bad in ((1024, 0), (0, 1)):
            if selected == 0:
                values.reshape(rows, 3)[:, 0] = 3.0
                invalid.reshape(rows, 3)[:, 1] = 1
            result.fill_(-99)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(
                result[: rows * 2 : 2], torch.full_like(result[:rows], selected)
            )
            torch.testing.assert_close(
                result[1 : rows * 2 : 2], torch.full_like(result[:rows], bad)
            )
            assert result[-2:].tolist() == [-99, -99]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure()
    rows, hidden = 3, 513
    kernel = residual_norm(rows, hidden=hidden)
    x = torch.randn(rows, hidden, device="cuda", dtype=torch.float16)
    r = torch.randn(rows, hidden, device="cuda", dtype=torch.float32)
    w = torch.randn(hidden, device="cuda", dtype=torch.float16)
    y = torch.empty_like(x)
    residual = torch.empty_like(r)
    for _ in range(3):
        kernel(x, r, w, y, residual)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        kernel(x, r, w, y, residual)
    for offset in (0, 1):
        x.add_(offset)
        graph.replay()
        torch.cuda.synchronize()
        expected = x.float() + r
        output = (
            expected
            * torch.rsqrt(expected.square().mean(-1, keepdim=True) + 1e-6)
            * (w.float() + 1)
        ).half()
        torch.testing.assert_close(residual, expected, atol=0, rtol=0)
        torch.testing.assert_close(y, output, atol=0.008, rtol=0.004)
    export_kernel(kernel, args.output / "norm")
    dynamic_greedy(args.output / "dynamic-greedy")
    write_json(
        args.output / "results.json",
        {
            "compiled": True,
            "tail_hidden": hidden,
            "graph_changed_input": True,
            "dynamic_rows": [3, 7],
            "affine_scalar_abi": True,
            "output_guard": True,
        },
    )


if __name__ == "__main__":
    main()
