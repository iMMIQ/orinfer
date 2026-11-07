"""Verify independent HC streams against the original ordered projection."""

import argparse
from pathlib import Path

import tilelang
import torch

from kernels.cuda.model.hc_prefill import (
    hc_projection_finish,
    hc_projection_partial,
    hc_projection_prefill,
)
from kernels.shared.rows import explicit_rows
from orinfer_build.compiler.cuda.common import configure, export_kernel, write_json


def compile_kernel(primitive):
    return tilelang.compile(
        explicit_rows(primitive),
        out_idx=[],
        execution_backend="nvrtc",
        target={"kind": "cuda", "arch": "sm_87"},
    )


def launch(kernel, *operands):
    kernel.adapter.func(*operands, stream=torch.cuda.current_stream().cuda_stream)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure()
    torch.manual_seed(20261002)
    partial = compile_kernel(hc_projection_partial.__wrapped__(4096, 320, 10240, True))
    finish = compile_kernel(hc_projection_finish.__wrapped__(4096, 320, True))
    reference = compile_kernel(
        hc_projection_prefill.__wrapped__(
            4096,
            320,
            10240,
            block_m=64,
            block_n=128,
            group_m=4,
            silu=True,
            reduction_groups=4,
            output_dtype="bfloat16",
            dynamic_rows=True,
        )
    )
    export_kernel(partial, args.output / "aot/partial")
    export_kernel(finish, args.output / "aot/finish")
    report = dict(complete=False, cases=[])
    weights = torch.randn((320, 10240), device="cuda", dtype=torch.bfloat16) * 0.01
    for rows in [2, 17, 129, 421, 512, 1025, 4096]:
        inputs = torch.randn((rows, 10240), device="cuda", dtype=torch.bfloat16) * 0.01
        parts = torch.empty((4, rows, 320), device="cuda")
        storage = torch.full((rows + 2, 320), 91, device="cuda", dtype=torch.bfloat16)
        output, expected = storage[:rows], torch.empty_like(storage[:rows])

        def candidate():
            launch(partial, inputs, weights, parts, rows)
            launch(finish, parts, output, rows)

        candidate()
        launch(reference, inputs, weights, expected, rows)
        torch.cuda.synchronize()
        assert torch.equal(output, expected), (rows, "direct")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            candidate()
        inputs.neg_()
        launch(reference, inputs, weights, expected, rows)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(output, expected), (rows, "changed-input replay")
        assert bool((storage[rows:] == 91).all()), (rows, "output tail")
        report["cases"].append(dict(rows=rows, exact=True, graph_replay=True))
        write_json(args.output / "results.json", report)
    report["complete"] = True
    write_json(args.output / "results.json", report)


if __name__ == "__main__":
    main()
