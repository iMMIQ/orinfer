"""Validate symbolic HC views, padded inputs, tail writes and graph replay."""

import argparse
from pathlib import Path

import torch
import tilelang

from kernels.cuda.model.hc_prefill import hc_up_mix_prefill
from kernels.shared.rows import explicit_rows
from orinfer_build.compiler.cuda.common import benchmark, configure, export_kernel, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure()
    torch.manual_seed(20261002)
    kernels = {}
    for name, options in [
        ("general", {}),
        ("aligned", {"aligned_rows": True}),
        ("padded", {"padded_inputs": True}),
    ]:
        fn = hc_up_mix_prefill.__wrapped__(4096, 2560, 320, dynamic_rows=True, **options)
        kernel = tilelang.compile(
            explicit_rows(fn),
            out_idx=[],
            execution_backend="nvrtc",
            target={"kind": "cuda", "arch": "sm_87"},
        )
        kernels[name] = kernel
        export_kernel(kernel, args.output / "aot" / name)
    report = dict(complete=False, cases=[])

    def launch(kernel, *inputs):
        kernel.adapter.func(*inputs, stream=torch.cuda.current_stream().cuda_stream)

    for rows in [17, 128, 129, 256, 421, 512, 588, 1025, 2048]:
        capacity = (rows + 127) // 128 * 128
        activation = torch.randn((capacity, 320), device="cuda", dtype=torch.bfloat16) * 0.02
        weights = torch.randn((10240, 320), device="cuda", dtype=torch.bfloat16) * 0.02
        residual = torch.randn((capacity, 4, 2560), device="cuda", dtype=torch.float16) * 0.1
        gamma = torch.randn((4, 2560), device="cuda", dtype=torch.float32) * 0.1
        inv = torch.rand((capacity, 4), device="cuda") + 0.1
        expected = torch.empty((rows, 2560), device="cuda", dtype=torch.float16)
        storage = torch.full((rows + 1, 2560), 99, device="cuda", dtype=torch.float16)
        output = storage[:rows]
        activation[rows:] = float("nan")
        residual[rows:] = float("nan")
        inv[rows:] = float("nan")
        inputs = [activation, weights, residual, gamma, inv]
        reference = hc_up_mix_prefill(rows, 2560, 320)
        launch(reference, *inputs, expected)
        for name, kernel in kernels.items():
            if name == "aligned" and rows % 128:
                continue

            def run():
                launch(kernel, *inputs, output, rows)

            timing, graph = benchmark(run, repetitions=7)
            assert torch.equal(output, expected), (rows, name)
            assert bool((storage[-1] == 99).all())
            residual[:rows].neg_()
            launch(reference, *inputs, expected)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(output, expected), (rows, name, "graph")
            assert bool((storage[-1] == 99).all())
            row = dict(
                rows=rows,
                implementation=name,
                exact=True,
                nan_padding_isolated=True,
                tail_unchanged=True,
                timing=timing,
            )
            report["cases"].append(row)
            write_json(args.output / "results.json", report)
            print("HC ALIGNMENT", row, flush=True)
    report["complete"] = True
    write_json(args.output / "results.json", report)


if __name__ == "__main__":
    main()
