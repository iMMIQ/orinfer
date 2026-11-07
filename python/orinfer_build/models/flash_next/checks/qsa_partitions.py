"""Verify fixed QSA partitions while skipping inactive attention and merge work."""

import argparse
from pathlib import Path

import torch
import tilelang

from kernels.cuda.model.qsa import sparse_merge
from kernels.cuda.model.qsa_attention import sparse_attention
from kernels.shared.rows import explicit_rows
from orinfer_build.compiler.cuda.common import benchmark, configure, export_kernel, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lengths", nargs="+", type=int, default=[2, 17, 129, 512, 2048])
    args = parser.parse_args()
    configure()
    torch.manual_seed(20261002)
    report = dict(complete=False, cases=[])
    capacity = 8192

    def launch(kernel, *inputs):
        kernel.adapter.func(*inputs, stream=torch.cuda.current_stream().cuda_stream)

    def compile_function(function):
        return tilelang.compile(
            explicit_rows(function),
            out_idx=[],
            execution_backend="nvrtc",
            target={"kind": "cuda", "arch": "sm_87"},
        )

    attention = compile_function(
        sparse_attention.__wrapped__(
            4096,
            capacity,
            packed=True,
            dynamic_rows=True,
            skip_empty=True,
        )
    )
    merge = compile_function(
        sparse_merge.__wrapped__(
            4096,
            dynamic_rows=True,
            split_size=256,
        )
    )
    export_kernel(attention, args.output / "aot/attention")
    export_kernel(merge, args.output / "aot/merge")
    key = torch.randint(-128, 128, (capacity, 2, 256), device="cuda", dtype=torch.int8).view(
        torch.uint32
    )
    value = torch.randint(-128, 128, (capacity, 2, 256), device="cuda", dtype=torch.int8).view(
        torch.uint32
    )
    ks = torch.full((capacity, 2, 4), 0.001, device="cuda", dtype=torch.float16)
    vs = ks.clone()
    position = torch.zeros(1, device="cuda", dtype=torch.int32)
    for rows in args.lengths:
        query = torch.randn((rows, 24, 256), device="cuda", dtype=torch.float16) * 0.02
        gate = query.clone()
        selected = (
            torch.arange(2051, device="cuda", dtype=torch.int32)[None, :]
            .expand(rows, -1)
            .contiguous()
        )
        fixed_attention = sparse_attention(rows, capacity, packed=True)
        fixed_merge = sparse_merge(rows)
        parts = [
            torch.empty((rows, 24, 8), device="cuda"),
            torch.empty((rows, 24, 8), device="cuda"),
            torch.empty((rows, 24, 8, 256), device="cuda"),
        ]
        candidate_parts = [torch.full_like(x, float("nan")) for x in parts]
        expected = torch.empty_like(query)
        output = torch.empty_like(query)
        for context in [0, 37, 4096]:
            position.fill_(context)
            for x in candidate_parts:
                x.fill_(float("nan"))

            def reference():
                launch(fixed_attention, query, key, value, ks, vs, selected, position, *parts)
                launch(fixed_merge, *parts, gate, expected)

            def run():
                launch(
                    attention, query, key, value, ks, vs, selected, position, *candidate_parts, rows
                )
                launch(merge, *candidate_parts, gate, output, position, rows)

            reference()
            before, _ = benchmark(reference, repetitions=3)
            after, graph = benchmark(run, repetitions=3)
            assert torch.equal(output, expected), (
                rows,
                context,
                (output - expected).abs().max().item(),
            )
            query.neg_()
            reference()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(output, expected), (rows, context, "graph")
            row = dict(
                rows=rows,
                context=context,
                exact=True,
                baseline_ms=before["median_ms"],
                optimized_ms=after["median_ms"],
            )
            report["cases"].append(row)
            write_json(args.output / "results.json", report)
            print("QSA PARTITIONS", row, flush=True)

    report["complete"] = True
    write_json(args.output / "results.json", report)


if __name__ == "__main__":
    main()
