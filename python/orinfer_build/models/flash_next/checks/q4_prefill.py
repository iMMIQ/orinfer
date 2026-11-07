"""Validate exact Q4 prefill implementations and temporary INT8 expansion."""

import argparse
from pathlib import Path

import torch
import tilelang

from kernels.cuda.model.int8_projection import int8_projection
from kernels.cuda.model.q4_projection import MAGNITUDES, q4_projection, unpack_q4
from kernels.shared.rows import explicit_rows
from orinfer_build.compiler.cuda.common import benchmark, configure, export_kernel, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lengths", nargs="+", type=int, default=[17, 129, 512])
    parser.add_argument("--outputs", nargs="+", type=int, default=[10240])
    args = parser.parse_args()
    if any(not 2 <= n <= 4096 for n in args.lengths):
        parser.error("Lengths must fit the dynamic prefill arena")
    configure()
    torch.manual_seed(20261002)
    report = dict(complete=False, cases=[])

    def compile_function(function):
        return tilelang.compile(
            explicit_rows(function),
            out_idx=[],
            execution_backend="nvrtc",
            target={"kind": "cuda", "arch": "sm_87"},
        )

    def launch(kernel, *inputs):
        kernel.adapter.func(*inputs, stream=torch.cuda.current_stream().cuda_stream)

    for outputs in args.outputs:
        width = 2560
        packed = torch.randint(0, 65536, (outputs, width // 4), device="cuda").to(torch.uint16)
        expanded = torch.empty((outputs, width), device="cuda", dtype=torch.int8)
        unpack = unpack_q4(outputs, width)
        launch(unpack, packed, expanded.view(torch.int32))
        # Independently verify all 16 codes, including signed packed stores.
        words = packed.to(torch.int32)
        codes = (words[:, :, None] >> (torch.arange(4, device="cuda") * 4)) & 15
        values = torch.tensor(MAGNITUDES, device="cuda", dtype=torch.int32)[codes & 7]
        expected_weights = torch.where(codes < 8, values, -values).flatten(1).to(torch.int8)
        assert torch.equal(expanded, expected_weights)
        candidates = {}
        for bm, bn in [(32, 64), (64, 64), (64, 128), (128, 128)]:
            name = f"fused-{bm}-{bn}"
            candidates[name] = compile_function(
                q4_projection.__wrapped__(
                    4096, outputs, width, BN=bn, block_m=bm, dynamic_rows=True
                )
            )
            export_kernel(candidates[name], args.output / "aot" / f"{outputs}-{name}")
        name = "fused-128-128-bk256"
        candidates[name] = compile_function(
            q4_projection.__wrapped__(
                4096,
                outputs,
                width,
                BN=128,
                BK=256,
                stages=1,
                block_m=128,
                dynamic_rows=True,
            )
        )
        export_kernel(candidates[name], args.output / "aot" / f"{outputs}-{name}")
        for bm, bn in [(64, 128), (128, 128)]:
            name = f"expanded-{bm}-{bn}"
            candidates[name] = compile_function(
                int8_projection.__wrapped__(
                    4096, outputs, width, block_m=bm, block_n=bn, block_k=128, dynamic_rows=True
                )
            )
            export_kernel(candidates[name], args.output / "aot" / f"{outputs}-{name}")
        export_kernel(unpack, args.output / "aot" / f"{outputs}-unpack")
        scales = torch.full((outputs,), 0.001, device="cuda", dtype=torch.float16)
        for rows in args.lengths:
            activation = torch.randint(-128, 128, (rows, width), device="cuda", dtype=torch.int8)
            token_scales = torch.full((rows,), 0.001, device="cuda", dtype=torch.float16)
            reference = q4_projection(rows, outputs, width, block_m=32)
            expected = torch.empty((rows, outputs), device="cuda", dtype=torch.float16)
            launch(reference, activation, packed, scales, token_scales, expected)
            for name, kernel in candidates.items():
                storage = torch.full((rows + 1, outputs), 99, device="cuda", dtype=torch.float16)
                result = storage[:rows]

                def run():
                    if name.startswith("expanded"):
                        launch(unpack, packed, expanded.view(torch.int32))
                        launch(kernel, activation, expanded, scales, token_scales, result, rows)
                    else:
                        launch(kernel, activation, packed, scales, token_scales, result, rows)

                timing, graph = benchmark(run, repetitions=7)
                assert torch.equal(result, expected), (name, rows, outputs)
                assert bool((storage[-1] == 99).all())
                activation.zero_()
                result.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                assert bool((result == 0).all())
                activation.random_(-128, 128)
                launch(reference, activation, packed, scales, token_scales, expected)
                graph.replay()
                torch.cuda.synchronize()
                assert torch.equal(result, expected)
                row = dict(
                    rows=rows, outputs=outputs, implementation=name, exact=True, timing=timing
                )
                report["cases"].append(row)
                write_json(args.output / "results.json", report)
                print("Q4 PREFILL", row, flush=True)
    report["complete"] = True
    write_json(args.output / "results.json", report)


if __name__ == "__main__":
    main()
