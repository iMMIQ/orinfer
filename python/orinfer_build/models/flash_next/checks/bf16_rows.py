"""Validate shared BF16 row tiles against the direct single-row MMA order."""

import argparse
from pathlib import Path
import torch
import tilelang
from kernels.cuda.model.bf16_single import bf16_single, bf16_rows
from kernels.shared.rows import explicit_rows
from orinfer_build.tuning.projection_tuning import profile_stream
from orinfer_build.compiler.cuda.common import configure, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure()
    torch.manual_seed(20261002)
    evidence = []
    for n, groups in [(256, 4), (320, 4), (512, 1)]:
        k = 2560
        weight = torch.randn(n, groups * k, device="cuda", dtype=torch.bfloat16) * 0.013
        x = torch.randn(512, groups, k, device="cuda", dtype=torch.float16) * 0.61
        for bm in [16, 32]:
            candidate = profile_stream(
                tilelang.compile(
                    explicit_rows(bf16_rows.__wrapped__(n, k, groups, rows=1024, block_m=bm)),
                    out_idx=[],
                    execution_backend="nvrtc",
                    target={"kind": "cuda", "arch": "sm_87"},
                )
            )
            for rows in [1, 2, 5, 17, 33, 128, 512]:
                expected = torch.empty(groups, rows, n, device="cuda", dtype=torch.float32)
                storage = torch.full(
                    (groups * rows * n + 31,), 91.0, device="cuda", dtype=torch.float32
                )
                actual = storage[:-31].view(groups, rows, n)
                ref = bf16_single(n, k, groups, rows=rows)

                def reference():
                    ref.adapter.func(
                        x[:rows], weight, expected, stream=torch.cuda.current_stream().cuda_stream
                    )

                def run():
                    candidate.adapter.func(x, weight, actual, rows)

                reference()
                run()
                torch.cuda.synchronize()
                error = float((actual - expected).abs().max())
                assert torch.equal(actual, expected), (n, groups, bm, rows, error)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run()
                x[:rows].mul_(-1)
                reference()
                graph.replay()
                torch.cuda.synchronize()
                assert torch.equal(actual, expected)
                assert bool((storage[-31:] == 91).all())
                evidence.append(
                    dict(outputs=n, groups=groups, tile=bm, rows=rows, exact=True, replay=True)
                )
                print("PASS", n, groups, bm, rows, flush=True)
    write_json(args.output / "results.json", dict(complete=True, cases=evidence))


if __name__ == "__main__":
    main()
