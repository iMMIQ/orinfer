"""Verify wide row MMA arithmetic against the original HC and router kernels."""

import argparse
from pathlib import Path
import numpy as np
import torch
import tilelang
from kernels.cuda.model.bf16_single import bf16_rows
from kernels.cuda.model.flash_next import dense_projection
from kernels.cuda.model.hyperconnection import hc_down_partial, hc_up_mix
from kernels.cuda.model.hc_bf16_cache import hc_up13_mix
from kernels.shared.rows import explicit_rows
from orinfer_build.tuning.projection_tuning import profile_stream
from orinfer_build.models.flash_next.hc_bf16 import pack
from orinfer_build.compiler.cuda.common import configure, write_json


def compile_rows(fn):
    return profile_stream(
        tilelang.compile(
            explicit_rows(fn),
            out_idx=[],
            execution_backend="nvrtc",
            target={"kind": "cuda", "arch": "sm_87"},
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure()
    torch.manual_seed(20261002)
    evidence = []
    for n, groups in [(320, 4), (512, 1)]:
        weight = torch.randn(n, groups * 2560, device="cuda", dtype=torch.bfloat16) * 0.013
        x = torch.randn(512, groups * 2560, device="cuda", dtype=torch.float16) * 0.61
        for bm in [16, 32, 64]:
            kernel = compile_rows(
                bf16_rows.__wrapped__(n, 2560, groups, rows=1024, block_m=bm, weight_left=False)
            )
            for rows in [17, 33, 128, 512]:
                expected = torch.empty(groups, rows, n, device="cuda", dtype=torch.float32)
                actual = torch.empty_like(expected)
                reference = (
                    hc_down_partial(rows, n, groups * 2560)
                    if groups == 4
                    else dense_projection(rows, n, 2560, "bfloat16", "float32")
                )
                reference.adapter.func(
                    x[:rows], weight, expected, stream=torch.cuda.current_stream().cuda_stream
                )
                kernel.adapter.func(x, weight, actual, rows)
                torch.cuda.synchronize()
                assert torch.equal(expected, actual), (
                    "down/router",
                    n,
                    bm,
                    rows,
                    float((expected - actual).abs().max()),
                )
                evidence.append(dict(kind="down/router", outputs=n, tile=bm, rows=rows, exact=True))
    rng = np.random.default_rng(20261002)
    raw = (
        (rng.integers(0, 2, (80, 5, 128, 64), dtype=np.uint16) << 15)
        | (rng.integers(110, 119, (80, 5, 128, 64), dtype=np.uint16) << 7)
        | rng.integers(0, 128, (80, 5, 128, 64), dtype=np.uint16)
    )
    raw.reshape(-1)[0] = np.uint16(2 << 7)
    cache = pack(raw)
    weight = torch.tensor(
        raw.reshape(80, 5, 4, 32, 64)
        .transpose(2, 0, 3, 1, 4)
        .copy()
        .reshape(10240, 320)
        .view(np.int16),
        device="cuda",
    ).view(torch.bfloat16)
    lo = torch.tensor(cache["low"].view(np.uint32), device="cuda", dtype=torch.uint32)
    hi = torch.tensor(cache["high"], device="cuda", dtype=torch.uint32)
    patches = torch.tensor(cache["patches"], device="cuda", dtype=torch.uint64)
    a = torch.randn(512, 320, device="cuda", dtype=torch.float16)
    normed = torch.randn(512, 4, 2560, device="cuda", dtype=torch.float16)
    kernel = compile_rows(
        hc_up13_mix.__wrapped__(
            1024,
            2560,
            320,
            dynamic_rows=True,
            base_exp=cache["base"],
            patch_count=cache["patch_count"],
            block_m=64,
            weight_left=False,
        )
    )
    packed = kernel
    raw_kernel = compile_rows(
        hc_up_mix.__wrapped__(1024, 2560, 320, dynamic_rows=True, block_m=128)
    )
    for kind, kernel in [("packed-up", packed), ("raw-up", raw_kernel)]:
        for rows in [17, 33, 128, 512]:
            expected = torch.empty(rows, 2560, device="cuda", dtype=torch.float16)
            storage = torch.full((rows * 2560 + 31,), 91.0, device="cuda", dtype=torch.float16)
            actual = storage[:-31].view(rows, 2560)
            ref = hc_up_mix(rows, 2560, 320)

            def reference():
                ref.adapter.func(
                    a[:rows],
                    weight,
                    normed[:rows],
                    expected,
                    stream=torch.cuda.current_stream().cuda_stream,
                )

            def run():
                if kind == "packed-up":
                    kernel.adapter.func(a, lo, hi, patches, normed, actual, rows)
                else:
                    kernel.adapter.func(a, weight, normed, actual, rows)

            reference()
            run()
            torch.cuda.synchronize()
            assert torch.equal(actual, expected), (
                "up",
                rows,
                float((actual - expected).abs().max()),
            )
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            a.mul_(-1)
            reference()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(actual, expected)
            assert bool((storage[-31:] == 91).all())
            evidence.append(dict(kind=kind, rows=rows, exact=True, replay=True))
    write_json(args.output / "results.json", dict(complete=True, cases=evidence))


if __name__ == "__main__":
    main()
