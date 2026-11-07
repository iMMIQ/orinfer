"""Check HC row tiling preserves the lossless cache and every FP16 boundary."""

import argparse
from pathlib import Path
import numpy as np
import torch
import tilelang
from kernels.cuda.model.hc_bf16_cache import hc_up13_mix
from kernels.shared.rows import explicit_rows
from orinfer_build.models.flash_next.hc_bf16 import pack
from orinfer_build.tuning.projection_tuning import profile_stream
from orinfer_build.compiler.cuda.common import configure, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    configure()
    torch.manual_seed(20261002)
    rng = np.random.default_rng(20261002)
    cases = []
    for rank in [256, 320]:
        raw = (
            (rng.integers(0, 2, (80, rank // 64, 128, 64), dtype=np.uint16) << 15)
            | (rng.integers(110, 119, (80, rank // 64, 128, 64), dtype=np.uint16) << 7)
            | rng.integers(0, 128, (80, rank // 64, 128, 64), dtype=np.uint16)
        )
        raw.reshape(-1)[0] = np.uint16(2 << 7)
        cache = pack(raw)
        lo = torch.tensor(cache["low"].view(np.uint32), device="cuda", dtype=torch.uint32)
        hi = torch.tensor(cache["high"], device="cuda", dtype=torch.uint32)
        patches = torch.tensor(cache["patches"], device="cuda", dtype=torch.uint64)
        a = torch.randn(512, rank, device="cuda", dtype=torch.float16)
        normed = torch.randn(512, 4, 2560, device="cuda", dtype=torch.float16)

        def build(bm):
            return profile_stream(
                tilelang.compile(
                    explicit_rows(
                        hc_up13_mix.__wrapped__(
                            1024,
                            2560,
                            rank,
                            dynamic_rows=True,
                            base_exp=cache["base"],
                            patch_count=cache["patch_count"],
                            block_m=bm,
                        )
                    ),
                    out_idx=[],
                    execution_backend="nvrtc",
                    target={"kind": "cuda", "arch": "sm_87"},
                )
            )

        reference = build(8)
        for bm in [16, 32]:
            candidate = build(bm)
            for rows in [1, 5, 17, 33, 129, 512]:
                expected = torch.empty(rows, 2560, device="cuda", dtype=torch.float16)
                storage = torch.full((rows * 2560 + 31,), 91.0, device="cuda", dtype=torch.float16)
                actual = storage[:-31].view(rows, 2560)

                def ref():
                    reference.adapter.func(a, lo, hi, patches, normed, expected, rows)

                def run():
                    candidate.adapter.func(a, lo, hi, patches, normed, actual, rows)

                ref()
                run()
                torch.cuda.synchronize()
                assert torch.equal(actual, expected), (
                    rank,
                    bm,
                    rows,
                    float((actual - expected).abs().max()),
                )
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run()
                a.mul_(-1)
                ref()
                graph.replay()
                torch.cuda.synchronize()
                assert torch.equal(actual, expected)
                assert bool((storage[-31:] == 91).all())
                cases.append(dict(rank=rank, tile=bm, rows=rows, exact=True, replay=True))
                print("PASS HC", rank, bm, rows, flush=True)
    write_json(args.output / "results.json", dict(complete=True, cases=cases))


if __name__ == "__main__":
    main()
