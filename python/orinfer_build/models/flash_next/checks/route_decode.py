"""Check fused decode routing against stable histogram, offsets and tile emission."""

import argparse
from pathlib import Path
import torch
import tilelang
from kernels.cuda.model.moe import route_metadata, expert_histogram, expert_offsets, expert_tiles
from kernels.shared.rows import explicit_rows
from orinfer_build.tuning.projection_tuning import profile_stream
from orinfer_build.compiler.cuda.common import configure, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    configure()
    torch.manual_seed(20261002)
    cases = []

    def compile(fn):
        return profile_stream(
            tilelang.compile(
                explicit_rows(fn),
                out_idx=[],
                execution_backend="nvrtc",
                target={"kind": "cuda", "arch": "sm_87"},
            )
        )

    for bm in [8, 16]:
        candidate = compile(route_metadata.__wrapped__(128, block_m=bm, dynamic_rows=True))
        for rows in [1, 2, 3, 4, 8, 9, 16, 17, 32, 33, 127, 128]:
            ids = torch.rand(rows, 512, device="cuda").topk(10, dim=1).indices.to(torch.int32)
            ids[:, 0] = 0
            # Ensure unique IDs while exercising the maximum possible hot-expert count.
            ids[:, 1:] = (
                torch.rand(rows, 511, device="cuda").topk(9, dim=1).indices.to(torch.int32) + 1
            )
            cap = (1280 + bm - 1) // bm + 512

            def outputs():
                return [
                    torch.empty(512, device="cuda", dtype=torch.int32),
                    torch.empty(rows, 10, device="cuda", dtype=torch.int32),
                    torch.empty(512, device="cuda", dtype=torch.int32),
                    torch.empty(512, device="cuda", dtype=torch.int32),
                    torch.empty(1, device="cuda", dtype=torch.int32),
                    torch.full((cap + 17,), 42, device="cuda", dtype=torch.int32),
                    torch.full((cap + 17,), 42, device="cuda", dtype=torch.int32),
                ]

            expected, actual = outputs(), outputs()
            hist = expert_histogram(rows)
            offsets = expert_offsets(512, bm)
            tiles = expert_tiles(rows, 512, 10, bm)

            def ref():
                hist.adapter.func(ids, *expected[:2])
                offsets.adapter.func(expected[0], expected[2], expected[3], expected[4])
                tiles.adapter.func(expected[0], expected[3], *expected[5:])

            def run():
                candidate.adapter.func(ids, *actual, rows)

            def check():
                torch.cuda.synchronize()
                n = int(expected[4].item())
                assert all(torch.equal(x, y) for x, y in zip(expected[:5], actual[:5]))
                assert all(
                    torch.equal(x[:n], y[:n]) and bool((y[n:] == 42).all())
                    for x, y in zip(expected[5:], actual[5:])
                ), (bm, rows)

            ref()
            run()
            check()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            ids.copy_(ids.flip(0).roll(1, dims=1))
            ref()
            graph.replay()
            check()
            cases.append(dict(rows=rows, bm=bm, exact=True, replay=True))
            print("PASS ROUTE", bm, rows, flush=True)
    write_json(a.output / "results.json", dict(complete=True, cases=cases))


if __name__ == "__main__":
    main()
