"""Compare single-token dispatch fusion, all metadata, tails and changed graphs."""

import argparse
from pathlib import Path

import torch

from kernels.cuda.model.moe import (
    expert_histogram,
    expert_offsets,
    expert_tiles,
    single_route_metadata,
)
from orinfer_build.compiler.cuda.common import configure, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure()
    cases = []
    for experts, top_k, block_m in [(512, 10, 16), (512, 10, 8), (37, 10, 16), (128, 1, 16)]:
        capacity = (top_k + block_m - 1) // block_m + top_k
        ids = torch.randperm(experts, device="cuda")[:top_k].reshape(1, top_k).to(torch.int32)
        extents = [experts, top_k, experts, experts, capacity, capacity, 1]

        def allocate():
            arenas = [torch.full((n + 1,), -91, device="cuda", dtype=torch.int32) for n in extents]
            return arenas, [x[:-1] for x in arenas]

        ref_arenas, ref = allocate()
        arenas, out = allocate()
        histogram = expert_histogram(1, experts, top_k)
        prefix = expert_offsets(experts, block_m)
        tiles = expert_tiles(1, experts, top_k, block_m)
        fused = single_route_metadata(experts, top_k, capacity)

        def baseline():
            histogram(ids, ref[0], ref[1].reshape(1, top_k))
            prefix(ref[0], ref[2], ref[3], ref[6])
            tiles(ref[0], ref[3], ref[4], ref[5])

        def run():
            fused(ids, out[0], out[1].reshape(1, top_k), *out[2:4], *out[4:6], out[6])

        def verify():
            torch.cuda.synchronize()
            for actual, expected in zip(arenas, ref_arenas, strict=True):
                assert torch.equal(actual, expected)
                assert actual[-1].item() == -91

        baseline()
        run()
        verify()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        for changed in [
            torch.arange(top_k, device="cuda", dtype=torch.int32),
            torch.arange(experts - top_k, experts, device="cuda", dtype=torch.int32),
            torch.randperm(experts, device="cuda")[:top_k].to(torch.int32),
        ]:
            ids.copy_(changed.reshape(1, top_k))
            baseline()
            graph.replay()
            verify()
        row = dict(
            experts=experts,
            top_k=top_k,
            block_m=block_m,
            all_metadata_exact=True,
            unchanged_unused_tails=True,
            capacity_guards=True,
            changed_graph=True,
        )
        cases.append(row)
        print(row, flush=True)
    write_json(args.output / "results.json", dict(complete=True, seed=20261002, cases=cases))


if __name__ == "__main__":
    main()
