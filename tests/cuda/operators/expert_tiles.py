"""Validate routed tile capacity and replay when expert distributions change."""

import argparse
from pathlib import Path

import torch

from kernels.cuda.model.moe import expert_offsets, expert_tiles
from orinfer_build.compiler.cuda.common import benchmark, configure, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure()
    cases = []
    for rows, experts, top_k, block_m in (
        (1, 512, 10, 8),
        (8, 512, 10, 8),
        (17, 37, 10, 8),
        (512, 512, 10, 16),
    ):
        capacity = (rows * top_k + block_m - 1) // block_m + min(experts, rows * top_k)
        counts = torch.zeros(experts, device="cuda", dtype=torch.int32)
        offsets, tile_offsets = torch.empty_like(counts), torch.empty_like(counts)
        tile_count = torch.empty(1, device="cuda", dtype=torch.int32)
        expert_storage = torch.full((capacity + 1,), -91, device="cuda", dtype=torch.int32)
        row_storage = torch.full_like(expert_storage, -91)
        tile_expert, tile_row = expert_storage[:-1], row_storage[:-1]
        prefix = expert_offsets(experts, block_m)
        tiles = expert_tiles(rows, experts, top_k, block_m)

        def run():
            prefix(counts, offsets, tile_offsets, tile_count)
            tiles(counts, tile_offsets, tile_expert, tile_row)

        def check():
            host = counts.cpu().tolist()
            expected = [(e, r) for e, count in enumerate(host) for r in range(0, count, block_m)]
            live = tile_count.item()
            assert live == len(expected) <= capacity
            assert (
                list(zip(tile_expert[:live].cpu().tolist(), tile_row[:live].cpu().tolist()))
                == expected
            )
            expected_offsets, cursor = [], 0
            for count in host:
                expected_offsets.append(cursor)
                cursor += count
            assert offsets.cpu().tolist() == expected_offsets
            assert expert_storage[-1].item() == row_storage[-1].item() == -91

        counts[:top_k] = rows
        timing, graph = benchmark(run, repetitions=8)
        check()
        # Unique top-k IDs per token keep each expert's count bounded by rows.
        ids = torch.stack([torch.randperm(experts, device="cuda")[:top_k] for _ in range(rows)])
        counts.copy_(torch.bincount(ids.flatten(), minlength=experts))
        graph.replay()
        torch.cuda.synchronize()
        check()
        counts.zero_()
        graph.replay()
        torch.cuda.synchronize()
        check()
        cases.append(
            dict(
                rows=rows,
                experts=experts,
                block_m=block_m,
                changed_routing_graph=True,
                capacity_guard=True,
                timing=timing,
            )
        )
    write_json(args.output / "results.json", dict(complete=True, cases=cases))


if __name__ == "__main__":
    main()
