"""Check dynamic E8P tails against the independent row-major exact recipe."""

import argparse
from pathlib import Path

import tilelang
import torch

from kernels.cuda.model.e8p_exact import e8p_exact_grouped
from kernels.cuda.model.e8p_prefill import e8p_prefill_grouped, split_prefill_a8
from kernels.cuda.model.moe import expert_offsets, expert_tiles
from kernels.shared.rows import explicit_rows
from orinfer_build.compiler.cuda.common import configure, export_kernel, write_json


def launch(kernel, *operands):
    kernel.adapter.func(*operands, stream=torch.cuda.current_stream().cuda_stream)


def compile_dynamic(primitive):
    return tilelang.compile(
        explicit_rows(primitive),
        out_idx=[],
        execution_backend="nvrtc",
        target={"kind": "cuda", "arch": "sm_87"},
        pass_configs={
            "tl.disable_thread_storage_sync": True,
            "tl.disable_shared_memory_reuse": True,
            "tl.if_stmt_binding_inline_replayable_binds": False,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure()
    torch.manual_seed(20261002)
    capacity, experts, columns = 40960, 64, 160
    tiles = (capacity + 15) // 16 + experts
    report = dict(complete=False, cases=[])
    for width, sparse in [(2560, True), (640, False)]:
        weights = torch.randint(
            0, 65536, (experts, width // 64, columns, 8), device="cuda", dtype=torch.int32
        ).to(torch.uint16)
        book = torch.randint(0, 2**32, (1, 256, 1), device="cuda", dtype=torch.int64).to(
            torch.uint32
        )
        scales = torch.full((experts, columns), 0.001, device="cuda", dtype=torch.float16)
        candidate = compile_dynamic(
            e8p_prefill_grouped.__wrapped__(
                capacity, experts, tiles, columns, width, dynamic_rows=True, sparse_rows=sparse
            )
        )
        pack = compile_dynamic(split_prefill_a8.__wrapped__(capacity, width, dynamic_rows=True))
        export_kernel(candidate, args.output / "aot" / str(width))
        for rows in [20, 170, 1290, 5120, 10250, capacity]:
            # Cover all 1..32 expert counts, zero experts, and one large expert.
            counts = torch.zeros(experts, device="cuda", dtype=torch.int32)
            remaining = rows
            for expert in range(1, experts):
                count = min(remaining, 1 + (expert - 1) % 32)
                counts[expert] = count
                remaining -= count
            counts[0] = remaining
            offsets, tile_offsets = torch.empty_like(counts), torch.empty_like(counts)
            total = torch.empty((1,), device="cuda", dtype=torch.int32)
            tile_expert = torch.empty(tiles, device="cuda", dtype=torch.int32)
            tile_row = torch.empty_like(tile_expert)
            launch(expert_offsets(experts), counts, offsets, tile_offsets, total)
            launch(expert_tiles(rows, experts), counts, tile_offsets, tile_expert, tile_row)
            activation = torch.randint(-128, 128, (rows, width), device="cuda", dtype=torch.int8)
            low = torch.empty((width // 64, rows, 8), device="cuda", dtype=torch.uint32)
            high, sums = torch.empty_like(low), torch.empty(rows, device="cuda", dtype=torch.int32)
            activation_scale = torch.full((rows,), 0.001, device="cuda", dtype=torch.float16)
            storage = torch.full((rows + 2, columns), 91, device="cuda", dtype=torch.float16)
            output, expected = storage[:rows], torch.empty_like(storage[:rows])
            reference = e8p_exact_grouped(rows, experts, tiles, columns, width)

            def prepare_reference():
                launch(pack, activation, low, high, sums, rows)
                row_low = (
                    low.permute(1, 0, 2).contiguous().view(torch.int8).reshape(rows, width // 2)
                )
                row_high = (
                    high.permute(1, 0, 2).contiguous().view(torch.int8).reshape(rows, width // 2)
                )
                launch(
                    reference,
                    row_low,
                    row_high,
                    sums,
                    weights,
                    book,
                    scales,
                    activation_scale,
                    counts,
                    offsets,
                    tile_expert,
                    tile_row,
                    total,
                    expected,
                )

            def execute():
                launch(
                    candidate,
                    low,
                    high,
                    sums,
                    weights,
                    book,
                    scales,
                    activation_scale,
                    counts,
                    offsets,
                    tile_expert,
                    tile_row,
                    total,
                    output,
                    rows,
                )

            prepare_reference()
            execute()
            torch.cuda.synchronize()
            assert torch.equal(output, expected), (width, rows, "direct")
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                execute()
            activation.bitwise_not_()
            prepare_reference()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(output, expected), (width, rows, "changed-input replay")
            assert bool((storage[rows:] == 91).all()), (width, rows, "output tail")
            case = dict(width=width, rows=rows, sparse=sparse, exact=True, graph_replay=True)
            report["cases"].append(case)
            write_json(args.output / "results.json", report)
            print(case, flush=True)
    report["complete"] = True
    write_json(args.output / "results.json", report)


if __name__ == "__main__":
    main()
