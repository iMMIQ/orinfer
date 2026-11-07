"""Check decode row tails and address changes across CUDA Graph replay."""

import argparse
from pathlib import Path

import tilelang
import torch
from kernels.cuda.model import decode_packet
from kernels.cuda.model.e8p_exact import e8p_exact_grouped, split_a8
from kernels.cuda.model.moe import expert_offsets, expert_tiles
from kernels.cuda.model.q4_projection import q4_gemv
from kernels.shared.rows import explicit_rows
from orinfer_build.compiler.cuda.common import configure, write_json


def compile_rows(fn):
    return tilelang.compile(
        explicit_rows(fn),
        out_idx=[],
        execution_backend="nvrtc",
        target={"kind": "cuda", "arch": "sm_87"},
    )


def launch(kernel, *args):
    kernel.adapter.func(*args, stream=torch.cuda.current_stream().cuda_stream)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    configure()
    torch.manual_seed(20261002)
    results = []
    for tile in [2, 8]:
        candidate = compile_rows(q4_gemv.__wrapped__(tile, 128, 2560, dynamic_rows=True))
        weight = torch.randint(0, 65536, (128, 640), device="cuda", dtype=torch.int32).to(
            torch.uint16
        )
        scale = torch.full((128,), 0.001, device="cuda", dtype=torch.float16)
        for rows in [1, 2] if tile == 2 else [3, 4, 5, 7, 8]:
            x = torch.randint(-128, 128, (rows, 2560), device="cuda", dtype=torch.int8)
            ts = torch.full((rows,), 0.001, device="cuda", dtype=torch.float16)
            storage = torch.full((rows + 2, 128), 91, device="cuda", dtype=torch.float16)
            out, expected = storage[:rows], torch.empty_like(storage[:rows])
            reference = q4_gemv(rows, 128, 2560)

            def execute():
                launch(candidate, x, weight, scale, ts, out, rows)

            launch(reference, x, weight, scale, ts, expected)
            execute()
            torch.cuda.synchronize()
            assert torch.equal(out, expected)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                execute()
            x.bitwise_not_()
            launch(reference, x, weight, scale, ts, expected)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out, expected) and torch.all(storage[rows:] == 91)
            results.append(dict(operator="q4", rows=rows, graph=True))
    capacity, e, n, k = 1280, 16, 128, 640
    tiles = (capacity + 7) // 8 + e
    candidate = compile_rows(
        e8p_exact_grouped.__wrapped__(capacity, e, tiles, n, k, block_m=8, dynamic_rows=True)
    )
    pack = compile_rows(split_a8.__wrapped__(capacity, k, dynamic_rows=True))
    weight = torch.randint(0, 65536, (e, k // 64, n, 8), device="cuda", dtype=torch.int32).to(
        torch.uint16
    )
    book = torch.randint(0, 2**32, (1, 256, 1), device="cuda", dtype=torch.int64).to(torch.uint32)
    ws = torch.full((e, n), 0.001, device="cuda", dtype=torch.float16)
    for rows in [10, 20, 30, 50, 70, 80, 90, 1270, 1280]:
        counts = torch.zeros(e, device="cuda", dtype=torch.int32)
        counts[0] = rows - 9
        counts[1:10] = 1
        offsets, to = torch.empty_like(counts), torch.empty_like(counts)
        tc = torch.empty(1, device="cuda", dtype=torch.int32)
        te = torch.empty(tiles, device="cuda", dtype=torch.int32)
        tr = torch.empty_like(te)
        launch(expert_offsets(e, 8), counts, offsets, to, tc)
        launch(expert_tiles(rows, e, top_k=1, block_m=8), counts, to, te, tr)
        x = torch.randint(-128, 128, (rows, k), device="cuda", dtype=torch.int8)
        low = torch.empty((rows, k // 8), device="cuda", dtype=torch.uint32)
        high = torch.empty_like(low)
        sums = torch.empty(rows, device="cuda", dtype=torch.int32)
        ts = torch.full((rows,), 0.001, device="cuda", dtype=torch.float16)
        storage = torch.full((rows + 2, n), 91, device="cuda", dtype=torch.float16)
        out, expected = storage[:rows], torch.empty_like(storage[:rows])
        reference = e8p_exact_grouped(rows, e, tiles, n, k, block_m=8)

        def reference_run():
            launch(split_a8(rows, k), x, low, high, sums)
            launch(
                reference,
                low,
                high,
                sums,
                weight,
                book,
                ws,
                ts,
                counts,
                offsets,
                te,
                tr,
                tc,
                expected,
            )

        def execute():
            launch(pack, x, low, high, sums, rows)
            launch(
                candidate,
                low,
                high,
                sums,
                weight,
                book,
                ws,
                ts,
                counts,
                offsets,
                te,
                tr,
                tc,
                out,
                rows,
            )

        reference_run()
        execute()
        torch.cuda.synchronize()
        assert torch.equal(out, expected)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            execute()
        x.bitwise_not_()
        reference_run()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, expected) and torch.all(storage[rows:] == 91)
        results.append(dict(operator="e8p", rows=rows, graph=True))
    stride, context, features, vocab, ring = 48 * 13, 32, 2, 1024, 16
    table = torch.zeros(
        128 * (stride + decode_packet.IO_COLUMNS), device="cuda", dtype=torch.uint64
    )
    ins = compile_rows(decode_packet.inputs.__wrapped__(stride, context, features, True))
    end = compile_rows(decode_packet.end.__wrapped__(stride, ring, True))
    outs = compile_rows(decode_packet.outputs.__wrapped__(stride, vocab))
    embedding = torch.empty((128, 2560), device="cuda", dtype=torch.float16)
    ple = torch.empty_like(embedding)
    hidden = torch.randn((128, 10240), device="cuda", dtype=torch.float16)
    logits = torch.randn((128, vocab), device="cuda")
    selected = torch.tensor([[32 + i, 0] for i in range(128)], device="cuda", dtype=torch.int32)
    for rows in [1, 3, 8, 17, 128]:
        graph = torch.cuda.CUDAGraph()
        owners = []

        def populate(reverse):
            nonlocal owners
            owners = []
            for lane in range(rows):
                feature = lane % 2 if reverse else -1
                values = [
                    torch.full((2560,), float(lane + reverse), device="cuda", dtype=torch.float16),
                    torch.full((2560,), float(-lane - reverse), device="cuda", dtype=torch.float16),
                    torch.tensor([lane % context], device="cuda", dtype=torch.int32),
                    torch.zeros(1, device="cuda", dtype=torch.int32),
                    torch.zeros(1, device="cuda", dtype=torch.int32),
                    torch.zeros(vocab, device="cuda"),
                    torch.full((features, 2560), 19.0, device="cuda", dtype=torch.float16),
                    torch.full((context,), feature, device="cuda", dtype=torch.int32),
                    torch.zeros((ring, 10240), device="cuda", dtype=torch.float16),
                    torch.zeros((9, 10240), device="cuda", dtype=torch.float16),
                ]
                owners.append(values)
            ptrs = torch.tensor(
                [t.data_ptr() for values in owners for t in values],
                device="cuda",
                dtype=torch.uint64,
            )
            table[128 * stride : 128 * stride + ptrs.numel()].copy_(ptrs)

        def execute():
            launch(ins, table, embedding, ple, rows)
            launch(end, table, hidden, rows)
            launch(outs, table, logits, selected, rows)

        populate(False)
        with torch.cuda.graph(graph):
            execute()
        # Keep captured owners alive while replacing all pointer-table contents.
        captured_owners = owners
        populate(True)
        graph.replay()
        torch.cuda.synchronize()
        for lane, values in enumerate(owners):
            assert torch.all(embedding[lane] == 19) and torch.all(ple[lane] == -lane - 1)
            assert values[2].item() == lane % context + 1
            assert values[3].item() == 32 + lane and values[4].item() == 0
            assert torch.equal(values[5], logits[lane])
            assert torch.equal(values[8][lane % context % ring], hidden[lane])
            assert torch.count_nonzero(captured_owners[lane][8]) == 0
        results.append(dict(operator="packet", rows=rows, membership_replay=True))
    write_json(args.output / "results.json", dict(complete=True, cases=results))


if __name__ == "__main__":
    main()
