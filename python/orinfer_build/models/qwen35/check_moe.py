"""Exact integer grouped projections, row tails, dispatch and changed-input replay."""

import argparse
from pathlib import Path
import numpy as np
import torch
from kernels.cuda.model import moe
from kernels.cuda.model.e8p_q4 import projection
from kernels.cuda.model.int8_projection import int8_projection
from orinfer_build.compiler.compile import primitive
from orinfer_build.tuning.row_tuning import launch
from orinfer_build.compiler.cuda.common import configure, write_json
from orinfer_build.quantization.e8p import basis
from orinfer_build.quantization.vq import e8p_decode, e8p_selector_table


def q4_pipeline_checks(rng):
    """Check both INT8 pipelines and direct INT4 against independent integer dots."""
    from kernels.cuda.model.e8p_prefill import e8p_prefill_grouped, split_prefill_a8

    capacity, experts = 256, 2
    grid = basis()
    book = torch.from_numpy(e8p_selector_table(grid)).cuda()
    reports = []
    for counts, n, k, tile in [
        ([1, 0], 129, 128, 16),
        ([7, 8], 160, 640, 16),
        ([8, 9], 160, 256, 16),
        ([17, 16], 257, 2560, 16),
        ([33, 32], 160, 512, 16),
        ([1, 0], 129, 128, 32),
        ([15, 16], 160, 640, 32),
        ([31, 32], 257, 256, 32),
        ([33, 32], 160, 2560, 32),
        ([1, 0], 129, 128, 64),
        ([63, 64], 257, 640, 64),
        ([127, 128], 257, 640, 128),
        ([193, 62], 129, 640, 256),
        ([63, 64], 257, 128, 256),
    ]:
        tiles = (capacity + tile - 1) // tile + experts
        rows = sum(counts)
        offsets = np.cumsum(counts, dtype=np.int32) - counts
        items = [(e, r) for e, count in enumerate(counts) for r in range(0, count, tile)]
        te, tr = np.zeros(tiles, np.int32), np.zeros(tiles, np.int32)
        te[: len(items)], tr[: len(items)] = zip(*items)
        routing = [
            torch.from_numpy(np.asarray(value, dtype=np.int32)).cuda()
            for value in (counts, offsets, te, tr, [len(items)])
        ]
        w = torch.from_numpy(
            rng.integers(0, 2**32, (experts, k // 128, n, 16), dtype=np.uint32)
        ).cuda()
        ws = torch.full((experts, n), 0.001, device="cuda", dtype=torch.float16)
        scale = torch.full((rows,), 0.002, device="cuda", dtype=torch.float16)
        a = torch.from_numpy(rng.integers(-128, 128, (rows, k), dtype=np.int8)).cuda()
        a[:, :4] = torch.tensor([-128, -1, 0, 127], device="cuda", dtype=torch.int8)
        low = torch.empty(rows * k // 2, device="cuda", dtype=torch.int8)
        high = torch.empty_like(low)
        sums = torch.empty(rows, device="cuda", dtype=torch.int32)
        storage = [
            torch.full((rows + 1, n), 91, device="cuda", dtype=torch.float16)
            for _ in range(2 if tile >= 64 else 4)
        ]
        outputs = [value[:-1] for value in storage]
        integer = [
            primitive(
                moe.integer_experts,
                capacity,
                n,
                k,
                experts,
                1,
                block_m=tile,
                block_n=32 if tile == 128 else 64,
                stages=stages,
                dynamic_rows=True,
                minimum_blocks=3 if tile == 256 and stages == 1 else 1,
                tail_m=64 if tile == 256 else 0,
                compiler_options=(
                    {"tl.enable_aggressive_shared_memory_merge": True} if tile == 256 else {}
                ),
            )
            for stages in (1, 2)
        ]
        pack = primitive(split_prefill_a8, capacity, k, dynamic_rows=True)
        direct = [
            primitive(
                e8p_prefill_grouped,
                capacity,
                experts,
                tiles,
                n,
                k,
                dynamic_rows=True,
                sparse_rows=sparse,
                codec="e8p_q4",
                minimum_blocks=6 if tile == 16 else 4,
                block_m=tile,
                compiler_options={
                    "tl.disable_thread_storage_sync": True,
                    "tl.disable_shared_memory_reuse": True,
                    "tl.if_stmt_binding_inline_replayable_binds": False,
                },
            )
            for sparse in ((False, True) if tile < 64 else ())
        ]

        def run():
            for kernel, out in zip(integer, outputs):
                launch(kernel, a, w, book, ws, scale, *routing, out, rows)
            launch(pack, a, low, high, sums, rows)
            for kernel, out in zip(direct, outputs[2:]):
                launch(kernel, low, high, sums, w, book, ws, scale, *routing, out, rows)

        def check():
            raw = w.cpu().numpy().transpose(0, 2, 1, 3).reshape(experts, n, k // 8)
            weights = (
                4 * e8p_decode(raw.astype(np.uint16), grid).astype(np.int64)
                + e8p_decode((raw >> 16).astype(np.uint16), grid).astype(np.int64)
            ).reshape(experts, n, k)
            av = a.cpu().numpy().astype(np.int64)
            wv, sv = ws.cpu().numpy().astype(np.float32), scale.cpu().numpy().astype(np.float32)
            expected = np.empty((rows, n), np.float16)
            for e, count in enumerate(counts):
                first, last = int(offsets[e]), int(offsets[e]) + count
                dot = av[first:last] @ weights[e].T
                expected[first:last] = (dot.astype(np.float32) * wv[e]) * sv[first:last, None]
            for out, backing in zip(outputs, storage):
                np.testing.assert_array_equal(out.cpu().numpy(), expected)
                assert bool(torch.all(backing[-1] == 91))

        run()
        check()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        for change in ("activation", "weights"):
            if change == "activation":
                a.neg_()
            else:
                w.view(torch.int32).bitwise_xor_(-1)
            graph.replay()
            check()
        reports.append(
            dict(counts=counts, n=n, k=k, tile=tile, exact=True, changed_input_replay=True)
        )
    return reports


def partition_checks(rng):
    """Independent dots after captured route/weight changes, including empty partitions."""
    from kernels.cuda.model import expert_partition
    from kernels.cuda.model.e8p_prefill import split_routed_prefill_a8
    from orinfer_build.tuning.expert_tuning import (
        cold_expert_config,
        expert_projection,
        expert_passes,
        tile_capacity,
    )

    capacity, experts, n, k, limit = 512, 32, 129, 640, 32
    grid = basis()
    book = torch.from_numpy(e8p_selector_table(grid)).cuda()
    cold = cold_expert_config()
    hot = dict(
        block_m=256,
        block_n=64,
        block_k=128,
        stages=1,
        tail_m=64,
        expand=False,
        minimum_blocks=3,
        reuse_shared=True,
    )
    prefix = primitive(expert_partition.offsets, experts, hot["block_m"], cold["block_m"], limit)
    tiles = primitive(
        expert_partition.tiles,
        capacity,
        experts,
        1,
        hot["block_m"],
        cold["block_m"],
        limit,
        dynamic_rows=True,
    )
    pack = primitive(
        split_routed_prefill_a8, capacity, k, experts, cold["block_m"], dynamic_rows=True
    )
    products = [
        primitive(
            expert_projection,
            capacity,
            n,
            k,
            experts,
            1,
            codec="e8p_q4",
            config=c,
            dynamic_rows=True,
            compiler_options=expert_passes("e8p_q4", c),
        )
        for c in (cold, hot)
    ]
    w = torch.from_numpy(rng.integers(0, 2**32, (experts, k // 128, n, 16), dtype=np.uint32)).cuda()
    ws = torch.full((experts, n), 0.001, device="cuda", dtype=torch.float16)
    reports = []
    for values in ([1] + [0] * 31, [0, 1, 16, 32, 33, 64, 128, 193] + [0] * 24):
        rows = sum(values)
        counts = torch.tensor(values, device="cuda", dtype=torch.int32)
        offsets = torch.empty_like(counts)
        to = [torch.empty_like(counts) for _ in range(2)]
        tc = [torch.full((2,), 91, device="cuda", dtype=torch.int32) for _ in range(2)]
        meta = []
        for c in (cold, hot):
            cap = tile_capacity(capacity, experts, 1, c["block_m"])
            meta.append(
                [torch.full((cap + 1,), 91, device="cuda", dtype=torch.int32) for _ in range(2)]
            )
        a = torch.from_numpy(rng.integers(-128, 128, (rows, k), dtype=np.int8)).cuda()
        a[:, :4] = torch.tensor([-128, -1, 0, 127], device="cuda", dtype=torch.int8)
        scale = torch.full((rows,), 0.002, device="cuda", dtype=torch.float16)
        planes = [torch.empty(rows * k // 2, device="cuda", dtype=torch.int8) for _ in range(2)]
        sums = torch.empty(rows, device="cuda", dtype=torch.int32)
        storage = torch.full((rows + 1, n), 91, device="cuda", dtype=torch.float16)
        out = storage[:-1]

        def run():
            launch(prefix, counts, offsets, to[1], tc[1][:1], to[0], tc[0][:1])
            launch(
                tiles,
                counts,
                to[1],
                *[x[:-1] for x in meta[1]],
                to[0],
                *[x[:-1] for x in meta[0]],
                rows,
            )
            launch(
                pack, a, counts, offsets, *[x[:-1] for x in meta[0]], tc[0][:1], *planes, sums, rows
            )
            launch(
                products[0],
                *planes,
                sums,
                w,
                book,
                ws,
                scale,
                counts,
                offsets,
                *[x[:-1] for x in meta[0]],
                tc[0][:1],
                out,
                rows,
            )
            launch(
                products[1],
                a,
                w,
                book,
                ws,
                scale,
                counts,
                offsets,
                *[x[:-1] for x in meta[1]],
                tc[1][:1],
                out,
                rows,
            )

        def check():
            c = counts.cpu().numpy()
            first = np.cumsum(c) - c
            np.testing.assert_array_equal(offsets.cpu().numpy(), first)
            for i, config in enumerate((cold, hot)):
                expected = [
                    (e, row)
                    for e, count in enumerate(c)
                    if (count <= limit) == (i == 0)
                    for row in range(0, int(count), config["block_m"])
                ]
                assert tc[i][0].item() == len(expected) and tc[i][1].item() == 91
                for col, table in enumerate(meta[i]):
                    np.testing.assert_array_equal(
                        table[: len(expected)].cpu().numpy(), [v[col] for v in expected]
                    )
                    assert table[-1].item() == 91
            raw = w.cpu().numpy().transpose(0, 2, 1, 3).reshape(experts, n, k // 8)
            weight = (
                4 * e8p_decode(raw.astype(np.uint16), grid).astype(np.int64)
                + e8p_decode((raw >> 16).astype(np.uint16), grid).astype(np.int64)
            ).reshape(experts, n, k)
            av = a.cpu().numpy().astype(np.int64)
            expected = np.empty((rows, n), np.float16)
            for e, count in enumerate(c):
                start, stop = int(first[e]), int(first[e] + count)
                expected[start:stop] = (
                    (av[start:stop] @ weight[e].T).astype(np.float32) * ws[e].float().cpu().numpy()
                ) * scale[start:stop].float().cpu().numpy()[:, None]
            np.testing.assert_array_equal(out.cpu().numpy(), expected)
            assert bool(torch.all(storage[-1] == 91))

        run()
        check()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        distributions = [values[::-1], [rows] + [0] * 31]
        if rows >= experts:
            distributions.append([rows // experts + (i < rows % experts) for i in range(experts)])
        for distribution in distributions:
            counts.copy_(torch.tensor(distribution, device="cuda", dtype=torch.int32))
            a.neg_()
            w.view(torch.int32).bitwise_xor_(-1)
            graph.replay()
            check()
        reports.append(
            dict(rows=rows, exact=True, changed_routes_graph=True, changed_weights_graph=True)
        )
    return reports


def routing_checks(rng):
    e, top, h, capacity, bm = 256, 8, 128, 4096, 16
    kernels = [
        primitive(moe.router_projection, capacity, h, e, dynamic_rows=True),
        primitive(moe.router_topk, capacity, e, top, dynamic_rows=True),
        primitive(moe.expert_histogram_local, capacity, e, top, dynamic_rows=True),
        primitive(moe.expert_histogram_offsets, capacity, e, top, dynamic_rows=True),
        primitive(moe.expert_histogram_finish, capacity, e, top, dynamic_rows=True),
        primitive(moe.expert_offsets, e, bm),
        primitive(moe.expert_tiles, capacity, e, top, bm, dynamic_rows=True),
        primitive(moe.expert_dispatch, capacity, h, e, top, h, dynamic_rows=True),
    ]
    reports = []
    for rows in (1, 3, 31, 512, 4096):

        def tensor(shape, dtype):
            return torch.empty(shape, device="cuda", dtype=dtype)

        x = torch.from_numpy(rng.normal(0, 0.2, (rows, h)).astype(np.float16)).cuda()
        w = torch.from_numpy(rng.normal(0, 0.2, (e + 1, h)).astype(np.float16)).cuda()
        logits = tensor((rows, e), torch.float32)
        gate = tensor((rows,), torch.float16)
        ids = tensor((rows, top), torch.int32)
        prob = tensor((rows, top), torch.float32)
        segments = (rows * top + 1023) // 1024
        partial = tensor((e, segments), torch.int32)
        seg_offsets = torch.empty_like(partial)
        relative = tensor((rows, top), torch.int32)
        counts = tensor((e,), torch.int32)
        offsets = torch.empty_like(counts)
        tile_offsets = torch.empty_like(counts)
        tiles = (capacity * top + bm - 1) // bm + min(e, capacity * top)
        te = torch.full((tiles,), -1, device="cuda", dtype=torch.int32)
        tr = torch.empty_like(te)
        tc = tensor((1,), torch.int32)
        a = torch.from_numpy(rng.integers(-127, 128, (rows, h), dtype=np.int8)).cuda()
        scale = torch.from_numpy(rng.uniform(0.001, 0.01, (rows, 1)).astype(np.float16)).cuda()
        d = tensor((rows * top, h), torch.int8)
        ds = tensor((rows * top, 1), torch.float16)
        slots = tensor((rows, top), torch.int32)
        args = [
            (x, w, logits, gate, rows),
            (logits, ids, prob, rows),
            (ids, partial, relative, rows),
            (partial, seg_offsets, counts, rows),
            (ids, seg_offsets, relative, rows),
            (counts, offsets, tile_offsets, tc),
            (counts, tile_offsets, te, tr, rows),
            (a, scale, ids, relative, offsets, d, ds, slots, rows),
        ]

        def run():
            for kernel, values in zip(kernels, args):
                launch(kernel, *values)

        def check():
            expected = x.float() @ w.float().T
            torch.testing.assert_close(logits, expected[:, :e], atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(gate, expected[:, e].half(), atol=1e-3, rtol=1e-3)
            selected = torch.argsort(logits, descending=True, stable=True)[:, :top]
            torch.testing.assert_close(ids.long(), selected, atol=0, rtol=0)
            expected_prob = torch.softmax(logits.gather(1, selected), dim=1)
            torch.testing.assert_close(prob, expected_prob, atol=1e-6, rtol=1e-6)
            cpu = ids.cpu().numpy()
            expected_counts = np.bincount(cpu.ravel(), minlength=e).astype(np.int32)
            expected_offsets = np.cumsum(expected_counts) - expected_counts
            np.testing.assert_array_equal(counts.cpu().numpy(), expected_counts)
            np.testing.assert_array_equal(offsets.cpu().numpy(), expected_offsets)
            counters = np.zeros(e, np.int32)
            expected_slots = np.empty_like(cpu)
            for row in range(rows):
                for rank in range(top):
                    expert = cpu[row, rank]
                    expected_slots[row, rank] = expected_offsets[expert] + counters[expert]
                    counters[expert] += 1
            np.testing.assert_array_equal(slots.cpu().numpy(), expected_slots)
            indices = torch.from_numpy(expected_slots).cuda().long()
            torch.testing.assert_close(
                d[indices], a[:, None, :].expand(rows, top, h), atol=0, rtol=0
            )
            torch.testing.assert_close(
                ds[indices], scale[:, None, :].expand(rows, top, 1), atol=0, rtol=0
            )
            expected_tiles = [
                (expert, row)
                for expert in range(e)
                for row in range(0, int(expected_counts[expert]), bm)
            ]
            assert tc.item() == len(expected_tiles)
            np.testing.assert_array_equal(
                te[: len(expected_tiles)].cpu().numpy(), [v[0] for v in expected_tiles]
            )
            np.testing.assert_array_equal(
                tr[: len(expected_tiles)].cpu().numpy(), [v[1] for v in expected_tiles]
            )

        run()
        check()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        x.neg_()
        a.neg_()
        graph.replay()
        check()
        reports.append(dict(rows=rows, routing_dispatch_exact=True, changed_input_replay=True))
    return reports


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    configure()
    rng = np.random.default_rng(20261002)
    e, top, n, k, capacity, bm = 4, 2, 96, 256, 33, 16
    codes = rng.integers(0, 2**32, (e, n, k // 8), dtype=np.uint32)
    grid = basis()
    q4 = (
        (
            4 * e8p_decode(codes.astype(np.uint16), grid).astype(np.int16)
            + e8p_decode((codes >> 16).astype(np.uint16), grid).astype(np.int16)
        )
        .reshape(e, n, k)
        .astype(np.int8)
    )
    book = torch.from_numpy(e8p_selector_table(grid)).cuda()
    scales = torch.from_numpy(rng.uniform(0.001, 0.01, (e, n)).astype(np.float16)).cuda()
    reports = []
    for codec in ("w8", "e8p_q4"):
        integer = rng.integers(-127, 128, (e, n, k), dtype=np.int8) if codec == "w8" else q4
        packed = np.ascontiguousarray(codes.reshape(e, n, k // 128, 16).transpose(0, 2, 1, 3))
        w = torch.from_numpy(integer if codec == "w8" else packed).cuda()
        grouped = primitive(
            moe.integer_experts, capacity, n, k, e, top, codec=codec, dynamic_rows=True
        )
        for rows in (1, 3, 17, 33):
            assignments = rows * top
            ids = rng.integers(0, e, (rows, top), dtype=np.int32)
            ids[:, 1] = (ids[:, 0] + 1) % e
            counts = np.bincount(ids.ravel(), minlength=e).astype(np.int32)
            offsets = np.cumsum(counts, dtype=np.int32) - counts
            tes, trs = [], []
            for expert in range(e):
                for row in range(0, int(counts[expert]), bm):
                    tes.append(expert)
                    trs.append(row)
            slots = (capacity * top + bm - 1) // bm + min(e, capacity * top)
            te = np.zeros(slots, np.int32)
            tr = te.copy()
            te[: len(tes)] = tes
            tr[: len(trs)] = trs
            a = torch.from_numpy(rng.integers(-127, 128, (assignments, k), dtype=np.int8)).cuda()
            scale = torch.from_numpy(
                rng.uniform(0.001, 0.01, (assignments,)).astype(np.float16)
            ).cuda()
            out = torch.full((assignments, n), float("nan"), device="cuda", dtype=torch.float16)
            values = [
                a,
                w,
                book,
                scales,
                scale,
                *[
                    torch.from_numpy(v).cuda()
                    for v in (counts, offsets, te, tr, np.array([len(tes)], np.int32))
                ],
                out,
            ]

            def check():
                ac = a.cpu().numpy().astype(np.int64)
                expected = np.empty((assignments, n), np.float16)
                ws = scales.cpu().numpy().astype(np.float32)
                token_scale = scale.cpu().numpy().astype(np.float32)
                for expert in range(e):
                    first = int(offsets[expert])
                    last = first + int(counts[expert])
                    dot = ac[first:last] @ integer[expert].astype(np.int64).T
                    expected[first:last] = (dot.astype(np.float32) * ws[expert]) * token_scale[
                        first:last, None
                    ]
                np.testing.assert_array_equal(out.cpu().numpy(), expected)

            launch(grouped, *values, rows)
            check()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                launch(grouped, *values, rows)
            a.copy_(-a)
            graph.replay()
            check()
            reports.append(dict(codec=codec, rows=rows, exact=True, changed_input_replay=True))
        # The same arithmetic body still serves the existing dense path.
        dense = (
            primitive(int8_projection, 3, n, k, block_m=16, block_n=64, block_k=128)
            if codec == "w8"
            else primitive(projection, 3, n, k, block_m=16, block_n=64, packed_k=128)
        )
        a = torch.from_numpy(rng.integers(-127, 128, (3, k), dtype=np.int8)).cuda()
        token_scale = torch.ones(3, device="cuda", dtype=torch.float16)
        out = torch.empty((3, n), device="cuda", dtype=torch.float16)
        launch(
            dense,
            *(
                [a, w[0], scales[0], token_scale, out]
                if codec == "w8"
                else [a, w[0], book, scales[0], token_scale, out]
            ),
        )
        expected = (a.cpu().numpy().astype(np.int64) @ integer[0].astype(np.int64).T).astype(
            np.float32
        ) * scales[0].cpu().numpy().astype(np.float32)
        np.testing.assert_array_equal(out.cpu().numpy(), expected.astype(np.float16))
    routing = routing_checks(rng)
    tails = q4_pipeline_checks(rng)
    write_json(
        args.output / "result.json",
        dict(
            checks=reports,
            routing=routing,
            projection_tails=tails,
            partition=partition_checks(rng),
            complete=True,
        ),
    )


if __name__ == "__main__":
    main()
