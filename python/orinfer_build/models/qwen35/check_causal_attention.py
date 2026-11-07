"""Exact single-query arithmetic across short causal widths and owner packets."""

import argparse
import json
from pathlib import Path

import torch
import tilelang

from kernels.cuda.model.attention_rows import causal_attention_int8, merge_attention_int8
from kernels.cuda.model.independent_rows import independent_rows
from orinfer_build.compiler.compile import primitive
from orinfer_build.models.qwen35.attention import COLUMNS
from orinfer_build.models.qwen35.parallel import factory
from orinfer_build.tuning.row_tuning import launch
from orinfer_build.compiler.cuda.common import SEED, configure


CONFIG = dict(block_m=16, block_n=64, head_group=6, max_splits=8, threads=128, stages=0)
HEADS, KV_HEADS, DIMENSION = 24, 4, 256


def pair(capacity, independent):
    common = {k: v for k, v in CONFIG.items() if k not in ("block_n", "threads", "stages")}
    common.update(query_heads=HEADS, kv_heads=KV_HEADS, independent_queries=independent)
    return (
        primitive(causal_attention_int8, capacity, 16384, block_n=64, **common),
        primitive(merge_attention_int8, capacity, **common),
    )


def scratch(tiles, queries=1):
    return (
        torch.zeros(queries * 4, tiles, 16, DIMENSION, device="cuda"),
        torch.zeros(queries * 4, tiles, 16, device="cuda"),
        torch.zeros(queries * 4, tiles, 16, device="cuda"),
    )


def execute(kernels, q, k, v, ks, vs, gate, positions, lengths, output, work, rows):
    partial, maximum, denom = work
    launch(
        kernels[0], q, k, v, ks, vs, gate, positions, lengths, output, partial, maximum, denom, rows
    )
    launch(kernels[1], partial, maximum, denom, gate, output, rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure()
    torch.manual_seed(SEED)
    reference, causal = pair(1, True), pair(8, True)
    old_work, work = scratch(32), scratch(32, 8)
    q = torch.randn(8, HEADS, DIMENSION, device="cuda", dtype=torch.float16) * 0.1
    gate = torch.randn_like(q)
    k, v = [
        torch.randint(-100, 101, (1, 16384, KV_HEADS * DIMENSION), device="cuda", dtype=torch.int8)
        for _ in range(2)
    ]
    ks, vs = [
        torch.rand(1, 16384, KV_HEADS * 4, device="cuda", dtype=torch.float16) * 0.002 + 0.001
        for _ in range(2)
    ]
    positions = torch.zeros(8, device="cuda", dtype=torch.int32)
    lengths = torch.zeros(1, device="cuda", dtype=torch.int32)
    output = torch.full((9, HEADS, DIMENSION), 91, device="cuda", dtype=torch.float16)
    expected = torch.empty(1, HEADS, DIMENSION, device="cuda", dtype=torch.float16)
    records = []
    for length in (
        1,
        7,
        8,
        9,
        31,
        32,
        33,
        39,
        40,
        41,
        63,
        64,
        65,
        127,
        128,
        129,
        255,
        256,
        257,
        8191,
        8192,
        8193,
        16384,
    ):
        for rows in range(1, min(8, length) + 1):
            positions[:rows].copy_(
                torch.arange(length - rows, length, device="cuda", dtype=torch.int32)
            )
            lengths.fill_(length)
            output.fill_(91)

            def run():
                execute(causal, q, k, v, ks, vs, gate, positions, lengths, output, work, rows)

            run()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            for replay in range(2):
                q.neg_()
                gate.neg_()
                output.fill_(91)
                graph.replay()
                for row in range(rows):
                    execute(
                        reference,
                        q[row:],
                        k,
                        v,
                        ks,
                        vs,
                        gate[row:],
                        positions[row:],
                        lengths,
                        expected,
                        old_work,
                        1,
                    )
                    assert torch.equal(output[row], expected[0]), (length, rows, row, replay)
                assert bool(torch.all(output[rows:] == 91))
            records.append(dict(context=length, rows=rows, exact=True, changed_input_replay=True))

    wrapped = []
    for role in ("attention", "attention_merge"):
        fn = factory(role, 16384, CONFIG | dict(independent_queries=True))
        modes = {
            b.name: ("private" if b.name in COLUMNS else "row", b.name)
            for b in fn.buffer_map.values()
        }
        fn, names = independent_rows(
            fn, 128, modes, COLUMNS, dynamic=True, batch_parameters=("independent_instances",)
        )
        wrapped.append(
            (
                tilelang.compile(
                    fn,
                    out_idx=[],
                    execution_backend="nvrtc",
                    target={"kind": "cuda", "arch": "sm_87"},
                ),
                names,
            )
        )
    owner_results = []
    for owners in (1, 2, 3, 4, 8, 17, 32, 64, 128):
        values = dict(
            Q=torch.randn(owners, HEADS, DIMENSION, device="cuda", dtype=torch.float16) * 0.1,
            Gate=torch.randn(owners, HEADS, DIMENSION, device="cuda", dtype=torch.float16),
            K=k,
            V=v,
            KS=ks,
            VS=vs,
            Positions=(128 - torch.arange(owners, device="cuda", dtype=torch.int32).remainder(17))[
                :, None
            ],
            Lengths=(129 - torch.arange(owners, device="cuda", dtype=torch.int32).remainder(17))[
                :, None
            ],
        )
        # Owner KV uses a short physical context; point every owner at the same
        # immutable KV bank to keep this address-table test bounded in memory.
        for name, elements in (
            ("Partial", 4 * 32 * 16 * 256),
            ("Maximum", 4 * 32 * 16),
            ("Denominator", 4 * 32 * 16),
        ):
            values[name] = torch.zeros(owners, elements, device="cuda")
        table = torch.tensor(
            [
                [
                    values[name].data_ptr()
                    if name in ("K", "V", "KS", "VS")
                    else values[name][i].data_ptr()
                    for name in COLUMNS
                ]
                for i in range(owners)
            ],
            device="cuda",
            dtype=torch.uint64,
        )
        result = torch.full((owners + 1, HEADS, DIMENSION), 91, device="cuda", dtype=torch.float16)

        def run_owners():
            for kernel, names in wrapped:
                launch(
                    kernel,
                    *[
                        table
                        if name is None
                        else owners
                        if name == "__rows__"
                        else result
                        if name == "Y"
                        else values[name]
                        for name in names
                    ],
                )

        run_owners()
        for reverse in (False, True):
            order = (
                torch.arange(owners - 1, -1, -1, device="cuda")
                if reverse
                else torch.arange(owners, device="cuda")
            )
            table.copy_(table.index_select(0, order))
            values["Q"].neg_()
            run_owners()
            for row, owner in enumerate(order.tolist()):
                execute(
                    reference,
                    values["Q"][row:],
                    k,
                    v,
                    ks,
                    vs,
                    values["Gate"][row:],
                    values["Positions"][owner],
                    values["Lengths"][owner],
                    expected,
                    old_work,
                    1,
                )
                assert torch.equal(result[row], expected[0]), (owners, reverse, row)
            assert bool(torch.all(result[owners:] == 91))
        owner_results.append(dict(owners=owners, exact=True, reverse_owners=True))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "result.json").write_text(
        json.dumps(dict(causal=records, owners=owner_results), indent=2)
    )
    print(
        "PASS exact causal widths, context tails, owner counts and changed-input replay", flush=True
    )


if __name__ == "__main__":
    main()
