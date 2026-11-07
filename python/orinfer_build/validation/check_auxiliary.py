"""Validate exact routing, QSA selection and PLE fusion on the target GPU."""

import argparse
import importlib.util
from pathlib import Path

import tilelang
import torch

from kernels.cuda.model import hyperconnection, moe, ple, qsa, qsa_selection
from kernels.shared.rows import explicit_rows
from orinfer_build.tuning.row_tuning import launch
from orinfer_build.compiler.cuda.common import benchmark, configure, export_kernel, write_json


def compile_kernel(fn):
    return tilelang.compile(
        explicit_rows(fn),
        out_idx=[],
        execution_backend="nvrtc",
        target={"kind": "cuda", "arch": "sm_87"},
    )


def reference(args, name, default):
    if args.reference_dir is None:
        return default
    path = args.reference_dir / ("reference_" + name + ".py")
    spec = importlib.util.spec_from_file_location("reference_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify(run, ref, actual, expected, change):
    def check():
        torch.cuda.synchronize()
        for a, e in zip(actual, expected, strict=True):
            if not torch.equal(a, e):
                delta = (a.float() - e.float()).abs()
                raise AssertionError(
                    dict(
                        shape=list(a.shape),
                        different=int((a != e).sum()),
                        max_abs=float(delta.max()),
                    )
                )

    ref()
    run()
    check()
    timing, graph = benchmark(run, repetitions=8, calls_per_replay=8)
    change()
    ref()
    graph.replay()
    check()
    baseline, _ = benchmark(ref, repetitions=8, calls_per_replay=8)
    return dict(
        exact=True,
        changed_input_replay=True,
        candidate_ms=timing["median_ms"],
        reference_ms=baseline["median_ms"],
    )


def routing(args, records):
    old = reference(args, "moe", moe)
    for experts, top in ((256, 8), (512, 10)):
        baseline = compile_kernel(
            old.router_topk.__wrapped__(4096, experts, top, dynamic_rows=True)
        )
        for warps in (4,) if args.quick else (1, 2, 4):
            candidate = compile_kernel(
                moe.router_topk_warp.__wrapped__(4096, experts, top, dynamic_rows=True, warps=warps)
            )
            export_kernel(candidate, args.output / f"router-e{experts}-w{warps}")
            for rows in [1, 3, 17] if args.quick else [1, 2, 3, 4, 8, 17, 33, 129, 512, 1023, 4096]:
                x = torch.randn(rows, experts, device="cuda")
                x[0].zero_()
                if rows > 1:
                    x[1].copy_(torch.arange(experts, device="cuda") % 3)
                if rows > 2:
                    x[2].mul_(100)
                expected = [
                    torch.full((rows + 2, top), 42, dtype=t, device="cuda")
                    for t in (torch.int32, torch.float32)
                ]
                actual = [t.clone() for t in expected]

                def ref():
                    return launch(baseline, x, *expected, rows)

                def run():
                    return launch(candidate, x, *actual, rows)

                result = verify(run, ref, actual, expected, lambda: x.copy_(x.flip(1)))
                records.append(
                    dict(kind="router", experts=experts, warps=warps, rows=rows, **result)
                )
                print("PASS", records[-1], flush=True)
                write_json(args.output / "results.json", dict(complete=False, cases=records))


def gating(args, records):
    old_hc, old_ple = reference(args, "hc", hyperconnection), reference(args, "ple", ple)
    for dtype in ("float16",):
        h, streams = 2560, 4
        tensor_dtype = getattr(torch, dtype)
        norm = compile_kernel(
            old_hc.hc_norm.__wrapped__(1024, h, streams, dtype=dtype, dynamic_rows=True)
        )
        gate = compile_kernel(old_ple.ple_gate.__wrapped__(1024, h, streams, dtype, True))
        candidate = compile_kernel(ple.ple_norm_gate.__wrapped__(1024, h, streams, dtype, True))
        export_kernel(candidate, args.output / f"ple-{dtype}")
        for rows in [1, 3, 17] if args.quick else [1, 2, 3, 4, 8, 17, 33, 129, 512, 1024]:
            key = torch.randn(rows, streams, h, device="cuda", dtype=tensor_dtype)
            query = torch.randn_like(key)
            query[0].zero_()
            kw = torch.randn(streams, h, device="cuda") * 0.1
            qw = torch.randn_like(kw) * 0.1
            value = torch.randn(rows, h, device="cuda", dtype=tensor_dtype)
            nk, nq = torch.empty_like(key), torch.empty_like(key)
            expected = torch.full((rows + 2, streams, h), 42, device="cuda", dtype=tensor_dtype)
            actual = expected.clone()

            def ref():
                launch(norm, key, kw, nk, rows)
                launch(norm, query, qw, nq, rows)
                launch(gate, nk, nq, value, expected, rows)

            def run():
                return launch(candidate, key, query, kw, qw, value, actual, rows)

            result = verify(run, ref, [actual], [expected], lambda: key.neg_())
            records.append(dict(kind="ple", dtype=dtype, rows=rows, **result))
            print("PASS", records[-1], flush=True)
            write_json(args.output / "results.json", dict(complete=False, cases=records))


def selection(args, records):
    capacity = 262144
    blocks, segments = capacity // 4, capacity // 4096
    hist = compile_kernel(qsa.radix_histogram.__wrapped__(4096, capacity, None, dynamic_rows=True))
    choose = compile_kernel(qsa.radix_choose.__wrapped__(4096, capacity, None, dynamic_rows=True))
    count = compile_kernel(qsa.selection_counts.__wrapped__(4096, capacity, dynamic_rows=True))
    offsets = compile_kernel(qsa.selection_offsets.__wrapped__(4096, capacity, dynamic_rows=True))
    scatter = compile_kernel(qsa.selection_scatter.__wrapped__(4096, capacity, dynamic_rows=True))
    configs = [
        (tile, threads, parts)
        for parts in args.qsa_parts
        for tile, threads in (
            [(512, 128)] if args.quick else [(256, 128), (512, 128), (1024, 128), (512, 256)]
        )
    ]
    for tile, threads, parts in configs:
        candidate = compile_kernel(
            qsa_selection.select.__wrapped__(4096, capacity, True, tile, threads, parts)
        )
        combine = compile_kernel(qsa_selection.merge.__wrapped__(4096, parts, True, threads))
        export_kernel(candidate, args.output / f"qsa-t{tile}-w{threads}-p{parts}")
        export_kernel(combine, args.output / f"qsa-merge-p{parts}-w{threads}")
        cases = (
            [(1, 5), (3, 2049), (17, 8193), (3, 262144)]
            if args.quick
            else [(1, 1), (1, 2048), (3, 2049), (17, 8193), (33, 32771), (129, 65539), (3, 262144)]
        )
        for rows, context in cases:
            for distribution in (
                ["ties"] if args.quick else ["random", "ties", "ascending", "descending"]
            ):
                scores = torch.rand(blocks, rows, device="cuda")
                if distribution == "ties":
                    scores.mul_(4).floor_()
                elif distribution in ("ascending", "descending"):
                    scores.copy_(torch.arange(blocks, device="cuda")[:, None].expand(-1, rows))
                    if distribution == "descending":
                        scores.copy_(scores.flip(0))
                position = torch.tensor([context - rows], device="cuda", dtype=torch.int32)
                prefix, remaining, greater = [
                    torch.zeros(rows, device="cuda", dtype=torch.int32) for _ in range(3)
                ]
                histogram = torch.empty(rows, segments, 256, device="cuda", dtype=torch.int32)
                counts = torch.empty(rows, 2, segments, device="cuda", dtype=torch.int32)
                places = torch.empty_like(counts)
                expected = torch.full((rows + 2, 2051), 42, device="cuda", dtype=torch.int32)
                actual = expected.clone()
                candidates = torch.full(
                    (rows + 2, parts * 1024), 42, device="cuda", dtype=torch.int32
                )

                def ref():
                    for shift in (24, 16, 8, 0):
                        launch(hist, scores, prefix, position, histogram, shift, rows)
                        launch(choose, histogram, prefix, remaining, position, shift, rows)
                    launch(count, scores, prefix, position, counts, rows)
                    launch(offsets, counts, places, greater, position, expected, rows)
                    launch(scatter, scores, prefix, places, greater, position, expected, rows)

                def run():
                    launch(candidate, scores, position, candidates, rows)
                    launch(combine, candidates, position, actual, rows)

                result = verify(
                    run, ref, [actual], [expected], lambda: scores.copy_(scores.flip(0))
                )
                assert bool((candidates[rows:] == 42).all()), "candidate tail guard"
                records.append(
                    dict(
                        kind="qsa",
                        tile=tile,
                        threads=threads,
                        parts=parts,
                        rows=rows,
                        context=context,
                        distribution=distribution,
                        **result,
                    )
                )
                print("PASS", records[-1], flush=True)
                write_json(args.output / "results.json", dict(complete=False, cases=records))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reference-dir", type=Path)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--qsa-parts", type=int, nargs="+", choices=(2, 4, 8), default=[2, 4])
    p.add_argument(
        "--families", nargs="+", choices=("router", "ple", "qsa"), default=["router", "ple", "qsa"]
    )
    args = p.parse_args()
    configure()
    records = []
    for family in args.families:
        {"router": routing, "ple": gating, "qsa": selection}[family](args, records)
    write_json(args.output / "results.json", dict(complete=True, cases=records))


if __name__ == "__main__":
    main()
