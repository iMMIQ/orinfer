"""Screen prefill INT8 GEMM with actual A8/W8 captures and an integer oracle.

The captured temporary W8 is checked against an independent W4 decoder. This
tool changes no quantization and reports projection timing, not model TPS.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.model.screen_prefill_ffn import decode_strict_w8, model_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--variant", choices=["tiles", "occupancy"], default="tiles")
    parser.add_argument(
        "--families", nargs="+", choices=["GateUp", "Down"], default=["GateUp", "Down"]
    )
    args = parser.parse_args()
    if args.rows <= 0:
        raise ValueError("rows must be positive")
    import numpy as np
    import torch
    from safetensors import safe_open
    from kernels.projections.candidates import int8_gemm
    from tools.operators.common import benchmark, configure, export_kernel, write_json

    configure()
    torch.empty(1, device="cuda")
    fingerprint = model_identity(args.model)
    captured = json.loads((args.activations / "capture.json").read_text())
    if captured["fingerprint"] != fingerprint or captured["seed"] != 20261002:
        raise ValueError("Capture belongs to another model/operator package or seed")
    weights = args.model / "cache/weights"
    index = json.loads((weights / "model.safetensors.index.json").read_text())["weight_map"]

    def tensor(name):
        with safe_open(str(weights / index[name]), framework="pt", device="cpu") as source:
            return source.get_tensor(name)

    def payload(prefix, suffix, digest):
        raw = (args.activations / (prefix + suffix)).read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("Capture payload digest mismatch: " + prefix + suffix)
        return raw

    report = dict(
        status="running",
        kind="expanded_int8_gemm",
        seed=20261002,
        fingerprint=fingerprint,
        rows=args.rows,
        cases=[],
        mapping_checks=[],
        scope="Actual captured prefill A8/W8; exact integer oracle; not end-to-end TPS.",
    )
    # Exercise every grouped launch mapping with a shortened final M group,
    # independent of which scheduling variant wins the real-model screen.
    for rows in [321, 513]:
        aa = torch.randint(-127, 128, (rows, 256), device="cuda", dtype=torch.int8)
        bb = torch.randint(-127, 128, (512, 256), device="cuda", dtype=torch.int8)
        scales_a = torch.full((rows,), 0.01, device="cuda", dtype=torch.float16)
        scales_b = torch.full((512,), 0.01, device="cuda", dtype=torch.float16)
        expected = (
            aa.float() @ bb.float().T * scales_a[:, None].float() * scales_b[None, :].float()
        ).half()
        for order in ["grouped4", "grouped8", "n1m2", "n1m4", "n1m8", "n2m2", "n2m4"]:
            kernel = int8_gemm(rows, 512, 256, 64, 64, 128, 2, threads=128, grid_order=order)
            guard = torch.full((rows * 512 + 256,), 123.0, device="cuda", dtype=torch.float16)
            result = guard[:-256].view(rows, 512)

            def mapping_run():
                kernel.adapter.func(
                    aa,
                    bb,
                    scales_a,
                    scales_b,
                    result,
                    stream=torch.cuda.current_stream().cuda_stream,
                )

            mapping_run()
            torch.cuda.synchronize()
            assert torch.equal(result, expected) and bool((guard[-256:] == 123.0).all())
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                mapping_run()
            saved = aa[-1].clone()
            aa[-1].zero_()
            result.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(result[:-1], expected[:-1]) and bool((result[-1] == 0).all())
            aa[-1].copy_(saved)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(result, expected) and bool((guard[-256:] == 123.0).all())
            report["mapping_checks"].append(
                dict(rows=rows, grid_order=order, exact=True, graph_last_row=True, guard=True)
            )
            kernel = None
            del guard
            result = None
            del graph
            del saved
        aa = None
        bb = None
        scales_a = None
        scales_b = None
        del expected
    for family in args.families:
        base = f"L{args.layer}_{family}"
        prefix = f"{base}_m{args.rows}"
        entry = captured["inputs"][prefix]
        n, k = entry["weight_shape"]
        if entry["rows"] != args.rows or entry["width"] != k:
            raise ValueError("Capture shape mismatch")
        raw = payload(prefix, ".w8", entry["weight_sha256"])
        logical = decode_strict_w8(
            tensor(base + "_P"), tensor(base + "_S"), tensor(base + "_Z"), tensor(base + "_WS")
        )
        if not np.array_equal(np.frombuffer(raw, dtype=np.int8).reshape(n, k), logical):
            raise ValueError("Temporary W8 differs from independent W4 reconstruction")
        w8 = torch.from_numpy(logical).cuda()
        del raw, logical
        raw = payload(prefix, ".a8", entry["activation_sha256"])
        aq = torch.from_numpy(np.frombuffer(raw, dtype=np.int8).copy().reshape(args.rows, k)).cuda()
        raw = payload(prefix, ".scale.f16", entry["scale_sha256"])
        asc = torch.from_numpy(np.frombuffer(raw, dtype=np.float16).copy()).cuda()
        ws = tensor(base + "_WS").cuda()
        del raw
        if k * 128 * 128 >= 2**31:
            raise ValueError("Projection can overflow the INT32 oracle")
        integer = torch.zeros((args.rows, n), device="cuda", dtype=torch.int32)
        for begin in range(0, k, 128):
            # Each partial integer dot is exactly representable in FP32.
            partial = aq[:, begin : begin + 128].float() @ w8[:, begin : begin + 128].float().T
            integer += partial.to(torch.int32)
        golden = (integer.float() * asc[:, None].float() * ws[None, :].float()).half()
        del integer, partial
        choices = []

        def measure(bm, bn, bk, stages, order, policy="default", threads=256):
            kernel = int8_gemm(
                args.rows,
                n,
                k,
                bm,
                bn,
                bk,
                stages,
                threads=threads,
                grid_order=order,
                cache_policy=policy,
            )
            guarded = torch.full((args.rows * n + 256,), 123.0, device="cuda", dtype=torch.float16)
            out = guarded[:-256].view(args.rows, n)

            def run():
                kernel.adapter.func(
                    aq, w8, asc, ws, out, stream=torch.cuda.current_stream().cuda_stream
                )

            timing, graph = benchmark(run, repetitions=8)
            assert torch.equal(out, golden), (family, bm, bn, bk, stages, order, "oracle mismatch")
            assert bool((guarded[-256:] == 123.0).all()), "Output guard changed"
            saved = aq.clone()
            aq.zero_()
            out.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            assert bool((out == 0).all()), "Graph ignored changed input"
            aq.copy_(saved)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out, golden) and bool((guarded[-256:] == 123.0).all())
            key = f"{base}-m{args.rows}-bm{bm}-bn{bn}-bk{bk}-s{stages}-t{threads}-{order}-{policy}"
            export_kernel(kernel, args.output / key)
            choice = dict(
                family=family,
                n=n,
                k=k,
                bm=bm,
                bn=bn,
                bk=bk,
                stages=stages,
                threads=threads,
                grid_order=order,
                cache_policy=policy,
                timing=timing,
                integer_oracle_equal=True,
                decoded_weight_equal=True,
                tail_guard=True,
                graph_zero_restore=True,
                export=key,
            )
            choices.append(choice)
            report["cases"].append(choice)
            write_json(args.output / "result.json", report)
            print(key, round(timing["median_ms"], 4), "ms", flush=True)

        tiles = [
            (256, 128, 128, 2),
            (128, 128, 128, 2),
            (128, 256, 128, 2),
            (256, 64, 128, 2),
            (128, 128, 64, 3),
            (256, 128, 64, 3),
        ]
        orders = ["nfirst", "mfirst", "n1m4", "n1m8", "grouped4", "grouped8"]
        if args.variant == "tiles":
            for tile in tiles:
                for order in orders:
                    measure(*tile, order)
        else:
            measure(256, 128, 128, 2, "nfirst")
            for tile in [
                (256, 128, 128, 2),
                (128, 256, 128, 2),
                (128, 128, 128, 2),
                (256, 64, 128, 2),
                (256, 128, 64, 3),
            ]:
                for threads in [128, 512]:
                    for order in ["nfirst", "n1m4"]:
                        measure(*tile, order, threads=threads)
        best = min(choices, key=lambda c: c["timing"]["median_ms"])
        for policy in ["a-last", "b-first", "a-last-b-first", "b-last", "a-first-b-last"]:
            measure(
                best["bm"],
                best["bn"],
                best["bk"],
                best["stages"],
                best["grid_order"],
                policy,
                best["threads"],
            )
        # Recheck finalists with longer measurements so a single short trial
        # does not determine publication. Retain the initial timing separately.
        finalists = sorted(choices, key=lambda c: c["timing"]["median_ms"])[:4]
        for choice in finalists:
            kernel = int8_gemm(
                args.rows,
                n,
                k,
                choice["bm"],
                choice["bn"],
                choice["bk"],
                choice["stages"],
                threads=choice["threads"],
                grid_order=choice["grid_order"],
                cache_policy=choice["cache_policy"],
            )
            out = torch.empty_like(golden)

            def run():
                kernel.adapter.func(
                    aq, w8, asc, ws, out, stream=torch.cuda.current_stream().cuda_stream
                )

            choice["screen_timing"] = choice["timing"]
            choice["timing"], graph = benchmark(run, repetitions=32)
            assert torch.equal(out, golden)
            choice["finalist_rechecked"] = True
            print("finalist", choice["export"], choice["timing"]["median_ms"], flush=True)
            kernel = None
            out = None
            del graph
        best = min(finalists, key=lambda c: c["timing"]["median_ms"])
        tail_a = torch.cat([aq, aq[:1]], dim=0)
        tail_scale = torch.cat([asc, asc[:1]], dim=0)
        tail_expected = torch.cat([golden, golden[:1]], dim=0)
        kernel = int8_gemm(
            args.rows + 1,
            n,
            k,
            best["bm"],
            best["bn"],
            best["bk"],
            best["stages"],
            threads=best["threads"],
            grid_order=best["grid_order"],
            cache_policy=best["cache_policy"],
        )
        guarded = torch.full(
            ((args.rows + 1) * n + 256,), 123.0, device="cuda", dtype=torch.float16
        )
        out = guarded[:-256].view(args.rows + 1, n)

        def tail_run():
            kernel.adapter.func(
                tail_a, w8, tail_scale, ws, out, stream=torch.cuda.current_stream().cuda_stream
            )

        tail_run()
        torch.cuda.synchronize()
        assert torch.equal(out, tail_expected) and bool((guarded[-256:] == 123.0).all())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            tail_run()
        tail_a[-1].zero_()
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out[:-1], golden) and bool((out[-1] == 0).all())
        tail_a[-1].copy_(aq[0])
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, tail_expected) and bool((guarded[-256:] == 123.0).all())
        best["nonaligned_tail_rows"] = args.rows + 1
        best["selected"] = True
        export_kernel(kernel, args.output / (best["export"] + "-tail"))
        print("best", family, best["export"], best["timing"]["median_ms"], flush=True)
        write_json(args.output / "result.json", report)
        kernel = None
        out = None
        del guarded
        del graph
        tail_a = None
        tail_scale = None
        del tail_expected
        aq = None
        asc = None
        ws = None
        w8 = None
        golden = None
    report["status"] = "passed"
    write_json(args.output / "result.json", report)


if __name__ == "__main__":
    main()
