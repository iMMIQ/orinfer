"""Paired complete gatednorm/A8 chain, numerical boundaries and graph replay."""

import argparse
import gc
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from common import benchmark, configure, environment, export_kernel, identity, write_json
from kernels.model.gdn_gated_norm_a8 import gdn_gated_norm_a8
from kernels.operators.op17_gdn_gated_norm import gdn_gated_norm
from kernels.operators.op30_activation_quantization import activation_quantization


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument(
        "--threads", type=int, nargs="+", choices=(128, 256, 512), default=[256, 128, 512]
    )
    ap.add_argument("--validation-only", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    configure()
    sources = (
        "kernels/model/gdn_gated_norm_a8.py",
        "kernels/operators/op17_gdn_gated_norm.py",
        "kernels/operators/op30_activation_quantization.py",
    )
    for path in sources:
        dest = args.output / "dependencies" / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    model = json.loads(args.model.read_text())
    spec = next(b for b in model["buffers"] if b["name"] == "L0_GatedWeight")
    path = args.model.parent / spec["data"]["file"]
    assert identity(path)["sha256"] == spec["data"]["sha256"]
    w = torch.from_numpy(np.fromfile(path, dtype=np.float16).reshape(spec["shape"])).cuda()
    assert w.numel() == 128
    old = gdn_gated_norm()
    quant = activation_quantization(6144)
    report = dict(
        status="running",
        environment=environment(),
        sources=[identity(p) for p in sources],
        weight=identity(path),
        model=identity(args.model),
        cases=[],
        scope="Real L0 ordinary norm weight, synthetic X/Z; complete gatednorm/A8 "
        "chain, no whole-model TPS or BF16/FP8 quality claim",
    )
    specs = [(512, "random"), (513, "random")]
    if not args.quick:
        specs += [(2048, "random"), (8192, "random"), (3, "zero"), (3, "tiny"), (3, "large")]
    for threads in args.threads:
        kernel = gdn_gated_norm_a8(threads)
        export_kernel(kernel, args.output / f"threads{threads}")
        for rows, mode in specs:
            x = torch.randn((rows, 48, 128), device="cuda", dtype=torch.float16)
            z = torch.randn_like(x)
            if mode == "zero":
                x.zero_()
                z.zero_()
            if mode == "tiny":
                x.mul_(2**-20)
                z.mul_(2**-20)
            if mode == "large":
                x.mul_(1000)
                z.mul_(10)
            y = torch.empty_like(x)
            q = torch.empty((rows, 6144), device="cuda", dtype=torch.int8)
            scale = torch.empty(rows, device="cuda", dtype=torch.float16)
            oq = torch.empty_like(q)
            os = torch.empty_like(scale)
            mask = torch.zeros(6144, device="cuda", dtype=torch.uint8)

            def paired():
                old.adapter.func(x, z, w, y, stream=torch.cuda.current_stream().cuda_stream)
                quant.adapter.func(y, mask, oq, os, stream=torch.cuda.current_stream().cuda_stream)

            def run():
                kernel.adapter.func(
                    x, z, w, q, scale, stream=torch.cuda.current_stream().cuda_stream
                )

            paired()
            run()
            torch.cuda.synchronize()
            assert bool(torch.isfinite(y).all() and torch.isfinite(scale).all())
            differences = (q.to(torch.int16) - oq.to(torch.int16)).abs()
            code_mismatches = int((differences != 0).sum())
            scale_mismatches = int((scale != os).sum())
            # Reduction reassociation may change FP16 half ties. Diagnose
            # codes/scales, do not turn token identity into a quality gate.
            assert int(differences.max()) <= 1, (threads, rows, mode, int(differences.max()))
            assert bool(torch.allclose(scale.float(), os.float(), rtol=0.002, atol=2**-24))
            if args.validation_only:
                timing = paired_timing = None
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run()
            else:
                timing, graph = benchmark(run, repetitions=8)
                paired_timing, paired_graph = benchmark(paired, repetitions=8)
                del paired_graph
            saved_x, saved_z = x.clone(), z.clone()
            saved_q, saved_scale = q.clone(), scale.clone()
            x.zero_()
            z.zero_()
            q.fill_(42)
            scale.zero_()
            graph.replay()
            torch.cuda.synchronize()
            assert bool((q == 0).all() and (scale == 1).all())
            x.copy_(saved_x)
            z.copy_(saved_z)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(q, saved_q) and torch.equal(scale, saved_scale)
            rec = dict(
                rows=rows,
                mode=mode,
                threads=threads,
                timing=timing,
                paired_current_timing=paired_timing,
                code_mismatches=code_mismatches,
                scale_mismatches=scale_mismatches,
                max_code_difference=int(differences.max()),
                graph_zero_restore_bitwise=True,
            )
            report["cases"].append(rec)
            write_json(args.output / "result.json", report)
            print(json.dumps(rec), flush=True)
            x = None
            z = None
            y = None
            q = None
            scale = None
            oq = None
            os = None
            mask = None
            del saved_x
            del saved_z
            del saved_q
            del saved_scale
            del differences
            del graph
            gc.collect()
    report["status"] = "passed"
    write_json(args.output / "result.json", report)


if __name__ == "__main__":
    main()
