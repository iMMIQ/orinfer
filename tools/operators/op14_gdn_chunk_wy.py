"""Offline WY mathematical verification, graph tests, timing and actual ABI."""
import argparse
import json
import shutil
import time
from pathlib import Path

import torch

from abi import parse_host
from common import benchmark, configure, environment, error, export_kernel, identity, write_json
from gdn_reference import chunk_matrices, triangle_transform, wy
from kernels.operators.op14_gdn_chunk_wy import gdn_chunk_wy, launch

ROOT = Path(__file__).resolve().parents[2]


def inputs(batch, tokens, bt, dtype, mode):
    chunks = (tokens + bt - 1) // bt
    k = torch.nn.functional.normalize(torch.randn((batch, 16, chunks, bt, 128), device="cuda"), dim=-1).to(getattr(torch, dtype))
    v = torch.randn((batch, 48, chunks, bt, 128), device="cuda").half()
    g = -torch.rand((batch, 48, chunks, bt), device="cuda") * .08
    beta = torch.rand_like(g)
    if mode == "g0":
        g.zero_()
    elif mode == "beta0":
        beta.zero_()
    elif mode == "beta1":
        beta.fill_(1.)
    elif mode == "strong":
        g.fill_(-1000.)
    valid = tokens - (chunks - 1) * bt
    if valid < bt:
        k[:, :, -1, valid:].zero_(); v[:, :, -1, valid:].zero_()
        g[:, :, -1, valid:].zero_(); beta[:, :, -1, valid:].zero_()
    gc = g.cumsum(-1)
    system, _ = chunk_matrices(k, k, gc, beta)
    # torch solve_triangular returns column-major matrix strides on CUDA;
    # the production API is explicitly contiguous row-major.
    a = triangle_transform(system).contiguous()
    if mode == "identity":
        a.copy_(torch.eye(bt, device="cuda").expand_as(a))
    if mode == "pad_garbage":
        assert valid < bt
        k[:, :, -1, valid:].fill_(float("nan"))
        v[:, :, -1, valid:].fill_(float("nan"))
        gc[:, :, -1, valid:].fill_(float("nan"))
        upper = torch.triu(torch.ones((bt, bt), device="cuda", dtype=torch.bool), diagonal=1)
        a[..., upper] = float("nan")
    if mode == "beta0":
        k.fill_(float("nan")); v.fill_(float("nan")); gc.fill_(float("nan"))
    return a, k, v, gc, beta


def reference(a, k, v, gc, beta):
    # Shared reference remains authoritative. Sanitize suppressed lanes before
    # calling it, because its intentionally simple beta*x expression has 0*NaN.
    inactive_k = (beta.reshape(beta.shape[0], 16, 3, *beta.shape[2:]) == 0).all(dim=2)
    safe_k = torch.where(inactive_k[..., None], 0., k).to(k.dtype)
    safe_v = torch.where((beta == 0)[..., None], 0., v).to(v.dtype)
    safe_gc = torch.where(beta == 0, 0., gc)
    return wy(torch.tril(a), safe_k, safe_v, safe_gc, beta)


def check(w, u, expected):
    result = {"W": error(w, expected[0]), "U": error(u, expected[1])}
    for metric in result.values():
        assert metric["finite"] and metric["relative_l2"] <= .002, result
    assert torch.allclose(w, expected[0], atol=5e-6, rtol=5e-4), result
    assert torch.allclose(u, expected[1], atol=2e-5, rtol=5e-4), result
    return result


def fp64_subset(a, k, v, gc, beta, w, u):
    aa = torch.tril(a[:1, :3, :1]).double()
    kk = k[:1, :1, :1].double().expand(1, 3, 1, k.shape[-2], 128)
    vv, gg, bb = v[:1, :3, :1].double(), gc[:1, :3, :1].double(), beta[:1, :3, :1].double()
    kk = torch.where((bb == 0)[..., None], 0., kk)
    vv = torch.where((bb == 0)[..., None], 0., vv)
    gg = torch.where(bb == 0, 0., gg)
    ww = aa @ ((bb[..., None] * kk) * gg.exp()[..., None])
    uu = aa @ (bb[..., None] * vv)
    def metric(actual, ref):
        delta = actual.double() - ref
        return {"finite": bool(torch.isfinite(delta).all()),
                "relative_l2": float(delta.norm() / ref.norm().clamp_min(1e-30)),
                "max_abs": float(delta.abs().max())}
    result = {"W": metric(w[:1, :3, :1], ww), "U": metric(u[:1, :3, :1], uu)}
    assert all(x["finite"] and x["relative_l2"] <= .002 for x in result.values()), result
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--value-tile", type=int, default=32)
    args = parser.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    configure(); env = environment()
    paths = ['kernels/operators/op14_gdn_chunk_wy.py', 'tools/operators/op14_gdn_chunk_wy.py', 'tools/operators/common.py', 'tools/operators/abi.py', 'tools/operators/gdn_reference.py']
    frozen = []
    for path in paths:
        dest = out / "source-freeze" / path; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / path, dest); frozen.append(identity(dest))
    report = {"environment": env, "source_freeze": frozen, "TF32": False,
              "reference": "shared gdn_reference FP32 WY and legal A from chunk_matrices/triangle_transform; same-input FP64 product subset; synthetic inputs, no model trace",
              "math": "FP32 SIMT, FP16 K/V promoted once; no A/BK/BV/output downcast; beta*K then exp(G); sequential FP32 FMA sum",
              "relative_l2_bugline": .002, "budget_B1_T512_ms": .250,
              "workspace_bytes": 0, "resident_parameter_bytes": 0,
              "value_tile": args.value_tile, "compilation": [], "cases": []}
    specs = [(1, 512, 64, "float16", "random")]
    if not args.quick:
        specs = [(b, bt + 1, bt, "float16", "random") for bt in (16, 32, 64) for b in (1, 2, 3, 4, 5, 7, 8)]
        specs += [(1, tokens, bt, "float16", "random") for bt in (16, 32, 64) for tokens in (511, 512, 513, 2048, 8192)]
        specs += [(3, bt + 1, bt, "float16", mode) for bt in (16, 32, 64) for mode in ("g0", "beta0", "beta1", "strong", "identity", "pad_garbage")]
        specs += [(3, bt + 1, bt, "float32", "random") for bt in (16, 32, 64)]
    kernels = {}
    for b, tokens, bt, dtype, mode in specs:
        key = bt, dtype
        if key not in kernels:
            start = time.perf_counter()
            kernels[key] = gdn_chunk_wy(bt, dtype, args.value_tile)
            report["compilation"].append({"BT": bt, "K_dtype": dtype, "compile_prepare_s": time.perf_counter() - start})
        kernel = kernels[key]
        start = time.perf_counter()
        a, k, v, gc, beta = inputs(b, tokens, bt, dtype, mode)
        assert all(t.is_contiguous() for t in (a, k, v, gc, beta))
        w = torch.empty_like(v, dtype=torch.float32); u = torch.empty_like(w)
        torch.cuda.synchronize(); prepare_s = time.perf_counter() - start
        run = lambda: launch(kernel, a, k, v, gc, beta, w, u, stream=torch.cuda.current_stream().cuda_stream)
        start = time.perf_counter(); run(); torch.cuda.synchronize()
        first_ms = 1000 * (time.perf_counter() - start)
        expected = reference(a, k, v, gc, beta)
        case = {"B": b, "T": tokens, "BT": bt, "C": k.shape[2], "K_dtype": dtype, "mode": mode,
                "prepare_input_A_output_s": prepare_s, "first_use_host_wall_ms": first_ms,
                "error_FP32": check(w, u, expected), "error_FP64_subset": fp64_subset(a, k, v, gc, beta, w, u),
                "input_bytes": sum(t.numel() * t.element_size() for t in (a, k, v, gc, beta)),
                "output_bytes": 8 * w.numel(), "workspace_bytes": 0,
                "logical_read_bytes_note": "input allocation bytes; A reloaded per dimension tile; shared K read with kh=vh//3; no repeated K allocation"}
        matrices = b * 48 * k.shape[2]
        case["logical_source_read_bytes"] = {
            "A_lower_reloaded_per_dimension_tile": matrices * bt * (bt + 1) // 2 * 4 * (128 // args.value_tile),
            "K_mapped_reads_before_beta_zero_suppression": matrices * bt * 128 * k.element_size(),
            "V_reads_before_beta_zero_suppression": v.numel() * v.element_size(),
            "G_Beta_unique_rows_per_CTA_minimum_before_suppression": matrices * bt * 8 * (128 // args.value_tile),
            "note": "logical load demand, not measured DRAM traffic; gate loads can be replicated across threads, cache may reuse shared K/A, exact beta=0 skips K/V/G"}
        case["timing"], graph = benchmark(run, repetitions=8, calls_per_replay=4)
        if tokens % bt:
            valid = tokens % bt
            assert bool((w[:, :, -1, valid:] == 0).all() and (u[:, :, -1, valid:] == 0).all())
            case["padded_rows_exact_zero"] = True
        if mode == "random" and (tokens == bt + 1 and b == 3 or b == 1 and tokens == 512 and bt == 64):
            originals = [t.clone() for t in (a, k, v, gc, beta)]
            case["graph_mutations"] = []
            for name, tensor, factor in zip(("A", "K", "V", "G", "Beta"), (a, k, v, gc, beta), (.75, -.5, -.75, 1.5, .5)):
                tensor.mul_(factor); w.fill_(float("nan")); u.fill_(float("nan"))
                graph.replay(); torch.cuda.synchronize()
                case["graph_mutations"].append({"input": name, "check": check(w, u, reference(a, k, v, gc, beta))})
                for dst, src in zip((a, k, v, gc, beta), originals):
                    dst.copy_(src)
            w.fill_(float("nan")); u.fill_(float("nan"))
            graph.replay(); torch.cuda.synchronize()
            case["graph_restored"] = check(w, u, expected)
            if b > 1:
                old_w, old_u = w.clone(), u.clone()
                v[0].mul_(2.); a[0].mul_(.5); run(); torch.cuda.synchronize()
                assert torch.equal(w[1:], old_w[1:]) and torch.equal(u[1:], old_u[1:])
                case["request_isolation_bitwise"] = True
                for dst, src in zip((a, k, v, gc, beta), originals):
                    dst.copy_(src)
            del originals
        report["cases"].append(case)
        write_json(out / "progress.json", report)
        print(json.dumps({"B": b, "T": tokens, "BT": bt, "dtype": dtype, "mode": mode,
                          "ms": case["timing"]["median_ms"], "errors": case["error_FP32"]}), flush=True)
        del graph, a, k, v, gc, beta, w, u, expected
    for (bt, dtype), kernel in kernels.items():
        dest = out / f"bt{bt}-{dtype}"; artifact = export_kernel(kernel, dest)
        write_json(dest / "abi.json", {"operator": "op14_gdn_chunk_wy", "sm": 87,
            "BT": bt, "K_dtype": dtype, "value_tile": args.value_tile,
            "logical_parameters": ["A_fp32[B,48,C,BT,BT]", f"K_{dtype}[B,16,C,BT,128]",
                "V_fp16[B,48,C,BT,128]", "G_fp32[B,48,C,BT]", "Beta_fp32[same]", "W_fp32[B,48,C,BT,128]", "U_fp32[same]"],
            "actual_generated_launches": parse_host((dest / "host.txt").read_text()),
            "layout": "contiguous row-major; kh=vh//3", "workspace_bytes": 0,
            "resident_parameter_bytes": 0, "cooperative_launch": False,
            "stream": "explicit caller stream", "alias_policy": "all seven buffers disjoint",
            "rounding": report["math"], "tail_policy": "finite lower A; identity padding; beta=0 suppresses reads of K/V/G; upper A ignored",
            "toolchain": env, "artifacts": artifact})
    report["peak_torch_validation_allocated_bytes"] = torch.cuda.max_memory_allocated()
    report["status"] = "passed"
    write_json(out / "results.json", report)
    print(json.dumps({"status": "passed", "cases": len(report["cases"])}), flush=True)


if __name__ == "__main__":
    main()
