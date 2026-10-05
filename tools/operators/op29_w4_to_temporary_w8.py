"""Offline op29 exact-code validation and real down full-path measurements."""
import argparse
import gc
import re
import shutil
import time
import traceback
from pathlib import Path
from tools.reference import ACTIVATIONS as REFERENCE_ACTIVATIONS
from tools.reference import CHECKPOINT, checkpoint_sha256

import torch
import tilelang.language as T

from common import (benchmark, configure, environment, error, export_kernel,
                    identity, tensor_sha, write_json)
from abi import parse_host
from kernels.operators.op29_w4_to_temporary_w8 import w4_to_temporary_w8, launch
from kernels.operators.op30_activation_quantization import activation_quantization, launch as launch_a8
from kernels.projections.candidates import int8_gemm
from tools.projections.screen import load_weights, activations

ROOT = Path(__file__).resolve().parents[2]


def dequant(p, s, z):
    n, k2 = p.shape
    q = torch.stack((p & 15, p >> 4), -1).reshape(n, k2 * 2)
    return ((q.reshape(n, -1, 128).float() - z[..., None].float()) * s[..., None].float()).reshape(n, -1).half()


def row_metadata(b):
    amax = b.float().abs().amax(1)
    return torch.where(amax > 0, (amax / 127).clamp_min(2**-24), 1.).half()


def codes(b, ws):
    result = torch.empty_like(b, dtype=torch.int8)
    for begin in range(0, len(b), 512):
        result[begin:begin + 512] = (b[begin:begin + 512].float() / ws[begin:begin + 512, None].float()).round().clamp(-127, 127).to(torch.int8)
    return result


def exact(actual, expected):
    mismatches = int((actual != expected).sum())
    assert mismatches == 0, {"code_mismatches": mismatches}
    return {"code_mismatches": mismatches, "elements": actual.numel()}


def run(kernel, p, s, z, ws, w8):
    launch(kernel, p, s, z, ws, w8, stream=torch.cuda.current_stream().cuda_stream)


def export(name, kernel, out, report, metadata):
    directory = out / "compiled" / name
    files = export_kernel(kernel, directory)
    host = (directory / "host.txt").read_text()
    cuda = (directory / "kernel.cu").read_text()
    entry = re.search(r'extern "C" __global__ void (\w+)\(([^;]+)\);', cuda)
    info = {"operator": "op29_w4_to_temporary_w8", "name": name, **metadata,
            "sm": 87, "toolchain": report["environment"], "exports": files,
            "entry_symbol": entry.group(1), "actual_host_launches": parse_host(host),
            "ordered_cuda_arguments": [a.strip() for a in entry.group(2).split(',')],
            "argument_order_source": "actual generated CUDA and NVRTC host wrapper",
            "cooperative": False, "stream": "explicit caller current stream at every invocation",
            "aliasing": "disjoint stable contiguous input/output buffers",
            "persistent_global_LUT_bytes": 0, "persistent_full_W8_bytes": 0}
    write_json(directory / "abi.json", info)
    report["exports"].append(info)


def graph_check(graph, p, s, z, ws, w8, expected):
    # Change every input class while retaining addresses. Each update of P/S/Z
    # recomputes canonical rowWS; metadata alone is additionally injected to
    # check that the graph observes its runtime pointer (not production policy).
    backup = [x[0].clone() for x in (p, s, z, ws)]
    records = []
    for field in ("P", "S", "Z", "WS"):
        if field == "P": p[0].bitwise_xor_(17)
        elif field == "S": s[0].mul_(.5)
        elif field == "Z": z[0].copy_((z[0].int() + 1).remainder(16).to(torch.int8))
        else: ws[0].mul_(2.)
        if field != "WS": ws[:1].copy_(row_metadata(dequant(p[:1], s[:1], z[:1])))
        expected[:1].copy_(codes(dequant(p[:1], s[:1], z[:1]), ws[:1]))
        w8.fill_(-128)
        graph.replay()
        torch.cuda.synchronize()
        records.append({"changed_input": field, **exact(w8, expected)})
        for tensor, old in zip((p, s, z, ws), backup): tensor[0].copy_(old)
        expected[:1].copy_(codes(dequant(p[:1], s[:1], z[:1]), ws[:1]))
        w8.fill_(-128)
        graph.replay()
        torch.cuda.synchronize()
        exact(w8, expected)
    return {"changed_P_S_Z_WS_and_output_poison": records, "restored_exact": True,
            "metadata_injection_note": "WS-only doubled scale tests graph dependency; production requires canonical metadata"}


def small_cases(builds, report):
    for n, k in ((1, 128), (17, 384), (65, 640)):
        p = torch.randint(0, 256, (n, k // 2), device="cuda", dtype=torch.uint8)
        s = (torch.randn(n, k // 128, device="cuda").abs() * .125).half()
        z = torch.randint(0, 16, s.shape, device="cuda", dtype=torch.int8)
        p[0].zero_(); z[0].zero_()
        b = dequant(p, s, z); ws = row_metadata(b); expected = codes(b, ws)
        assert float(ws[0]) == 1.
        for name, kernel in builds.items():
            w8 = torch.empty((n, k), device="cuda", dtype=torch.int8)
            fn = lambda: run(kernel, p, s, z, ws, w8)
            fn(); torch.cuda.synchronize()
            timing, graph = benchmark(fn, repetitions=3, calls_per_replay=16)
            row = {"N": n, "K": k, "route": name, "kind": "random_and_all_zero_row",
                   "exact": exact(w8, expected), "timing": timing,
                   "graph": graph_check(graph, p, s, z, ws, w8, expected)}
            report["edge_cases"].append(row)
    # Canonical WS=1 from maximum127 in group0; group1 explicitly hits
    # positive/negative FP32 half-integer ties. Group2 exercises half subnormals.
    n, k = 17, 384
    q = torch.ones((n, k), device="cuda", dtype=torch.uint8)
    q[:, 128:256] = torch.arange(128, device="cuda", dtype=torch.uint8)[None] % 16
    q[:, 256:] = q[:, 128:256]
    p = q[:, ::2] | (q[:, 1::2] << 4)
    s = torch.tensor([127., .5, 2**-24], device="cuda", dtype=torch.float16).repeat(n, 1)
    z = torch.tensor([0, 8, 8], device="cuda", dtype=torch.int8).repeat(n, 1)
    b = dequant(p, s, z); ws = row_metadata(b); expected = codes(b, ws)
    ratio = b.float() / ws[:, None].float()
    ties = (ratio.abs().remainder(1) == .5).sum().item()
    assert ties > 0 and bool((ws == 1).all())
    for name, kernel in builds.items():
        w8 = torch.empty_like(q, dtype=torch.int8)
        run(kernel, p, s, z, ws, w8); torch.cuda.synchronize()
        report["edge_cases"].append({"N": n, "K": k, "route": name,
          "kind": "canonical_scale_positive_negative_ties_and_half_subnormals",
          "ties": ties, "exact": exact(w8, expected)})


    # Nonzero underflow floor and largest safe half source magnitudes.
    p = torch.full((2, 64), 255, device="cuda", dtype=torch.uint8)
    s = torch.tensor([[2**-24], [4096.]], device="cuda", dtype=torch.float16)
    z = torch.zeros((2, 1), device="cuda", dtype=torch.int8)
    b = dequant(p, s, z); ws = row_metadata(b); expected = codes(b, ws)
    assert float(ws[0]) == 2**-24 and bool(torch.isfinite(b).all())
    for name, kernel in builds.items():
        w8 = torch.empty((2, 128), device="cuda", dtype=torch.int8)
        run(kernel, p, s, z, ws, w8); torch.cuda.synchronize()
        report["edge_cases"].append({"N": 2, "K": 128, "route": name,
            "kind": "nonzero_rowWS_subnormal_floor_and_safe_large_half_weights",
            "rowWS": ws.tolist(), "exact": exact(w8, expected)})


def full_chain(p, s, z, ws, w8, b, expand, layer, out, report):
    n, k = w8.shape
    sources = activations(REFERENCE_ACTIVATIONS, "down", layer)
    pre = next(x for m, x, source in sources if m == 512)
    dec = next(x for m, x, source in sources if m == 8)
    for _, _, source in sources:
        for meta in source["sources"]:
            path = REFERENCE_ACTIVATIONS / meta["file"]
            if str(path) not in {item["file"]["path"] for item in report["input_sources"]}:
                info = {"file": identity(path), "tensor_sha256": tensor_sha(torch.load(path, map_location="cpu", weights_only=True)), "metadata": meta}
                assert info["file"]["sha256"] == meta["file_sha256"]
                assert info["tensor_sha256"] == meta["tensor_sha256"]
                report["input_sources"].append(info)
    mask = torch.zeros(k, device="cuda", dtype=torch.uint8)
    aq = activation_quantization(k)
    export(f"a8-layer{layer}", aq, out, report, {"dependency": "op30 dynamic-M per-token A8", "K": k})
    gems = {}
    for bm, bn, threads in ((16, 64, 128), (128, 128, 256), (256, 128, 256)):
        key = f"BM{bm}BN{bn}"
        started = time.perf_counter()
        gems[key] = int8_gemm(T.dynamic("M"), n, k, bm, bn, 128, 2, threads)
        report["compile"].append({"name": f"gemm-layer{layer}-{key}", "seconds": time.perf_counter() - started})
        export(f"gemm-layer{layer}-{key}", gems[key], out, report, {"dependency": "existing TileLang int8_gemm, INT32 MMA then AS/WS scaling", "dynamic_M": True, "N": n, "K": k})
    # b/bw are validation-only materializations, not resident production weights.
    bf = b.float(); bw = w8.float() * ws[:, None].float()
    for m in (1, 2, 3, 4, 5, 7, 8, 511, 512, 513, 2048, 8192):
        if m <= 8: cpu = dec[:m].contiguous(); origin = "consecutive captured M1 decode rows, not simultaneous model batch"
        elif m == 511: cpu = pre[:511].contiguous(); origin = "captured512prefill prefix511, tail verification"
        elif m == 512: cpu = pre; origin = "actual captured512prefill"
        elif m == 513: cpu = torch.cat((pre, dec[:1])); origin = "captured512prefill + firstdecode, boundary verification"
        else: cpu = pre.repeat(m // 512, 1); origin = f"actual512 activation repeated{m // 512}x for{m}-row kernel shape; not genuine long-context model trace/quality"
        x = cpu.cuda(); q = torch.empty((m, k), device="cuda", dtype=torch.int8)
        asc = torch.empty((m, 1), device="cuda", dtype=torch.float16)
        y = torch.empty((m, n), device="cuda", dtype=torch.float16)
        def quant(): launch_a8(aq, x, mask, q, asc, stream=torch.cuda.current_stream().cuda_stream)
        quant(); torch.cuda.synchronize()
        refscale = torch.where(x.float().abs().amax(1) > 0, (x.float().abs().amax(1) / 127).clamp_min(2**-24), 1.).half()
        refcodes = (x.float() / refscale[:, None].float()).round().clamp(-127, 127).to(torch.int8)
        exact(q, refcodes); assert torch.equal(asc[:, 0], refscale)
        ah = q.float() * asc.float()
        quantref = ah @ bw.T; orig = x.float() @ bf.T
        row = {"layer": layer, "M": m, "N": n, "K": k, "origin": origin,
               "activation_sha256": tensor_sha(cpu), "A8_code_exact": exact(q, refcodes),
               "quantization_loss_vs_original_W4": {"A8_only": error(ah @ bf.T, orig),
                 "W8_only": error(x.float() @ bw.T, orig), "combined": error(quantref, orig)},
               "candidates": [], "resident_bytes": p.numel() + 2 * s.numel() + z.numel() + 2 * ws.numel(),
               "workspace_bytes": w8.numel() + q.numel() + 2 * asc.numel() + 2 * y.numel()}
        choices = ("BM16BN64",) if m < 511 else ("BM128BN128", "BM256BN128")
        for name in choices:
            gem = gems[name]
            def multiply(): gem.adapter.func(q, w8, asc[:, 0], ws, y, stream=torch.cuda.current_stream().cuda_stream)
            def chain(): run(expand, p, s, z, ws, w8); quant(); multiply()
            started = time.perf_counter(); chain(); torch.cuda.synchronize()
            entry = {"gemm_route": name, "kernel_count_per_call": 3,
                     "first_use_ms": (time.perf_counter() - started) * 1000,
                     "kernel_error_vs_explicit_quantized_FP32": error(y, quantref)}
            assert entry["kernel_error_vs_explicit_quantized_FP32"]["relative_l2"] <= .002
            entry["complete_chain"], graph = benchmark(chain, repetitions=3, calls_per_replay=4)
            entry["gemm_and_scales"], _ = benchmark(multiply, repetitions=3, calls_per_replay=4)
            saved = x.clone(); x.zero_(); w8.fill_(-128); q.fill_(-128); y.fill_(float('nan'))
            graph.replay(); torch.cuda.synchronize(); assert bool((y == 0).all())
            x.copy_(saved); w8.fill_(-128); q.fill_(-128); y.fill_(float('nan'))
            graph.replay(); torch.cuda.synchronize()
            assert error(y, quantref)["relative_l2"] <= .002
            entry["graph_changed_activation_poison_W8_and_restore"] = True
            row["candidates"].append(entry)
        row["A8_stage"], _ = benchmark(quant, repetitions=3, calls_per_replay=4)
        row["best_measured_complete_ms"] = min(a["complete_chain"]["median_ms"] for a in row["candidates"])
        row["linear_budget_ms"] = .28 if m <= 8 else 1.8 * (m / 512)
        row["quality_policy"] = "isolated math diagnostic; M512 known model failure prevents enable/accept; 2K repeats are shape timing only"
        report["full_chains"].append(row)
        write_json(out / "results.json", report)
        print({"layer": layer, "M": m, "chain_ms": row["best_measured_complete_ms"]}, flush=True)
        del x, q, asc, y, saved, ah, quantref, orig, refcodes, refscale, graph
        gc.collect()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--output", required=True); ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(); out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    configure()
    report = {"environment": environment(), "status": "running", "exports": [], "compile": [],
              "edge_cases": [], "real_weights": [], "full_chains": [], "input_sources": [], "failures": [],
              "checkpoint": {"path": str(CHECKPOINT / 'model.safetensors'), "sha256": checkpoint_sha256(CHECKPOINT / 'model.safetensors'),
                "identity_note": "Supplied checkpoint hashed once; read source tensors separately hashed"},
              "budget": "conditional29 expansion and conditional30 A8 charged within complete linear budget; no standalone29 allocation",
              "quality_policy": "no model quality acceptance or enable; M512 already has known quality degradation",
              "persistent_global_LUT_bytes": 0, "persistent_full_W8_bytes": 0,
              "memory_policy": "P/S/Z + rowWS resident; explicit W8 workspace overwritten each layercall; Torch full weights only validation"}
    paths = ['kernels/operators/op29_w4_to_temporary_w8.py', 'tools/operators/op29_w4_to_temporary_w8.py', 'kernels/operators/op30_activation_quantization.py', 'kernels/operators/op04_swiglu.py', 'kernels/projections/candidates.py', 'tools/projections/screen.py', 'tools/operators/common.py', 'tools/operators/abi.py']
    report["sources"] = []
    for path in paths:
        dest = out / "measurement-source" / path; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / path, dest); report["sources"].append(identity(dest))
    write_json(out / "results.json", report)
    try:
        builds = {}
        for name, route, bn in (("inline64", "inline", 64), ("inline32", "inline", 32), ("direct16", "direct", 16)):
            started = time.perf_counter(); builds[name] = w4_to_temporary_w8(route=route, BN=bn)
            report["compile"].append({"name": name, "seconds": time.perf_counter() - started})
            export(name, builds[name], out, report, {"route": route, "BN": bn, "BK": 256,
                "dynamic_N": True, "dynamic_K": True, "K_multiple": 128,
                "tensor_ABI": "P[N,K/2]u8,S[N,K/128]f16,Z[N,K/128]i8,WS[N]f16,W8[N,K]i8",
                "workspace_bytes": "N*K (W8)", "resident_weight_bytes": "N*K/2+3*N*K/128+2*N"})
        small_cases(builds, report)
        for layer in (0, 32):
            for case in ("down", "gate_up", "gdn_qkvz"):
                started = time.perf_counter()
                p, s, z, b, tensors, cost = load_weights(case, layer)
                n, k = b.shape
                metadata_start = time.perf_counter(); ws = row_metadata(b); torch.cuda.synchronize()
                metadata_ms = (time.perf_counter() - metadata_start) * 1000
                expected = codes(b, ws)
                row = {"layer": layer, "case": case, "N": n, "K": k,
                    "source_tensors": tensors, "offline_preparation": cost,
                    "rowWS_sha256": tensor_sha(ws), "logical_tensors": {"P": tensor_sha(p), "S": tensor_sha(s), "Z": tensor_sha(z)},
                    "rowWS_metadata_preparation_ms": metadata_ms,
                    "resident_bytes": {"P_S_Z": p.numel() + 2 * s.numel() + z.numel(), "rowWS": 2 * n, "persistent_LUT": 0, "persistent_full_W8": 0},
                    "effective_bits_per_parameter": 8 * (p.numel() + 2 * s.numel() + z.numel() + 2*n) / (n*k),
                    "if_old_LUT_persisted_extra_bytes": n * (k//128) * 16,
                    "if_old_LUT_persisted_extra_bits_per_parameter": 1.0,
                    "workspace_bytes": n*k, "routes": []}
                w8 = torch.empty_like(b, dtype=torch.int8)
                for name, kernel in builds.items():
                    fn = lambda: run(kernel, p, s, z, ws, w8)
                    start = time.perf_counter(); fn(); torch.cuda.synchronize()
                    entry = {"route": name, "first_use_ms": 1000*(time.perf_counter()-start), "exact": exact(w8, expected)}
                    entry["expansion"], graph = benchmark(fn, repetitions=5, calls_per_replay=4)
                    row["routes"].append(entry)
                selected = min(row["routes"], key=lambda e: e["expansion"]["median_ms"])["route"]
                row["selected_route"] = selected
                fn = lambda: run(builds[selected], p, s, z, ws, w8)
                _, graph = benchmark(fn, repetitions=3, calls_per_replay=4)
                row["graph"] = graph_check(graph, p, s, z, ws, w8, expected)
                row["W8_reconstruction_loss_vs_original_W4_half"] = error(w8.float() * ws.float()[:, None], b)
                report["real_weights"].append(row); write_json(out / "results.json", report)
                print({"layer": layer, "case": case, "route": selected, "expansion_ms": min(e["expansion"]["median_ms"] for e in row["routes"])}, flush=True)
                if case == "down" and not args.quick: full_chain(p, s, z, ws, w8, b, builds[selected], layer, out, report)
                del p, s, z, b, ws, expected, w8, graph
                gc.collect()
        report["status"] = "passed"; report["peak_cuda_allocated_bytes_reference_inclusive"] = torch.cuda.max_memory_allocated()
    except Exception as exc:
        report["status"] = "failed"; report["failures"].append({"exception": repr(exc), "traceback": traceback.format_exc()})
        raise
    finally: write_json(out / "results.json", report)


if __name__ == "__main__": main()
