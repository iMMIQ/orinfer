"""Offline exact A8 validation, complete SwiGLU->A8 timing and SM87 export."""
import argparse
import gc
import json
import math
import re
import shutil
import time
from pathlib import Path

import torch

from common import (benchmark, configure, environment, error, export_kernel,
                    identity, tensor_sha, write_json)
from abi import parse_host
from kernels.operators.op04_swiglu import launch as launch_swiglu, swiglu
from kernels.operators.op30_activation_quantization import (
    activation_quantization, launch, swiglu_activation_quantization)

ROOT = Path(__file__).resolve().parents[2]
ROWS = [1, 2, 3, 4, 5, 7, 8, 511, 512, 513, 2048, 8192]


def reference(x, group, mask=None):
    """Normative FP32 arithmetic, FP16 scale, nearest-even integer codes."""
    k = x.shape[1]
    values = x.float()
    if mask is not None:
        values = torch.where(mask[None] != 0, 0., values)
    groups = math.ceil(k / group)
    padded = torch.nn.functional.pad(values, (0, groups * group - k))
    amax = padded.reshape(len(x), groups, group).abs().amax(-1)
    scale = torch.where(amax > 0, (amax / 127).clamp_min(2**-24), 1.).half()
    codes = (padded.reshape(len(x), groups, group) / scale.float()[..., None]).round().clamp(-127, 127).to(torch.int8)
    return codes.reshape(len(x), -1)[:, :k].contiguous(), scale


def exact(q, s, expected):
    qe, se = expected
    result = {"code_mismatches": int((q != qe).sum()),
              "scale_bit_mismatches": int((s.view(torch.int16) != se.view(torch.int16)).sum()),
              "finite_scale": bool(torch.isfinite(s).all()),
              "minimum_scale": float(s.min())}
    assert result["code_mismatches"] == result["scale_bit_mismatches"] == 0, result
    assert result["finite_scale"] and result["minimum_scale"] >= 2**-24, result
    return result


def outputs(m, k, group):
    return (torch.empty((m, k), device="cuda", dtype=torch.int8),
            torch.empty((m, math.ceil(k / group)), device="cuda", dtype=torch.float16))


def run_quant(kernel, x, mask, q, s):
    # Capture can change this stream; never cache it outside the run closure.
    launch(kernel, x, mask, q, s, stream=torch.cuda.current_stream().cuda_stream)


def manifest(kernel, directory, metadata, report):
    exported = export_kernel(kernel, directory)
    host = (directory / "host.txt").read_text()
    cuda = (directory / "kernel.cu").read_text()
    entry = re.search(r'extern "C" __global__ void (\w+)\(([^;]+)\);', cuda)
    assert entry
    actual = [a.strip() for a in entry.group(2).split(",")]
    info = {"operator": "op30_activation_quantization", **metadata,
            "sm": 87, "toolchain": report["environment"], "exports": exported,
            "entry_symbol": entry.group(1),
            "ordered_arguments": [{"index": i, "declaration": arg,
                                   "driver_type": "device_ptr:u64" if "*" in arg else "int32"}
                                  for i, arg in enumerate(actual)],
            "host_wrapper": host, "argument_order_source": "actual generated CUDA/host",
            "actual_host_launches": parse_host(host),
            "cooperative": False, "workspace_bytes": 0, "resident_weight_bytes": 0,
            "aliasing": "all input/output buffers disjoint, stable addresses in graph",
            "dynamic_M": True, "shared_memory_source": "host_wrapper dynamic_smem_bytes actual generated value"}
    write_json(directory / "abi.json", info)
    return info


def checked_case(kernel, x, mask, group, masked, report, label):
    q, s = outputs(len(x), x.shape[1], group)
    run = lambda: run_quant(kernel, x, mask, q, s)
    expected = reference(x, group, mask if masked else None)
    started = time.perf_counter()
    run()
    torch.cuda.synchronize()
    case = {"label": label, "M": len(x), "K": x.shape[1], "group": group,
            "masked": masked, "first_use_ms": (time.perf_counter() - started) * 1000,
            "exact": exact(q, s, expected)}
    timing, graph = benchmark(run, repetitions=5 if len(x) <= 513 else 2, calls_per_replay=16)
    case["timing"] = timing
    original = x.clone()
    x.mul_(-.75)
    q.fill_(-128)
    s.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    case["graph_changed"] = exact(q, s, reference(x, group, mask if masked else None))
    x.copy_(original)
    q.fill_(-128)
    s.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    case["graph_restored"] = exact(q, s, expected)
    scales = s.float().repeat_interleave(group, dim=1)[:, :x.shape[1]]
    reconstructed = q.float() * scales
    target = torch.where(mask[None] != 0, 0., x.float()) if masked else x.float()
    case["activation_quantization_loss"] = error(reconstructed, target)
    case["logical_io_bytes"] = x.numel() * 3 + s.numel() * 2 + (mask.numel() if masked else 0)
    case["workspace_bytes"] = 0
    report["cases"].append(case)
    del graph, original, q, s, expected, scales, reconstructed, target
    gc.collect()
    return case


def captures(kind, layer, report):
    folder = ROOT / "artifacts/experimental-vllm/activations"
    records = []
    for p in sorted(folder.glob("*.json")):
        meta = json.loads(p.read_text())
        if f"layers.{layer}." in meta["kind"] and meta["kind"].endswith(kind):
            path = folder / meta["file"]
            t = torch.load(path, map_location="cpu", weights_only=True).half()
            source = {"metadata": identity(p), "file": identity(path),
                      "tensor_sha256": tensor_sha(t), "mode": meta["mode"],
                      "computed_tokens_before": meta["computed_tokens_before"],
                      "origin": meta["activation_origin"], "layer": layer, "kind": kind}
            assert source["file"]["sha256"] == meta["file_sha256"]
            assert source["tensor_sha256"] == meta["tensor_sha256"]
            report["input_sources"].append(source)
            records.append((meta, t))
    pref = next(t for meta, t in records if meta["mode"] == "prefill" and len(t) == 512)
    dec = sorted([(meta, t) for meta, t in records if meta["mode"] == "decode"],
                 key=lambda pair: pair[0]["computed_tokens_before"])
    return pref, torch.cat([t for _, t in dec[:8]])


def fused_case(separate, fused, sw, x, mask, group, masked, report, origin):
    m, k = len(x), x.shape[1] // 2
    assert x.shape[1] == 34816 and k == 17408, "op04 fixed hidden shape; capture gate_up input is NOT its output"
    assert bool(torch.isfinite(x).all()), "finite input contract"
    y = torch.empty((m, k), dtype=torch.float16, device="cuda")
    q, s = outputs(m, k, group)
    fq, fs = outputs(m, k, group)
    def independent():
        launch_swiglu(sw, x, y, stream=torch.cuda.current_stream().cuda_stream)
        run_quant(separate, y, mask, q, s)
    def fusion():
        run_quant(fused, x, mask, fq, fs)
    started = time.perf_counter()
    independent()
    fusion()
    torch.cuda.synchronize()
    if not torch.equal(fq, q) or not torch.equal(fs, s):
        diagnostic = {"input": x.cpu(), "op04_output": y.cpu(), "q": q.cpu(),
                      "s": s.cpu(), "fq": fq.cpu(), "fs": fs.cpu(),
                      "mappings": {"op04": str(sw.adapter.kernels),
                                   "a8": str(separate.adapter.kernels),
                                   "fused": str(fused.adapter.kernels)}}
        path = Path(report["environment"]["output_root"]) / "fusion-failure.pt"
        torch.save(diagnostic, path)
        print("FUSION_DIAGNOSTIC", diagnostic["mappings"],
              "scales", s.cpu().float().tolist(), fs.cpu().float().tolist(),
              "op04_finite", bool(torch.isfinite(y).all()), flush=True)
    row = {"M": m, "K": k, "group": group, "masked": masked, "origin": origin,
           "first_pair_use_ms": 1000 * (time.perf_counter() - started),
           "same_op04_fp16_codes_scale": exact(fq, fs, (q, s)),
           "independent_normative": exact(q, s, reference(y, group, mask if masked else None))}
    row["independent_complete_chain"], graph = benchmark(independent, repetitions=3, calls_per_replay=16)
    row["fused_complete_chain"], fg = benchmark(fusion, repetitions=3, calls_per_replay=16)
    original = x.clone()
    for tag, changed in (("changed", True), ("restored", False)):
        x.copy_(original * -.5 if changed else original)
        y.fill_(float("nan"))
        q.fill_(-128)
        fq.fill_(-128)
        s.fill_(float("nan"))
        fs.fill_(float("nan"))
        graph.replay()
        fg.replay()
        torch.cuda.synchronize()
        row[f"graph_{tag}_fusion_exact"] = exact(fq, fs, (q, s))
        row[f"graph_{tag}_normative"] = exact(q, s, reference(y, group, mask if masked else None))
    scale_bytes = s.numel() * 2
    row["logical_bytes"] = {"independent_read_write": m * k * 9 + scale_bytes,
                            "fused_read_write": m * k * 5 + scale_bytes,
                            "avoided_fp16_materialization": m * k * 4,
                            "independent_intermediate_workspace": m * k * 2,
                            "fused_workspace": 0, "mask_read_optional": k if masked else 0}
    row["chain_scope"] = "entire op04 SwiGLU + A8, no GEMM or W4 expansion; not complete down/model"
    row["speedup"] = row["independent_complete_chain"]["median_ms"] / row["fused_complete_chain"]["median_ms"]
    report["fusion_cases"].append(row)
    del graph, fg, original, y, q, s, fq, fs
    gc.collect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    configure()
    report = {"environment": environment(), "cases": [], "fusion_cases": [],
              "input_sources": [], "exports": [], "compile": [], "edge_cases": [],
              "status": "running", "weight_checkpoint_sha256": "15c5b07049149c73236254d53eca1d2f3274f9fb6803540ca47b1ce657dcf583",
              "weight_identity_note": "reuse locked prior full-file hash; no weight tensor used by this operator",
              "quality_policy": "A8 exactness is not model quality; formal M512 native threshold unchanged; no model started",
              "budget": "conditional op30 charged inside SwiGLU/linear path; no independent allocation or double counting"}
    freeze = out / "measurement-source"
    source_paths = ['kernels/operators/op04_swiglu.py', 'kernels/operators/op30_activation_quantization.py', 'tools/operators/op30_activation_quantization.py', 'tools/operators/common.py', 'tools/operators/abi.py']
    report["sources"] = []
    for path in source_paths:
        destination = freeze / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / path, destination)
        report["sources"].append(identity(destination))
    def save():
        write_json(out / "progress.json", report)
    cache = {}
    def build(k, group, masked=False, fused=False):
        key = k, group, masked, fused
        if key not in cache:
            started = time.perf_counter()
            fn = swiglu_activation_quantization if fused else activation_quantization
            kernel = fn(k, group, masked=masked)
            cache[key] = kernel
            tag = f"{'fused' if fused else 'a8'}-k{k}-g{group}-mask{int(masked)}"
            report["compile"].append({"variant": tag, "prepare_s": time.perf_counter() - started})
            report["exports"].append(manifest(kernel, out / tag,
                {"K": k, "group": group, "masked": masked, "fused": fused,
                 "grid": ["M", math.ceil(k / group), 1], "block": [256, 1, 1],
                 "input_shape": ["M", k * (2 if fused else 1)], "input_dtype": "float16",
                 "mask_shape": [k], "mask_dtype": "uint8", "code_shape": ["M", k],
                 "code_dtype": "int8", "scale_shape": ["M", math.ceil(k / group)],
                 "scale_dtype": "float16", "layout": "contiguous row-major; fused split [gate|up]"}, report))
            save()
        return cache[key]
    for k in (5120, 6144, 17408):
        mask = torch.zeros(k, device="cuda", dtype=torch.uint8)
        for group in (k, 512):
            kernel = build(k, group)
            for m in ([1, 7, 512] if args.quick else ROWS):
                x = torch.randn((m, k), device="cuda", dtype=torch.float16)
                checked_case(kernel, x, mask, group, False, report, "synthetic FP16 real model dimensions")
                del x
                save()
        del mask
    # Tail groups, exact integer half ties, subnormal scales and maximum finite.
    for k, group, masked in ((513, 128, False), (513, 128, True), (17408, 17408, False)):
        mask = torch.zeros(k, device="cuda", dtype=torch.uint8)
        x = torch.zeros((7, k), device="cuda", dtype=torch.float16)
        x[1, :8] = torch.tensor([127, .5, 1.5, 2.5, -.5, -1.5, -2.5, -127], device="cuda").half()
        x[2].fill_(2**-24)
        x[3, :4] = torch.tensor([65504, -65504, 65504-32, -65504+32], device="cuda").half()
        x[4, 0] = 2**-14
        x[5, -1] = 2**-24
        x[6, :8] = x[1, :8]
        if masked:
            mask[:32] = 1
            mask[-1] = 1
        kernel = build(k, group, masked)
        q, s = outputs(7, k, group)
        run_quant(kernel, x, mask, q, s)
        cpu = reference(x.cpu(), group, mask.cpu() if masked else None)
        report["edge_cases"].append({"K": k, "group": group, "masked": masked,
                                     "cpu_normative_exact": exact(q, s, tuple(t.cuda() for t in cpu)),
                                     "ties_first8_codes": q[1, :8].cpu().tolist(),
                                     "scales": s.cpu().float().tolist()})
        checked_case(kernel, x, mask, group, masked, report, "zero/ties/subnormal/finite extrema/group-tail")
        del x, mask, q, s
        save()
    for layer in (0, 32):
        pref, dec = captures("mlp.down_proj", layer, report)
        mask = torch.zeros(17408, device="cuda", dtype=torch.uint8)
        mask[pref[:256].abs().amax(0).topk(32).indices.cuda()] = 1
        for group, masked in ((17408, False), (512, False), (17408, True)):
            kernel = build(17408, group, masked)
            for m in ([1, 512] if args.quick else ROWS):
                if m <= 8:
                    cpu = dec[:m]
                    origin = "consecutive real M1 decode rows; not concurrent batch"
                elif m <= 512:
                    cpu = pref[:m]
                    origin = "real prefill512 rows/subset"
                elif m == 513:
                    cpu = torch.cat((pref, dec[:1]))
                    origin = "real512 + first decode row tail"
                else:
                    cpu = pref.repeat((m // 512, 1))
                    origin = "repeated real512 data at actual 2K/8K shape; not true long-context trace"
                x = cpu.cuda()
                case = checked_case(kernel, x, mask, group, masked, report, f"layer{layer}: {origin}")
                case["input_tensor_sha256"] = tensor_sha(cpu)
                if masked:
                    case["mask_sha256"] = tensor_sha(mask)
                    case["mask_policy"] = "calibration prefill rows0:256 absmax top32; no compensation"
                del x
                save()
        del pref, dec, mask
    started = time.perf_counter()
    sw = swiglu(1024, 256, "split")
    report["op04_prepare_s"] = time.perf_counter() - started
    report["op04_export"] = export_kernel(sw, out / "independent-op04")
    fusion_mask = torch.zeros(17408, device="cuda", dtype=torch.uint8)
    for group in (17408, 512):
        separate, fused = build(17408, group), build(17408, group, False, True)
        edge = torch.zeros((7, 34816), device="cuda", dtype=torch.float16)
        g = torch.tensor([-65504., -1000., -20., -1., 0., 1., 20., 65504.], device="cuda", dtype=torch.float16)
        u = torch.tensor([1., -1., 100., -3., 1., -3., 10., .5], device="cuda", dtype=torch.float16)
        edge[1:, :17408] = g.repeat(17408 // 8)
        edge[1:, 17408:] = u.repeat(17408 // 8)
        edge[2, 17408:].fill_(2**-24)
        edge[3, :17408].fill_(20.)
        edge[3, 17408:].fill_(2**-24)
        fused_case(separate, fused, sw, edge, fusion_mask, group, False, report,
                   "synthetic safe extrema, zero group, tiny finite and half output materialization")
        random = torch.randn((7, 34816), device="cuda", dtype=torch.float16)
        fused_case(separate, fused, sw, random, fusion_mask, group, False, report,
                   "seeded random FP16 gates/up; exact independent op04->A8")
        del edge, random
        save()
    del fusion_mask
    for layer in (0, 32):
        pref, dec = captures("mlp.down_proj", layer, report)
        mask = torch.zeros(17408, device="cuda", dtype=torch.uint8)
        mask[:32] = 1
        for group, masked in ((17408, False), (512, False), (17408, True)):
            separate, fused = build(17408, group, masked), build(17408, group, masked, True)
            for m in ([1, 512] if args.quick else ROWS):
                if m <= 8:
                    cpu, origin = dec[:m], "consecutive real down decode rows"
                elif m <= 512:
                    cpu, origin = pref[:m], "real512 down prefill subset"
                elif m == 513:
                    cpu, origin = torch.cat((pref, dec[:1])), "real512 down + decode tail"
                else:
                    cpu, origin = pref.repeat((m // 512, 1)), "repeated real512 down at actual 2K/8K shape"
                # Captures of gate_up_proj are its INPUT hidden5120, not output
                # split gate/up34816. Construct a disclosed finite fusion test
                # from real down values rather than mislabel a captured tensor.
                gate = torch.tensor([-1., 1., 3., 20.], dtype=torch.float16).repeat(17408 // 4)
                up = (cpu.float() / torch.nn.functional.silu(gate.float())).half()
                split = torch.cat((gate.expand(m, -1), up), dim=1).contiguous()
                x = split.cuda()
                fused_case(separate, fused, sw, x, mask, group, masked, report,
                           f"layer{layer}: synthetic split gate[-1,1,3,20], up=half(real_down/silu(gate)); {origin}")
                del x
                save()
        del pref, dec, mask
    report["validation_process_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    report["status"] = "passed"
    write_json(out / "results.json", report)
    print(json.dumps({"status": report["status"], "cases": len(report["cases"]),
                      "fusion_cases": len(report["fusion_cases"]), "exports": len(report["exports"])}))


if __name__ == "__main__":
    main()
