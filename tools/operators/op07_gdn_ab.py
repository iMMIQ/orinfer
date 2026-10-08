"""Compile/validate/time op07 using checkpoint high precision a/b and real hidden."""

import argparse
import json
import shutil
import time
from pathlib import Path
from tools.reference import ACTIVATIONS as REFERENCE_ACTIVATIONS
from tools.reference import CHECKPOINT, SOURCE

import torch
import tilelang.language as T
from safetensors import safe_open

from common import (
    benchmark,
    configure,
    environment,
    error,
    export_kernel,
    identity,
    tensor_sha,
    write_json,
)
from abi import parse_host
from kernels.operators.op07_gdn_ab import gdn_ab_simt, gdn_ab_tensorcore, launch

ROOT = Path(__file__).resolve().parents[2]
MODEL = CHECKPOINT / "model.safetensors"
LOCK = ROOT / "artifacts/reference/reference-lock.json"
ACT = REFERENCE_ACTIVATIONS
NATIVE = SOURCE / "model_executor/layers/mamba/gdn_linear_attn.py"
ROWS = (1, 2, 3, 4, 5, 7, 8, 511, 512, 513, 2048, 8192)


def checked(actual, ref):
    e = error(actual, ref)
    assert e["finite"] and e["relative_l2"] <= 0.002, e
    return e


def load_weights(out):
    started = time.perf_counter()
    lock = json.loads(LOCK.read_text())
    file_id = next(x for x in lock["files"] if x["name"] == "model.safetensors")
    assert MODEL.stat().st_size == file_id["bytes"]
    weights, tensors = {}, []
    with safe_open(str(MODEL), framework="pt", device="cpu") as f:
        layers = sorted(
            {
                int(k.split(".layers.")[1].split(".")[0])
                for k in f.keys()
                if k.endswith("linear_attn.in_proj_a.weight")
            }
        )
        assert len(layers) == 48
        for layer in layers:
            parts = []
            for part in ("a", "b"):
                name = f"model.language_model.layers.{layer}.linear_attn.in_proj_{part}.weight"
                raw = f.get_tensor(name)
                assert raw.dtype == torch.bfloat16 and list(raw.shape) == [48, 5120]
                cast = raw.half()
                tensors.append(
                    {
                        "name": name,
                        "shape": list(raw.shape),
                        "storage_dtype": str(raw.dtype),
                        "storage_sha256": tensor_sha(raw),
                        "fp16_sha256": tensor_sha(cast),
                        "bf16_to_fp16": error(cast, raw.float()),
                        "changed_elements": int((cast.float() != raw.float()).sum()),
                        "range": [float(raw.min()), float(raw.max())],
                    }
                )
                parts.append(raw)
            if layer in (0, 32):
                weights[layer] = torch.cat(parts).contiguous()
    source = NATIVE.read_text()
    assert "b, a = ba.chunk(2, dim=-1)" in source
    assert "in_proj_b and in_proj_a" in source
    shutil.copyfile(NATIVE, out / "native-gdn-source.py")
    shutil.copyfile(LOCK, out / "reference-lock.json")
    return weights, {
        "locked_model_identity": file_id,
        "size_checked": True,
        "full_hash_policy": "Reuse locked full-file SHA; read/hash only 96 actual unquantized a/b tensors",
        "lock": identity(LOCK),
        "native_source": identity(NATIVE),
        "tensors": tensors,
        "native_output_order": "b then a: b,a=ba.chunk(2,dim=-1)",
        "api_output_order": "a then b: W_ab=cat([in_proj_a.weight,in_proj_b.weight],dim=0)",
        "read_hash_cast_s": time.perf_counter() - started,
    }


def input_rows(layer, m):
    mode = "decode" if m <= 8 else "prefill"
    found = []
    for path in ACT.glob("*.json"):
        meta = json.loads(path.read_text())
        if meta["mode"] == mode and meta["kind"].endswith(
            f"layers.{layer}.linear_attn.in_proj_qkvz"
        ):
            found.append((meta["computed_tokens_before"], path, meta))
    found.sort(key=lambda item: item[0])
    if mode == "decode":
        chosen = found[:m]
        assert len(chosen) == m
    else:
        chosen = next(([item] for item in found if item[2]["shape"][0] >= m), None)
    if chosen:
        values, sources = [], []
        for _, path, meta in chosen:
            tensor_path = ACT / meta["file"]
            assert identity(tensor_path)["sha256"] == meta["file_sha256"]
            value = torch.load(tensor_path, map_location="cpu", weights_only=True)
            assert tensor_sha(value) == meta["tensor_sha256"]
            values.append(value)
            sources.append(
                {
                    "metadata": str(path),
                    "file_sha256": meta["file_sha256"],
                    "position": meta["computed_tokens_before"],
                }
            )
        x = torch.cat(values)[:m].contiguous()
        assert list(x.shape) == [m, 5120] and x.dtype == torch.float16
        return x, {
            "origin": "Actual normalized hidden input to same GDN qkvz and ba projections; decode rows stacked, not simultaneous model batch",
            "sources": sources,
        }
    return torch.randn((m, 5120), dtype=torch.float16), {
        "origin": "Synthetic seeded hidden; no recorded prefill at this length"
    }


def gate_diagnostics(actual, ref, al, dt):
    a, b = actual.float().chunk(2, dim=-1)
    ra, rb = ref.chunk(2, dim=-1)
    return {
        "g": error(
            -al.exp() * torch.nn.functional.softplus(a + dt),
            -al.exp() * torch.nn.functional.softplus(ra + dt),
        ),
        "beta": error(torch.sigmoid(b), torch.sigmoid(rb)),
        "scope": "Diagnostic FP32 gate reference applied offline; op09 TileLang execution not included",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--outputs-only",
        action="store_true",
        help="Supplement FP32/mixed output semantics without repeating main suite",
    )
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    configure()
    raw_weights, weight_identity = load_weights(out)
    report = {
        "environment": environment(),
        "source": identity(ROOT / "kernels/operators/op07_gdn_ab.py"),
        "weights": weight_identity,
        "tuning": [],
        "cases": [],
        "boundary_cases": [],
        "failures": [],
        "workspace_bytes": 0,
        "resident_weight_bytes_per_layer": 983040,
        "input_output_dtypes": ["float16", "bfloat16"],
        "accumulation_dtype": "float32",
        "rounding": "Input and weight storage dtype -> FP32 accumulation -> one output cast; no bias/gate in timed path",
        "budget": {
            "decode_M1_ms": 0.010,
            "prefill512_ms": 0.040,
            "prefill2048_ms": 0.160,
            "prefill8192_ms": 0.640,
        },
    }
    if args.outputs_only:
        report["scope"] = (
            "Additional FP32 and BF16-input/FP16-output validation; main suite remains run02"
        )
        with safe_open(str(MODEL), framework="pt", device="cpu") as f:
            gate = tuple(
                f.get_tensor(f"model.language_model.layers.0.linear_attn.{suffix}").float().cuda()
                for suffix in ("A_log", "dt_bias")
            )
        for dtype, odtype in (
            ("float16", "float32"),
            ("bfloat16", "float32"),
            ("bfloat16", "float16"),
        ):
            w = raw_weights[0].to(device="cuda", dtype=getattr(torch, dtype))
            for name, build, kwargs in (
                ("simt256", gdn_ab_simt, {"threads": 256}),
                ("tc32x64x128", gdn_ab_tensorcore, {"BM": 32, "BN": 64, "BK": 128, "stages": 2}),
            ):
                started = time.perf_counter()
                kernel = build(T.dynamic("M"), dtype=dtype, output_dtype=odtype, **kwargs)
                prepare_s = time.perf_counter() - started
                dest = out / f"{dtype}-to-{odtype}-{name}"
                exported = export_kernel(kernel, dest)
                write_json(
                    dest / "abi.json",
                    {
                        "operator": "op07_gdn_ab",
                        "sm": 87,
                        "input_weight_dtype": dtype,
                        "output_dtype": odtype,
                        "shape": ["M", 96, 5120],
                        "parameters": kwargs,
                        "layout": "contiguous X[M,K], W_ab[N,K], Y[M,N]; a first 48, b second 48",
                        "actual_launches": parse_host((dest / "host.txt").read_text()),
                        "artifacts": exported,
                        "workspace_bytes": 0,
                        "cooperative": False,
                        "toolchain": report["environment"],
                        "source": report["source"],
                    },
                )
                for m in (1, 513):
                    cpu, origin = input_rows(0, m)
                    x = cpu.to(device="cuda", dtype=getattr(torch, dtype))
                    saved = x.clone()
                    y = torch.empty((m, 96), device="cuda", dtype=getattr(torch, odtype))

                    def run():
                        return launch(kernel, x, w, y)

                    started = time.perf_counter()
                    run()
                    torch.cuda.synchronize()
                    first_ms = 1000 * (time.perf_counter() - started)
                    ref = x.float() @ w.float().T
                    e = checked(y, ref)
                    timing, graph = benchmark(
                        run, repetitions=5, calls_per_replay=16 if m == 1 else 1
                    )
                    x.mul_(0.75).add_(0.125)
                    y.fill_(float("nan"))
                    graph.replay()
                    torch.cuda.synchronize()
                    changed = checked(y, x.float() @ w.float().T)
                    x.copy_(saved)
                    y.fill_(float("nan"))
                    graph.replay()
                    torch.cuda.synchronize()
                    restored = checked(y, ref)
                    report["cases"].append(
                        {
                            "dtype": dtype,
                            "output_dtype": odtype,
                            "implementation": name,
                            "M": m,
                            "prepare_s": prepare_s,
                            "first_use_ms": first_ms,
                            "input": origin,
                            "error": e,
                            "timing": timing,
                            "gates": gate_diagnostics(y, ref, *gate),
                            "graph_changed": changed,
                            "graph_restored": restored,
                        }
                    )
                    del graph
                    write_json(out / "progress.json", report)
        report["status"] = "passed"
        write_json(out / "results.json", report)
        print(
            json.dumps({"status": "passed", "supplemental_cases": len(report["cases"])}), flush=True
        )
        return
    kernels, selected = {}, {}
    options = {
        "simt256": ("simt", {"threads": 256}),
        "tc16x64x64": ("tc", {"BM": 16, "BN": 64, "BK": 64, "stages": 2}),
        "tc32x64x128": ("tc", {"BM": 32, "BN": 64, "BK": 128, "stages": 2}),
    }
    for dtype in ("float16", "bfloat16"):
        td = getattr(torch, dtype)
        w = raw_weights[0].to(device="cuda", dtype=td)
        for name, (kind, kwargs) in options.items():
            started = time.perf_counter()
            build = gdn_ab_simt if kind == "simt" else gdn_ab_tensorcore
            kernel = build(T.dynamic("M"), dtype=dtype, output_dtype=dtype, **kwargs)
            kernels[dtype, name] = kernel
            trial = {
                "dtype": dtype,
                "implementation": name,
                "parameters": kwargs,
                "prepare_s": time.perf_counter() - started,
                "shapes": {},
            }
            for m in (1,) if kind == "simt" else (1, 512):
                cpu, origin = input_rows(0, m)
                x = cpu.to(device="cuda", dtype=td)
                y = torch.empty((m, 96), device="cuda", dtype=td)

                def run():
                    return launch(kernel, x, w, y)

                started = time.perf_counter()
                run()
                torch.cuda.synchronize()
                first_ms = 1000 * (time.perf_counter() - started)
                ref = x.float() @ w.float().T
                trial["shapes"][str(m)] = {
                    "first_use_ms": first_ms,
                    "error": checked(y, ref),
                    "timing": benchmark(run, repetitions=10, calls_per_replay=16)[0],
                }
            report["tuning"].append(trial)
            print(json.dumps({"tuning": trial}), flush=True)
            write_json(out / "progress.json", report)
        for mode, m in (("decode", 1), ("prefill", 512)):
            best = min(
                (
                    item
                    for item in report["tuning"]
                    if item["dtype"] == dtype and str(m) in item["shapes"]
                ),
                key=lambda item: item["shapes"][str(m)]["timing"]["median_ms"],
            )
            selected[dtype, mode] = best["implementation"]
    report["selected"] = {"/".join(k): v for k, v in selected.items()}
    # Actual ABI/export for every candidate; stable address buffers, explicit stream.
    for (dtype, name), kernel in kernels.items():
        dest = out / f"{dtype}-{name}"
        exported = export_kernel(kernel, dest)
        host = (dest / "host.txt").read_text()
        write_json(
            dest / "abi.json",
            {
                "operator": "op07_gdn_ab",
                "dtype": dtype,
                "sm": 87,
                "logical_api": ["X[M,5120]", "W_ab[96,5120]", "Y[M,96]"],
                "layout": "contiguous row-major; a first 48, b second 48",
                "actual_launches": parse_host(host),
                "parameters": options[name][1],
                "workspace_bytes": 0,
                "resident_weight_bytes": 983040,
                "cooperative": False,
                "aliasing": "X/W/Y disjoint; graph requires stable addresses and fixed shape",
                "artifacts": exported,
                "toolchain": report["environment"],
                "source": report["source"],
            },
        )
    with safe_open(str(MODEL), framework="pt", device="cpu") as f:
        gates = {
            layer: tuple(
                f.get_tensor(f"model.language_model.layers.{layer}.linear_attn.{suffix}")
                .float()
                .cuda()
                for suffix in ("A_log", "dt_bias")
            )
            for layer in (0, 32)
        }
    for dtype in ("float16", "bfloat16"):
        td = getattr(torch, dtype)
        for layer in (0, 32):
            w = raw_weights[layer].to(device="cuda", dtype=td)
            for m in ROWS:
                cpu, origin = input_rows(layer, m)
                started = time.perf_counter()
                x = cpu.to(device="cuda", dtype=td)
                saved = x.clone()
                y = torch.empty((m, 96), device="cuda", dtype=td)
                torch.cuda.synchronize()
                allocation_s = time.perf_counter() - started
                name = selected[dtype, "decode" if m <= 8 else "prefill"]
                kernel = kernels[dtype, name]

                def run():
                    return launch(kernel, x, w, y)

                run()
                torch.cuda.synchronize()
                ref = x.float() @ w.float().T
                e = checked(y, ref)
                original = y.clone()
                timing, graph = benchmark(
                    run, repetitions=5 if m > 513 else 10, calls_per_replay=1 if m > 513 else 16
                )
                x.mul_(0.75).add_(0.125)
                y.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                changed = checked(y, x.float() @ w.float().T)
                assert not torch.equal(y, original)
                x.copy_(saved)
                y.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                restored = checked(y, ref)
                assert torch.equal(y, original)
                case = {
                    "dtype": dtype,
                    "layer": layer,
                    "M": m,
                    "N": 96,
                    "K": 5120,
                    "implementation": name,
                    "input": origin,
                    "input_sha256": tensor_sha(x),
                    "error": e,
                    "per_half": {
                        "a": checked(y[:, :48], ref[:, :48]),
                        "b": checked(y[:, 48:], ref[:, 48:]),
                    },
                    "gates": gate_diagnostics(y, ref, *gates[layer]),
                    "timing": timing,
                    "allocation_and_input_upload_s": allocation_s,
                    "graph_changed": changed,
                    "graph_restored": restored,
                    "input_bytes": x.numel() * x.element_size(),
                    "output_bytes": y.numel() * y.element_size(),
                    "weight_bytes": w.numel() * w.element_size(),
                    "full_path": "one TileLang linear kernel; no runtime preparation or intermediate buffer",
                }
                report["cases"].append(case)
                print(
                    json.dumps(
                        {
                            "case": {
                                "dtype": dtype,
                                "layer": layer,
                                "M": m,
                                "ms": timing["median_ms"],
                                "l2": e["relative_l2"],
                            }
                        }
                    ),
                    flush=True,
                )
                del graph
                write_json(out / "progress.json", report)
        # Zero, NaN, infinity and tail rows; assert IEEE propagation, not finite tolerance.
        for name in {selected[dtype, "decode"], selected[dtype, "prefill"]}:
            kernel = kernels[dtype, name]
            x = torch.zeros((3, 5120), device="cuda", dtype=td)
            y = torch.empty((3, 96), device="cuda", dtype=td)
            w = raw_weights[0].to(device="cuda", dtype=td)
            launch(kernel, x, w, y)
            torch.cuda.synchronize()
            assert bool((y == 0).all())
            x[1, 0] = float("nan")
            x[2, 1] = float("inf")
            launch(kernel, x, w, y)
            torch.cuda.synchronize()
            ref = x.float() @ w.float().T
            assert torch.equal(torch.isnan(y), torch.isnan(ref))
            assert torch.equal(torch.isposinf(y), torch.isposinf(ref))
            assert torch.equal(torch.isneginf(y), torch.isneginf(ref))
            assert bool((y[0] == 0).all())
            report["boundary_cases"].append(
                {
                    "dtype": dtype,
                    "implementation": name,
                    "zero_exact": True,
                    "nan_inf_classification_matches_FP32_reference": True,
                    "M": 3,
                }
            )
    report["peak_torch_validation_allocated_bytes"] = torch.cuda.max_memory_allocated()
    report["status"] = "passed"
    write_json(out / "results.json", report)
    print(
        json.dumps(
            {"status": "passed", "cases": len(report["cases"]), "selected": report["selected"]}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
