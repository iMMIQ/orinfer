"""Offline op23 correctness, gather exactness, graph, ABI and latency checks."""

import argparse
import gc
import json
import re
import time
from pathlib import Path
from tools.reference import ACTIVATIONS as REFERENCE_ACTIVATIONS
from tools.reference import CHECKPOINT

import torch
from safetensors import safe_open

from common import (
    ROOT,
    benchmark,
    configure,
    environment,
    error,
    export_kernel,
    identity,
    tensor_sha,
    write_json,
)

# Reuse the explicitly source-grounded op02 reference, not its kernel/benchmark.
from tools.operators.op02_residual_norm import source_reference, SOURCE
from kernels.operators.op23_final_norm import (
    final_norm,
    final_norm_presummed,
    last_hidden_gather,
    validate_last_indices,
)

MODEL = CHECKPOINT
ACTIVATIONS = REFERENCE_ACTIVATIONS
BATCHES = (1, 2, 3, 4, 5, 7, 8)
MODES = ("residual_fp32", "presummed_fp32", "no_residual_fp16")


def binding(output):
    started = time.perf_counter()
    lock = json.loads((ROOT / "artifacts/reference/reference-lock.json").read_text())
    checkpoint = next(f for f in lock["files"] if f["name"] == "model.safetensors")
    assert checkpoint["bytes"] == (MODEL / "model.safetensors").stat().st_size
    config = json.loads((MODEL / "config.json").read_text())["text_config"]
    assert config["hidden_size"] == 5120 and config["rms_norm_eps"] == 1e-6
    name = "model.language_model.norm.weight"
    with safe_open(str(MODEL / "model.safetensors"), framework="pt", device="cpu") as f:
        raw = f.get_tensor(name)
    weight = raw.half().contiguous()
    assert list(raw.shape) == [5120] and torch.equal(raw, weight.to(raw.dtype))
    info = {
        "checkpoint": dict(
            checkpoint,
            path=str(MODEL / "model.safetensors"),
            identity_policy="reuse locked full-file SHA256; no repeated 18GB scan",
        ),
        "reference_lock": identity(ROOT / "artifacts/reference/reference-lock.json"),
        "config": identity(MODEL / "config.json"),
        "reference_sources": [identity(p) for p in sorted(SOURCE.rglob("*.py"))],
        "reference_function": "op02 source_reference executing frozen GemmaRMSNorm.forward_cuda/native",
        "reference_helper": identity(ROOT / "tools/operators/op02_residual_norm.py"),
        "weight": {
            "name": name,
            "shape": list(raw.shape),
            "source_dtype": str(raw.dtype),
            "source_tensor_sha256": tensor_sha(raw),
            "runtime_dtype": "float16",
            "runtime_tensor_sha256": tensor_sha(weight),
            "conversion": "exact BF16 to FP16",
        },
        "epsilon": config["rms_norm_eps"],
        "hidden_size": config["hidden_size"],
    }
    captured, sample_info = [], []
    for position in range(512, 520):
        path = ACTIVATIONS / f"capture-512-{position}-lm_head.pt"
        meta_path = path.with_suffix(".json")
        if not path.exists() or not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        sample = torch.load(path, map_location="cpu", weights_only=True)
        assert tuple(sample.shape) == (1, 5120) and sample.dtype == torch.float16
        assert tensor_sha(sample) == meta["tensor_sha256"]
        assert identity(path)["sha256"] == meta["file_sha256"]
        captured.append(sample)
        sample_info.append(
            {
                "file": identity(path),
                "metadata": identity(meta_path),
                "position": position,
                "tensor_sha256": tensor_sha(sample),
            }
        )
    info["real_hidden_samples"] = {
        "available_rows": len(captured),
        "samples": sample_info,
        "origin": "LM-head input after native final norm; NOT pre-final-norm hidden/residual",
        "batch_semantics": "stack of consecutive real M1 decode activations; not simultaneous requests",
    }
    gpu_weight = weight.cuda()
    gpu_hidden = torch.cat(captured).cuda() if captured else None
    torch.cuda.synchronize()
    info["load_and_gpu_transfer_s"] = time.perf_counter() - started
    write_json(output / "binding.json", info)
    return gpu_weight, gpu_hidden, info


def export(kernel, directory, mode, threads=256):
    exported = export_kernel(kernel, directory)
    cuda = (directory / "kernel.cu").read_text()
    host = (directory / "host.txt").read_text()
    entries = [
        {"symbol": name, "parameters_verbatim": args}
        for name, args in re.findall(r"__global__\s+void\s+(\w+)\s*\(([^)]*)\)", cuda)
        if name in exported["symbols"]
    ]
    assert entries, "actual CUDA entries must be recorded"
    shared = re.search(r"config.sharedMemBytes\s*=\s*(\d+)", host)
    dtypes = {"I": "int32", "W": "float16", "Y": "float16"}
    if mode == "residual_fp32":
        order = ["X", "R", "I", "W", "Y"]
        dtypes.update(X="float16", R="float32")
    elif mode.startswith("gather_"):
        order = ["X", "I", "G"]
        dtypes = {"I": "int32", "X": mode[7:], "G": mode[7:]}
    else:
        order = ["U", "I", "W", "Y"]
        dtypes["U"] = "float32" if mode == "presummed_fp32" else "float16"
    abi = {
        "schema_version": 1,
        "operator": "op23_final_norm",
        "mode": mode,
        "tensor_api_order": order,
        "tensor_dtypes": dtypes,
        "dimensions": {"rows": "runtime int32 M", "batch": "runtime int32 B", "hidden": 5120},
        "layout": "contiguous row-major inputs[M,5120], outputs[B,5120], I[B], W[5120]",
        "actual_cuda_entries": entries,
        "symbols": exported["symbols"],
        "host_arguments_verbatim": [
            l.strip() for l in host.splitlines() if "arg_values =" in l or "arg_types =" in l
        ],
        "host_launch_lines_verbatim": [
            l.strip()
            for l in host.splitlines()
            if any(word in l.lower() for word in ("launch", "grid", "block", "shared"))
        ],
        "grid": ["B", 1, 1],
        "block": [threads, 1, 1],
        "dynamic_shared_memory_bytes": int(shared.group(1)) if shared else 0,
        "static_shared_memory_bytes": 0,
        "workspace_bytes": 0,
        "persistent_weight_bytes": 0 if mode.startswith("gather_") else 10240,
        "cooperative_launch": False,
        "sm": 87,
        "index_contract": "caller validates every int32 I[b] in [0,M) before launch and each change; duplicates legal",
        "finite_contract": "finite inputs and FP32 squared-reduction sum without overflow",
        "alias_contract": "all tensors distinct; read-only inputs and indices; stable graph addresses",
        "stream": "explicit stream in real host wrapper; resolve capture-current stream every invocation",
        "math": "gather bit-exact copy"
        if mode.startswith("gather_")
        else "FP32 u, square/reduction/rsqrt eps1e-6 and (1+FP32(W)); final FP16 store",
        "toolchain": environment(),
        "files": exported["files"],
    }
    write_json(directory / "abi.json", abi)
    return abi


def exact_bits(actual, expected):
    dtype = torch.int16 if actual.dtype == torch.float16 else torch.int32
    return torch.equal(actual.view(dtype), expected.view(dtype))


def arguments(mode, x, r, indices, weight, y):
    return (x, r, indices, weight, y) if mode == "residual_fp32" else (x, indices, weight, y)


def reference(mode, x, r, indices, weight, native):
    selected = x.index_select(0, indices.long())
    residual = r.index_select(0, indices.long()) if mode == "residual_fp32" else None
    return native(selected, residual, weight, 1e-6)[0].half()


def checked_case(mode, kernel, gather, x, r, indices, weight, native, repetitions):
    y = torch.empty((indices.numel(), 5120), device="cuda", dtype=torch.float16)
    call = arguments(mode, x, r, indices, weight, y)

    def run():
        return kernel(*call, stream=torch.cuda.current_stream().cuda_stream)

    begun = time.perf_counter()
    run()
    torch.cuda.synchronize()
    first_s = time.perf_counter() - begun
    expected = reference(mode, x, r, indices, weight, native)
    initial = error(y, expected)
    row_errors = [error(y[i], expected[i]) for i in range(indices.numel())]
    assert all(e["finite"] and e["relative_l2"] <= 0.001 for e in row_errors), row_errors
    # Exact standalone gather proves row indexing independently of norm tolerance.
    g = torch.empty((indices.numel(), 5120), dtype=x.dtype, device=x.device)
    gather(x, indices, g, stream=torch.cuda.current_stream().cuda_stream)
    assert exact_bits(g, x.index_select(0, indices.long()))
    timing, graph = benchmark(run, repetitions=repetitions, calls_per_replay=16)
    saved_x, saved_i = x.clone(), indices.clone()
    saved_r = r.clone() if r is not None else None
    original = y.clone()
    y.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(y, original), "poison replay failed"
    x.mul_(-0.375)
    if r is not None:
        r.mul_(0.25)
    changed_i = ((saved_i.flip(0).long() + 1) % x.shape[0]).int()
    validate_last_indices(changed_i.cpu().tolist(), x.shape[0], indices.numel())
    indices.copy_(changed_i)
    changed_ref = reference(mode, x, r, indices, weight, native)
    y.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    changed_errors = [error(y[i], changed_ref[i]) for i in range(indices.numel())]
    assert all(e["finite"] and e["relative_l2"] <= 0.001 for e in changed_errors)
    if original.abs().max() > 0:
        assert not torch.equal(y, original), "graph did not consume changed inputs"
    x.copy_(saved_x)
    indices.copy_(saved_i)
    if r is not None:
        r.copy_(saved_r)
    y.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(y, original), "restored replay differs"
    return {
        "norm_error": initial,
        "row_errors": row_errors,
        "gather_bit_exact": True,
        "graph": {
            "poison_output": True,
            "changed_inputs": True,
            "changed_indices": not torch.equal(saved_i, changed_i),
            "index_change_possible": x.shape[0] > 1,
            "changed_row_errors": changed_errors,
            "restore_bit_exact": True,
        },
        "first_launch_s": first_s,
        "timing": timing,
        "budget_ms": 0.010,
        "budget_met": timing["median_ms"] <= 0.010,
        "latency_over_budget_ratio": timing["median_ms"] / 0.010,
        "allocated_io_bytes": x.numel() * x.element_size()
        + (0 if r is None else r.numel() * r.element_size())
        + indices.numel() * 4
        + y.numel() * 2,
        "minimum_selected_io_bytes": indices.numel()
        * (5120 * (x.element_size() + (0 if r is None else r.element_size()) + 2) + 4),
        "weight_bytes": 10240,
        "workspace_bytes": 0,
    }


def rejected_cases():
    invalid = [
        ([], 8, None),
        ([-1], 8, None),
        ([8], 8, None),
        ([1.0], 8, None),
        ([True], 8, None),
        ([0], 0, None),
        ([0], 2147483648, None),
        ([0, 1], 8, 1),
        ([0], 8, 0),
        ([0], 8.0, None),
    ]
    results = []
    for indices, rows, batch in invalid:
        try:
            validate_last_indices(indices, rows, batch)
        except ValueError as exc:
            results.append(
                {"indices": indices, "rows": rows, "batch": batch, "rejection": str(exc)}
            )
        else:
            raise AssertionError("invalid indices accepted")
    assert validate_last_indices([7, 0, 7, 3], 8) == (7, 0, 7, 3)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--repetitions", type=int, default=30)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    configure()
    weight, real, bind = binding(output)
    native = source_reference()
    results = {
        "environment": environment(),
        "binding": bind,
        "cases": [],
        "exports": [],
        "invalid_indices_rejected": rejected_cases(),
        "status": "in_progress",
    }
    kernels, gathers = {}, {}
    for mode in MODES:
        started = time.perf_counter()
        kernel = (
            final_norm()
            if mode == "residual_fp32"
            else final_norm_presummed(
                input_dtype="float32" if mode == "presummed_fp32" else "float16"
            )
        )
        prepare_s = time.perf_counter() - started
        kernels[mode] = kernel
        results["exports"].append(
            {
                "mode": mode,
                "prepare_compile_s": prepare_s,
                "abi": export(kernel, output / "aot" / mode, mode),
            }
        )
    for dtype in ("float16", "float32"):
        gathers[dtype] = last_hidden_gather(dtype=dtype)
        results["exports"].append(
            {
                "mode": f"gather_{dtype}",
                "abi": export(
                    gathers[dtype], output / "aot" / f"gather_{dtype}", f"gather_{dtype}"
                ),
            }
        )

    # All B tails on decode M=B and every representative prompt M.
    for mode in MODES:
        dtype = torch.float32 if mode == "presummed_fp32" else torch.float16
        for m, b in [(b, b) for b in BATCHES] + [
            (m, b) for m in (511, 512, 513, 2048, 8192) for b in BATCHES
        ]:
            x = torch.randn((m, 5120), dtype=dtype, device="cuda")
            r = (
                torch.randn((m, 5120), dtype=torch.float32, device="cuda")
                if mode == "residual_fp32"
                else None
            )
            # Ragged request lengths with cumulative last-row indices; reverse
            # request order to ensure kernel does not rely on monotonic offsets.
            ends = [m - 1] if b == 1 else [int((k / (b - 1)) ** 1.4 * (m - 1)) for k in range(b)]
            ends = list(reversed(ends))
            validate_last_indices(ends, m, b)
            indices = torch.tensor(ends, dtype=torch.int32, device="cuda")
            check = checked_case(
                mode,
                kernels[mode],
                gathers[str(dtype).split(".")[-1]],
                x,
                r,
                indices,
                weight,
                native,
                args.repetitions,
            )
            results["cases"].append(
                {
                    "mode": mode,
                    "rows_M": m,
                    "batch_B": b,
                    "last_indices": ends,
                    "origin": "seeded random hidden and optional FP32 residual",
                    **check,
                }
            )
            write_json(output / "results.json", results)
            print(
                f"{mode} M{m} B{b} l2={check['norm_error']['relative_l2']:.3g} ms={check['timing']['median_ms']:.6f}",
                flush=True,
            )
            del x, r, indices
            gc.collect()

    results["boundaries"] = []
    for mode in MODES:
        dtype = torch.float32 if mode == "presummed_fp32" else torch.float16
        x = torch.randn((8, 5120), dtype=dtype, device="cuda")
        x[0].zero_()
        x[1].fill_(2**-24)
        x[2].fill_(65504)
        x[3].zero_()
        x[3, 13] = 65504
        x[4].fill_(0.125)
        x[5].fill_(-0.125)
        x[6].zero_()
        x[6, ::2] = -0.0
        r = (
            torch.randn((8, 5120), dtype=torch.float32, device="cuda")
            if mode == "residual_fp32"
            else None
        )
        if r is not None:
            r[0].zero_()
            r[1].zero_()
            r[2].fill_(1e8)
            r[3].zero_()
            r[4] = -x[4].float() + 1e-6
            r[5].fill_(0.25)
            r[6].zero_()
        indices = torch.tensor([7, 0, 7, 1, 2, 3, 4, 5, 6], dtype=torch.int32, device="cuda")
        validate_last_indices(indices.cpu().tolist(), 8, 9)
        check = checked_case(
            mode,
            kernels[mode],
            gathers[str(dtype).split(".")[-1]],
            x,
            r,
            indices,
            weight,
            native,
            args.repetitions,
        )
        results["boundaries"].append(
            {
                "mode": mode,
                "input_rows": [
                    "zero",
                    "FP16_subnormal",
                    "max_FP16_plus_large_residual",
                    "single_outlier",
                    "constant_or_cancellation",
                    "negative_constant",
                    "signed_zero",
                    "random",
                ],
                "last_indices": indices.cpu().tolist(),
                **check,
            }
        )

    results["real_cases"] = []
    if real is not None:
        for b in (1, 8):
            if real.shape[0] < b:
                continue
            for mode in MODES:
                base = real[:b].clone()
                r = base.float() * 0.75 + 0.0001 if mode == "residual_fp32" else None
                x = (
                    base.float() + (base.float() * 0.75 + 0.0001)
                    if mode == "presummed_fp32"
                    else base
                )
                indices = torch.arange(b - 1, -1, -1, device="cuda", dtype=torch.int32)
                check = checked_case(
                    mode,
                    kernels[mode],
                    gathers[str(x.dtype).split(".")[-1]],
                    x,
                    r,
                    indices,
                    weight,
                    native,
                    args.repetitions,
                )
                results["real_cases"].append(
                    {
                        "mode": mode,
                        "rows_M": b,
                        "batch_B": b,
                        "origin": bind["real_hidden_samples"]["origin"],
                        "residual_origin": "synthetic 0.75*captured_hidden+0.0001; absent for FP16 no-residual",
                        "input_tensor_sha256": tensor_sha(x),
                        **check,
                    }
                )

    # FP32 pre-summed must match the fused residual path on identical u.
    x = torch.randn((513, 5120), dtype=torch.float16, device="cuda")
    r = torch.randn((513, 5120), dtype=torch.float32, device="cuda")
    u = x.float() + r
    indices = torch.tensor([512, 0, 237, 237, 11], device="cuda", dtype=torch.int32)
    ya = torch.empty((5, 5120), device="cuda", dtype=torch.float16)
    yb = torch.empty_like(ya)
    kernels["residual_fp32"](
        x, r, indices, weight, ya, stream=torch.cuda.current_stream().cuda_stream
    )
    kernels["presummed_fp32"](
        u, indices, weight, yb, stream=torch.cuda.current_stream().cuda_stream
    )
    assert exact_bits(ya, yb)
    results["fused_vs_fp32_presummed_bit_exact"] = True
    torch.cuda.synchronize()
    results["memory"] = {
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    results["implementation_identity"] = [
        identity(ROOT / "kernels/operators/op23_final_norm.py"),
        identity(Path(__file__)),
    ]
    results["status"] = (
        "passed numerical, bit-exact gather, index contract and graph checks; standalone operator only"
    )
    write_json(output / "results.json", results)
    print("op23 complete; wrapper will release GPU lock after cleanup", flush=True)


if __name__ == "__main__":
    main()
