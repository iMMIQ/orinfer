"""Standalone op02 validation, real checkpoint binding, AOT export and timing."""

import argparse
import ast
import gc
import json
import re
import time
import types
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
from kernels.operators.op02_residual_norm import first_norm, residual_norm

MODEL = CHECKPOINT
SOURCE = ROOT / "artifacts/operators/op02_residual_norm/reference-source"
ACTIVATIONS = REFERENCE_ACTIVATIONS
ROWS = (1, 2, 3, 4, 5, 7, 8, 511, 512, 513, 2048, 8192)


def source_reference():
    """Execute the exact frozen native methods, without importing vLLM."""
    norm_source = SOURCE / "model_executor/layers/layernorm.py"
    ir_source = SOURCE / "ir/ops/layernorm.py"
    ir_tree = ast.parse(ir_source.read_text())
    ir_node = next(
        n for n in ir_tree.body if isinstance(n, ast.FunctionDef) and n.name == "rms_norm"
    )
    ir_node.decorator_list = []
    namespace = {"torch": torch, "Tensor": torch.Tensor}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[ir_node], type_ignores=[])),
            str(ir_source),
            "exec",
        ),
        namespace,
    )
    norm_tree = ast.parse(norm_source.read_text())
    klass = next(
        n for n in norm_tree.body if isinstance(n, ast.ClassDef) and n.name == "GemmaRMSNorm"
    )
    methods = [
        n
        for n in klass.body
        if isinstance(n, ast.FunctionDef) and n.name in ("forward_native", "forward_cuda")
    ]
    namespace["ir"] = types.SimpleNamespace(
        ops=types.SimpleNamespace(rms_norm=namespace["rms_norm"])
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[])),
            str(norm_source),
            "exec",
        ),
        namespace,
    )

    def reference(x, r, w, eps):
        instance = types.SimpleNamespace(weight=w, variance_epsilon=eps)
        instance.forward_native = types.MethodType(namespace["forward_native"], instance)
        answer = namespace["forward_cuda"](instance, x, r)
        return (answer, x.clone()) if r is None else answer

    return reference


def binding(output):
    started = time.perf_counter()
    lock = json.loads((ROOT / "artifacts/reference/reference-lock.json").read_text())
    checkpoint = next(f for f in lock["files"] if f["name"] == "model.safetensors")
    assert checkpoint["bytes"] == (MODEL / "model.safetensors").stat().st_size
    config = json.loads((MODEL / "config.json").read_text())["text_config"]
    assert config["hidden_size"] == 5120 and config["rms_norm_eps"] == 1e-6
    meta_path = (
        ACTIVATIONS / "capture-512-0-language_model_model_layers_0_linear_attn_in_proj_qkvz.json"
    )
    metadata = json.loads(meta_path.read_text())
    tokens = metadata["prompt_token_ids"]
    weights, identities = {}, []
    with safe_open(str(MODEL / "model.safetensors"), framework="pt", device="cpu") as f:
        for layer in (0, 32):
            for position in ("input_layernorm", "post_attention_layernorm"):
                name = f"model.language_model.layers.{layer}.{position}.weight"
                raw = f.get_tensor(name)
                converted = raw.half().contiguous()
                assert torch.equal(raw, converted.to(raw.dtype))
                weights[f"{layer}_{position}"] = converted.cuda()
                identities.append(
                    {
                        "name": name,
                        "shape": list(raw.shape),
                        "source_dtype": str(raw.dtype),
                        "source_tensor_sha256": tensor_sha(raw),
                        "runtime_dtype": "float16",
                        "runtime_tensor_sha256": tensor_sha(converted),
                        "conversion": "exact BF16 to FP16",
                    }
                )
        embeddings = f.get_slice("model.language_model.embed_tokens.weight")
        unique = sorted(set(tokens))
        selected = {token: embeddings[token : token + 1].clone() for token in unique}
        hidden = torch.cat([selected[token] for token in tokens]).half().contiguous()
        identities.append(
            {
                "name": "model.language_model.embed_tokens.weight",
                "read_scope": "only the unique actual 512-prompt token rows",
                "token_ids": unique,
                "selected_rows_source_sha256": tensor_sha(
                    torch.cat([selected[token] for token in unique])
                ),
                "assembled_runtime_sha256": tensor_sha(hidden),
            }
        )
    source_id = [identity(path) for path in sorted(SOURCE.rglob("*.py"))]
    info = {
        "checkpoint": dict(
            checkpoint,
            path=str(MODEL / "model.safetensors"),
            identity_policy="reuse locked full-file SHA256; do not rehash 18GB",
        ),
        "config": identity(MODEL / "config.json"),
        "reference_lock": identity(ROOT / "artifacts/reference/reference-lock.json"),
        "reference_image": lock["config"]["image"],
        "reference_sources": source_id,
        "weights": identities,
        "hidden": {
            "origin": "actual prompt token embedding rows: genuine first-norm input",
            "prompt_metadata": identity(meta_path),
            "rows": 512,
            "runtime_dtype": "float16",
            "sha256": tensor_sha(hidden),
        },
        "load_slice_and_gpu_transfer_s": time.perf_counter() - started,
        "epsilon": config["rms_norm_eps"],
        "hidden_size": config["hidden_size"],
    }
    write_json(output / "binding.json", info)
    return weights, hidden.cuda(), info


def export(kernel, directory, mode, r_dtype, ro_dtype, rows, threads):
    result = export_kernel(kernel, directory)
    cuda = (directory / "kernel.cu").read_text()
    host = (directory / "host.txt").read_text()
    entries = [
        (name, args)
        for name, args in re.findall(r"__global__\s+void\s+(\w+)\s*\(([^)]*)\)", cuda)
        if name in result["symbols"]
    ]
    shared_match = re.search(r"config.sharedMemBytes\s*=\s*(\d+)", host)
    argument_lines = [
        line.strip()
        for line in host.splitlines()
        if "arg_values =" in line or "arg_types =" in line
    ]
    launch_lines = [
        line.strip()
        for line in host.splitlines()
        if "launch" in line.lower()
        or "grid" in line.lower()
        or "block" in line.lower()
        or "shared" in line.lower()
    ]
    abi = {
        "schema_version": 1,
        "operator": "op02_residual_norm",
        "mode": mode,
        "sm": 87,
        "rows": rows or "runtime rows",
        "hidden": 5120,
        "epsilon": 1e-6,
        "zero_centered": True,
        "tensor_api_order": ["X", "W", "Y", "RO"]
        if mode == "first"
        else ["X", "R", "W", "Y", "RO"],
        "tensor_dtypes": {
            "X": "float16",
            "R": r_dtype,
            "W": "float16",
            "Y": "float16",
            "RO": ro_dtype,
        },
        "layout": "contiguous row-major [rows,5120], W[5120]",
        "actual_cuda_entries": [
            {"symbol": name, "parameters_verbatim": args} for name, args in entries
        ],
        "host_arguments_verbatim": argument_lines,
        "host_launch_lines_verbatim": launch_lines,
        "symbols": result["symbols"],
        "grid": ["rows", 1, 1],
        "block": [threads, 1, 1],
        "dynamic_shared_memory_bytes": int(shared_match.group(1)) if shared_match else None,
        "static_shared_memory_bytes": 0,
        "shared_memory_note": "dynamic reduction scratch emitted by TileLang; no global workspace",
        "cooperative_launch": False,
        "stream": "explicit CUDA stream argument in actual host wrapper",
        "alias_contract": "X, R, Y, RO distinct allocations; W immutable; stable addresses in each captured graph",
        "workspace_bytes": 0,
        "persistent_weight_bytes": 10240,
        "first_layer_residual": "FP16 exact X copy",
        "norm_rounding": "FP32 sum and reduction; multiply (1+FP32(W)); round norm to FP16 at final store",
        "residual_rounding": f"FP32 sum cast to {ro_dtype} at RO store; norm retains unrounded sum",
        "toolchain": environment(),
        "files": result["files"],
    }
    write_json(directory / "abi.json", abi)
    return abi


def checked_graph(kernel, arguments, reference, x, r, w, eps, y, ro, ro_dtype, repetitions):
    # Capture switches to a side stream. Resolve its handle on every invocation.
    def run():
        return kernel(*arguments, stream=torch.cuda.current_stream().cuda_stream)

    begun = time.perf_counter()
    run()
    torch.cuda.synchronize()
    first_s = time.perf_counter() - begun
    ref_y, ref_ro = reference(x, r, w, eps)
    ref_ro = ref_ro.to(ro_dtype)
    initial_error = error(y, ref_y)
    assert initial_error["finite"] and initial_error["relative_l2"] <= 0.001, initial_error
    assert torch.equal(ro, ref_ro), "residual rounding differs from FP32 sum reference"
    # Short single-node graphs can be host-submission bound. Include the
    # original single-node measurement, then measure contiguous graph nodes.
    single_timing, _ = benchmark(run, repetitions=repetitions)
    launches_per_graph = 16 if x.shape[0] <= 8 else 4

    def graph_work():
        for _ in range(launches_per_graph):
            run()

    timing, graph = benchmark(graph_work, repetitions=repetitions)
    timing["whole_graph_median_ms"] = timing["median_ms"]
    timing["median_ms"] /= launches_per_graph
    timing["trials_ms"] = [v / launches_per_graph for v in timing["trials_ms"]]
    timing["launches_per_graph"] = launches_per_graph
    timing["single_node_graph"] = single_timing
    timing["timing_scope"] = (
        "CUDA events; contiguous same-operation graph nodes; per-node division; no state reset required"
    )
    saved_x = x.clone()
    saved_r = r.clone() if r is not None else None
    expected = y.clone()
    y.fill_(float("nan"))
    ro.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(y, expected) and torch.equal(ro, ref_ro)
    x.mul_(-0.375)
    if r is not None:
        r.mul_(0.25)
    changed_y, changed_ro = reference(x, r, w, eps)
    y.fill_(float("nan"))
    ro.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    changed_error = error(y, changed_y)
    assert changed_error["finite"] and changed_error["relative_l2"] <= 0.001
    assert torch.equal(ro, changed_ro.to(ro_dtype))
    assert not torch.equal(y, expected), "graph failed to consume changed input"
    x.copy_(saved_x)
    if r is not None:
        r.copy_(saved_r)
    y.fill_(float("nan"))
    ro.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(y, expected) and torch.equal(ro, ref_ro)
    return {
        "norm_error": initial_error,
        "residual_bit_exact": True,
        "graph": {
            "poison_both_outputs": True,
            "changed_x_and_residual": True,
            "restore_replay_bit_exact": True,
            "changed_norm_error": changed_error,
        },
        "first_launch_s": first_s,
        "timing": timing,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--repetitions", type=int, default=30)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    configure()
    weights, real_hidden, bind = binding(output)
    reference = source_reference()
    w = weights["0_input_layernorm"]
    eps = bind["epsilon"]
    modes = [
        ("first", None, torch.float16),
        ("fp16_to_fp32", torch.float16, torch.float32),
        ("fp32_to_fp32", torch.float32, torch.float32),
        ("fp16_to_fp16", torch.float16, torch.float16),
        ("fp32_to_fp16", torch.float32, torch.float16),
    ]
    results = {
        "environment": environment(),
        "binding": bind,
        "cases": [],
        "exports": [],
        "tuning": [],
        "semantics": "matched frozen forward_cuda/native FP32 residual; FP16 output alternate",
        "status": "in_progress",
    }
    kernels = {}
    # One reusable runtime-row kernel per dtype contract, with all real model rows.
    for mode, r_dtype, ro_dtype in modes:
        started = time.perf_counter()
        kernel = (
            first_norm()
            if r_dtype is None
            else residual_norm(
                residual_dtype=str(r_dtype).split(".")[-1],
                output_residual_dtype=str(ro_dtype).split(".")[-1],
            )
        )
        compile_s = time.perf_counter() - started
        kernels[mode] = kernel
        directory = output / "aot" / mode
        abi = export(
            kernel,
            directory,
            "first" if r_dtype is None else "residual",
            None if r_dtype is None else str(r_dtype).split(".")[-1],
            str(ro_dtype).split(".")[-1],
            None,
            256,
        )
        results["exports"].append({"mode": mode, "prepare_compile_s": compile_s, "abi": abi})
        for m in ROWS:
            # M512 is exact genuine embedding input; larger M tiles those actual
            # values for shape testing and is explicitly not a real 8K forward.
            x = real_hidden.repeat((m + 511) // 512, 1)[:m].clone()
            r = (x.float() * 0.75 + 0.0001).to(r_dtype) if r_dtype is not None else None
            y = torch.empty_like(x)
            ro = torch.empty_like(x, dtype=ro_dtype)
            arguments = (x, w, y, ro) if r is None else (x, r, w, y, ro)
            check = checked_graph(
                kernel, arguments, reference, x, r, w, eps, y, ro, ro_dtype, args.repetitions
            )
            budget = {1: 0.006, 512: 0.070, 2048: 0.280, 8192: 1.120}.get(m)
            case = {
                "mode": mode,
                "rows": m,
                "input_origin": "genuine embedding hidden"
                if m <= 512
                else "tiled genuine 512-prompt embeddings; shape test only",
                "residual_origin": "none" if r is None else "synthetic scaled embedding residual",
                "input_tensor_sha256": tensor_sha(x),
                "allocated_io_bytes": x.numel()
                * (
                    x.element_size()
                    + y.element_size()
                    + ro.element_size()
                    + (r.element_size() if r is not None else 0)
                ),
                "weight_bytes": w.numel() * w.element_size(),
                "workspace_bytes": 0,
                "budget_ms": budget,
                **check,
            }
            if budget is not None:
                case["latency_over_budget_ratio"] = check["timing"]["median_ms"] / budget
                case["budget_met"] = check["timing"]["median_ms"] <= budget
            results["cases"].append(case)
            write_json(output / "results.json", results)
            print(
                f"{mode} M{m} l2={check['norm_error']['relative_l2']:.3g} ms={check['timing']['median_ms']:.6f}",
                flush=True,
            )
            del x, r, y, ro, arguments
            gc.collect()

    # Additional real norm tensors and well-conditioned/extreme synthetic rows.
    for weight_name, weight in weights.items():
        for mode in ("first", "fp16_to_fp32", "fp32_to_fp32", "fp16_to_fp16"):
            x = torch.randn((7, 5120), device="cuda", dtype=torch.float16)
            x[0].zero_()
            x[1].fill_(2**-24)
            x[2].fill_(1000)
            x[3, 0] = 10000
            x[4].fill_(0.125)
            r_dtype = dict((a, b) for a, b, _ in modes)[mode]
            ro_dtype = dict((a, c) for a, _, c in modes)[mode]
            r = None if r_dtype is None else (-x.float() + 0.000001).to(r_dtype)
            if r is not None:
                r[2].fill_(500)
                r[3].zero_()
                r[4].fill_(0.125)
            y = torch.empty_like(x)
            ro = torch.empty_like(x, dtype=ro_dtype)
            args_call = (x, weight, y, ro) if r is None else (x, r, weight, y, ro)
            stream = torch.cuda.current_stream().cuda_stream
            kernels[mode](*args_call, stream=stream)
            ref_y, ref_ro = reference(x, r, weight, eps)
            row_errors = [error(y[i], ref_y[i]) for i in range(7)]
            assert all(e["finite"] and e["relative_l2"] <= 0.001 for e in row_errors), row_errors
            assert torch.equal(ro, ref_ro.to(ro_dtype))
            results.setdefault("boundary_cases", []).append(
                {
                    "weight": weight_name,
                    "mode": mode,
                    "rows": [
                        "zero",
                        "tiny_subnormal",
                        "large_finite",
                        "single_outlier",
                        "constant_half",
                        "random",
                        "random",
                    ],
                    "row_errors": row_errors,
                    "residual_bit_exact": True,
                }
            )

    # Parameterized non-model hidden tests exercise irregular reduction tails.
    for hidden in (127, 513):
        kernel = residual_norm(hidden=hidden, residual_dtype="float16")
        x = torch.randn((3, hidden), device="cuda", dtype=torch.float16)
        r = torch.randn_like(x)
        weight = torch.randn((hidden,), device="cuda", dtype=torch.float16)
        y = torch.empty_like(x)
        ro = torch.empty_like(x, dtype=torch.float32)
        kernel(x, r, weight, y, ro, stream=torch.cuda.current_stream().cuda_stream)
        ref_y, ref_ro = reference(x, r, weight, eps)
        e = error(y, ref_y)
        assert e["finite"] and e["relative_l2"] <= 0.001
        assert torch.equal(ro, ref_ro)
        results.setdefault("parameterization_cases", []).append(
            {"hidden": hidden, "rows": 3, "norm_error": e, "residual_bit_exact": True}
        )

    # Static row specialization and thread-count choices; same exact reference.
    for m in (1, 512):
        for threads in (128, 256, 512):
            started = time.perf_counter()
            kernel = residual_norm(M=m, residual_dtype="float32", threads=threads)
            compile_s = time.perf_counter() - started
            x = real_hidden[:m].clone()
            r = x.float() * 0.75 + 0.0001
            y = torch.empty_like(x)
            ro = torch.empty_like(r)
            check = checked_graph(
                kernel,
                (x, r, w, y, ro),
                reference,
                x,
                r,
                w,
                eps,
                y,
                ro,
                torch.float32,
                args.repetitions,
            )
            directory = output / "aot" / f"static_m{m}_t{threads}"
            export(kernel, directory, "residual", "float32", "float32", m, threads)
            results["tuning"].append(
                {"rows": m, "threads": threads, "prepare_compile_s": compile_s, **check}
            )
            print(f"static M{m} t{threads} ms={check['timing']['median_ms']:.6f}", flush=True)

    torch.cuda.synchronize()
    results["status"] = "passed numerical and graph checks; inspect budgets individually"
    results["memory"] = {
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    results["implementation_identity"] = [
        identity(ROOT / "kernels/operators/op02_residual_norm.py"),
        identity(Path(__file__)),
    ]
    write_json(output / "results.json", results)
    print("op02 complete; wrapper will release GPU lock after cleanup", flush=True)


if __name__ == "__main__":
    main()
