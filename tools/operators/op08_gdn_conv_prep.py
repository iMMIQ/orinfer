"""Offline op08 state, pinned-native, graph, ABI and performance checks."""

import argparse
import ast
import gc
import json
import re
import shutil
import time
from pathlib import Path
from tools.reference import ACTIVATIONS as REFERENCE_ACTIVATIONS
from tools.reference import CHECKPOINT, SOURCE

import torch
import triton
import triton.language as tl
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
from kernels.operators.op08_gdn_conv_prep import gdn_conv_decode, gdn_conv_prep, launch

MODEL = CHECKPOINT
NATIVE = SOURCE
MODES = {
    "native_prefill": dict(
        normalize_round_fp16=True,
        q_scale=1.0,
        qk_output_dtype="float16",
        conv_product_round_fp16=True,
    ),
    "native_staged_decode": dict(
        normalize_round_fp16=True,
        q_scale=1.0,
        qk_output_dtype="float16",
        conv_product_round_fp16=True,
    ),
    "native_fused_decode": dict(
        normalize_round_fp16=False,
        q_scale=128**-0.5,
        qk_output_dtype="float32",
        conv_product_round_fp16=True,
    ),
    "scaled_half_candidate": dict(
        normalize_round_fp16=True,
        q_scale=128**-0.5,
        qk_output_dtype="float16",
        conv_product_round_fp16=True,
    ),
    "fp32_conv_candidate": dict(
        normalize_round_fp16=True,
        q_scale=1.0,
        qk_output_dtype="float16",
        conv_product_round_fp16=False,
    ),
}


def binding(out):
    lock_path = ROOT / "artifacts/reference/reference-lock.json"
    lock = json.loads(lock_path.read_text())
    file = next(f for f in lock["files"] if f["name"] == "model.safetensors")
    assert file["bytes"] == (MODEL / "model.safetensors").stat().st_size
    cfg = json.loads((MODEL / "config.json").read_text())["text_config"]
    assert [
        cfg[k]
        for k in (
            "linear_conv_kernel_dim",
            "linear_num_key_heads",
            "linear_num_value_heads",
            "linear_key_head_dim",
            "linear_value_head_dim",
        )
    ] == [4, 16, 48, 128, 128]
    weights, names = {}, []
    with safe_open(str(MODEL / "model.safetensors"), framework="pt", device="cpu") as f:
        keys = f.keys()
        for layer in (0, 32):
            name = f"model.language_model.layers.{layer}.linear_attn.conv1d.weight"
            assert name.replace(".weight", ".bias") not in keys
            raw = f.get_tensor(name)
            assert list(raw.shape) == [10240, 1, 4]
            w = raw[:, 0].half().contiguous()
            names.append(
                {
                    "name": name,
                    "shape": list(raw.shape),
                    "dtype": str(raw.dtype),
                    "raw_tensor_sha256": tensor_sha(raw),
                    "runtime_tensor_sha256": tensor_sha(w),
                    "conversion": "checkpoint to FP16 to match locked --dtype float16 runtime",
                    "conversion_bit_exact": torch.equal(raw[:, 0], w.to(raw.dtype)),
                    "conversion_error": error(w, raw[:, 0]),
                    "bias_exists": False,
                }
            )
            weights[layer] = w.cuda()
    source_dir = out / "reference-source"
    source_dir.mkdir()
    sources = []
    for relative in (
        "model_executor/layers/mamba/gdn_linear_attn.py",
        "model_executor/layers/mamba/ops/causal_conv1d.py",
        "model_executor/layers/fla/ops/l2norm.py",
        "model_executor/layers/fla/ops/fused_recurrent.py",
        "model_executor/layers/fla/ops/fused_gdn_prefill_post_conv.py",
    ):
        source = NATIVE / relative
        target = source_dir / Path(relative).name
        shutil.copyfile(source, target)
        sources.append(identity(target))
    metadata_path = (
        REFERENCE_ACTIVATIONS
        / "capture-512-0-language_model_model_layers_0_linear_attn_in_proj_qkvz.json"
    )
    meta = json.loads(metadata_path.read_text())
    assert meta["shape"] == [512, 5120]
    info = {
        "checkpoint": dict(file, path=str(MODEL / "model.safetensors")),
        "file_hash_policy": "reuse locked full-file hash and check size; tensor slices hashed",
        "reference_lock": identity(lock_path),
        "config": identity(MODEL / "config.json"),
        "weights": names,
        "native_sources": sources,
        "input_origin": "synthetic projected FP16 QKV, seed20261002; no actual 10240-wide projection capture available",
        "rejected_capture": {
            "identity": identity(metadata_path),
            "shape": meta["shape"],
            "reason": "captured in_proj_qkvz input is hidden5120, not projected QKV10240",
        },
        "epsilon": 1e-6,
        "conv_width": 4,
        "conv_bias": None,
    }
    write_json(out / "binding.json", info)
    return weights, info


def reference(x, w, hi, lengths, positions, mode):
    """Same FP32 math, explicit native rounding points; no production fallback."""
    b, t, _ = x.shape
    ho = hi.clone()
    conv = torch.zeros_like(x)
    for request in range(b):
        n, pos = int(lengths[request]), int(positions[request])
        if n == 0:
            continue
        prior = hi[request].clone()
        if pos < 3:
            prior[: 3 - pos].zero_()
        extended = torch.cat((prior, x[request, :n]))
        acc = torch.zeros((n, 10240), device=x.device, dtype=torch.float32)
        for tap in range(4):
            if mode["conv_product_round_fp16"]:
                acc = acc + (extended[tap : tap + n].float() * w[:, tap].float()).half().float()
            else:
                acc = torch.addcmul(acc, extended[tap : tap + n].float(), w[:, tap].float())
        conv[request, :n] = (acc / (1 + torch.exp(-acc))).half()
        ho[request] = extended[-3:]
    q = conv[:, :, :2048].reshape(b, t, 16, 128).float()
    k = conv[:, :, 2048:4096].reshape(b, t, 16, 128).float()
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    if mode["normalize_round_fp16"]:
        q, k = q.half().float(), k.half().float()
    dtype = getattr(torch, mode["qk_output_dtype"])
    q = (q * mode["q_scale"]).to(dtype).permute(0, 2, 1, 3).contiguous()
    k = k.to(dtype).permute(0, 2, 1, 3).contiguous()
    v = conv[:, :, 4096:].reshape(b, t, 48, 128).permute(0, 2, 1, 3).contiguous()
    return q, k, v, ho, positions + lengths, conv


def outputs(x, dtype):
    b, t, _ = x.shape
    return (
        torch.empty((b, 16, t, 128), device="cuda", dtype=getattr(torch, dtype)),
        torch.empty((b, 16, t, 128), device="cuda", dtype=getattr(torch, dtype)),
        torch.empty((b, 48, t, 128), device="cuda", dtype=torch.float16),
        torch.empty((b, 3, 10240), device="cuda", dtype=torch.float16),
        torch.empty((b,), device="cuda", dtype=torch.int32),
    )


def check(result, ref):
    errors = {name: error(a, r) for name, a, r in zip(("q", "k", "v"), result, ref)}
    assert all(e["finite"] and e["relative_l2"] <= 0.001 for e in errors.values()), errors
    assert torch.equal(result[3], ref[3]), "raw history mismatch"
    assert torch.equal(result[4], ref[4]), "positions mismatch"
    return {"errors": errors, "history_bit_exact": True, "positions_bit_exact": True}


def checked_case(kernel, x, w, hi, lengths, positions, mode, repetitions, graph_check=True):
    result = outputs(x, mode["qk_output_dtype"])

    def run():
        return launch(
            kernel,
            x,
            w,
            hi,
            lengths,
            positions,
            *result,
            stream=torch.cuda.current_stream().cuda_stream,
        )

    started = time.perf_counter()
    run()
    torch.cuda.synchronize()
    first = time.perf_counter() - started
    expected = reference(x, w, hi, lengths, positions, mode)
    report = check(result, expected)
    timing, graph = benchmark(
        run, repetitions=repetitions, calls_per_replay=16 if x.shape[1] <= 8 else 4
    )
    report.update(
        first_launch_s=first,
        timing=timing,
        input_sha256=tensor_sha(x),
        history_input_sha256=tensor_sha(hi),
        memory={
            "input_bytes": x.numel() * 2,
            "weight_bytes": w.numel() * 2,
            "state_input_bytes": hi.numel() * 2 + positions.numel() * 4,
            "state_output_bytes": result[3].numel() * 2 + result[4].numel() * 4,
            "output_qkv_bytes": sum(r.numel() * r.element_size() for r in result[:3]),
            "lengths_bytes": lengths.numel() * 4,
            "workspace_bytes": 0,
        },
    )
    if graph_check:
        saved_x, saved_hi = x.clone(), hi.clone()
        original = [r.clone() for r in result]

        def poison():
            for r in result[:4]:
                r.fill_(float("nan"))
            result[4].fill_(-999)

        poison()
        graph.replay()
        torch.cuda.synchronize()
        assert all(torch.equal(a, b) for a, b in zip(result, original))
        x.mul_(-0.375)
        hi.mul_(0.25)
        changed = reference(x, w, hi, lengths, positions, mode)
        poison()
        graph.replay()
        torch.cuda.synchronize()
        changed_check = check(result, changed)
        assert not torch.equal(result[2], original[2])
        x.copy_(saved_x)
        hi.copy_(saved_hi)
        poison()
        graph.replay()
        torch.cuda.synchronize()
        assert all(torch.equal(a, b) for a, b in zip(result, original))
        report["graph"] = {
            "poison_all_outputs": True,
            "mutated_input_and_history": True,
            "changed_check": changed_check,
            "restore_bit_exact": True,
        }
    return report


def export(kernel, output, mode, tile):
    exported = export_kernel(kernel, output)
    cuda, host = (output / "kernel.cu").read_text(), (output / "host.txt").read_text()
    entries = [
        {"symbol": name, "parameters_verbatim": args}
        for name, args in re.findall(r"__global__\s+void\s+(\w+)\s*\(([^)]*)\)", cuda)
        if name in exported["symbols"]
    ]
    abi = {
        "schema_version": 1,
        "operator": "op08_gdn_conv_prep",
        "sm": 87,
        "mode": mode,
        "build_parameters": MODES[mode],
        "runtime_shape": ["B", "T", 10240],
        "tensor_api_order": [
            "X",
            "W",
            "HI",
            "lengths",
            "positions",
            "Q",
            "K",
            "V",
            "HO",
            "positions_out",
        ],
        "tensor_dtypes": {
            "X": "float16",
            "W": "float16",
            "HI": "float16",
            "lengths": "int32",
            "positions": "int32",
            "Q": MODES[mode]["qk_output_dtype"],
            "K": MODES[mode]["qk_output_dtype"],
            "V": "float16",
            "HO": "float16",
            "positions_out": "int32",
        },
        "layout": {
            "X": "[B,T,10240], Q2048 K2048 V6144",
            "W": "[10240,4], oldest to current tap",
            "HI/HO": "[B,3,10240], chronological raw projection values",
            "Q/K": "[B,16,T,128]",
            "V": "[B,48,T,128]",
            "head_mapping": "kh=vh//3",
        },
        "q_scale": MODES[mode]["q_scale"],
        "Q_already_scaled": MODES[mode]["q_scale"] != 1.0,
        "epsilon": 1e-6,
        "actual_cuda_entries": entries,
        "host_arguments_verbatim": [
            l.strip() for l in host.splitlines() if "arg_values =" in l or "arg_types =" in l
        ],
        "host_launch_lines_verbatim": [
            l.strip()
            for l in host.splitlines()
            if any(s in l.lower() for s in ("grid", "block", "shared", "launch"))
        ],
        "grid": [f"ceil(T/{tile})", 80, "B"],
        "block": [128, 1, 1],
        "cooperative_launch": False,
        "dynamic_shared_memory_bytes": [
            int(v) for v in re.findall(r"config.sharedMemBytes\s*=\s*(\d+)", host)
        ],
        "static_shared_memory_bytes": 0,
        "workspace_bytes": 0,
        "persistent_weight_bytes": 81920,
        "private_state_per_request_bytes": 61444,
        "alias_contract": "all tensor buffers distinct; inputs immutable; caller preallocates outputs; repeated launches overwrite from unchanged HI",
        "stream": "explicit at each run; acquire current capture stream",
        "toolchain": environment(),
        **exported,
    }
    write_json(output / "abi.json", abi)
    return abi


def native_function(path, name):
    node = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == name
    )
    node.decorator_list = [
        ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="triton", ctx=ast.Load()), attr="jit", ctx=ast.Load()
            ),
            args=[],
            keywords=[],
        )
    ]
    namespace = {"triton": triton, "tl": tl, "__name__": "op08_pinned_native"}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), "exec"
        ),
        namespace,
    )
    return namespace[name]


def native_checks(out, w, kernel):
    source = out / "reference-source"
    conv = native_function(source / "causal_conv1d.py", "_causal_conv1d_update_kernel")
    norm = native_function(source / "l2norm.py", "l2norm_fwd_kernel2")
    reports = []
    for b, t in ((1, 1), (3, 2), (7, 7)):
        x = torch.randn((b, t, 10240), device="cuda", dtype=torch.float16)
        hi = torch.randn((b, 3, 10240), device="cuda", dtype=torch.float16)
        lens = torch.full((b,), t, device="cuda", dtype=torch.int32)
        pos = torch.full((b,), 29, device="cuda", dtype=torch.int32)
        ref = reference(x, w, hi, lens, pos, MODES["native_prefill"])
        native_x = x.permute(0, 2, 1).contiguous()
        native_hi = hi.permute(0, 2, 1).contiguous()
        indices = torch.arange(b, device="cuda", dtype=torch.int32)
        out_conv = torch.empty_like(native_x)
        conv[(b, triton.cdiv(10240, 256))](
            native_x,
            w,
            None,
            native_hi,
            indices,
            None,
            None,
            None,
            None,
            out_conv,
            b,
            10240,
            t,
            3,
            b,
            *native_x.stride(),
            *w.stride(),
            *native_hi.stride(),
            1,
            *out_conv.stride(),
            -1,
            HAS_BIAS=False,
            KERNEL_WIDTH=4,
            SILU_ACTIVATION=True,
            IS_VARLEN=False,
            IS_APC_ENABLED=False,
            IS_SPEC_DECODING=False,
            NP2_STATELEN=4,
            HAS_NULL_BLOCK=False,
            BLOCK_N=256,
        )
        native_conv = out_conv.permute(0, 2, 1).contiguous()
        conv_error = error(native_conv, ref[5])
        assert conv_error["relative_l2"] <= 0.001
        assert torch.equal(native_hi.permute(0, 2, 1), ref[3])
        qkv = []
        for offset in (0, 2048):
            values = native_conv[:, :, offset : offset + 2048].reshape(-1, 128).contiguous()
            normalized = torch.empty_like(values)
            norm[(triton.cdiv(values.shape[0], 32),)](
                values, normalized, 1e-6, values.shape[0], 128, 128, 32
            )
            qkv.append(normalized.reshape(b, t, 16, 128).permute(0, 2, 1, 3).contiguous())
        native_err = {
            key: error(actual, expected) for key, actual, expected in zip(("q", "k"), qkv, ref)
        }
        assert all(v["relative_l2"] <= 0.001 for v in native_err.values())
        result = outputs(x, "float16")
        launch(kernel, x, w, hi, lens, pos, *result, stream=torch.cuda.current_stream().cuda_stream)
        v = native_conv[:, :, 4096:].reshape(b, t, 48, 128).permute(0, 2, 1, 3).contiguous()
        tl_vs_native = check(result, tuple(qkv) + (v, ref[3], ref[4]))
        reports.append(
            {
                "B": b,
                "T": t,
                "conv_error": conv_error,
                "qk_error": native_err,
                "tilelang_vs_native": tl_vs_native,
                "history_bit_exact": True,
            }
        )
    return reports


def state_checks(kernel, w, mode):
    b = 3
    total = 37
    x = torch.randn((b, total, 10240), device="cuda", dtype=torch.float16)
    initial = torch.randn((b, 3, 10240), device="cuda", dtype=torch.float16)
    positions = torch.tensor([0, 1, 91], device="cuda", dtype=torch.int32)

    def chain(chunks, data=x, history=initial, pos=positions):
        h, p = history.clone(), pos.clone()
        outputs_list = []
        offset = 0
        for n in chunks:
            part = data[:, offset : offset + n].contiguous()
            length = torch.full((b,), n, device="cuda", dtype=torch.int32)
            out = outputs(part, mode["qk_output_dtype"])
            launch(
                kernel, part, w, h, length, p, *out, stream=torch.cuda.current_stream().cuda_stream
            )
            check(out, reference(part, w, h, length, p, mode))
            outputs_list.append(out[:3])
            h, p = out[3:]
            offset += n
        return tuple(torch.cat([o[i] for o in outputs_list], 2) for i in range(3)) + (h, p)

    full = chain([total])
    sequential = chain([1] * total)
    split = chain([1, 2, 3, 7, 11, 13])
    reports = {}
    for name, actual in (("sequential_decode", sequential), ("mixed_chunks", split)):
        reports[name] = check(actual, full)
    prefix = chain([9], data=x[:, :9])
    saved_h, saved_p = prefix[3].clone(), prefix[4].clone()
    tail = x[:, 9:].contiguous()
    restored = chain([2, 1, 25], data=tail, history=saved_h.clone(), pos=saved_p.clone())
    for i in range(3):
        e = error(restored[i], full[i][:, :, 9:])
        assert e["relative_l2"] <= 0.001
    assert torch.equal(restored[3], full[3]) and torch.equal(restored[4], full[4])
    alternate = tail.clone()
    alternate[0].mul_(-2)
    branch = chain([28], data=alternate, history=saved_h.clone(), pos=saved_p.clone())
    assert torch.equal(branch[3][1:], full[3][1:])
    assert not torch.equal(branch[3][0], full[3][0])
    assert torch.equal(prefix[3], saved_h) and torch.equal(prefix[4], saved_p)
    # Different request lengths include an empty chunk and a short <3 chunk.
    padded = torch.randn((3, 7, 10240), device="cuda", dtype=torch.float16)
    lengths = torch.tensor([0, 1, 2], device="cuda", dtype=torch.int32)
    pos = torch.tensor([0, 0, 17], device="cuda", dtype=torch.int32)
    result = outputs(padded, mode["qk_output_dtype"])
    launch(
        kernel,
        padded,
        w,
        initial,
        lengths,
        pos,
        *result,
        stream=torch.cuda.current_stream().cuda_stream,
    )
    check(result, reference(padded, w, initial, lengths, pos, mode))
    original = [o.clone() for o in result]
    for request, n in enumerate((0, 1, 2)):
        padded[request, n:].fill_(float("nan"))
    launch(
        kernel,
        padded,
        w,
        initial,
        lengths,
        pos,
        *result,
        stream=torch.cuda.current_stream().cuda_stream,
    )
    assert all(torch.equal(a, z) for a, z in zip(result, original)), "padding consumed"
    reports.update(
        prefix_restore=True,
        branch_request_isolation=True,
        positions_isolation=True,
        short_chunks=[0, 1, 2],
        padding_nan_ignored_bit_exact=True,
        empty_history_preserved=True,
        initial_history_immutable=True,
        checkpoint_bytes_per_request=61444,
    )
    return reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--prefill-tile", type=int, default=16)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    configure()
    weights, bind = binding(out)
    report = {
        "environment": environment(),
        "binding": bind,
        "cases": [],
        "exports": [],
        "status": "in_progress",
        "input_kind": "synthetic QKV projection output, real layer0/32 conv weights",
    }
    kernels = {}
    for mode, parameters in MODES.items():
        tile = 1 if mode in ("native_fused_decode", "native_staged_decode") else args.prefill_tile
        start = time.perf_counter()
        kernel = (
            gdn_conv_decode(**parameters)
            if tile == 1
            else gdn_conv_prep(tile_tokens=tile, **parameters)
        )
        kernels[mode] = kernel
        report["exports"].append(
            {
                "mode": mode,
                "prepare_compile_s": time.perf_counter() - start,
                "abi": export(kernel, out / "aot" / mode, mode, tile),
            }
        )
        shapes = [(b, 1) for b in (1, 2, 3, 4, 5, 7, 8)]
        if mode == "native_prefill":
            shapes = [(1, t) for t in (511, 512, 513, 2048, 8192)] + [(3, 513)]
        elif mode in ("scaled_half_candidate", "fp32_conv_candidate"):
            shapes = [(1, 1), (1, 512)]
        for b, t in shapes:
            x = torch.randn((b, t, 10240), device="cuda", dtype=torch.float16)
            hi = torch.randn((b, 3, 10240), device="cuda", dtype=torch.float16)
            lengths = torch.full((b,), t, device="cuda", dtype=torch.int32)
            positions = torch.arange(b, device="cuda", dtype=torch.int32) * 71
            case = checked_case(
                kernel, x, weights[0], hi, lengths, positions, parameters, args.repetitions
            )
            budget = {1: 0.010, 512: 0.120, 2048: 0.480, 8192: 1.920}.get(t) if b == 1 else None
            case.update(mode=mode, B=b, T=t, budget_ms=budget)
            if budget is not None:
                case.update(
                    budget_met=case["timing"]["median_ms"] <= budget,
                    over_budget_ratio=case["timing"]["median_ms"] / budget,
                )
            report["cases"].append(case)
            write_json(out / "results.json", report)
            print(
                f"{mode} B{b} T{t} ms={case['timing']['median_ms']:.6f} maxl2={max(e['relative_l2'] for e in case['errors'].values()):.3g}",
                flush=True,
            )
            del x, hi, lengths, positions
            gc.collect()
        report.setdefault("state_checks", {})[mode] = state_checks(kernel, weights[0], parameters)
    report["native_checks"] = native_checks(out, weights[0], kernels["native_prefill"])
    report["tuning"] = []
    for tile in (4, 8):
        started = time.perf_counter()
        kernel = gdn_conv_prep(tile_tokens=tile, **MODES["native_prefill"])
        compile_s = time.perf_counter() - started
        x = torch.randn((1, 512, 10240), device="cuda", dtype=torch.float16)
        hi = torch.zeros((1, 3, 10240), device="cuda", dtype=torch.float16)
        lengths = torch.tensor([512], device="cuda", dtype=torch.int32)
        positions = torch.tensor([0], device="cuda", dtype=torch.int32)
        tested = checked_case(
            kernel, x, weights[0], hi, lengths, positions, MODES["native_prefill"], args.repetitions
        )
        report["tuning"].append(
            {
                "tile_tokens": tile,
                "B": 1,
                "T": 512,
                "compile_s": compile_s,
                "abi": export(kernel, out / "aot" / f"prefill_tile{tile}", "native_prefill", tile),
                **tested,
            }
        )
        print(f"prefill tile{tile} P512 ms={tested['timing']['median_ms']:.6f}", flush=True)
    # Second actual conv tensor, zero/subnormal/large finite activation rows.
    mode = MODES["native_prefill"]
    kernel = kernels["native_prefill"]
    x = torch.randn((3, 7, 10240), device="cuda", dtype=torch.float16)
    x[0].zero_()
    x[1].fill_(2**-24)
    x[2].mul_(30)
    hi = torch.zeros((3, 3, 10240), device="cuda", dtype=torch.float16)
    lens = torch.tensor([7, 7, 7], device="cuda", dtype=torch.int32)
    pos = torch.zeros(3, device="cuda", dtype=torch.int32)
    report["layer32_boundary"] = checked_case(
        kernel, x, weights[32], hi, lens, pos, mode, args.repetitions
    )
    report["status"] = (
        "passed same-math, pinned native, state and graph checks; standalone budgets separately reported"
    )
    report["memory_peak"] = {
        "allocated_bytes": torch.cuda.max_memory_allocated(),
        "reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    report["implementation_identity"] = [
        identity(Path(__file__)),
        identity(ROOT / "kernels/operators/op08_gdn_conv_prep.py"),
    ]
    write_json(out / "results.json", report)
    print("op08 complete; wrapper cleanup releases GPU lock", flush=True)


if __name__ == "__main__":
    main()
