"""Offline op01: real sampled BF16 rows, full-shape allocation, exact math, ABI."""

import argparse
import gc
import json
import time
from pathlib import Path
from tools.reference import ACTIVATIONS as REFERENCE_ACTIVATIONS
from tools.reference import CHECKPOINT

import torch
from safetensors import safe_open

from common import (
    ROOT,
    SEED,
    benchmark,
    configure,
    environment,
    error,
    export_kernel,
    identity,
    tensor_sha,
    write_json,
)
from tools.operators.abi import parse_host
from kernels.operators.op01_embedding import embedding_gather, embedding_u4, validate_token_ids

MODEL = CHECKPOINT
META = (
    REFERENCE_ACTIVATIONS
    / "capture-512-0-language_model_model_layers_0_linear_attn_in_proj_qkvz.json"
)
VOCAB, HIDDEN = 248320, 5120
ROWS = (1, 2, 3, 4, 5, 7, 8, 511, 512, 513, 2048, 8192)


def pack_u4(raw, group=128):
    """Explicit asymmetric round-even min/max quantizer; new lossy candidate.

    Include zero in range, FP16-rounded positive scale, then derive/clamp zero
    and q using that stored scale. Zero groups use scale1. Padding quantizes0.
    No calibration or model-quality acceptance is implied.
    """
    h = raw.shape[1]
    padded = ((h + group - 1) // group) * group
    x = torch.nn.functional.pad(raw.float(), (0, padded - h)).reshape(raw.shape[0], -1, group)
    assert torch.isfinite(x).all()
    lo = x.amin(-1).clamp(max=0)
    hi = x.amax(-1).clamp(min=0)
    s = ((hi - lo) / 15).half()
    s = torch.where(s > 0, s, torch.ones_like(s))
    assert torch.isfinite(s).all()
    z = (-lo / s.float()).round().clamp(0, 15).to(torch.int8)
    q = (
        (x / s.float().unsqueeze(-1) + z.float().unsqueeze(-1))
        .round()
        .clamp(0, 15)
        .to(torch.uint8)
        .reshape(raw.shape[0], padded)
    )
    p = q[:, ::2] | (q[:, 1::2] << 4)
    return p.contiguous(), s.contiguous(), z.contiguous(), q


def dequant(p, s, z, hidden, group=128):
    q = torch.stack((p & 15, p >> 4), -1).reshape(p.shape[0], -1)
    return (
        (
            (q.float().reshape(p.shape[0], -1, group) - z.float().unsqueeze(-1))
            * s.float().unsqueeze(-1)
        )
        .reshape(p.shape[0], -1)[:, :hidden]
        .half()
    )


def source_rows(out, sample_rows):
    started = time.perf_counter()
    lock_path = ROOT / "artifacts/reference/reference-lock.json"
    lock = json.loads(lock_path.read_text())
    file = next(f for f in lock["files"] if f["name"] == "model.safetensors")
    assert (MODEL / file["name"]).stat().st_size == file["bytes"]
    config = json.loads((MODEL / "config.json").read_text())["text_config"]
    assert config["vocab_size"] == VOCAB and config["hidden_size"] == HIDDEN
    meta = json.loads(META.read_text())
    prompt = list(validate_token_ids(meta["prompt_token_ids"], VOCAB))
    generator = torch.Generator().manual_seed(SEED)
    sampled = torch.randperm(VOCAB, generator=generator)[:sample_rows].tolist()
    ids = sorted(set(sampled + prompt + [0, 1, VOCAB - 2, VOCAB - 1]))
    name = "model.language_model.embed_tokens.weight"
    chunks = []
    with safe_open(str(MODEL / file["name"]), framework="pt", device="cpu") as f:
        tensor = f.get_slice(name)
        assert tensor.get_shape() == [VOCAB, HIDDEN] and tensor.get_dtype() == "BF16"
        # Only these selected rows are read. No full-tensor materialization/hash.
        for token in ids:
            chunks.append(tensor[token : token + 1])
    raw = torch.cat(chunks).contiguous()
    info = {
        "checkpoint_locked_identity": file,
        "checkpoint_path": str(MODEL / file["name"]),
        "full_hash_policy": "Reuse locked full-file SHA256; size checked; never rescan whole18GB",
        "reference_lock": identity(lock_path),
        "config": identity(MODEL / "config.json"),
        "prompt_metadata": identity(META),
        "tensor_name": name,
        "source_shape": [VOCAB, HIDDEN],
        "source_dtype": "bfloat16",
        "sampled_row_ids": ids,
        "sampled_row_ids_sha256": tensor_sha(torch.tensor(ids, dtype=torch.int32)),
        "sampled_rows_sha256": tensor_sha(raw),
        "actual_source_tensor_bytes_read": raw.numel() * raw.element_size(),
        "FP16_cast_sampled_rows_sha256": tensor_sha(raw.half()),
        "read_scope": "Only listed row fragments; this is NOT a full embedding tensor hash or full-vocabulary quality evaluation",
        "source_read_and_hash_s": time.perf_counter() - started,
    }
    write_json(out / "binding.json", info)
    return ids, raw, prompt, info


def export(kernel, directory, mode, hidden, vocab, block, stride=None):
    files = export_kernel(kernel, directory)
    host = (directory / "host.txt").read_text()
    launches = parse_host(host)
    padded = ((hidden + 127) // 128) * 128
    resident = (
        vocab * (padded // 2 + padded // 128 * 3)
        if mode == "u4"
        else vocab * (stride or hidden) * 2
    )
    abi = {
        "schema_version": 1,
        "operator": "op01_embedding",
        "mode": mode,
        "tensor_api_order": ["P", "S", "Z", "I", "Y"] if mode == "u4" else ["W", "I", "Y"],
        "actual_generated_launches": launches,
        "files": files["files"],
        "symbols": files["symbols"],
        "dimensions": {
            "V": vocab,
            "H": hidden,
            "M": "runtime rows:int32",
            "weight_stride": stride or hidden,
            "Kp": padded,
        },
        "layout": "row-major; adjacent low/high U4, group128 S FP16/Z int8"
        if mode == "u4"
        else "row-major " + mode + " weight",
        "output": "FP16[M,H]",
        "indices": "int32[M], CPU reject outside[0,V) before each upload/change; duplicates valid",
        "math": "FP32(q-Z)*FP32(S), final FP16 RNE"
        if mode == "u4"
        else mode + " weight cast to FP16 RNE",
        "workspace_bytes": 0,
        "persistent_weight_bytes": resident,
        "grid": f"ceildiv(M*{(hidden + 1) // 2 if mode == 'u4' else hidden},{block})",
        "block": [128, 1, 1],
        "shared_memory_bytes": 0,
        "cooperative_launch": False,
        "stream_contract": "explicit stream; resolve capture-current stream at each invocation",
        "alias_contract": "all tensors disjoint; inputs immutable during launch; graph addresses stable",
        "target": "sm_87",
        "toolchain": environment(),
    }
    write_json(directory / "abi.json", abi)
    return abi


def exact(y, ref):
    result = error(y, ref)
    result["bit_exact"] = torch.equal(y.view(torch.int16), ref.view(torch.int16))
    assert result["bit_exact"] and result["finite"], result
    return result


def invalid_tests():
    rejected = []
    for ids, vocab, rows in [
        ([-1], VOCAB, None),
        ([VOCAB], VOCAB, None),
        ([2147483647], VOCAB, None),
        ([True], VOCAB, None),
        ([1.0], VOCAB, None),
        ([], VOCAB, None),
        ([0], 0, None),
        ([0], VOCAB, 2),
    ]:
        try:
            validate_token_ids(ids, vocab, rows)
        except ValueError:
            rejected.append({"ids": ids, "vocab": vocab, "rows": rows})
        else:
            raise AssertionError("invalid CPU request accepted")
    assert validate_token_ids([0, VOCAB - 1, 0], VOCAB, 3) == (0, VOCAB - 1, 0)
    return rejected


def checked(kernel, table, ids_cpu, reference_rows, row_map, replacements, repetitions):
    m = len(ids_cpu)
    validate_token_ids(ids_cpu, table[0].shape[0], m)
    ids = torch.tensor(ids_cpu, device="cuda", dtype=torch.int32)
    y = torch.empty((m, reference_rows.shape[1]), device="cuda", dtype=torch.float16)

    def run():
        kernel(*table, ids, y, stream=torch.cuda.current_stream().cuda_stream)

    def expected(values):
        return reference_rows[torch.tensor([row_map[t] for t in values], device="cuda")]

    begun = time.perf_counter()
    run()
    torch.cuda.synchronize()
    first_ms = 1000 * (time.perf_counter() - begun)
    initial = exact(y, expected(ids_cpu))
    timing, graph = benchmark(run, repetitions=repetitions, calls_per_replay=16)
    original = y.clone()
    y.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(y, original)
    changed = [replacements[(i + 1) % len(replacements)] for i in range(m)]
    validate_token_ids(changed, table[0].shape[0], m)
    ids.copy_(torch.tensor(changed, device="cuda", dtype=torch.int32))
    y.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    exact(y, expected(changed))
    assert not torch.equal(y, original), "graph ignored changed IDs"
    # Mutate only the rows currently read, no whole-table clone or expansion.
    selected = torch.tensor(sorted(set(changed)), device="cuda", dtype=torch.long)
    target = table[1] if len(table) == 3 else table[0]  # S or W
    saved = target.index_select(0, selected)
    target.index_copy_(0, selected, -saved if len(table) == 1 else saved * 2)
    changed_reference = -expected(changed) if len(table) == 1 else expected(changed) * 2
    y.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    exact(y, changed_reference)
    target.index_copy_(0, selected, saved)
    ids.copy_(torch.tensor(ids_cpu, device="cuda", dtype=torch.int32))
    y.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(y, original), "restore replay failed"
    return {
        "M": m,
        "same_format": initial,
        "first_launch_ms": first_ms,
        "hot": timing,
        "graph_poison_changed_ids_changed_data_restore": True,
        "output_bytes": y.numel() * 2,
        "index_bytes": ids.numel() * 4,
        "workspace_bytes": 0,
        "unique_read_rows": len(set(ids_cpu)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-rows", type=int, default=8192)
    parser.add_argument("--repetitions", type=int, default=20)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    configure()
    context_configure_s = time.perf_counter() - started
    report = {
        "environment": environment(),
        "source": identity(ROOT / "kernels/operators/op01_embedding.py"),
        "invalid_requests_rejected": invalid_tests(),
        "cases": [],
        "tuning": [],
        "exports": [],
        "workspace_bytes": 0,
        "model_integration": False,
        "context_configure_s": context_configure_s,
        "load_scope": "Context, source-row I/O/hash, sparse quantization, allocation/upload, JIT/module and first launch recorded separately; OS page-cache state uncontrolled; no full model load",
        "abi_parser_source": identity(ROOT / "tools/operators/abi.py"),
        "quality_status": "Not accepted: new U4 needs pinned official BF16 model-level evaluation",
        "allocation_scope": "Full-size tensors; only listed true-source rows initialized/read; all other rows deliberately undefined",
        "performance_scope": "Hot repeated graph replay; sparse sampled real rows in full-address-space table; not cold/random full-vocab bandwidth",
    }
    ids, raw, prompt, binding = source_rows(out, args.sample_rows)
    report["binding"] = binding
    row_map = {t: i for i, t in enumerate(ids)}
    begun = time.perf_counter()
    p, s, z, q = pack_u4(raw)
    unpacked_q = torch.stack((p & 15, p >> 4), -1).reshape(q.shape)
    assert torch.equal(unpacked_q, q)
    dq = dequant(p, s, z, HIDDEN)
    report["pack"] = {
        "selected_rows_pack_s": time.perf_counter() - begun,
        "code_roundtrip_exact": True,
        "P_sha256": tensor_sha(p),
        "S_sha256": tensor_sha(s),
        "Z_sha256": tensor_sha(z),
        "sample_format_loss_vs_source_BF16": error(dq, raw),
        "sample_BF16_to_FP16_cast_loss": error(raw.half(), raw),
        "sample_coverage_rows": len(ids),
        "vocab_rows": VOCAB,
        "full_table_prepack_completed": False,
    }
    generator = torch.Generator().manual_seed(SEED)
    random_order = torch.randperm(len(ids), generator=generator).tolist()
    replacements = [ids[i] for i in random_order]
    workload = {}
    for m in ROWS:
        # Prompt M512 uses original real IDs. Larger sizes mix independent real rows;
        # not presented as real model histories or submitted concurrency.
        workload[m] = (
            ([0] if m == 1 else ([0, VOCAB - 1, 0] + replacements)[:m])
            if m < 511
            else (prompt if m == 512 else [replacements[i % len(replacements)] for i in range(m)])
        )
    write_json(out / "workload-ids.json", {str(m): values for m, values in workload.items()})
    for mode in ("float16", "bfloat16", "u4"):
        begun = time.perf_counter()
        gpu_ids = torch.tensor(ids, device="cuda", dtype=torch.long)
        if mode == "u4":
            table = [
                torch.empty((VOCAB, HIDDEN // 2), dtype=torch.uint8, device="cuda"),
                torch.empty((VOCAB, HIDDEN // 128), dtype=torch.float16, device="cuda"),
                torch.empty((VOCAB, HIDDEN // 128), dtype=torch.int8, device="cuda"),
            ]
            for dst, src in zip(table, (p, s, z)):
                dst.index_copy_(0, gpu_ids, src.cuda())
            ref = dq.cuda()
        else:
            table = [torch.empty((VOCAB, HIDDEN), dtype=getattr(torch, mode), device="cuda")]
            table[0].index_copy_(0, gpu_ids, raw.to(getattr(torch, mode)).cuda())
            ref = raw.half().cuda()
        torch.cuda.synchronize()
        prepare = time.perf_counter() - begun
        resident = sum(t.numel() * t.element_size() for t in table)
        candidates = (256, 512, 1024) if mode != "u4" else (128, 256, 512)
        kernels = {}
        scores = []
        for block in candidates:
            begun = time.perf_counter()
            kernel = (
                embedding_u4(block=block)
                if mode == "u4"
                else embedding_gather(dtype=mode, block=block)
            )
            compile_s = time.perf_counter() - begun
            kernels[block] = kernel
            trial = {
                "mode": mode,
                "block": block,
                "threads": 128,
                "compile_load_s": compile_s,
                "timings": {},
            }
            for m in (1, 512):
                ids_gpu = torch.tensor(workload[m], device="cuda", dtype=torch.int32)
                output = torch.empty((m, HIDDEN), device="cuda", dtype=torch.float16)

                def run():
                    kernel(*table, ids_gpu, output, stream=torch.cuda.current_stream().cuda_stream)

                begun = time.perf_counter()
                run()
                torch.cuda.synchronize()
                trial.setdefault("first_actual_launch_ms", {})[str(m)] = 1000 * (
                    time.perf_counter() - begun
                )
                exact(output, ref[torch.tensor([row_map[t] for t in workload[m]], device="cuda")])
                trial["timings"][str(m)], _ = benchmark(
                    run, repetitions=args.repetitions, calls_per_replay=16
                )
            scores.append(trial)
            report["tuning"].append(trial)
        best = min(
            scores,
            key=lambda t: (
                t["timings"]["1"]["median_ms"] / 0.005 + t["timings"]["512"]["median_ms"] / 0.1
            ),
        )
        block = best["block"]
        kernel = kernels[block]
        report["exports"].append(export(kernel, out / mode, mode, HIDDEN, VOCAB, block))
        for m in ROWS:
            print(f"{mode} M{m}", flush=True)
            case = checked(kernel, table, workload[m], ref, row_map, replacements, args.repetitions)
            case.update(
                mode=mode,
                block=block,
                persistent_weight_bytes=resident,
                effective_weight_bits=8 * resident / (VOCAB * HIDDEN),
                allocation_and_sparse_upload_s=prepare,
                requested_weight_bytes=m * (HIDDEN // 2 + HIDDEN // 128 * 3)
                if mode == "u4"
                else m * HIDDEN * 2,
                byte_note="Unique format bytes per requested row; repeated IDs/cache reuse may reduce DRAM; no hardware byte counter",
            )
            case["budget_ms"] = {1: 0.005, 512: 0.1, 2048: 0.4, 8192: 1.6}.get(m)
            if case["budget_ms"]:
                case["budget_met"] = case["hot"]["median_ms"] <= case["budget_ms"]
            report["cases"].append(case)
            write_json(out / "result.json", report)
        # M512 real prompt has only21 distinct rows; separately expose a512-row
        # varied-ID workload so the repeated prompt cannot hide memory cost.
        case = checked(
            kernel, table, replacements[:512], ref, row_map, replacements, args.repetitions
        )
        case.update(
            mode=mode,
            block=block,
            workload="512 distinct sampled real rows",
            persistent_weight_bytes=resident,
            effective_weight_bits=8 * resident / (VOCAB * HIDDEN),
            budget_ms=0.1,
            budget_met=case["hot"]["median_ms"] <= 0.1,
            requested_weight_bytes=512 * (HIDDEN // 2 + HIDDEN // 128 * 3)
            if mode == "u4"
            else 512 * HIDDEN * 2,
        )
        report.setdefault("varied_512_cases", []).append(case)
        del table, ref, gpu_ids, kernels
        gc.collect()
        torch.cuda.empty_cache()
    # Non-aligned hidden, weight padding, negative zero and numeric cast checks.
    report["padding_cases"] = []
    for h, stride in ((129, 136), (513, 520)):
        raw_small = torch.randn((17, h), dtype=torch.float32).to(torch.bfloat16)
        raw_small[0, 0] = -0.0
        raw_small[1].zero_()
        small_ids = [0, 16, 0, 1, 8, 2, 16]
        for mode in ("float16", "bfloat16", "u4"):
            if mode == "u4":
                pp, ss, zz, qq = pack_u4(raw_small)
                assert torch.equal(torch.stack((pp & 15, pp >> 4), -1).reshape(qq.shape), qq)
                table = [v.cuda() for v in (pp, ss, zz)]
                ref = dequant(pp, ss, zz, h).cuda()
                kernel = embedding_u4(vocab=17, hidden=h, block=128)
            else:
                weight = torch.full((17, stride), float("nan"), dtype=getattr(torch, mode))
                weight[:, :h] = raw_small.to(getattr(torch, mode))
                table = [weight.cuda()]
                ref = raw_small.half().cuda()
                kernel = embedding_gather(vocab=17, hidden=h, stride=stride, dtype=mode, block=256)
            # The large-table graph mutation has sign-preserving +/-zero math;
            # run direct bit checks here because zero rows can change zero sign.
            i = torch.tensor(small_ids, device="cuda", dtype=torch.int32)
            y = torch.empty((7, h), device="cuda", dtype=torch.float16)
            kernel(*table, i, y, stream=torch.cuda.current_stream().cuda_stream)
            result = exact(y, ref[i.long()])
            result.update(mode=mode, hidden=h, stride=stride)
            report["padding_cases"].append(result)
    report["memory"] = {
        "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_torch_reserved_bytes": torch.cuda.max_memory_reserved(),
        "torch_stats_scope": "Does not count external context/module/graph driver allocations",
    }
    report["runner"] = identity(Path(__file__))
    write_json(out / "result.json", report)
    print("op01 complete; wrapper cleanup releases GPU lock", flush=True)


if __name__ == "__main__":
    main()
