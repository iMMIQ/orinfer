"""Offline same-logits full-vocabulary op26 reference, ABI and graph checks."""

import argparse
import gc
import json
import math
import time
from pathlib import Path
from tools.reference import CHECKPOINT

import torch

from common import ROOT, benchmark, configure, environment, export_kernel, identity, write_json
from abi import parse_host
from kernels.operators.op26_probability_score import (
    VOCAB,
    launch,
    probability_merge,
    probability_partials,
    validate_query_ids,
)
from tools.eval.scoring_common import context_hash, pair_probes, validate_probes
from tools.eval.quick_quality import analyze

CHUNK = 4096
BLOCKS = (VOCAB + CHUNK - 1) // CHUNK
DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def allocate(x, ids):
    m, q = ids.shape
    return (
        torch.empty((m, BLOCKS, 3), device="cuda", dtype=torch.float32),
        torch.empty(m, device="cuda", dtype=torch.float32),
        torch.empty((m, q), device="cuda", dtype=torch.float32),
        torch.empty((m, q), device="cuda", dtype=torch.float32),
        torch.empty(m, device="cuda", dtype=torch.int32),
        torch.empty((m, q), device="cuda", dtype=torch.int32),
    )


def reference(x, ids):
    # Double reference over the ENTIRE vocab, bounded-size batches, no top3 norm.
    ls, lp, pr = [], [], []
    for start in range(0, x.shape[0], 16):
        z = x[start : start + 16].double()
        mx = z.max(dim=1).values
        logmass = torch.log(torch.exp(z - mx[:, None]).sum(dim=1))
        selected = z.gather(1, ids[start : start + 16].long())
        scores = (selected - mx[:, None]) - logmass[:, None]
        ls.append((mx + logmass).float())
        lp.append(scores.float())
        pr.append(scores.exp().float())
    return torch.cat(ls), torch.cat(lp), torch.cat(pr)


def score_error(actual, reference):
    # Offline FP64 metric accumulation prevents huge finite test logits from
    # overflowing error norms; production buffers/arithmetic remain FP32.
    a, b = actual.double(), reference.double()
    delta = a - b
    norm = float(b.norm())
    return {
        "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
        "relative_l2": float(delta.norm()) / max(norm, 1e-30),
        "reference_l2": norm,
        "max_abs": float(delta.abs().max()),
        "rms_abs": float(delta.square().mean().sqrt()),
    }


def check(buffers, refs):
    _, lse, lp, prob, rs, qs = buffers
    assert not bool(rs.any()) and not bool(qs.any()), (rs, qs)
    errs = {
        "lse": score_error(lse, refs[0]),
        "logprob": score_error(lp, refs[1]),
        "prob": score_error(prob, refs[2]),
    }
    assert all(e["finite"] for e in errs.values()), errs
    assert errs["logprob"]["max_abs"] <= 0.001, errs
    assert errs["prob"]["max_abs"] <= 2e-6, errs
    assert errs["prob"]["relative_l2"] <= 0.0001, errs
    # Huge common offsets make absolute LSE error unhelpful; FP32 representation.
    assert torch.allclose(lse, refs[0], rtol=2e-7, atol=2e-5), errs
    return errs


def queries(x, q=4):
    top = x.topk(3, dim=1).indices.int()
    target = torch.full((x.shape[0], 1), VOCAB - 1, device="cuda", dtype=torch.int32)
    ids = torch.cat((top, target), 1)
    if q > 4:
        extra = torch.arange(q - 4, device="cuda", dtype=torch.int32)[None].expand(x.shape[0], -1)
        ids = torch.cat((ids, extra), 1).contiguous()
    validate_query_ids(ids.cpu().tolist())
    return ids


def checked_case(kernels, x, ids, repetitions):
    buffers = allocate(x, ids)

    def run():
        return launch(*kernels, x, ids, *buffers, stream=torch.cuda.current_stream().cuda_stream)

    started = time.perf_counter()
    run()
    torch.cuda.synchronize()
    first = time.perf_counter() - started
    refs = reference(x, ids)
    errs = check(buffers, refs)
    timing, graph = benchmark(run, repetitions=repetitions, calls_per_replay=16)
    original = tuple(t.clone() for t in buffers[1:])
    # Poison every workspace/output, prove replay actually computes all buffers.
    for t in buffers:
        t.fill_(float("nan") if t.is_floating_point() else -123)
    graph.replay()
    torch.cuda.synchronize()
    assert all(torch.equal(t, saved) for t, saved in zip(buffers[1:], original))
    saved_ids = ids.clone()
    saved_tail = x[:, -1].clone()
    # Mutate all logits reversibly plus one queried tail logit; tail is masked in partial.
    x.neg_()
    x[:, -1].add_(0.25)
    ids[:, :3].copy_(x.topk(3, dim=1).indices.int())
    ids[:, 3] = 0
    changed_refs = reference(x, ids)
    for t in buffers:
        t.fill_(float("nan") if t.is_floating_point() else -123)
    graph.replay()
    torch.cuda.synchronize()
    changed_errs = check(buffers, changed_refs)
    assert not torch.equal(buffers[2], original[1]), "changed input/query graph not consumed"
    x.neg_()
    x[:, -1].copy_(saved_tail)
    ids.copy_(saved_ids)
    restored_refs = reference(x, ids)
    graph.replay()
    torch.cuda.synchronize()
    restored = check(buffers, restored_refs)
    assert all(torch.equal(t, saved) for t, saved in zip(buffers[1:], original))
    return {
        "errors": errs,
        "changed_errors": changed_errs,
        "restored_errors": restored,
        "graph": {
            "poison_all_buffers": True,
            "changed_logits": True,
            "changed_query_ids": True,
            "restore_bit_exact": True,
            "nodes_per_replay": 32,
        },
        "first_launch_s": first,
        "timing": timing,
        "ms_per_scored_row": timing["median_ms"] / x.shape[0],
        "target_ms_per_row": 0.080,
        "target_met": timing["median_ms"] / x.shape[0] <= 0.080,
        "workspace_bytes": buffers[0].numel() * 4,
        "weights_bytes": 0,
        "input_bytes": x.numel() * x.element_size() + ids.numel() * 4,
        "output_bytes": sum(t.numel() * t.element_size() for t in buffers[1:]),
    }


def export(kernel, path, stage, dtype):
    data = export_kernel(kernel, path)
    host = (path / "host.txt").read_text()
    abi = {
        "operator": "op26_probability_score",
        "stage": stage,
        "actual_host_abi": parse_host(host),
        "tensor_layout": "contiguous row major",
        "logits_dtype": dtype,
        "output_dtype": "float32 except statuses int32",
        "vocab": VOCAB,
        "chunk": CHUNK,
        "partial_blocks": BLOCKS,
        "M": "runtime int32 rows",
        "Q": "runtime int32 queries (merge only)",
        "SM": 87,
        "cooperative_launch": False,
        "persistent_weights_bytes": 0,
        "workspace_bytes": f"M*{BLOCKS}*3*4",
        "alias_contract": "all tensors disjoint; stable addresses and fixed shapes per graph",
        "stream": "explicit current/capture stream resolved at every invocation",
        "status_contract": "RS=1 nonfinite row; QS bits 1 row invalid,2 ID invalid,4 FP32 logprob overflow; must check",
        "toolchain": environment(),
        **data,
    }
    write_json(path / "abi.json", abi)
    return abi


def rejected_cpu():
    bad = [[], [[]], [[-1]], [[VOCAB]], [[1.0]], [[True]], [[1], [1, 2]], [1]]
    result = []
    for ids in bad:
        try:
            validate_query_ids(ids)
        except ValueError as exc:
            result.append({"ids": ids, "rejection": str(exc)})
        else:
            raise AssertionError(f"accepted {ids}")
    assert validate_query_ids([[VOCAB - 1, 0, 0, 4095]])
    return result


def invalid_device(kernels, dtype):
    x = torch.zeros((5, VOCAB), dtype=dtype, device="cuda")
    x[0, 0] = float("nan")
    x[1, VOCAB - 1] = float("inf")
    x[2, 4096] = -float("inf")
    ids = torch.tensor(
        [[0, 1, -1, VOCAB], [0, 1, 2, 3], [0, 1, 2, 3], [-1, VOCAB, 0, VOCAB - 1], [0, 1, 1, 2]],
        dtype=torch.int32,
        device="cuda",
    )
    b = allocate(x, ids)
    launch(*kernels, x, ids, *b, stream=torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    assert b[4].cpu().tolist() == [1, 1, 1, 0, 0]
    assert b[5].cpu().tolist() == [[1, 1, 3, 3], [1] * 4, [1] * 4, [2, 2, 0, 0], [0] * 4]
    rejected = b[5] != 0
    assert bool(torch.isneginf(b[2][rejected]).all()) and bool((b[3][rejected] == 0).all())
    assert not bool(torch.isnan(b[1]).any()) and not bool(torch.isnan(b[2]).any())
    return {
        "nan_positive_negative_inf_rejected": True,
        "invalid_id_guarded": True,
        "row_status": b[4].cpu().tolist(),
        "query_status": b[5].cpu().tolist(),
        "no_nan_output": True,
        "duplicate_query_exact": bool(b[2][4, 1] == b[2][4, 2]),
    }


def protocol_case(kernels, output):
    baseline = torch.randn((1, VOCAB), dtype=torch.float32, device="cuda")
    baseline.clamp_(-7, 7)
    baseline[0, :3] = 8
    candidate = baseline.clone()
    candidate[0, :3] = -7
    candidate[0, 10:13] = 9
    ids = torch.tensor([[0, 1, 2, VOCAB - 1]], dtype=torch.int32, device="cuda")
    bufs = allocate(candidate, ids)
    launch(*kernels, candidate, ids, *bufs, stream=torch.cuda.current_stream().cuda_stream)
    check(bufs, reference(candidate, ids))
    # This synthetic candidate has three exactly tied largest logits. Apply
    # the scoring protocol's smallest-token-ID tie break in the offline adapter.
    top = candidate.topk(3, dim=1).indices.sort(dim=1).values.int()
    other = allocate(candidate, top)
    launch(*kernels, candidate, top, *other, stream=torch.cuda.current_stream().cuda_stream)
    check(other, reference(candidate, top))
    baseref = reference(baseline, ids)
    meta = {
        "case_id": "op26-synthetic-teacher-forced",
        "execution_mode": "prefill",
        "position": 0,
        "seed": 20261002,
        "context_sha256": context_hash([100, 200]),
        "reference_token_id": VOCAB - 1,
    }
    a = dict(
        meta,
        reference_logprob=float(baseref[1][0, 3]),
        top3=[{"token_id": i, "logprob": float(baseref[1][0, i])} for i in range(3)],
        queried_logprobs={str(int(i)): float(lp) for i, lp in zip(ids[0], baseref[1][0])},
    )
    b = dict(
        meta,
        reference_logprob=float(bufs[2][0, 3]),
        top3=[{"token_id": int(i), "logprob": float(lp)} for i, lp in zip(top[0], other[2][0])],
        queried_logprobs={str(int(i)): float(lp) for i, lp in zip(ids[0], bufs[2][0])},
    )
    validate_probes([a], [100, 200], [VOCAB - 1], meta["case_id"], "prefill")
    validate_probes([b], [100, 200], [VOCAB - 1], meta["case_id"], "prefill")
    paired = pair_probes([a], [b])
    report = analyze(paired)
    write_json(
        output / "synthetic-quality-protocol.json",
        {
            "baseline": a,
            "candidate": b,
            "paired": paired,
            "diagnostics": report,
            "scope": "synthetic protocol validation only; no model/task quality conclusion",
        },
    )
    assert set(top[0].cpu().tolist()).isdisjoint([0, 1, 2])
    assert sum(math.exp(t["logprob"]) for t in b["top3"]) < 0.5
    return {
        "baseline_ids_outside_candidate_top3_queried": True,
        "full_vocab_normalized": True,
        "temperature": 0,
        "repetition_penalty": 1,
        "presence_frequency_penalty": 0,
        "teacher_forced_history_identity_checked": True,
        "synthetic_only": True,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--repetitions", type=int, default=20)
    args = ap.parse_args()
    output = Path(args.output)
    configure()
    lock = ROOT / "artifacts/reference/reference-lock.json"
    config = CHECKPOINT / "config.json"
    cfg = json.loads(config.read_text())
    assert cfg["text_config"]["vocab_size"] == VOCAB
    results = {
        "status": "in_progress",
        "environment": environment(),
        "cases": [],
        "exports": [],
        "cpu_rejected": rejected_cpu(),
        "binding": {
            "config": identity(config),
            "reference_lock": identity(lock),
            "vocab": VOCAB,
            "weights_loaded": False,
            "input_origin": "seeded synthetic logits; no real-model quality claim",
        },
    }
    kernels = {}
    for dtype, tdtype in DTYPES.items():
        start = time.perf_counter()
        partial = probability_partials(dtype=dtype)
        partial_compile = time.perf_counter() - start
        start = time.perf_counter()
        merge = probability_merge(dtype=dtype)
        merge_compile = time.perf_counter() - start
        kernels[dtype] = (partial, merge)
        for stage, kernel, secs in [
            ("partial", partial, partial_compile),
            ("merge", merge, merge_compile),
        ]:
            results["exports"].append(
                {
                    "dtype": dtype,
                    "stage": stage,
                    "prepare_compile_s": secs,
                    "abi": export(kernel, output / "aot" / dtype / stage, stage, dtype),
                }
            )
        for m in (1, 2, 3, 4, 5, 7, 8, 512):
            x = (torch.randn((m, VOCAB), device="cuda") * 3).to(tdtype)
            ids = queries(x, q=4 if m != 7 else 9)
            out = checked_case(
                kernels[dtype], x, ids, min(args.repetitions, 5) if m == 512 else args.repetitions
            )
            results["cases"].append(
                {"dtype": dtype, "M": m, "Q": ids.shape[1], "kind": "random", **out}
            )
            print(
                f"{dtype} M{m} Q{ids.shape[1]} LPerr={out['errors']['logprob']['max_abs']:.3g} ms={out['timing']['median_ms']:.6f}",
                flush=True,
            )
            write_json(output / "results.json", results)
            del x, ids
            gc.collect()
        x = torch.randn((3, VOCAB), dtype=tdtype, device="cuda")
        ids = queries(x, q=129)
        out = checked_case(kernels[dtype], x, ids, args.repetitions)
        results["cases"].append(
            {"dtype": dtype, "M": 3, "Q": 129, "kind": "query_thread_tail", **out}
        )
        del x, ids
        # Uniform ties, tail-only peak, extreme finite offsets/ranges, chunk boundaries.
        x = torch.zeros((8, VOCAB), dtype=tdtype, device="cuda")
        x[1].fill_(60000)
        x[2].fill_(-60000)
        x[3].fill_(-1000)
        x[3, -1] = 1000
        x[4].fill_(10)
        x[4, [4095, 4096, VOCAB - 1]] = 20
        x[5].fill_(-100)
        x[5, 0] = 100
        if dtype == "float32":
            x[6].fill_(1e30)
            x[7].fill_(-1e30)
            x[7, 0] = 1e30
        else:
            x[6].fill_(0.0001)
            x[7].fill_(-0.0001)
        ids = torch.tensor([[0, 4095, 4096, VOCAB - 1]] * 8, dtype=torch.int32, device="cuda")
        # Huge positive/negative extremes differ from graph mutation by <ULP; random rows still change.
        out = checked_case(kernels[dtype], x, ids, args.repetitions)
        results["cases"].append(
            {"dtype": dtype, "M": 8, "Q": 4, "kind": "ties_tail_extreme_finite", **out}
        )
        results.setdefault("invalid_device", {})[dtype] = invalid_device(kernels[dtype], tdtype)
        write_json(output / "results.json", results)
        del x, ids
        gc.collect()
    # Explicit finite-logit subtraction overflow policy (query status bit4).
    x = torch.zeros((1, VOCAB), dtype=torch.float32, device="cuda")
    x[0, 0] = 3e38
    x[0, 1] = -3e38
    ids = torch.tensor([[0, 1, 2, VOCAB - 1]], device="cuda", dtype=torch.int32)
    b = allocate(x, ids)
    launch(*kernels["float32"], x, ids, *b, stream=torch.cuda.current_stream().cuda_stream)
    assert b[4].item() == 0 and b[5][0, 1].item() == 4
    assert torch.isneginf(b[2][0, 1]) and b[3][0, 1] == 0
    results["finite_logprob_overflow_explicit_status"] = True
    results["eval_protocol"] = protocol_case(kernels["float32"], output)
    results["implementation_identity"] = [
        identity(ROOT / "kernels/operators/op26_probability_score.py"),
        identity(Path(__file__)),
    ]
    results["memory"] = {
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    results["status"] = (
        "passed isolated full-vocab numerical/status/graph/protocol tests; synthetic only"
    )
    write_json(output / "results.json", results)
    print("op26 complete; wrapper cleanup releases GPU lock", flush=True)


if __name__ == "__main__":
    main()
