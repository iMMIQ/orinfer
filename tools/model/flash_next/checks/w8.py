"""Paired complete-chain checks for unchanged-weight Flash Next W8 kernels.

Compilation, prefill and prefix restore are outside decode timings. All CPU
work and MTP draft/verify/commit/refresh are inside them. Results are offline
request measurements, not online serving or a full model-quality benchmark.
"""

import argparse
import gc
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from tools.model.flash_next.checkpoint import Checkpoint
from tools.model.flash_next.mtp import Session
from tools.model.flash_next.native import Model, generate, prefill, state_checks
from tools.model.flash_next.policy import code_vocabulary
from tools.model.flash_next.cuda_profile import PhaseTrace
from tools.model.flash_next.scenes import scenes
from tools.model.flash_next.validation import baseline_probes, probe
from tools.model.flash_next.workloads import requests, session_state
from tools.operators.common import configure, environment, identity, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("artifacts/models/flash-next-e8p-a8")
    )
    parser.add_argument(
        "--draft", type=Path, default=Path("artifacts/models/flash-next-e8p-a8-mtp")
    )
    parser.add_argument("--compile-cache", type=Path)
    parser.add_argument("--bf16-reference", type=Path)
    parser.add_argument("--decode", type=int, default=256)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--trace-only", action="store_true")
    args = parser.parse_args()
    configure()
    if args.decode < 32 or args.trials < 1:
        parser.error("Decode budget must be >=32 and trials positive")
    if args.compile_cache:
        cache = args.output / "cache/0.1.15"
        if cache.exists():
            if any(p.is_file() for p in cache.rglob("*")):
                raise ValueError("Refusing to replace a populated compile cache")
            shutil.rmtree(cache)
        cache.symlink_to(args.compile_cache.resolve(), target_is_directory=True)
    files = (
        "tools/model/flash_next/native.py",
        "kernels/model/int8_swiglu.py",
        "kernels/model/int8_projection.py",
        "tools/model/flash_next/mtp.py",
        "tools/model/flash_next/policy.py",
        "tools/model/flash_next/checks/w8.py",
    )
    report = dict(
        complete=False,
        seed=20261002,
        scope=__doc__,
        environment=environment(),
        sources=[identity(p) for p in files],
        decode=[],
        prefill=[],
        quality=[],
        profiles=[],
    )

    def save():
        write_json(args.output / "results.json", report)

    save()
    started = time.perf_counter()
    source = Checkpoint(args.checkpoint, verify_hashes=False)
    target = Model(source, 262144, args.output / "target", decode_w8_optimized=False)
    draft = Model(
        Checkpoint(args.draft, verify_hashes=False),
        262144,
        args.output / "draft",
        target=target,
        decode_w8_optimized=False,
    )
    tokenizer, cases = requests(args.checkpoint, [2048, 8192], ["python"], [False, True])
    vocab = code_vocabulary(tokenizer, Path.cwd(), 65536)
    draft.set_draft_vocab(vocab)
    write_json(args.output / "draft-vocabulary.json", vocab)
    session = Session(target, draft)
    eos = json.loads((args.checkpoint / "generation_config.json").read_text())["eos_token_id"]
    eos = {eos} if isinstance(eos, int) else set(eos)
    report["load_s"] = time.perf_counter() - started
    save()
    print("LOADED", report["load_s"], flush=True)
    references, exact_states, probe_logits = {}, {}, {}
    _, quality_cases = scenes(args.checkpoint)
    bf16 = (
        baseline_probes(
            json.loads(args.bf16_reference.read_text()),
            source.config["quantization_config"],
            quality_cases,
        )
        if args.bf16_reference
        else []
    )
    bf16 = {(p["case_id"], p["position"]): p for p in bf16}

    for optimized in (False, True):
        mode = "optimized" if optimized else "baseline"
        for owner in (target, draft):
            owner.reset()
            owner.transaction = owner.last_plan = None
            owner.plans.clear()
            owner.workspaces.clear()
            owner.prefill_workspace_rows = 0
            owner.decode_w8_optimized = optimized
        gc.collect()
        torch.cuda.empty_cache()
        if not args.trace_only:
            for case in quality_cases:
                target.reset()
                logits = prefill(target, case["prompt_ids"], 2048)
                for position, token in enumerate(case["target_ids"]):
                    key = case["id"], position
                    if not optimized:
                        probe_logits[key] = logits.copy()
                    assert np.array_equal(logits, probe_logits[key]), (mode, key, "logits")
                    queries = [p["token_id"] for p in bf16[key]["top3"]] if key in bf16 else []
                    score = probe(
                        logits,
                        case["prompt_ids"],
                        case["target_ids"],
                        position,
                        case["id"],
                        queries,
                    )
                    report["quality"].append(
                        dict(
                            mode=mode,
                            logits_exact=True,
                            chosen_in_BF16_top3=score["top3"][0]["token_id"] in queries
                            if queries
                            else None,
                            reference_logprob=score["reference_logprob"],
                            case=case["id"],
                            position=position,
                        )
                    )
                    if position + 1 < len(case["target_ids"]):
                        logits = target.execute([token])
            report.setdefault("state_checks", []).append(
                dict(mode=mode, result=state_checks(target, quality_cases[0]["prompt_ids"][:7]))
            )
            save()
        session.warm(727, (3, 7))
        for case in cases[:1] if args.trace_only else cases:
            session.prefill(case["prompt_ids"])
            selected = session.pending
            torch.cuda.synchronize()
            begin = time.perf_counter()
            session.prefill(case["prompt_ids"])
            torch.cuda.synchronize()
            seconds = time.perf_counter() - begin
            assert session.pending == selected
            report["prefill"].append(
                dict(
                    mode=mode,
                    case=case["id"],
                    tokens=len(case["prompt_ids"]),
                    prefill_s=seconds,
                    prefill_tps=len(case["prompt_ids"]) / seconds,
                )
            )
            prefix = session.snapshot(cpu=True)
            logits = target.last_plan["output"][-1].cpu().numpy().copy()
            if not optimized:
                references[case["id"]] = generate(target, logits, eos, args.decode)
            for depth in (7,) if args.trace_only else (0, 3, 7):

                def run(trace=None):
                    if depth:
                        return session.generate(
                            args.decode if trace is None else 64, eos, drafts=depth, trace=trace
                        )
                    return generate(target, logits, eos, args.decode)

                session.restore(prefix)
                tokens, reason = run()
                if not args.trace_only:
                    assert (tokens, reason) == references[case["id"]], (
                        mode,
                        case["id"],
                        depth,
                        "greedy",
                    )
                key = case["id"], depth
                states = session_state(session)
                if not optimized:
                    exact_states[key] = states
                assert states == exact_states[key], (mode, key, "private-state")
                for trial in range(0 if args.trace_only else args.trials):
                    session.restore(prefix)
                    torch.cuda.synchronize()
                    begin = time.perf_counter()
                    output, finish = run()
                    torch.cuda.synchronize()
                    seconds = time.perf_counter() - begin
                    assert (output, finish) == references[case["id"]]
                    assert session_state(session) == exact_states[key]
                    row = dict(
                        mode=mode,
                        case=case["id"],
                        drafts=depth,
                        trial=trial,
                        decode_tokens=len(output) - 1,
                        decode_s=seconds,
                        decode_tps=(len(output) - 1) / seconds,
                        greedy_exact=True,
                        private_state_exact=True,
                        gpu_bytes=torch.cuda.memory_allocated(),
                        statistics=dict(session.statistics) if depth else None,
                    )
                    report["decode"].append(row)
                    save()
                    print("DECODE", row, flush=True)
                if depth:
                    session.restore(prefix)
                    trace = PhaseTrace()
                    if args.trace_only:
                        torch.cuda.cudart().cudaProfilerStart()
                    run(trace)
                    torch.cuda.synchronize()
                    if args.trace_only:
                        torch.cuda.cudart().cudaProfilerStop()
                    report["profiles"].append(
                        dict(mode=mode, case=case["id"], drafts=depth, phases=trace.summary())
                    )
                    save()
            del prefix
            logits = None
            gc.collect()
    report["sources_unchanged"] = all(
        identity(s["path"])["sha256"] == s["sha256"] for s in report["sources"]
    )
    assert report["sources_unchanged"]
    report["complete"] = True
    save()
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
