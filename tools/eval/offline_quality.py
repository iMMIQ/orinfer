"""Prepare fixed-answer histories and pair independent offline scoring reports."""

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from tools.eval.scoring_common import SEED, pair_probes, validate_probes
from tools.eval.quick_quality import analyze

TARGETS = {
    "grounded_fact": "ORIN-64",
    "unknown_fact": "资料未说明",
    "arithmetic": "641",
    "json_format": '{"ok":true,"count":3}',
    "instruction_order": "BLUE",
    "table_lookup": "Chen,done",
    "python_code": "def stable_unique(xs):\n    seen = set()\n    result = []\n    for x in xs:\n        if x not in seen:\n            seen.add(x)\n            result.append(x)\n    return result",
    "rust_code": "fn sum_even(xs: &[i32]) -> i64 {\n    xs.iter().filter(|&&x| x % 2 == 0).map(|&x| i64::from(x)).sum()\n}",
    "english_summary": "The prototype uses Rust for scheduling, TileLang for GPU kernels, and offline weight preparation.",
    "translation": "The first request requires warm-up, while subsequent requests can reuse the cache.",
    "chat_state": "Orchid",
    "numeric_units": "12.5",
}


def long_context_case(tokenizer, budget):
    """Deterministic retrieval with a fact in the middle of a text prompt."""
    if not 256 <= budget <= 262144:
        raise ValueError("Long-context token budgets must be in 256..262144")

    def prompt(records):
        lines = [f"普通记录{i:06d}：测试数据，不含校验码。" for i in range(records)]
        lines.insert(records // 2, "唯一有效资料：校验码是CODE-641。")
        text = "只根据以下资料回答。\n" + "\n".join(lines) + "\n问题：校验码是什么？只输出校验码。"
        return tokenizer.apply_chat_template(
            [dict(role="user", content=text)],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False,
            enable_thinking=False,
        )

    low, high = 0, budget
    while low < high:
        middle = (low + high + 1) // 2
        if len(prompt(middle)) <= budget:
            low = middle
        else:
            high = middle - 1
    ids = prompt(low)
    if not ids or len(ids) > budget:
        raise ValueError("Retrieval prompt does not fit its context budget")
    return dict(
        id=f"context_retrieval_{budget}",
        prompt_ids=ids,
        target_ids=tokenizer.encode("CODE-641", add_special_tokens=False),
        query_ids=[],
    )


def prepare(checkpoint, fixtures, long_context_tokens=()):
    from transformers import AutoTokenizer

    if fixtures["seed"] != SEED:
        raise ValueError("The quality suite requires the fixed seed")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    cases = []
    for case in fixtures["cases"]:
        prompt = tokenizer.apply_chat_template(
            case["messages"],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False,
            enable_thinking=False,
        )
        if isinstance(prompt, Mapping):
            prompt = prompt["input_ids"]
        if (
            not isinstance(prompt, list)
            or not prompt
            or any(type(i) is not int or i < 0 for i in prompt)
        ):
            raise ValueError("Chat template must return one nonempty token-ID history")
        target_text = case.get("target_text", TARGETS.get(case["id"]))
        if not isinstance(target_text, str) or not target_text:
            raise ValueError(f"{case['id']}: provide a nonempty target_text")
        targets = tokenizer.encode(target_text, add_special_tokens=False)
        cases.append(dict(id=case["id"], prompt_ids=prompt, target_ids=targets, query_ids=[]))
    if len(set(long_context_tokens)) != len(long_context_tokens):
        raise ValueError("Duplicate long-context token budgets")
    cases.extend(long_context_case(tokenizer, n) for n in long_context_tokens)
    return dict(seed=fixtures["seed"], cases=cases)


def add_queries(requests, baseline):
    if not baseline.get("complete") or baseline["seed"] != requests["seed"]:
        raise ValueError("Reference is incomplete or uses another seed")
    expected = {c["id"] for c in requests["cases"]}
    if len(expected) != len(requests["cases"]) or any(
        p["case_id"] not in expected for p in baseline["probes"]
    ):
        raise ValueError("Duplicate request IDs or unexpected reference cases")
    for case in requests["cases"]:
        probes = [p for p in baseline["probes"] if p["case_id"] == case["id"]]
        validate_probes(
            probes,
            case["prompt_ids"],
            case["target_ids"],
            case["id"],
            "prefill",
            case.get("images", ()),
        )
        case["query_ids"] = [
            [i["token_id"] for i in p["top3"]] for p in sorted(probes, key=lambda p: p["position"])
        ]
    return requests


def compare(requests, baseline, candidate):
    add_queries(requests, baseline)
    if candidate["seed"] != requests["seed"]:
        raise ValueError("Candidate uses another seed")
    expected = {c["id"] for c in requests["cases"]}
    if any(p["case_id"] not in expected for p in candidate["probes"]):
        raise ValueError("Unexpected candidate cases")
    paired = []
    for case in requests["cases"]:
        a = [p for p in baseline["probes"] if p["case_id"] == case["id"]]
        b = [p for p in candidate["probes"] if p["case_id"] == case["id"]]
        a = validate_probes(
            a, case["prompt_ids"], case["target_ids"], case["id"], "prefill", case.get("images", ())
        )
        b = validate_probes(
            b, case["prompt_ids"], case["target_ids"], case["id"], "decode", case.get("images", ())
        )
        # Preserve both actual modes in the report. The pairing label names this
        # cross-path comparison; it does not claim BF16 autoregressive execution.
        a = [dict(p, execution_mode="decode_vs_bf16_prefill") for p in a]
        b = [dict(p, execution_mode="decode_vs_bf16_prefill") for p in b]
        paired.extend(pair_probes(a, b))
    return dict(
        analyze(paired),
        reference_execution=baseline["execution"],
        reference_revision=baseline["revision"],
        candidate_manifest_sha256=candidate["manifest_sha256"],
        candidate_execution="sequential teacher-forced decode after normal prefill",
        history_source="fixed valid answers; token divergence is diagnostic, not an accuracy gate",
        task_accuracy="Evaluate independent free generations; target NLL alone is insufficient",
        paired=paired,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["prepare", "queries", "compare"])
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--fixtures", type=Path, default=Path("fixtures/quick-quality-scenarios.json"))
    p.add_argument("--requests", type=Path)
    p.add_argument("--baseline", type=Path)
    p.add_argument("--candidate", type=Path)
    p.add_argument("--long-context-tokens", type=int, nargs="*", default=[])
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    if a.action == "prepare":
        result = prepare(a.checkpoint, json.loads(a.fixtures.read_text()), a.long_context_tokens)
    elif a.action == "queries":
        result = add_queries(json.loads(a.requests.read_text()), json.loads(a.baseline.read_text()))
    else:
        result = compare(
            json.loads(a.requests.read_text()),
            json.loads(a.baseline.read_text()),
            json.loads(a.candidate.read_text()),
        )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
