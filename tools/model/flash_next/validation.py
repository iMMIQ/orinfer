"""CPU identities, fixed scenes and state guards for native Flash validation."""

import math

import numpy as np

from tools.eval.scoring_common import SEED, context_hash, validate_probes


CASES = (
    ("zh", "只输出三个汉字：中国的首都是哪里？", "北京市"),
    ("en", "Answer with one word: what is the capital of France?", "Paris"),
    ("math", "只输出整数：17乘以23等于多少？", "391"),
    ("json", "只输出JSON：一个对象，name为Alice，age为30。", '{"name":"Alice","age":30}'),
    (
        "code",
        "Only output Python code defining add(a, b) which returns their sum.",
        "def add(a, b):\n    return a + b",
    ),
    ("read", "记录：甲有12本书，乙有7本书。只输出乙的书本数量。", "7"),
)


def validate_snapshot(
    snapshot, states, capacity, vocab_size, history_limit, *, prefix_divisors=None, allow_cpu=False
):
    """Validate the entire snapshot before a single live state is mutated."""
    if not isinstance(snapshot, dict) or set(snapshot) != {"states", "history", "position"}:
        raise ValueError("Invalid native prefix snapshot fields")
    position, history = snapshot["position"], snapshot["history"]
    if type(position) is not int or not 0 <= position <= capacity:
        raise ValueError("Invalid native prefix position")
    if (
        not isinstance(history, list)
        or len(history) > min(history_limit, position)
        or any(type(token) is not int or not 0 <= token < vocab_size for token in history)
    ):
        raise ValueError("Invalid private PLE history")
    saved = snapshot["states"]
    if not isinstance(saved, dict) or set(saved) != set(states):
        raise ValueError("Invalid prefix state set")
    prefix_divisors = prefix_divisors or {}
    for name, value in saved.items():
        live = states[name]
        expected = live.shape
        if name in prefix_divisors:
            expected = (position // prefix_divisors[name], *live.shape[1:])
        device_matches = value.device == live.device or allow_cpu and str(value.device) == "cpu"
        if value.shape != expected or value.dtype != live.dtype or not device_matches:
            raise ValueError("Prefix state geometry or device changed")


def probe(logits, prompt_ids, targets, position, case_id, query_ids=()):
    """Score an actual next-token distribution with an exact forced-history ID."""
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 1 or not np.isfinite(logits).all():
        raise ValueError("Expected finite vector of vocabulary logits")
    if not 0 <= position < len(targets):
        raise ValueError("Invalid teacher-forcing position")
    reference = targets[position]
    ids = set(query_ids) | {reference}
    if any(type(i) is not int or not 0 <= i < len(logits) for i in ids):
        raise ValueError("Invalid probability query token")
    log_z = float(logits.max()) + math.log(float(np.exp(logits - logits.max()).sum()))
    order = np.argsort(-logits, kind="stable")[:3]
    return {
        "case_id": case_id,
        "execution_mode": "flash-native",
        "position": position,
        "seed": SEED,
        "context_sha256": context_hash(prompt_ids + targets[:position]),
        "reference_token_id": reference,
        "reference_logprob": float(logits[reference] - log_z),
        "top3": [{"token_id": int(i), "logprob": float(logits[i] - log_z)} for i in order],
        "queried_logprobs": {str(i): float(logits[i] - log_z) for i in sorted(ids)},
    }


def baseline_probes(report, quantization, cases):
    """Require original precision and the exact pinned model/forced histories."""
    if report.get("complete") is not True or report.get("baseline_precision") not in (
        "original-bf16",
        "original-fp8",
    ):
        raise ValueError("Completed original BF16/FP8 reference required")
    contract = report.get("contract", {})
    if (contract.get("source"), contract.get("revision"), contract.get("seed")) != (
        quantization.get("source"),
        quantization.get("source_revision"),
        SEED,
    ):
        raise ValueError("Baseline refers to different original weights or seed")
    probes = report["probes"]
    ids = {(row["case_id"], row["position"]) for row in probes}
    if len(ids) != len(probes):
        raise ValueError("Duplicate original probe identity")
    result = []
    for case in cases:
        rows = [row for row in probes if row["case_id"] == case["id"]]
        if not rows:
            raise ValueError("Baseline lacks selected scene")
        result.extend(
            validate_probes(
                rows, case["prompt_ids"], case["target_ids"], case["id"], rows[0]["execution_mode"]
            )
        )
    return result
