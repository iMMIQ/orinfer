"""Compare KV storage on identical weights and teacher-forced token histories.

This isolates KV error; it does not replace a BF16/FP8 weight-quality benchmark.
"""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def compare(reference, candidate, reference_requests, candidate_requests):
    baseline = {r["id"]: r for r in reference["requests"]}
    observed = {r["id"]: r for r in candidate["requests"]}
    requests = {r["id"]: r for r in reference_requests["requests"]}
    rows = []
    for request in candidate_requests["requests"]:
        if not request.get("logits_steps"):
            continue
        key = request["id"]
        base = baseline[key]
        cand = observed[key]
        original = requests[key]
        if (
            request["input_tokens"] != original["input_tokens"]
            or request.get("forced_tokens")
            != base["output_tokens"][: request["max_new_tokens"] - 1]
        ):
            raise ValueError(f"{key}: a fixed reference history is required")
        for step in request["logits_steps"]:
            if step not in original.get("logits_steps", []):
                raise ValueError("Reference position missing")

            def load(report):
                path = next(
                    Path(p) for p in report["logits_files"] if Path(p).name == f"{key}-{step}.f32"
                )
                data = path.read_bytes()
                logits = np.frombuffer(data, dtype="<f4").astype(np.float64)
                if not len(logits) or not np.isfinite(logits).all():
                    raise ValueError("Invalid logits")
                logits -= logits.max()
                logprob = logits - np.log(np.exp(logits).sum())
                return logprob, hashlib.sha256(data).hexdigest()

            ref, rhash = load(base)
            cur, chash = load(cand)
            if ref.shape != cur.shape:
                raise ValueError("Vocabulary mismatch")
            top = np.argsort(ref)[-3:][::-1]
            ctop = np.argsort(cur)[-3:][::-1]
            target = base["output_tokens"][step]
            history = request["input_tokens"] + request["forced_tokens"][:step]
            row = dict(
                request=key,
                position=step,
                context_tokens=len(history),
                history_sha256=hashlib.sha256(
                    json.dumps(history, separators=(",", ":")).encode()
                ).hexdigest(),
                reference_logits_sha256=rhash,
                candidate_logits_sha256=chash,
                kl_nats=float((np.exp(ref) * (ref - cur)).sum()),
                reference_token_nll_delta=float(ref[target] - cur[target]),
                top1_agrees=bool(top[0] == ctop[0]),
                top3_overlap=len(set(top) & set(ctop)) / 3,
                max_top3_probability_error=float(np.abs(np.exp(ref[top]) - np.exp(cur[top])).max()),
                reference_top3=[
                    dict(
                        token=int(i),
                        probability=float(np.exp(ref[i])),
                        candidate_probability=float(np.exp(cur[i])),
                    )
                    for i in top
                ],
                candidate_top3=[
                    dict(token=int(i), probability=float(np.exp(cur[i]))) for i in ctop
                ],
            )
            rows.append(row)
    if not rows:
        raise ValueError("No same-history positions")
    return dict(
        seed=20261002,
        scope="KV storage only; same weights, FP16 KV reference; teacher-forced histories",
        positions=len(rows),
        mean_kl_nats=float(np.mean([r["kl_nats"] for r in rows])),
        max_kl_nats=max(r["kl_nats"] for r in rows),
        mean_reference_token_nll_delta=float(
            np.mean([r["reference_token_nll_delta"] for r in rows])
        ),
        top1_agreement=float(np.mean([r["top1_agrees"] for r in rows])),
        mean_top3_overlap=float(np.mean([r["top3_overlap"] for r in rows])),
        mean_top3_probability_error=float(np.mean([r["max_top3_probability_error"] for r in rows])),
        max_top3_probability_error=max(r["max_top3_probability_error"] for r in rows),
        rows=rows,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "candidate", "reference-requests", "candidate-requests", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()

    def read(path):
        return json.loads(path.read_text())

    result = compare(
        read(a.reference), read(a.candidate), read(a.reference_requests), read(a.candidate_requests)
    )
    with a.output.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
