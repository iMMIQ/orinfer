"""Full-vocabulary same-history top3 comparison to frozen native AWQ reference."""

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True, type=Path)
    ap.add_argument("--baseline", required=True, type=Path)
    ap.add_argument("--requests", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    a = ap.parse_args()
    report = json.loads(a.report.read_text())
    baseline = json.loads(a.baseline.read_text())
    requests = json.loads(a.requests.read_text())
    rows = []
    for request in requests["requests"]:
        if not request.get("logits_steps"):
            continue
        case = next(c for c in baseline["cases"] if c["prompt_ids"] == request["input_tokens"])
        assert (
            request.get("forced_tokens", []) == case["output_ids"][: request["max_new_tokens"] - 1]
        ), "A fixed teacher-forced history is required"
        observed = next(q for q in report["requests"] if q["id"] == request["id"])
        for step in request["logits_steps"]:
            source = case["decode"][step]
            path = Path(
                next(
                    p
                    for p in observed["logits_files"]
                    if Path(p).name == f"{request['id']}-{step}.f32"
                )
            )
            raw = path.read_bytes()
            assert len(raw) == 248320 * 4
            logits = struct.unpack("<248320f", raw)
            assert all(math.isfinite(v) for v in logits)
            maximum = max(logits)
            lse = maximum + math.log(sum(math.exp(v - maximum) for v in logits))
            top = sorted(range(len(logits)), key=lambda i: (-logits[i], i))[:3]
            ids = request["input_tokens"] + request.get("forced_tokens", [])[:step]
            history = {"token_ids": ids, "positions": list(range(len(ids)))}
            digest = hashlib.sha256(
                json.dumps(history, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            assert digest == source["context_sha256"], "Reference history mismatch"
            candidate = {str(x["token_id"]): logits[x["token_id"]] - lse for x in source["top3"]}
            rows.append(
                {
                    "request": request["id"],
                    "position": step,
                    "context_sha256": digest,
                    "baseline_top3": source["top3"],
                    "candidate_top3": [{"token_id": i, "logprob": logits[i] - lse} for i in top],
                    "candidate_logprobs_on_baseline_top3": candidate,
                    "top1_agrees": top[0] == source["top3"][0]["token_id"],
                    "top3_overlap": len(set(top) & {x["token_id"] for x in source["top3"]}) / 3,
                    "max_top3_probability_error": max(
                        abs(math.exp(candidate[str(x["token_id"])]) - math.exp(x["logprob"]))
                        for x in source["top3"]
                    ),
                    "logits_sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
    assert rows
    result = {
        "seed": 20261002,
        "scope": "Frozen native community AWQ same-history decode reference; not official BF16 quantization acceptance",
        "baseline_sha256": hashlib.sha256(a.baseline.read_bytes()).hexdigest(),
        "report_sha256": hashlib.sha256(a.report.read_bytes()).hexdigest(),
        "positions": len(rows),
        "top1_agreement": sum(r["top1_agrees"] for r in rows) / len(rows),
        "mean_top3_overlap": sum(r["top3_overlap"] for r in rows) / len(rows),
        "max_top3_probability_error": max(r["max_top3_probability_error"] for r in rows),
        "rows": rows,
    }
    with a.output.open("x") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
