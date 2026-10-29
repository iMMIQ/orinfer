"""Gate paired native A2 complete requests by workload, without averaging away tails."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

from tools.build.files import atomic_json, file_hash


def compare(reference, candidate, maximum_regression=0.05):
    if not math.isfinite(maximum_regression) or not 0 <= maximum_regression <= 1:
        raise ValueError("Invalid regression bound")
    groups = [defaultdict(list), defaultdict(list)]
    metrics = ["ttft_s", "decode_s", "e2e_s"]
    for metric in ("request_ttft_s", "request_e2e_s"):
        present = [metric in q for r in (reference, candidate) for q in r["requests"]]
        if any(present):
            if not all(present):
                raise ValueError("Reference and candidate timing scopes differ")
            metrics.append(metric)
    for field in ("model_layers", "tensor_parallel", "capacity", "worker_cpus", "logits_dump"):
        if reference.get(field) != candidate.get(field):
            raise ValueError("Reference and candidate execution environments differ: " + field)
    for report, target in zip((reference, candidate), groups, strict=True):
        for request in report["requests"]:
            key = (
                request["name"],
                request["tokens_sha256"],
                request["input_tokens"],
                request["output_tokens"],
                tuple(request["prefill_chunks"]),
            )
            for metric in metrics:
                if not math.isfinite(request[metric]) or request[metric] <= 0:
                    raise ValueError("Invalid complete-request timing")
            target[key].append(request)
    if not groups[0] or set(groups[0]) != set(groups[1]):
        raise ValueError("Reference and candidate workload coverage differ")
    results = []
    for key in groups[0]:
        old, new = groups[0][key], groups[1][key]
        if len(old) != len(new) or [q["repeat"] for q in old] != [q["repeat"] for q in new]:
            raise ValueError("Reference and candidate repetitions differ")
        timings = {}
        for metric in metrics:
            before = statistics.median(q[metric] for q in old)
            after = statistics.median(q[metric] for q in new)
            timings[metric] = dict(
                reference=before,
                candidate=after,
                ratio=after / before,
                passed=after <= before * (1 + maximum_regression),
            )
        results.append(
            dict(
                name=key[0],
                input_tokens=key[2],
                prefill_chunks=key[4],
                repeats=len(old),
                metrics=timings,
                passed=all(v["passed"] for v in timings.values()),
                output_differences=sum(
                    a["output_token_ids"] != b["output_token_ids"]
                    for a, b in zip(old, new, strict=True)
                ),
            )
        )
    return dict(
        passed=all(r["passed"] for r in results),
        maximum_regression=maximum_regression,
        workloads=results,
        scope="Paired complete-request medians, including request preparation/reset when recorded. Same numerical policy and externally controlled device environment required. Output differences are diagnostics; quality and first-shape preparation require separate acceptance.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-regression", type=float, default=0.05)
    args = parser.parse_args()
    result = compare(
        json.loads(args.reference.read_text()),
        json.loads(args.candidate.read_text()),
        args.maximum_regression,
    )
    result["inputs"] = {
        k: dict(file=str(p), sha256=file_hash(p))
        for k, p in (("reference", args.reference), ("candidate", args.candidate))
    }
    atomic_json(args.output, result)
    print(json.dumps(dict(passed=result["passed"], workloads=len(result["workloads"]))))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
