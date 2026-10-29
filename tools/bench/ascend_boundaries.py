"""Check adjacent prompt lengths using complete native A2 request measurements."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

from tools.build.files import atomic_json, file_hash


def summarize(inputs, report, maximum_regression=0.10, minimum_samples=5):
    if not math.isfinite(maximum_regression) or not 0 <= maximum_regression < 1:
        raise ValueError("Invalid boundary regression limit")
    if minimum_samples < 1:
        raise ValueError("Invalid minimum sample count")
    cases = {c["name"]: c for c in inputs["cases"]}
    if len(cases) != len(inputs["cases"]):
        raise ValueError("Duplicate input case name")
    grouped = defaultdict(list)
    for q in report["requests"]:
        if q["name"] not in cases:
            raise ValueError("Measurement has no matching input case")
        grouped[q["name"]].append(q)
    if set(grouped) != set(cases):
        raise ValueError("Incomplete measurement coverage")
    lengths, histories, budgets, first_shapes, excluded = {}, {}, {}, {}, []
    preparations = {p["name"]: p for p in report["preparations"]}
    if len(preparations) != len(report["preparations"]) or set(preparations) != set(cases):
        raise ValueError("Incomplete or duplicate first-shape measurements")
    for name, case in cases.items():
        samples = grouped[name]
        if sorted(q["repeat"] for q in samples) != list(range(case.get("repeats", 1))):
            raise ValueError("Missing or duplicate measurement repetition")
        requests = case.get("requests", [])
        if len(requests) != 1 or case.get("frames") or not requests[0].get("output", False):
            excluded.append(name)
            continue
        tokens = requests[0]["tokens"]
        n, budget = len(tokens), case.get("decode_steps", 0)
        if not n or n + budget > report["context_capacity"]:
            raise ValueError("Invalid prompt/context length")
        if n in histories and (histories[n] != tokens or budgets[n] != budget):
            raise ValueError("Different workloads at the same prompt length")
        histories[n], budgets[n] = tokens, budget
        for q in samples:
            if q["batch"] != 1 or q["input_tokens"] != n:
                raise ValueError("Measurement differs from the input workload")
            for metric in ("prefill_s", "request_s"):
                if not math.isfinite(q[metric]) or q[metric] <= 0:
                    raise ValueError("Invalid complete-request timing")
            if q["request_s"] < q["prefill_s"]:
                raise ValueError("Complete request is shorter than its prefill")
        lengths.setdefault(n, []).extend(samples)
        p = preparations[name]
        cold_parts = [
            p[k] for k in ("owner_admission_s", "reset_s", "graph_preparation_s", "first_prefill_s")
        ]
        if any(not math.isfinite(v) or v < 0 for v in cold_parts):
            raise ValueError("Invalid first-shape timing component")
        cold = sum(cold_parts)
        if not math.isfinite(cold) or cold <= 0:
            raise ValueError("Invalid first-shape timing")
        first_shapes[n] = max(first_shapes.get(n, 0), cold)
    points = []
    for n, samples in sorted(lengths.items()):
        if len(samples) < minimum_samples:
            raise ValueError("Too few samples for a prompt length")
        prefill = statistics.median(q["prefill_s"] for q in samples)
        points.append(
            dict(
                length=n,
                samples=len(samples),
                prefill_s=prefill,
                prefill_p95_s=sorted(q["prefill_s"] for q in samples)[
                    math.ceil(0.95 * len(samples)) - 1
                ],
                request_s=statistics.median(q["request_s"] for q in samples),
                tokens_per_s=n / prefill,
                first_shape_request_s=first_shapes[n],
            )
        )
    boundaries = []
    for before, after in zip(points, points[1:]):
        a, b = before["length"], after["length"]
        if b != a + 1:
            continue
        if histories[a] != histories[b][:a] or budgets[a] != budgets[b]:
            raise ValueError("Adjacent lengths do not share the same prefix/output budget")
        # Compare time per input token, retaining the real extra packet cost.
        ratios = {k: after[k] / before[k] * a / b for k in ("prefill_s", "request_s")}
        throughputs = {k: 1 / v for k, v in ratios.items()}
        boundaries.append(
            dict(
                before=a,
                after=b,
                normalized_latency_ratios=ratios,
                normalized_throughput_ratios=throughputs,
                passed=all(v >= 1 - maximum_regression for v in throughputs.values()),
            )
        )
    if not boundaries:
        raise ValueError("No adjacent prompt lengths were measured")
    return dict(
        passed=all(p["passed"] for p in boundaries),
        maximum_regression=maximum_regression,
        points=points,
        boundaries=boundaries,
        excluded_cases=excluded,
        scope="Measured adjacent single-owner prompt lengths with identical prefixes/output budgets. Gate the relative input throughput drop from warm complete forward/request medians; first-shape admission/reset/Graph/TTFT reported separately. Unmeasured lengths and numerical quality require separate acceptance.",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--maximum-regression", type=float, default=0.10)
    p.add_argument("--minimum-samples", type=int, default=5)
    args = p.parse_args()
    result = summarize(
        json.loads(args.inputs.read_text()),
        json.loads(args.report.read_text()),
        args.maximum_regression,
        args.minimum_samples,
    )
    result["inputs"] = {
        k: dict(file=str(v), sha256=file_hash(v))
        for k, v in (("workloads", args.inputs), ("measurements", args.report))
    }
    atomic_json(args.output, result)
    print(
        json.dumps(
            dict(
                passed=result["passed"],
                lengths=len(result["points"]),
                boundaries=len(result["boundaries"]),
            )
        )
    )
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
