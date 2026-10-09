"""Measure committed output, reasoning delay and repeated-prompt cache reuse."""

import argparse
import json
from pathlib import Path
import statistics
import urllib.request

from tools.bench.concurrency import counter_delta, request


def snapshot(base_url):
    url = base_url.rstrip("/").removesuffix("/v1") + "/health"
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.load(response)


def median_field(rows, field):
    values = [r[field] for r in rows if r[field] is not None]
    return statistics.median(values) if values else None


def summarize(rows):
    groups = {}
    for row in rows:
        if row["warmup"]:
            continue
        groups.setdefault(row["case"], []).append(row)
    return {
        name: {
            "median_output_tps": statistics.median(r["output_tps"] for r in group),
            "median_ttft_s": median_field(group, "ttft_s"),
            "median_first_content_s": median_field(group, "first_content_s"),
            "prefix_token_hit_rate": sum(
                r["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0) for r in group
            )
            / max(1, sum(r["usage"]["prompt_tokens"] for r in group)),
            "empty_content_requests": sum(not r["content"].strip() for r in group),
            "truncated_requests": sum(r["finish_reason"] == "length" for r in group),
            "completion_tokens": [r["usage"]["completion_tokens"] for r in group],
        }
        for name, group in groups.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8088/v1")
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if args.output.exists() or args.repetitions < 1:
        parser.error("Require a fresh output and positive repetition count")
    cases = json.loads(args.requests.read_text())["cases"]
    report = {
        "scope": "Committed usage completion tokens / entire HTTP wall time; includes reasoning, prefill and transport. First content is time until the answer begins. SSE chunks are not tokens. Each case repeats the exact request after one cold/warmup run; cache state and model behavior are reported, not assumed.",
        "server": snapshot(args.base_url),
        "requests": cases,
        "rows": [],
        "status": "running",
    }
    for case in cases:
        for repetition in range(args.repetitions + 1):
            before = snapshot(args.base_url)
            row = request(args.base_url, case["request"], collect_arrivals=True)
            after = snapshot(args.base_url)
            row.update(
                case=case["id"],
                repetition=repetition,
                warmup=repetition == 0,
                output_tps=row["usage"]["completion_tokens"] / row["wall_s"],
                scheduler_delta=counter_delta(
                    before.get("scheduler_statistics"), after.get("scheduler_statistics")
                ),
            )
            report["rows"].append(row)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            print(case["id"], repetition, round(row["output_tps"], 3), "TPS", flush=True)
    report.update(status="complete", summary=summarize(report["rows"]))
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
