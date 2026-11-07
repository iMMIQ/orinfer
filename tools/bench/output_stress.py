"""Timed Chat pressure with natural EOS, output checks and retained responses.

Uses the shared streaming client. No deployment changes or GPU lock acquisition.
Cases supply id, request and optional contains/json_equals checks; generated code
and general answers remain available for manual inspection. Repetition signals
are diagnostics, not a universal task-quality metric.
"""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics
import threading
import time
import urllib.request

from tokenizers import Tokenizer

from tools.bench.concurrency import percentile, request


def repetition(ids):
    """Find long adjacent cycles anywhere in the generated token history."""
    positions = defaultdict(list)
    for i in range(max(0, len(ids) - 31)):
        positions[tuple(ids[i : i + 32])].append(i)
    best = dict(period_tokens=0, repeated_tokens=0, repeats=0)
    for starts in positions.values():
        if len(starts) < 3:
            continue
        # Bound work even for one-token loops. Earliest occurrences expose the
        # beginning of a sustained cycle; later ones diagnose a repeated tail.
        selected = sorted(set(starts[:8] + starts[-8:]))
        for a, b, c in zip(selected, selected[1:], selected[2:]):
            period = b - a
            if c - b != period or len(ids) - a < 3 * period:
                continue
            unit = ids[a:b]
            end = b
            while ids[end : end + period] == unit:
                end += period
            size = end - a
            if size >= 3 * period and size > best["repeated_tokens"]:
                best = dict(period_tokens=period, repeated_tokens=size, repeats=size // period)
    return best


def inspect_output(row, checks, tokenizer):
    issues = []
    for field in ["reasoning", "content"]:
        text = row[field]
        repeat = repetition(tokenizer.encode(text, add_special_tokens=False).ids)
        row[field + "_repetition"] = repeat
        if repeat["repeated_tokens"] >= 384 and repeat["repeats"] >= 3:
            issues.append(field + ":repeated_cycle")
        if "\ufffd" in text:
            issues.append(field + ":replacement_character")
    if not row["content"].strip():
        issues.append("empty_answer")
    for expected in checks.get("contains", []):
        if expected not in row["content"]:
            issues.append("missing:" + expected)
    if "json_equals" in checks:
        try:
            if json.loads(row["content"]) != checks["json_equals"]:
                issues.append("json_value_mismatch")
        except ValueError:
            issues.append("invalid_json")
    return issues


def summarize(rows, wall):
    good = [r for r in rows if not r.get("error")]
    groups = {}
    for case in sorted({r["case"] for r in rows}):
        values = [r for r in good if r["case"] == case]
        decode = [r["decode_tps"] for r in values if r["decode_tps"] is not None]
        groups[case] = dict(
            completed=len(values),
            median_decode_tps=statistics.median(decode) if decode else None,
            max_wall_s=max((r["wall_s"] for r in values), default=None),
        )
    return dict(
        requests=len(rows),
        errors=sum(bool(r.get("error")) for r in rows),
        output_issues=sum(bool(r.get("issues")) for r in rows),
        truncated=sum(r.get("finish_reason") == "length" for r in good),
        completion_tokens=sum(r["usage"]["completion_tokens"] for r in good),
        full_wall_s=wall,
        completed_output_tps=sum(r["usage"]["completion_tokens"] for r in good) / max(wall, 1e-9),
        ttft_p95_s=percentile([r["ttft_s"] for r in good if r["ttft_s"] is not None], 0.95)
        if any(r["ttft_s"] is not None for r in good)
        else None,
        by_case=groups,
    )


def run(args):
    cases = json.loads(args.cases.read_text())["cases"]
    if not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Require nonempty unique cases")
    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    tokenizer.no_truncation()
    health_url = args.base_url.rstrip("/").removesuffix("/v1") + "/health"
    rows, phases, health_errors = [], [], []
    lock = threading.Lock()
    stop = threading.Event()

    def monitor():
        with (args.output / "health.jsonl").open("w") as file:
            while not stop.is_set():
                try:
                    with urllib.request.urlopen(health_url, timeout=5) as response:
                        value = json.load(response)
                    file.write(json.dumps(dict(epoch=time.time(), health=value)) + "\n")
                    file.flush()
                    if value.get("failure"):
                        health_errors.append(value["failure"])
                        stop.set()
                except Exception as error:
                    health_errors.append(repr(error))
                    file.write(json.dumps(dict(epoch=time.time(), error=repr(error))) + "\n")
                    file.flush()
                stop.wait(5)

    started = time.monotonic()
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    (args.output / "cases.json").write_text(json.dumps(dict(cases=cases), ensure_ascii=False))
    try:
        with (args.output / "responses.jsonl").open("w") as file:
            for phase, spec in enumerate(args.phases):
                concurrency, seconds = spec.split(":")
                concurrency, seconds = int(concurrency), float(seconds)
                if concurrency < 1 or seconds <= 0:
                    raise ValueError("Phases require positive concurrency:seconds")
                at = time.monotonic()
                deadline = at + seconds

                def worker(lane):
                    iteration = 0
                    while time.monotonic() < deadline and not stop.is_set():
                        case = cases[(lane + iteration * concurrency) % len(cases)]
                        row = dict(
                            case=case["id"],
                            phase=phase,
                            concurrency=concurrency,
                            lane=lane,
                            iteration=iteration,
                            issued_epoch=time.time(),
                        )
                        try:
                            row.update(
                                request(
                                    args.base_url,
                                    case["request"],
                                    collect_arrivals=True,
                                    timeout_s=args.timeout,
                                )
                            )
                            row["issues"] = inspect_output(row, case.get("checks", {}), tokenizer)
                            decode_s = row["wall_s"] - (row["ttft_s"] or row["wall_s"])
                            row["decode_tps"] = (
                                (row["usage"]["completion_tokens"] - 1) / decode_s
                                if decode_s > 0
                                else None
                            )
                        except Exception as error:
                            row["error"] = repr(error)
                        with lock:
                            rows.append(row)
                            file.write(json.dumps(row, ensure_ascii=False) + "\n")
                            file.flush()
                            print(
                                json.dumps(
                                    {
                                        k: row.get(k)
                                        for k in [
                                            "case",
                                            "phase",
                                            "lane",
                                            "wall_s",
                                            "decode_tps",
                                            "finish_reason",
                                            "error",
                                            "issues",
                                        ]
                                    }
                                ),
                                flush=True,
                            )
                        iteration += 1
                        if row.get("error"):
                            stop.set()
                            return

                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    list(pool.map(worker, range(concurrency)))
                phases.append(
                    dict(
                        concurrency=concurrency, requested_s=seconds, actual_s=time.monotonic() - at
                    )
                )
                if stop.is_set():
                    break
    finally:
        stop.set()
        monitor_thread.join(timeout=10)
    summary = summarize(rows, time.monotonic() - started)
    summary["phases"] = phases
    summary["health_errors"] = health_errors
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return int(bool(summary["errors"] or summary["output_issues"] or health_errors or not rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8088/v1")
    parser.add_argument("--phases", nargs="+", default=["1:30", "8:120", "16:210", "32:240"])
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("Timeout must be positive")
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
