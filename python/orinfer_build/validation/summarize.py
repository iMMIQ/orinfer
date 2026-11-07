"""Summarize actual fixed-work Rust model runs; never sum isolated operators."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--report", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    report = json.loads(a.report.read_text())
    r = report["requests"]
    rows = []
    for length in (512, 2048, 8192):
        trials = [q for q in r if q["id"].startswith(f"len{length}-run")]
        assert len(trials) == 3 and all(
            q["input_tokens"] == length and len(q["output_tokens"]) == 256 and not q["diagnostic"]
            for q in trials
        )
        assert all(q["final_position"] == length + 255 for q in trials)
        row = {
            "input_tokens": length,
            "output_tokens": 256,
            "repetitions": len(trials),
            "median_prefill_tps": statistics.median(q["prefill_tps"] for q in trials),
            "median_decode_tps": statistics.median(q["decode_tps"] for q in trials),
            "median_ttft_s": statistics.median(q["ttft_s"] for q in trials),
            "median_prefill_s": statistics.median(q["prefill_s"] for q in trials),
            "median_head_s": statistics.median(q["head_s"] for q in trials),
            "median_decode_s": statistics.median(q["decode_s"] for q in trials),
            "repeat_tokens_identical": all(
                q["output_tokens"] == trials[0]["output_tokens"] for q in trials
            ),
            "trials": [
                {
                    "id": q["id"],
                    "prefill_tps": q["prefill_tps"],
                    "decode_tps": q["decode_tps"],
                    "ttft_s": q["ttft_s"],
                }
                for q in trials
            ],
        }
        row["prefill_target_met"] = row["median_prefill_tps"] >= 800
        row["decode_target_met"] = row["median_decode_tps"] >= 10
        if all("prefill_chunk_tokens" in q for q in trials):
            plans = {q["prefill_chunk_tokens"] for q in trials}
            assert len(plans) == 1
            row["prefill_chunk_tokens"] = plans.pop()
            row["prefill_program"] = trials[0]["prefill_program"]
            assert all(q["prefill_program"] == row["prefill_program"] for q in trials)
        rows.append(row)
    result = {
        "status": "measured",
        "report_sha256": hashlib.sha256(a.report.read_bytes()).hexdigest(),
        "seed": report["seed"],
        "load_to_ready_s": report["load_to_ready_s"],
        "weight_io_hash_s": report["weight_io_hash_s"],
        "weight_upload_s": report["weight_upload_s"],
        "module_load_bind_s": report["module_load_bind_s"],
        "graph_capture_s": report["graph_capture_s"],
        "weight_bytes": report["weight_bytes"],
        "weight_gib": report["weight_bytes"] / 2**30,
        "effective_weight_bits": report["effective_weight_bits"],
        "buffer_bytes": report["buffer_bytes"],
        "buffer_scope": "Explicit CUDA allocations; excludes driver/module/graph internal overhead, not peak device process memory",
        "weight_scope": report["weight_scope"],
        "first_request": r[0],
        "lengths": rows,
        "timing_scope": report["timing_scope"],
        "os_page_cache": "Uncontrolled; historical/build reads warm some files. No cold disk claim.",
        "limitations": [
            "Single resident request, fixed-shape prefill plans (reported per length where available); no prefix/no concurrent scheduler/no MTP",
            "RNE embedding/head candidate requires quality acceptance; these performance results do not assert quality",
            "No tokenization or HTTP/server/queue latency in native Rust benchmark",
            "Prefill includes H2D/sync, excludes last-position head; TTFT includes head; decode excludes first token",
        ],
        "all_performance_targets_met": all(
            row["prefill_target_met"] and row["decode_target_met"] for row in rows
        ),
    }
    with a.output.open("x") as f:
        json.dump(result, f, indent=2)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("first_request", "lengths")}, indent=2
        )
    )
    for row in rows:
        print(
            row["input_tokens"],
            round(row["median_prefill_tps"], 2),
            round(row["median_decode_tps"], 3),
            "TTFT",
            round(row["median_ttft_s"], 3),
            "repeated IDs identical",
            row["repeat_tokens_identical"],
        )


if __name__ == "__main__":
    main()
