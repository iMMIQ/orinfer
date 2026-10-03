"""Validate coverage of user-confirmed acceptance requirements; no GPU needed."""
import json
from pathlib import Path


def validate():
    root = Path(__file__).resolve().parents[2]
    plan = json.loads((root / "configs/benchmark.json").read_text())
    protocol = json.loads((root / "configs/quick-quality.json").read_text())
    scenes = json.loads((root / "fixtures/quick-quality-scenarios.json").read_text())
    checks = {
        "text-only phase one": plan["requirements"]["modalities"] == plan["workloads"]["modalities"] == protocol["modalities"] == ["text"],
        "phase-one weight budget": "vision" not in plan["weight_accounting"]["included"]
        and {"text trunk", "embedding", "LM head", "scales"}.issubset(plan["weight_accounting"]["included"]),
        "input lengths": plan["workloads"]["input_tokens"] == [512, 2048, 8192],
        "all submitted concurrency 2..128": plan["workloads"]["submitted_concurrency"] == list(range(2, 129)),
        "common specialized rows": {1, 2, 4, 8}.issubset(plan["workloads"]["primary_optimized_rows"]),
        "single stream": plan["workloads"]["single_stream"] is True,
        "single-stream acceptance": plan["requirements"]["performance_acceptance"]["single_stream_prefill_tps"] == 800
        and plan["requirements"]["performance_acceptance"]["single_stream_decode_tps"] == 10
        and plan["requirements"]["performance_acceptance"]["mtp"] is False,
        "lifecycle/prefix/concurrency metrics": {"native_load", "import", "first_request", "warm_prefill", "warm_decode", "prefix", "concurrency"}.issubset(plan["metrics"]),
        "seed alignment": protocol["seed"] == scenes["seed"] == 20261002,
        "greedy non-MTP quality": scenes["generation"]["temperature"] == 0 and scenes["generation"]["mtp"] is False,
        "unique 12 scenarios": len(scenes["cases"]) == len({case["id"] for case in scenes["cases"]}) == 12,
    }
    for name, ok in checks.items():
        if not ok:
            raise ValueError(f"Acceptance requirement missing: {name}")
    print(f"Validated {len(checks)} acceptance/fixture coverage checks; no performance measurements.")


if __name__ == "__main__":
    validate()
