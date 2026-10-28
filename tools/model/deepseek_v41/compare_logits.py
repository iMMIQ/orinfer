"""Compare native vocabulary shards with official logits on fixed histories."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.build.files import atomic_json


def log_probabilities(values):
    if values.shape != (129280,) or not np.isfinite(values).all():
        raise ValueError("Expected one finite full-vocabulary logit vector")
    values = values.astype(np.float64)
    maximum = values.max()
    return values - (maximum + np.log(np.exp(values - maximum).sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--logits", type=Path, required=True)
    parser.add_argument("--baseline-logits", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads((args.reference / "report.json").read_text())
    identity = reference["identity"]
    positions = reference["positions"]
    gold = np.load(args.reference / "logits.npy", mmap_mode="r")
    if not identity["same_teacher_history"] or identity["seed"] != 20261002:
        raise ValueError("Reference must declare the fixed seeded teacher history")
    if len(args.cases) != len(positions) or gold.shape != (len(positions), 129280):
        raise ValueError("Provide one ordered native case per reference position")
    inputs = json.loads(args.inputs.read_text())["cases"]
    cases = {case["name"]: case for case in inputs}
    first = cases[args.cases[0]]["tokens"]
    if (
        len(first) != identity["input_tokens"]
        or hashlib.sha256(json.dumps(first).encode()).hexdigest() != identity["input_tokens_sha256"]
    ):
        raise ValueError("Native prompt differs from the official reference")
    teacher = [position["target"] for position in positions]
    if (
        hashlib.sha256(json.dumps(teacher).encode()).hexdigest()
        != identity["teacher_tokens_sha256"]
    ):
        raise ValueError("Reference teacher identity differs")

    def load(directory, rows):
        shards = [
            np.fromfile(directory / f"r{rank}-p0-m{rows}.bin", dtype="<f4") for rank in range(8)
        ]
        if any(shard.shape != (16160,) for shard in shards):
            raise ValueError("Missing or truncated native vocabulary shard")
        return log_probabilities(np.concatenate(shards))

    report = []
    for i, name in enumerate(args.cases):
        tokens = cases[name]["tokens"]
        if tokens != first + teacher[:i] or positions[i]["position"] != i:
            raise ValueError(f"{name} does not contain the fixed teacher history")
        # p0/mN identifies a single complete prefill packet, not a resumed tail.
        if cases[name].get("prefill_chunk_tokens", len(tokens)) < len(tokens):
            raise ValueError("Native logit comparison requires a single prefill packet")
        official = log_probabilities(gold[i])
        candidate = load(args.logits, len(tokens))
        target = teacher[i]
        row = dict(
            position=i,
            case=name,
            target=target,
            official_nll=float(-official[target]),
            candidate_nll=float(-candidate[target]),
            candidate_argmax=int(candidate.argmax()),
            official_argmax=int(official.argmax()),
            kl_official_candidate=float(np.sum(np.exp(official) * (official - candidate))),
        )
        if args.baseline_logits:
            baseline = load(args.baseline_logits, len(tokens))
            row.update(
                baseline_nll=float(-baseline[target]),
                baseline_argmax=int(baseline.argmax()),
                delta_nll_candidate_baseline=float(baseline[target] - candidate[target]),
                kl_baseline_candidate=float(np.sum(np.exp(baseline) * (baseline - candidate))),
            )
        report.append(row)
    result = dict(
        reference_identity=identity,
        positions=report,
        mean_candidate_nll=float(np.mean([r["candidate_nll"] for r in report])),
        mean_official_nll=float(np.mean([r["official_nll"] for r in report])),
        scope="Fixed-history diagnostic positions; task and long-context quality require separate acceptance",
    )
    if args.baseline_logits:
        result["mean_baseline_nll"] = float(np.mean([r["baseline_nll"] for r in report]))
    atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
