"""Bounded, fixed-layout W4 gate calibration on native routed activations."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path

import numpy as np

from orinfer_build.models.deepseek_v41.expert_precision import block_rotation
from orinfer_build.models.deepseek_v41.floating import bf16_parts
from orinfer_build.models.deepseek_v41.official import Checkpoint
from orinfer_build.models.deepseek_v41.prepare import Checkpoint as IntegerCheckpoint
from orinfer_build.quantization.activation import (
    fit_output_scales,
    quantize_rows,
    refine_integer_codes,
)


def bf16(values):
    return (bf16_parts(values)[0].astype("<u4") << 16).view("<f4")


def swiglu(gate, up):
    gate = np.minimum(gate, 10)
    return bf16(gate / (1 + np.exp(np.clip(-gate, -80, 80))) * np.clip(up, -10, 10))


def unpack_rows(packed):
    low, high = (packed & 15).astype("i1"), (packed >> 4).astype("i1")
    codes = np.empty((packed.shape[0] * 2, packed.shape[1]), dtype="i1")
    codes[0::2] = np.where(low >= 8, low - 16, low)
    codes[1::2] = np.where(high >= 8, high - 16, high)
    return codes


def collect(trace, rank, layer, *, validation_owners, fit_rows=128, validation_rows=64):
    """Reservoir-sample routed rows using an explicit, conversation-disjoint owner split."""
    validation_owners = set(validation_owners)
    if not validation_owners or min(fit_rows, validation_rows) < 1:
        raise ValueError("Provide validation owners and positive row budgets")
    rows = max(fit_rows, validation_rows)
    capacities = [fit_rows, validation_rows]
    x = np.zeros((2, 48, rows, 5120), dtype="i1")
    token_scale = np.zeros((2, 48, rows), dtype="f4")
    y = np.zeros((2, 48, rows, 4608), dtype="<u2")
    priorities = np.full((2, 48, rows), -1.0, dtype="f8")
    observed = np.zeros((2, 48), dtype="i8")
    rng = np.random.default_rng(20261002 + layer * 17 + rank)
    clipped = 0
    for frame in sorted(Path(trace).iterdir()):
        if not frame.is_dir():
            continue
        owner = int(frame.name.split("-")[0])
        fold = int(owner in validation_owners)
        folder = frame if rank == 0 else frame / f"rank{rank}"
        stem = f"p0-P3.L{layer:02}."
        count_file = folder / (stem + "routing.counts.bin")
        if not count_file.exists():
            continue
        counts = np.fromfile(count_file, dtype="<i8")
        if counts.shape != (48,) or np.any(counts < 0):
            raise ValueError("Invalid routed expert counts")
        source_x = np.memmap(folder / (stem + "expert.up.input.bin"), mode="r", dtype="i1").reshape(
            -1, 5120
        )
        source_s = np.memmap(folder / (stem + "expert.up.token_scale.bin"), mode="r", dtype="<f4")
        source_y = np.memmap(
            folder / (stem + "expert.up.output.bin"), mode="r", dtype="<u2"
        ).reshape(-1, 4608)
        capacity = min(len(source_x), len(source_s), len(source_y))
        start = 0
        for expert, count in enumerate(counts):
            end = min(start + int(count), capacity)
            available = max(0, end - start)
            clipped += int(count) - available
            if available:
                limit = capacities[fold]
                old = int(min(observed[fold, expert], limit))
                scores = np.concatenate((priorities[fold, expert, :old], rng.random(available)))
                keep = np.argsort(scores)[-limit:]
                combined_x = np.concatenate((x[fold, expert, :old], source_x[start:end]))
                combined_s = np.concatenate((token_scale[fold, expert, :old], source_s[start:end]))
                combined_y = np.concatenate((y[fold, expert, :old], source_y[start:end]))
                size = len(keep)
                x[fold, expert, :size] = combined_x[keep]
                token_scale[fold, expert, :size] = combined_s[keep]
                y[fold, expert, :size] = combined_y[keep]
                priorities[fold, expert, :size] = scores[keep]
                observed[fold, expert] += available
            start += int(count)
    return x, token_scale, y, np.minimum(observed, np.array(capacities)[:, None]), observed, clipped


def fit_bank(arguments):
    (
        trace,
        output,
        official_root,
        integer_root,
        layer,
        rank,
        minimum,
        validation_owners,
        rounding_steps,
    ) = arguments
    output = Path(output) / f"layer{layer}" / f"rank{rank}"
    output.mkdir(parents=True, exist_ok=False)
    xs, ts, ys, lengths, observed, clipped = collect(
        trace, rank, layer, validation_owners=validation_owners
    )
    names = [
        f"layers.{layer}.ffn.experts.{e}.{part}.{suffix}"
        for e in range(rank * 48, (rank + 1) * 48)
        for part in ("w1", "w2")
        for suffix in ("weight", "weight_scale")
    ]
    integer = IntegerCheckpoint(
        Path(integer_root), "quant_model_weights.safetensors.index.json", names
    )
    official = Checkpoint(official_root)
    rotation = block_rotation(integer_root)
    norm = official.read(f"layers.{layer}.ffn_norm.weight")
    scales = np.zeros((48, 2304), dtype="f4")
    sums = np.zeros((48, 2304), dtype="i4")
    records = []
    try:
        for local in range(48):
            expert = rank * 48 + local
            prefix = f"layers.{layer}.ffn.experts.{expert}"
            codes = unpack_rows(integer.read(prefix + ".w1.weight", "I8", [1152, 5120]))
            original = integer.read(prefix + ".w1.weight_scale", "F32", [2304, 1])[:, 0]
            scales[local] = original
            sums[local] = codes.sum(1, dtype="i4")
            train_n, valid_n = map(int, lengths[:, local])
            record = dict(expert=expert, fit_rows=train_n, validation_rows=valid_n, selected=False)
            if train_n < minimum or valid_n < 16:
                records.append(record)
                continue
            x = [
                xs[f, local, :n].astype("f4") * ts[f, local, :n, None]
                for f, n in enumerate((train_n, valid_n))
            ]
            native = [
                (ys[f, local, :n].astype("u4") << 16).view("f4")
                for f, n in enumerate((train_n, valid_n))
            ]
            canonical = (
                (official.matrix(prefix + ".w1.weight") * norm).reshape(2304, 160, 32) @ rotation
            ).reshape(2304, 5120)
            products = [a @ codes.astype("f4").T for a in x]
            reference = [a @ canonical.T for a in x]
            down_codes = unpack_rows(integer.read(prefix + ".w2.weight", "I8", [2560, 2304]))
            down_scale = integer.read(prefix + ".w2.weight_scale", "F32", [5120, 1])[:, 0]
            down = down_codes.astype("f4") * down_scale[:, None]
            influence = np.square(down, dtype="f8").sum(0)
            influence /= max(float(influence.mean()), 1e-30)
            gate = native[0][:, :2304]
            sig = 1 / (1 + np.exp(np.clip(-gate, -80, 80)))
            derivative = (sig + gate * sig * (1 - sig)) * np.clip(native[0][:, 2304:], -10, 10)
            derivative[gate >= 10] = 0
            importance = np.square(derivative) * influence
            gold = quantize_rows(swiglu(bf16(reference[1]), native[1][:, 2304:])) @ down.T
            energy = max(float(np.square(gold, dtype="f8").sum()), 1e-30)

            def score(value):
                hidden = quantize_rows(swiglu(value, native[1][:, 2304:]))
                error = hidden @ down.T - gold
                return float(np.square(error, dtype="f8").sum()) / energy

            baseline = score(native[1][:, :2304])
            best, best_scale, best_alpha = baseline, original, 0.0
            for alpha in (0.25, 0.5, 1.0):
                candidate = fit_output_scales(
                    products[0], reference[0], original, importance, shrink=alpha, limit=0.05
                )
                loss = score(bf16(products[1] * candidate))
                if loss < best:
                    best, best_scale, best_alpha = loss, candidate, alpha
            best_codes = codes
            if rounding_steps:
                rounded = refine_integer_codes(
                    x[0], reference[0], codes, best_scale, importance, steps=rounding_steps
                )
                loss = score(bf16((x[1] @ rounded.astype("f4").T) * best_scale))
                if loss < best:
                    best, best_codes = loss, rounded
            if best < baseline * 0.995:
                scales[local] = best_scale
                record.update(selected=True, shrink=best_alpha)
                updates = int(np.count_nonzero(best_codes != codes))
                record["integer_code_updates"] = updates
                if updates:
                    sums[local] = best_codes.sum(1, dtype="i4")
                    packed = (best_codes[0::2].astype("u1") & 15) | (
                        (best_codes[1::2].astype("u1") & 15) << 4
                    )
                    np.save(output / f"codes{local}.npy", packed)
            else:
                best = baseline
            record.update(
                reference_scope="Conditional complete FFN shadow, fixed baseline W3 and W2; BF16 SwiGLU and per-token A8 boundary",
                baseline_nmse=baseline,
                candidate_nmse=best,
            )
            records.append(record)
    finally:
        integer.close()
        official.close()
    np.save(output / "scale.npy", scales)
    np.save(output / "sum-codes.npy", sums)
    report = dict(
        layer=layer, rank=rank, records=records, observed=observed.tolist(), clipped_rows=clipped
    )
    (output / "fit.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(dict(layer=layer, rank=rank, selected=sum(r["selected"] for r in records))),
        flush=True,
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--integer-source", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--minimum-rows", type=int, default=64)
    parser.add_argument(
        "--rounding-steps",
        type=int,
        default=0,
        help="0 retains codes; 1-16 enables bounded INT4 coordinate reconstruction",
    )
    parser.add_argument(
        "--validation-owners",
        type=int,
        nargs="+",
        required=True,
        help="Owners from whole held-out calibration conversations; never split one conversation across folds",
    )
    args = parser.parse_args()
    if (
        args.workers < 1
        or not 1 <= args.minimum_rows <= 128
        or any(not 0 <= l < 40 for l in args.layers)
        or not 0 <= args.rounding_steps <= 16
    ):
        parser.error("Require positive workers, 1-128 fit rows, and target layers 0-39")
    tasks = [
        (
            str(args.trace),
            str(args.output),
            str(args.official),
            str(args.integer_source),
            l,
            r,
            args.minimum_rows,
            args.validation_owners,
            args.rounding_steps,
        )
        for l in sorted(set(args.layers))
        for r in range(8)
    ]
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        result = list(pool.map(fit_bank, tasks))
    (args.output / "fit.json").write_text(
        json.dumps(
            dict(
                layers=args.layers,
                validation_owners=sorted(set(args.validation_owners)),
                rounding_steps=args.rounding_steps,
                banks=result,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
