"""Fit fixed-layout DSpark expert scales from native routed activations."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import re

import numpy as np

from tools.model.deepseek_v41.export_release import TensorRelease
from tools.model.deepseek_v41.official import Checkpoint, Decoder
from tools.model.deepseek_v41.prepare import Checkpoint as IntegerCheckpoint
from tools.model.deepseek_v41.w4_calibration import bf16, swiglu, unpack_rows
from tools.model.deepseek_v41.w4_scale_release import reconstruct_scales
from tools.quantization.activation import fit_output_scales, quantize_rows


def fit_bank(arguments):
    folder, output, official, integer, layer, rank = arguments
    folder, output = Path(folder), Path(output) / f"layer{layer}/rank{rank}"
    output.mkdir(parents=True, exist_ok=False)
    data = np.load(folder / f"layer{layer}.npz")
    fold_labels, indices = data["fold"], data["indices"]
    names = [
        f"mtp.{layer - 40}.ffn.experts.{e}.{part}.{suffix}"
        for e in range(rank * 16, (rank + 1) * 16)
        for part in ("w1", "w3")
        for suffix in ("weight", "weight_scale")
    ]
    source = IntegerCheckpoint(Path(integer), "quant_model_weights.safetensors.index.json", names)
    reference = Checkpoint(official, Decoder(Path(output) / "decoder"))
    scales = [np.empty((16, 2304), dtype="f4") for _ in range(2)]
    sums = [np.empty((16, 2304), dtype="i4") for _ in range(2)]
    records = []
    try:
        for local in range(16):
            expert = rank * 16 + local
            rows = np.nonzero(np.any(indices == expert, axis=1))[0]
            folds = [rows[fold_labels[rows] == 0], rows[fold_labels[rows] == 1]]
            # Deterministic evenly spaced rows preserve owner/position diversity.
            folds = [x[np.linspace(0, len(x) - 1, min(len(x), 64)).astype(int)] for x in folds]
            record = dict(
                expert=expert, fit_rows=len(folds[0]), validation_rows=len(folds[1]), selected=False
            )
            codes, original = [], []
            prefix = f"mtp.{layer - 40}.ffn.experts.{expert}"
            for section, part in enumerate(("w1", "w3")):
                code = unpack_rows(source.read(prefix + f".{part}.weight", "I8", [1152, 5120]))
                scale = source.read(prefix + f".{part}.weight_scale", "F32", [2304, 1])[:, 0]
                codes.append(code)
                original.append(scale)
                scales[section][local] = scale
                sums[section][local] = code.sum(1, dtype="i4")
            if len(folds[0]) < 8 or len(folds[1]) < 4:
                records.append(record)
                continue
            xs = [data["inputs"][x].astype("f4") for x in folds]
            token_scale = [data["scale"][x, None] for x in folds]
            products = [
                [(x @ c.astype("f4").T) * s for x, s in zip(xs, token_scale, strict=True)]
                for c in codes
            ]
            gold = [
                [
                    bf16((x * s) @ reference.matrix(prefix + f".{part}.weight").T)
                    for x, s in zip(xs, token_scale, strict=True)
                ]
                for part in ("w1", "w3")
            ]
            native = [[bf16(x * s) for x in products[i]] for i, s in enumerate(original)]
            gate, up = native[0][0], native[1][0]
            sig = 1 / (1 + np.exp(np.clip(-gate, -80, 80)))
            derivatives = [
                (sig + gate * sig * (1 - sig)) * np.clip(up, -10, 10),
                np.minimum(gate, 10) * sig,
            ]
            derivatives[0][gate >= 10] = 0
            importance = [np.square(d) * data["importance"][folds[0], None] for d in derivatives]
            target = quantize_rows(swiglu(gold[0][1], gold[1][1]))
            energy = max(float(np.square(target, dtype="f8").sum()), 1e-30)

            def score(candidate):
                value = quantize_rows(
                    swiglu(bf16(products[0][1] * candidate[0]), bf16(products[1][1] * candidate[1]))
                )
                return float(np.square(value - target, dtype="f8").sum()) / energy

            best, selected = score(original), original
            baseline = best
            # Reconstruct the fused gate/up boundary jointly; leave codes fixed.
            for alpha in (0.25, 0.5, 1.0):
                candidate = [
                    fit_output_scales(
                        products[i][0],
                        gold[i][0],
                        original[i],
                        importance[i],
                        shrink=alpha,
                        limit=0.05,
                    )
                    for i in range(2)
                ]
                loss = score(candidate)
                if loss < best:
                    best, selected = loss, candidate
            if best < baseline * 0.995:
                for i in range(2):
                    scales[i][local] = selected[i]
                record["selected"] = True
            record.update(baseline_nmse=baseline, candidate_nmse=best)
            records.append(record)
    finally:
        source.close()
        reference.close()
    for i, suffix in enumerate(("", "-w3")):
        np.save(output / f"scale{suffix}.npy", scales[i])
        np.save(output / f"sum-codes{suffix}.npy", sums[i])
    report = dict(layer=layer, rank=rank, records=records, fixed_codes=True)
    (output / "fit.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(dict(layer=layer, rank=rank, selected=sum(x["selected"] for x in records))),
        flush=True,
    )
    return report


class DraftScaleRelease(TensorRelease):
    def __init__(self, *args, fitted, **kwargs):
        super().__init__(*args, **kwargs)
        self.fitted = Path(fitted)

    def plan(self):
        changed = 0
        for ri, rank in enumerate(self.model["ranks"]):
            definitions = {b["name"]: b for b in rank["buffers"]}
            for definition in rank["buffers"]:
                match = re.fullmatch(
                    r"dspark\.draft\.w\.L(4[0-2])\.\['w1', 'w3'\]\.scale", definition["name"]
                )
                if not match:
                    continue
                layer = int(match[1])
                assist = definitions[definition["name"].removesuffix(".scale") + ".assist"]
                values = self.array(definition["prepared_tensor"]["name"])
                bias = self.array(assist["prepared_tensor"]["name"])
                if values.shape != (16, 1, 4608):
                    raise ValueError("Require the released draft fused W1/W3 bank")
                folder = self.fitted / f"layer{layer}/rank{ri}"
                for section, suffix in enumerate(("", "-w3")):
                    values, bias, count = reconstruct_scales(
                        values,
                        bias,
                        np.load(folder / f"scale{suffix}.npy"),
                        np.load(folder / f"sum-codes{suffix}.npy"),
                        section=section,
                    )
                    changed += count
                for bank, payload in ((definition, values), (assist, bias)):
                    self.replace_tensor(
                        bank,
                        payload,
                        suffix=".draft_calibrated",
                        group="weights/calibration/draft-scales.safetensors",
                    )
        if not changed:
            raise ValueError("No selected draft scale changes")
        self.prepare_overrides()
        self.manifest["draft_calibration"] = dict(
            method="native routed fixed-code gate/up reconstruction",
            changed_expert_sections=changed,
            programs_unchanged=True,
            banks_unchanged=True,
        )
        return self


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("collect")
    for name in ("trace", "points", "samples", "output"):
        collect.add_argument("--" + name, type=Path, required=True)
    fit = commands.add_parser("fit")
    for name in ("inputs", "output", "official", "integer-source"):
        fit.add_argument("--" + name, type=Path, required=True)
    fit.add_argument("--workers", type=int, default=8)
    export = commands.add_parser("export")
    for name in ("source", "source-execution", "fitted", "output", "execution-output"):
        export.add_argument("--" + name, type=Path, required=True)
    export.add_argument("--overlay", action="store_true")
    export.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.command == "collect":
        from tools.model.deepseek_v41.calibration import collect_routed

        print(collect_routed(args.trace, args.points, args.samples, args.output))
        return
    if args.command == "export":
        job = DraftScaleRelease(
            args.source,
            args.output,
            source_execution=args.source_execution,
            fitted=args.fitted,
            execution_output=args.execution_output,
            overlay=args.overlay,
        ).plan()
        print(json.dumps(job.manifest["draft_calibration"]), flush=True)
        job.export(args.workers)
        return
    tasks = [
        (
            str(args.inputs),
            str(args.output),
            str(args.official),
            str(args.integer_source),
            layer,
            rank,
        )
        for layer in range(40, 43)
        for rank in range(8)
    ]
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        reports = list(pool.map(fit_bank, tasks))
    (args.output / "report.json").write_text(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
