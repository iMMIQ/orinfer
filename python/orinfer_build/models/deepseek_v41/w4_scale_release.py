"""Export calibrated W4 codes/scales in the existing layout and native programs."""

import argparse
import json
from pathlib import Path
import re

import numpy as np

from orinfer_build.models.deepseek_v41.export_release import TensorRelease
from orinfer_build.models.deepseek_v41.weight_layout import transposed_nz_blocks


def reconstruct_scales(
    scales, assist, fitted, sums, *, strength=1.0, changed_codes=None, section=0
):
    """Replace one fused projection's scale and nibble-offset compensation."""
    scales, assist = np.array(scales), np.array(assist)
    fitted, sums = np.asarray(fitted), np.asarray(sums)
    if (
        scales.dtype != np.dtype("<u8")
        or assist.dtype != np.dtype("<f4")
        or scales.ndim != 3
        or scales.shape[1] != 1
        or assist.shape != (scales.shape[0], scales.shape[2])
        or scales.shape[2] % 2
    ):
        raise ValueError("Released W4 scale/assist geometry differs")
    if np.any(scales >> 32) or not 0 < strength <= 1 or section not in (0, 1):
        raise ValueError("Require low FP32 scale bits and strength within (0,1]")
    width = scales.shape[2] // 2
    columns = slice(section * width, (section + 1) * width)
    original = scales[:, 0, columns].astype("<u4").view("<f4")
    if (
        fitted.shape != original.shape
        or sums.shape != original.shape
        or not np.issubdtype(sums.dtype, np.integer)
        or not np.isfinite(fitted).all()
        or np.any(fitted <= 0)
        or not np.isfinite(original).all()
        or np.any(original <= 0)
    ):
        raise ValueError("Calibrated W4 scale values differ")
    if np.any(abs(fitted / original - 1) > 0.0501):
        raise ValueError("Calibration exceeds its bounded scale adjustment")
    fitted = (original + strength * (fitted - original)).astype("<f4")
    changed = fitted.view("u4") != original.view("u4")
    if changed_codes is not None:
        changed_codes = np.asarray(changed_codes)
        if changed_codes.shape != changed.shape or changed_codes.dtype != np.dtype("bool"):
            raise ValueError("Changed-code mask must match W1 output channels")
        changed |= changed_codes
    scales[:, 0, columns] = fitted.view("<u4").astype("<u8")
    bias = (sums.astype("f4") * np.float32(8)) * fitted
    assist[:, columns][changed] = bias[changed]
    return scales, assist, int(np.any(changed, axis=1).sum())


def reconstruct_gate_bank(bank, replacements):
    """Replace W1 in the original fused NZ bank; retain all W3 bytes."""
    bank = np.array(bank).view("u1")
    if bank.ndim != 5 or bank.shape[-2:] != (16, 32) or bank.shape[2] % 2:
        raise ValueError("Require the fused W1/W3 transposed-NZ bank")
    half = bank.shape[2] // 2
    changed = np.zeros((len(bank), half * 16), dtype=bool)
    for expert, packed in replacements.items():
        if (
            not 0 <= expert < len(bank)
            or packed.dtype != np.dtype("u1")
            or packed.shape != (half * 8, bank.shape[1] * 64)
        ):
            raise ValueError("W1 replacement geometry differs")
        converted = np.stack(list(transposed_nz_blocks(np.ascontiguousarray(packed.T)[None])))
        changed[expert] = np.any(bank[expert, :, :half] != converted, axis=(0, 3)).reshape(-1)
        bank[expert, :, :half] = converted
    return bank, changed


class ScaleRelease(TensorRelease):
    def __init__(self, *args, fitted, layers, strength=1.0, overlay=False, **kwargs):
        super().__init__(*args, overlay=overlay, **kwargs)
        self.fitted, self.layers = Path(fitted), sorted(set(layers))
        self.strength, self.overlay = strength, overlay
        if not self.layers or any(not 0 <= l < 40 for l in self.layers) or not 0 < strength <= 1:
            raise ValueError("Select target layers and a strength within (0,1]")

    def plan(self):
        selected = 0
        changed_banks = 0
        for ri, rank in enumerate(self.model["ranks"]):
            definitions = {b["name"]: b for b in rank["buffers"]}
            for layer in self.layers:
                calls = [
                    s
                    for packet in rank["packets"]
                    for s in packet["body"]
                    if s["op"] in ("w4", "w4_a4")
                    and re.search(rf"\.L{layer}\.up$", s.get("name", ""))
                ]
                if not calls or any(
                    (s["input"], s["output"], s["experts"]) != (5120, 4608, 48)
                    or s["buffers"][1:4] != calls[0]["buffers"][1:4]
                    for s in calls
                ):
                    raise ValueError("Require the released TP8 DSv4.1 up-bank contract")
                buffers = [definitions[n] for n in calls[0]["buffers"][2:4]]
                scale_name, assist_name = [b["prepared_tensor"]["name"] for b in buffers]
                scales, assist = np.array(self.array(scale_name)), np.array(self.array(assist_name))
                folder = self.fitted / f"layer{layer}/rank{ri}"
                fitted = np.load(folder / "scale.npy")
                sums = np.load(folder / "sum-codes.npy")
                replacements = {
                    e: np.load(folder / f"codes{e}.npy")
                    for e in range(48)
                    if (folder / f"codes{e}.npy").exists()
                }
                changed_codes = None
                banks, values_to_write = list(buffers), [scales, assist]
                if replacements:
                    if self.strength != 1:
                        raise ValueError("Rounded integer codes require strength=1")
                    weight = definitions[calls[0]["buffers"][1]]
                    if calls[0].get("weight_layout") != "transposed_nz":
                        raise ValueError("Rounded W1 requires the published NZ layout")
                    values, changed_codes = reconstruct_gate_bank(
                        self.array(weight["prepared_tensor"]["name"]), replacements
                    )
                    banks.append(weight)
                    values_to_write.append(values)
                    changed_banks += 1
                scales, assist, count = reconstruct_scales(
                    scales,
                    assist,
                    fitted,
                    sums,
                    strength=self.strength,
                    changed_codes=changed_codes,
                )
                values_to_write[:2] = [scales, assist]
                selected += count
                for bank, values in zip(banks, values_to_write, strict=True):
                    group = (
                        f"weights/calibration/layer{layer}/rank{ri}.safetensors"
                        if values.dtype.itemsize == 1
                        else "weights/calibration/w1-scales.safetensors"
                    )
                    self.replace_tensor(bank, values, suffix=".w4_calibrated", group=group)
        if not selected:
            raise ValueError("No selected W4 scale changes")
        self.prepare_overrides()
        self.manifest["w4_calibration"] = dict(
            layers=self.layers,
            selected_experts=selected,
            method="weighted INT4 coordinate/scale reconstruction"
            if changed_banks
            else "fixed-code weighted-scale reconstruction",
            strength=self.strength,
            experimental_overlay=self.overlay,
            banks_unchanged=not changed_banks,
            programs_unchanged=True,
        )
        return self


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "source-execution", "fitted", "output", "execution-output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="Experimental overrides; final publication uses semantic export",
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    job = ScaleRelease(
        args.source,
        args.output,
        source_execution=args.source_execution,
        execution_output=args.execution_output,
        fitted=args.fitted,
        layers=args.layers,
        strength=args.strength,
        overlay=args.overlay,
    ).plan()
    print(json.dumps(job.manifest["w4_calibration"]), flush=True)
    job.export(args.workers)


if __name__ == "__main__":
    main()
