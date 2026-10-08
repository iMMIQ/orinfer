"""Import native Qwen3_5 MTP weights with bounded CPU FP8 dequantization.

Draft W4 quantization is separate from target-model quantization: the target
verifier decides every committed token. A W4 draft's numerical error must be
evaluated through acceptance rates, not treated as target output degradation.
This module never uploads a tensor to the GPU or modifies the input checkpoint.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from tools.quantization.w4_warp_pack import LAYOUT, pack_array


def identity(path, root):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for data in iter(lambda: f.read(8 << 20), b""):
            digest.update(data)
    return dict(
        file=str(path.relative_to(root)), sha256=digest.hexdigest(), bytes=path.stat().st_size
    )


def tensor_rows(source, name, start, end):
    """FP8 E4M3 times native inverse scale, in FP32; BF16/FP16 unchanged."""
    value = source.get_slice(name)[start:end].float()
    scale_name = name.removesuffix(".weight") + ".weight_scale_inv"
    if scale_name in source.keys():
        scales = source.get_tensor(scale_name).float()
        rows = torch.arange(start, end) // 128
        columns = torch.arange(value.shape[1]) // 128
        value *= scales[rows[:, None], columns[None, :]]
    if not torch.isfinite(value).all():
        raise ValueError(f"Nonfinite weight {name}")
    return value


def import_weights(checkpoint, output, format="w4"):
    output.mkdir(parents=True, exist_ok=True)
    weights = []
    parameters = 0
    with safe_open(checkpoint, framework="pt", device="cpu") as source:
        names = sorted(
            name for name in source.keys() if name.startswith("mtp.") and name.endswith(".weight")
        )
        if len(names) != 15:
            raise ValueError("Expected native one-layer dense Qwen3_5 MTP weights")
        for key in names:
            name = key.removeprefix("mtp.").removesuffix(".weight").replace(".", "_")
            shape = source.get_slice(key).get_shape()
            parameters += int(np.prod(shape))
            if len(shape) == 1 or format == "f16":
                path = output / f"{name}.f16"
                if path.exists():
                    raise ValueError(f"Refusing to overwrite {path}")
                with path.open("wb") as f:
                    if len(shape) == 1:
                        source.get_tensor(key).half().numpy().tofile(f)
                    else:
                        for start in range(0, shape[0], 256):
                            tensor_rows(
                                source, key, start, min(start + 256, shape[0])
                            ).half().numpy().tofile(f)
                weights.append(
                    dict(
                        tensor=key,
                        shape=shape,
                        dtype="f16",
                        layout="contiguous",
                        data=identity(path, output),
                    )
                )
                continue
            n, k = shape
            if k % 128 or n % 64:
                raise ValueError("W4 draft projection dimensions require N/64 and K/128")
            paths = {kind: output / f"{name}_{kind}.bin" for kind in ("P", "S", "Z")}
            if any(path.exists() for path in paths.values()):
                raise ValueError(f"Refusing existing weight {name}")
            # Whole U4 storage is only N*K/2 bytes; FP32 decoded weights stay
            # bounded to 256 rows. Packing permutes integer codes losslessly.
            codes = np.empty((n, k // 2), dtype=np.uint8)
            scales = np.empty((n, k // 128), dtype=np.float16)
            zeros = np.empty((n, k // 128), dtype=np.int8)
            numerator = denominator = 0.0
            for start in range(0, n, 256):
                end = min(start + 256, n)
                original = tensor_rows(source, key, start, end).reshape(-1, k // 128, 128)
                low = original.amin(-1).clamp(max=0)
                high = original.amax(-1).clamp(min=0)
                scale = ((high - low) / 15).clamp(min=2**-24).half()
                zero = torch.round(-low / scale.float()).clamp(0, 15).to(torch.int8)
                q = (
                    torch.round(original / scale.float()[..., None] + zero[..., None])
                    .clamp(0, 15)
                    .to(torch.uint8)
                )
                actual = (
                    ((q.float() - zero[..., None].float()) * scale[..., None].float())
                    .half()
                    .float()
                )
                numerator += float((actual - original).double().square().sum())
                denominator += float(original.double().square().sum())
                q = q.reshape(-1, k)
                codes[start:end] = (q[:, 0::2] | (q[:, 1::2] << 4)).numpy()
                scales[start:end] = scale.numpy()
                zeros[start:end] = zero.numpy()
            packed = pack_array(codes)
            packed.tofile(paths["P"])
            scales.tofile(paths["S"])
            zeros.tofile(paths["Z"])
            weights.append(
                dict(
                    tensor=key,
                    shape=shape,
                    dtype="w4a16",
                    layout=LAYOUT,
                    packed_shape=list(packed.shape),
                    data={kind: identity(path, output) for kind, path in paths.items()},
                    relative_l2=(numerator / max(denominator, 1e-30)) ** 0.5,
                    method="Uncalibrated asymmetric group128 RNE U4, FP16 scales, numeric I8 zero",
                )
            )
            print("Imported", key, shape, flush=True)
    total_bytes = sum(
        sum(value["bytes"] for value in w["data"].values())
        if w["dtype"] == "w4a16"
        else w["data"]["bytes"]
        for w in weights
    )
    report = dict(
        checkpoint=str(checkpoint),
        format=format,
        weights=weights,
        weight_parameters=parameters,
        weight_bytes=total_bytes,
        effective_weight_bits=8 * total_bytes / parameters,
        shared_embedding_and_head=True,
        source=identity(checkpoint, checkpoint.parent),
    )
    (output / "weights.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--format", choices=("w4", "f16"), default="w4")
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = import_weights(args.checkpoint.resolve(), args.output.resolve(), args.format)
    print(
        json.dumps(
            {k: report[k] for k in ("weight_parameters", "weight_bytes", "effective_weight_bits")}
        )
    )


if __name__ == "__main__":
    main()
