"""Write full-checkpoint vision fixtures using the model author's PyTorch implementation."""

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


def main():
    import torch
    import torch_npu  # noqa: F401
    from safetensors.torch import load_file

    from orinfer_build.common.files import atomic_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=7)
    parser.add_argument("--large-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads((args.source / "config.json").read_text())
    v = config["vision_config"]
    spec = importlib.util.spec_from_file_location(
        "checkpoint_vision", args.source / "inference/vision.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    architecture = SimpleNamespace(
        vision_patch_size=v["patch_size"],
        vision_dim=v["hidden_size"],
        vision_inter_dim=v["intermediate_size"],
        vision_n_heads=v["num_attention_heads"],
        vision_n_layers=v["num_hidden_layers"],
        vision_rope_theta=v["rope_theta"],
        vision_downsample_ratio=v["downsample_ratio"],
        dim=config["text_config"]["hidden_size"],
    )
    weights = load_file(args.weights / "vision.safetensors")
    torch.npu.set_device(args.device)
    device = f"npu:{args.device}"
    # The checkpoint norms are BF16 tensors; the author's RMSNorm promotes to FP32.
    encoder = module.ViT(architecture).to(dtype=torch.bfloat16)
    aligner = module.Aligner(architecture).to(dtype=torch.bfloat16)
    vision = {
        name.removeprefix("vision."): value
        for name, value in weights.items()
        if name.startswith("vision.")
    }
    vision["patch_embed.proj.weight"] = vision["patch_embed.proj.weight"][:, :588].contiguous()
    encoder.load_state_dict(vision, strict=True)
    aligner.load_state_dict(
        {
            name.removeprefix("aligner."): value
            for name, value in weights.items()
            if name.startswith("aligner.")
        },
        strict=True,
    )
    encoder, aligner = encoder.to(device).eval(), aligner.to(device).eval()
    cases = []
    torch.manual_seed(20261002)
    grids = (
        [("image-63x63", 63, 63), ("image-87x90", 87, 90)]
        if args.large_only
        else [
            ("odd-2x4", 2, 4),
            ("odd-5x7", 5, 7),
            ("changed-5x7", 5, 7),
            ("image-39x39", 39, 39),
            ("changed-aspect-7x5", 7, 5),
        ]
    )
    for name, h, w in grids:
        pixels = torch.rand((h * w, 588), dtype=torch.float32) * 2 - 1
        x = pixels.to(device=device, dtype=torch.bfloat16)
        cos, sin = module.get_vision_cos_sin(h, w, 32, v["rope_theta"])
        cos, sin = cos.to(device), sin.to(device)
        with torch.inference_mode():
            x = encoder.patch_embed(x)
            for block in encoder.blocks:
                x = block(x, cos, sin)
            x = encoder.norm(x)
            aligned = aligner(x, h, w).cpu()
        span = [weights["image_start"].reshape(1, -1)]
        for row in aligned.reshape((h + 2) // 3, (w + 2) // 3, -1):
            span.extend([row, weights["image_newline"].reshape(1, -1)])
        span.append(weights["image_end"].reshape(1, -1))
        expected = torch.cat(span).contiguous()
        filename = name + ".bf16"
        (args.output / filename).write_bytes(expected.view(torch.uint8).numpy().tobytes())
        # JSON fixtures are offline-only and deliberately contain exact pre-BF16 input.
        cases.append(
            dict(
                name=name,
                image=dict(grid_height=h, grid_width=w, pixels=pixels.flatten().tolist()),
                expected=filename,
            )
        )
        print(name, list(expected.shape), flush=True)
    if not args.large_only:
        # Same graph geometry, different 2-D RoPE/grid, then restore the old
        # input. This catches stale grid uploads when graph slots are reused.
        cases.append(dict(cases[1], name="restored-5x7"))
    atomic_json(args.output / "fixtures.json", dict(cases=cases))


if __name__ == "__main__":
    main()
