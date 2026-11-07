"""Encode Nemotron-3.5-Embed-1B with the shared two-stage E8P Q4A8 encoder.

Text and vision matrices and embeddings use one compressed bank. Attention
output projections and the final multimodal projection retain row-scaled W8;
patch convolution, norms and small parameters retain their original precision.
"""

import argparse
import json
from pathlib import Path


from orinfer_build.models.nemotron_embed.config import ROTATED_W8_LAYERS, configuration, signature
from orinfer_build.quantization.qwen_e8p import encode


def upgrade(args):
    from orinfer_build.quantization.promotion import upgrade as promote

    return promote(
        args,
        configuration=configuration,
        signature=signature,
        layers=ROTATED_W8_LAYERS,
        layer_prefix="language_model.layers.",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--upgrade-from",
        type=Path,
        help="Retain a completed E8P bank and promote the protected BF16 matrices",
    )
    p.add_argument("--shard")
    p.add_argument("--rows", type=int, default=512)
    p.add_argument("--gpu-lock", type=Path, default=Path("artifacts/gpu-experiment.lock"))
    args = p.parse_args()
    if not 1 <= args.rows <= 1024:
        p.error("Row block must be in 1..1024")
    if args.upgrade_from:
        upgrade(args)
    else:
        configuration(json.loads((args.source_dir / "config.json").read_text()))
        encode(args)


if __name__ == "__main__":
    main()
