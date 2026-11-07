"""Encode Qwen3-Reranker-0.6B with shared E8P Q4A8 and exact two-class head."""

import argparse
import json
from pathlib import Path
import numpy as np
from safetensors.numpy import save_file

from orinfer_build.common.files import file_hash, write_json
from orinfer_build.models.qwen_reranker.config import configuration, signature, ROTATED_W8_LAYERS
from orinfer_build.weights.safetensors_source import Source
from orinfer_build.quantization.qwen_e8p import encode


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--upgrade-from", type=Path)
    p.add_argument("--shard")
    p.add_argument("--rows", type=int, default=512)
    p.add_argument("--gpu-lock", type=Path, default=Path("artifacts/gpu-experiment.lock"))
    args = p.parse_args()
    if not 1 <= args.rows <= 1024:
        p.error("Row block must be in 1..1024")
    if args.upgrade_from:
        from orinfer_build.quantization.promotion import upgrade

        upgrade(
            args,
            configuration=configuration,
            signature=signature,
            layers=ROTATED_W8_LAYERS,
            layer_prefix="model.layers.",
        )
        return
    configuration(json.loads((args.source_dir / "config.json").read_text()))
    encode(args)
    progress = args.output / "progress.json"
    state = json.loads(progress.read_text())
    if not state.get("complete"):
        return
    source = Source(args.source_dir / "model.safetensors.index.json", directory=args.source_dir)
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(args.source_dir / "tokenizer.json"))
    ids = [tokenizer.token_to_id(t) for t in ["no", "yes"]]
    if any(i is None for i in ids) or len(set(ids)) != 2:
        raise ValueError("Missing yes/no token ids")
    name = "model.embed_tokens.weight"
    if source.tensor(name)[2]["dtype"] != "BF16":
        raise ValueError("Classifier requires original BF16")
    rows = []
    for token in ids:
        raw = source.rows(name, token, 1)
        row = (
            (np.frombuffer(raw, "<u2").astype(np.uint32) << 16).view(np.float32).astype(np.float16)
        )
        if not np.isfinite(row).all():
            raise ValueError("Classifier FP16 overflow")
        rows.append(row)
    path = args.output / "classifier.safetensors"
    key = "orinfer.reranker.classifier.weight"
    save_file(
        {key: np.stack(rows)}, str(path), metadata={f"orin.layout.{key}": "native_contiguous"}
    )
    index = json.loads((args.output / "model.safetensors.index.json").read_text())
    index["weight_map"][key] = path.name
    index["metadata"]["orinfer.sha256"][path.name] = file_hash(path)
    index["metadata"]["total_size"] += 4096
    write_json(args.output / "model.safetensors.index.json", index)
    config = json.loads((args.output / "config.json").read_text())
    config["quantization_config"]["classifier_token_ids"] = ids
    config["quantization_config"]["classifier"] = "fp16-original-two-rows"
    write_json(args.output / "config.json", config)
    state["weight_bytes"] = index["metadata"]["total_size"]
    write_json(progress, state)
    print("CLASSIFIER", ids, "COMPLETE", state["weight_bytes"], flush=True)


if __name__ == "__main__":
    main()
