"""Prepare small original-BF16 embedding/PLE fixtures for fixed histories."""

import argparse
import bisect
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

from tools.eval.scoring_common import SEED
from tools.model.flash_next.reference.original import Original
from tools.model.flash_next.ple import PleLookup
from tools.model.safetensors_source import Source
from tools.quantization.flash_next import atomic_json, digest


class Inputs:
    def __init__(self, directory, *, source, revision, config_sha256, scenes_sha256, index_sha256):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "inputs.json").read_text())
        wanted = {
            "source": source,
            "revision": revision,
            "config_sha256": config_sha256,
            "scenes_sha256": scenes_sha256,
            "index_sha256": index_sha256,
            "seed": SEED,
            "format": "orinfer.original-bf16-inputs.v1",
        }
        if self.manifest.get("complete") is not True or self.manifest["contract"] != wanted:
            raise ValueError("Original input fixture identity changed or is incomplete")
        if digest(self.directory / "inputs.safetensors") != self.manifest["sha256"]:
            raise ValueError("Original input payload changed")

    def case(self, case, width):
        tokens = case["prompt_ids"] + case["target_ids"][:-1]
        if self.manifest["tokens"].get(case["id"]) != tokens:
            raise ValueError("Different teacher-forced history")
        with safe_open(self.directory / "inputs.safetensors", framework="np") as f:
            values = [f.get_tensor(kind + "." + case["id"]) for kind in ("embedding", "ple")]
        for value in values:
            if (
                value.shape != (len(tokens), width)
                or value.dtype != np.float32
                or not np.isfinite(value).all()
                or (value.view(np.uint32) & 0xFFFF).any()
            ):
                raise ValueError("Expected exact original BF16 values in FP32 containers")
        return values


def prepare(original, cases, output, contract, *, workers=4):
    output = Path(output)
    if (output / "inputs.json").exists():
        cached = Inputs(
            output,
            **{
                key: contract[key]
                for key in ("source", "revision", "config_sha256", "scenes_sha256", "index_sha256")
            },
        )
        for case in cases:
            cached.case(case, original.config["text_config"]["hidden_size"])
        return cached.manifest
    marker = output / "inputs-contract.json"
    if output.exists() and any(output.iterdir()) and not marker.exists():
        raise ValueError("Refusing unrelated nonempty input directory")
    output.mkdir(parents=True, exist_ok=True)
    if marker.exists() and json.loads(marker.read_text()) != contract:
        raise ValueError("Incomplete input fixture contract changed")
    atomic_json(marker, contract)
    tokens = {case["id"]: case["prompt_ids"] + case["target_ids"][:-1] for case in cases}
    if len(tokens) != len(cases):
        raise ValueError("Duplicate original scene ID")
    embedding_name = "model.language_model.embed_tokens.weight"
    if original.info(embedding_name)["dtype"] != "BF16":
        raise ValueError("Original BF16 embedding required")
    lookup = PleLookup(original)
    if any(original.info(name)["dtype"] != "BF16" for name in lookup.names):
        raise ValueError("Original BF16 PLE required")
    ids = {name: lookup.row_ids(values, [])[0] for name, values in tokens.items()}
    rows = {}
    for row in sorted({row for values in ids.values() for row in values}):
        shard = bisect.bisect_right(lookup.starts, row) - 1
        rows[row] = (lookup.names[shard], row - lookup.starts[shard], 1)
    embedding_requests = {
        t: (embedding_name, t, 1) for t in sorted({t for values in tokens.values() for t in values})
    }
    print(
        json.dumps(
            {
                "phase": "fetch-original-inputs",
                "embedding_rows": len(embedding_requests),
                "ple_rows": len(rows),
            }
        ),
        flush=True,
    )
    fetched = original.batch_rows([*embedding_requests.values(), *rows.values()], workers=workers)
    tensors = {}
    for name, values in tokens.items():
        tensors["embedding." + name] = np.stack([fetched[embedding_requests[t]][0] for t in values])
        tensors["ple." + name] = np.stack([fetched[rows[row]][0] for row in ids[name]]).reshape(
            len(values), -1
        )
    temporary = output / "inputs.safetensors.tmp"
    save_file(
        tensors,
        str(temporary),
        metadata={"format": contract["format"], "precision": "original-bf16"},
    )
    temporary.replace(output / "inputs.safetensors")
    manifest = {
        "contract": contract,
        "tokens": tokens,
        "sha256": digest(output / "inputs.safetensors"),
        "source_ranges": list(original.reads.values()),
        "complete": True,
    }
    atomic_json(output / "inputs.json", manifest)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("index", "config", "scenes", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--repo", default="Qwen/Qwen3.8-Flash-Next")
    p.add_argument("--revision", required=True)
    p.add_argument("--header-cache", type=Path)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    scenes = json.loads(a.scenes.read_text())
    if scenes["seed"] != SEED:
        raise ValueError("Different original input seed")
    source = Original(
        Source(a.index, repo=a.repo, revision=a.revision, cache=a.header_cache),
        json.loads(a.config.read_text()),
    )
    contract = {
        "source": a.repo,
        "revision": a.revision,
        "config_sha256": digest(a.config),
        "scenes_sha256": digest(a.scenes),
        "index_sha256": digest(a.index),
        "seed": SEED,
        "format": "orinfer.original-bf16-inputs.v1",
    }
    result = prepare(source, scenes["cases"], a.output, contract, workers=a.workers)
    print(
        json.dumps(
            {
                "complete": result["complete"],
                "cases": len(result["tokens"]),
                "source_ranges": len(result["source_ranges"]),
                "bytes": (a.output / "inputs.safetensors").stat().st_size,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
