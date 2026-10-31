"""Export DSpark physical tensors while sharing the released target shards.

Draft weights use three-layer rank shards and one shared shard. Operators and
execution plans are published into the separate execution cache.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from copy import deepcopy
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

from tools.build.files import atomic_json
from tools.model.deepseek_v41.export_release import Export, DTYPES, physical_contract
from tools.model.safetensors_source import SIZES


class DraftExport(Export):
    def __init__(self, source, base_model, plan, output, execution_output):
        self.source, self.output = Path(source).resolve(), Path(output).resolve()
        self.execution_output = Path(execution_output).resolve()
        if self.execution_output.is_relative_to(self.output) or self.output.is_relative_to(
            self.execution_output
        ):
            raise ValueError("Weights and execution outputs must be separate")
        self.base_model = Path(base_model).resolve()
        self.attribution = self.base_model
        self.package = Path(plan).resolve().parent
        self.model = json.loads(Path(plan).read_text())
        self.block_size = max(
            p["coordinates"]["draft_block"]
            for p in self.model["ranks"][0]["packets"]
            if p.get("role") == "draft"
        )
        self.old_index = json.loads((self.source / "model.safetensors.index.json").read_text())
        self.headers, self.entries, self.groups = {}, {}, defaultdict(list)
        self.base_index = json.loads((self.base_model / "model.safetensors.index.json").read_text())
        self.base_manifest = json.loads((self.base_model / "release-manifest.json").read_text())

    def plan(self):
        self.index = deepcopy(self.base_index)
        self.manifest = deepcopy(self.base_manifest)
        self.manifest["dspark"] = dict(
            layers=[40, 41, 42],
            block_size=self.block_size,
            target_layers=[37, 38, 39],
            kv="int8_group64_fp16_scale",
        )
        for name, file in self.base_index["weight_map"].items():
            if self.old_index["weight_map"].get(name) != file:
                raise ValueError("DSpark source changes an existing target tensor")
        specs = {}
        for rank in self.model["ranks"]:
            physical = physical_contract(rank, self.package, specs)
            for b in rank["buffers"]:
                if b.get("prepared_tensor"):
                    continue
                if "prepared_weight" in b:
                    key = "bank:" + b["prepared_weight"]
                    dtype, shape = physical[b["name"]]
                    source = dict(file=str(self.package / b["prepared_weight"]), offset=0)
                elif "weight" in b:
                    w = b["weight"]
                    if (
                        w["kind"] == "matrix"
                        and self.tensor(w["name"])["dtype"] != DTYPES[w["dtype"]]
                    ):
                        raise ValueError("DSpark source matrix dtype differs: " + w["name"])
                    key = "transform:" + json.dumps(w, sort_keys=True)
                    dtype = (
                        "F32" if w.get("float32") or w["kind"] != "matrix" else DTYPES[w["dtype"]]
                    )
                    shape = (
                        [w["rows"], w["columns"]]
                        if w["kind"] == "matrix"
                        else [b["bytes"] // SIZES[dtype]]
                    )
                    source = dict(transform=w)
                elif "data" in b:
                    key = "data:" + b["data"]["file"]
                    if b["name"] not in physical:
                        continue
                    dtype, shape = physical[b["name"]]
                    source = dict(file=str(self.package / b["data"]["file"]), offset=0)
                else:
                    continue
                if math.prod(shape) * SIZES[dtype] != b["bytes"]:
                    raise ValueError("DSpark tensor extent differs")
                if key not in self.entries:
                    self.entries[key] = dict(
                        source=source, dtype=dtype, shape=shape, bytes=b["bytes"], refs=[]
                    )
                entry = self.entries[key]
                if (entry["dtype"], entry["shape"]) != (dtype, shape):
                    raise ValueError("Conflicting DSpark tensor interpretation")
                entry["refs"].append((rank["rank"], b))
        for i, entry in enumerate(self.entries.values()):
            ranks = sorted({r for r, _ in entry["refs"]})
            tensor = f"dspark.{i:05}." + entry["refs"][0][1]["name"]
            group = (
                "weights/dspark/shared.safetensors"
                if len(ranks) > 1
                else f"weights/dspark/rank-{ranks[0]:02}-layers-40-42.safetensors"
            )
            self.add(group, tensor, entry)
            for _, b in entry["refs"]:
                for field in ("prepared_weight", "weight", "data"):
                    b.pop(field, None)
                b["prepared_tensor"] = dict(name=tensor, dtype=entry["dtype"], shape=entry["shape"])
        embed = self.tensor("mtp.0.embed.weight")
        self.add(
            "weights/dspark/embedding.safetensors",
            "mtp.0.embed.weight",
            dict(
                source=embed,
                dtype=embed["dtype"],
                shape=embed["shape"],
                bytes=math.prod(embed["shape"]) * SIZES[embed["dtype"]],
            ),
        )
        if any(sum(e["bytes"] for _, e in group) > 4 << 30 for group in self.groups.values()):
            raise ValueError("DSpark shard exceeds the transfer bound")
        return self

    def export(self, workers):
        self.output.mkdir(parents=True, exist_ok=False)
        for name, record in self.base_manifest["files"].items():
            source, target = self.base_model / name, self.output / name
            if source.stat().st_size != record["bytes"]:
                raise ValueError("Base shard size differs")
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(source, target)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for name, record in pool.map(self.write_shard, self.groups):
                self.manifest["files"][name] = record
                print(json.dumps(dict(file=name, bytes=record["bytes"])), flush=True)
        self.finish()
        atomic_json(
            self.output / "dspark.json",
            dict(
                block_size=self.block_size,
                draft_layers=3,
                draft_sampling="deterministic_markov",
                verification="target_probability_preserving",
                base_shards="unchanged",
            ),
        )


def prepare_source(base_model, checkpoint, temporary):
    base_model, checkpoint = Path(base_model), Path(checkpoint)
    for filename in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
        "chat_template.jinja",
        "LICENSE",
        "SOURCE-README.md",
    ):
        source = base_model / filename
        if source.exists():
            shutil.copyfile(source, temporary / filename)
    (temporary / "cache").mkdir()
    shutil.copyfile(base_model / "cache/model.json", temporary / "cache/model.json")
    index = json.loads((base_model / "model.safetensors.index.json").read_text())
    for name in sorted(set(index["weight_map"].values())):
        target = temporary / name
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(base_model / name, target)
    extra = json.loads((checkpoint / "quant_model_weights.safetensors.index.json").read_text())[
        "weight_map"
    ]
    draft = {
        name: "dspark-source/" + file for name, file in extra.items() if name.startswith("mtp.")
    }
    for filename in sorted(set(draft.values())):
        target = temporary / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(checkpoint / filename.removeprefix("dspark-source/"), target)
    index["weight_map"].update(draft)
    atomic_json(temporary / "model.safetensors.index.json", index)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execution-output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".dspark-export-", dir=args.output.parent) as temporary:
        source = Path(temporary)
        prepare_source(args.base_model, args.checkpoint, source)
        DraftExport(
            source, args.base_model, args.plan, args.output, args.execution_output
        ).plan().export(args.workers)


if __name__ == "__main__":
    main()
