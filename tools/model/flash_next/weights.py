"""Audit and losslessly publish Flash Next quantized safetensors shards.

The converter's progress files are offline inputs only. The published directory
has an HF config and standard safetensors index with globally unique tensor
names. Publishing consumes our temporary converted shards after their exact
payload has been verified in the destination; it never keeps a second model.
"""

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct

import numpy as np
from safetensors import safe_open

from tools.model.safetensors_source import Source, local_header, relative_name
from tools.quantization.flash_next import atomic_json, digest
from tools.quantization.flash_next_aux import policy
from tools.quantization.e8p import basis


FRONTEND = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
    "merges.txt",
    "vocab.json",
)


def header(path):
    return local_header(path)


def payload_hash(path):
    begin, _ = header(path)
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        f.seek(begin)
        for raw in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(raw)
    return h.hexdigest()


def source_and_records(converted, index):
    converted = Path(converted)
    expert = json.loads((converted / "experts-progress.json").read_text())
    auxiliary = json.loads((converted / "aux-progress.json").read_text())
    contract = json.loads(expert["contract"])
    aux_contract = json.loads(auxiliary["contract"])
    if (
        contract["source_index_sha256"] != digest(index)
        or aux_contract["source_contract"] != expert["contract"]
    ):
        raise ValueError("Source index or conversion contracts disagree")
    source = Source(
        index,
        repo=contract["source"],
        revision=contract["revision"],
        cache=converted / "source-headers",
    )
    records = []
    filenames = set()
    for group, items in (("experts", expert["shards"]), ("aux", auxiliary["shards"])):
        for filename, record in items.items():
            relative_name(filename)
            if filename != record["filename"] or filename in filenames:
                raise ValueError("Duplicate or inconsistent converted filename")
            filenames.add(filename)
            item = dict(record, group=group)
            if group == "experts":
                item.update(
                    kind="e8p-expert", first=record["first_expert"], count=record["shape"][0]
                )
            else:
                item.update(first=record["first_row"], count=record["rows"])
            records.append(item)
    return source, contract, sorted(records, key=lambda r: (r["tensor"], r["first"]))


def coverage(source, records, *, component="text"):
    if component not in ("text", "mtp"):
        raise ValueError("Invalid Flash component")
    expected = {
        n
        for n in source.weight_map
        if (
            n.startswith("mtp.")
            if component == "mtp"
            else n.startswith("model.language_model.") or n == "lm_head.weight"
        )
    }
    groups = defaultdict(list)
    for r in records:
        if r["tensor"] not in expected:
            raise ValueError("Unexpected converted tensor")
        if (
            type(r["first"]) is not int
            or type(r["count"]) is not int
            or r["first"] < 0
            or r["count"] <= 0
        ):
            raise ValueError("Invalid converted tensor span")
        groups[r["tensor"]].append(r)
    missing, parameters = [], 0
    for name in sorted(expected):
        _, _, info = source.tensor(name)
        shape = info["shape"]
        parameters += math.prod(shape)
        rows = shape[0] if shape else 1
        cursor = 0
        for r in sorted(groups[name], key=lambda r: r["first"]):
            if r["first"] < cursor or r["first"] + r["count"] > rows:
                raise ValueError(f"Overlapping or oversized span: {name}")
            if r["first"] != cursor:
                missing.append({"tensor": name, "first": cursor, "count": r["first"] - cursor})
            wanted = "e8p-expert" if policy(name, info) == "experts" else policy(name, info)
            if r["kind"] != wanted:
                raise ValueError("Unexpected mixed-precision policy")
            if r["group"] == "experts":
                if r["shape"] != [r["count"], *shape[1:]]:
                    raise ValueError("Expert logical shape mismatch")
            elif r["source_shape"] != shape or r["dtype"] != info["dtype"]:
                raise ValueError("Auxiliary source geometry changed")
            cursor = r["first"] + r["count"]
        if cursor != rows:
            missing.append({"tensor": name, "first": cursor, "count": rows - cursor})
    return {
        "complete": not missing,
        "source_tensors": len(expected),
        "parameters": parameters,
        "missing": missing,
    }


def check_piece(path, record, source):
    if digest(path) != record["sha256"]:
        raise ValueError(f"Converted shard hash mismatch: {path.name}")
    begin, data = header(path)
    meta = data.get("__metadata__", {})
    name = record["tensor"]
    _, _, info = source.tensor(name)
    if (
        meta.get("source_tensor") != name
        or meta.get("source_range_sha256") != record["source_range_sha256"]
    ):
        raise ValueError("Converted shard source identity mismatch")
    shape = info["shape"]
    count = record["count"]
    kind = record["kind"]
    tensors = {k: v for k, v in data.items() if k != "__metadata__"}
    if kind == "e8p-expert":
        _, n, k = shape
        wanted = {
            "indices": ("U16", [count, k // 128, n, 16]),
            "scales": ("F16", [count, n]),
            "table": ("I8", [256, 8]),
            "signs": ("I8", [k]),
        }
        if (
            meta.get("first_expert") != str(record["first"])
            or json.loads(meta.get("logical_shape", "null")) != record["shape"]
        ):
            raise ValueError("Expert shard metadata span mismatch")
    elif kind == "int8-row":
        wanted = {"weight": ("I8", [count, shape[1]]), "scale": ("F16", [count])}
    elif kind == "e8p-embedding":
        wanted = {"rows": ("U8", [count, 42]), "table": ("I8", [256, 8]), "signs": ("I8", [160])}
    else:
        wanted = {name: (info["dtype"], shape)}
        if payload_hash(path) != record["source_range_sha256"]:
            raise ValueError("Original-precision tensor bytes changed")
    if set(tensors) != set(wanted) or any(
        tensors[k]["dtype"] != dtype or tensors[k]["shape"] != size
        for k, (dtype, size) in wanted.items()
    ):
        raise ValueError("Converted physical layout mismatch")
    if record["group"] == "aux" and (
        meta.get("first_row") != str(record["first"]) or meta.get("rows") != str(count)
    ):
        raise ValueError("Auxiliary shard metadata span mismatch")
    if kind != "original":
        with safe_open(path, framework="np") as f:
            if kind.startswith("e8p"):
                if not np.array_equal(f.get_tensor("table"), basis()):
                    raise ValueError("E8P basis changed")
                if not np.isin(f.get_tensor("signs"), [-1, 1]).all():
                    raise ValueError("Invalid input rotation")
            if kind == "e8p-embedding":
                packets = f.get_tensor("rows")
                scales = np.ascontiguousarray(packets[:, :2]).view("<f2")
            else:
                scales = f.get_tensor("scales" if kind == "e8p-expert" else "scale")
            if not np.isfinite(scales).all() or (scales <= 0).any():
                raise ValueError("Invalid quantized scale")
    return begin, data


def remap(path, target, record, source):
    """Rename tensor keys only; copy the payload without conversion or reorder."""
    begin, data = check_piece(path, record, source)
    old_meta = data.pop("__metadata__", {})
    _, _, info = source.tensor(record["tensor"])
    prefix = f"{record['tensor']}.part_{record['first']:09d}"
    keys = {k: k if record["kind"] == "original" else f"{prefix}.{k}" for k in data}
    renamed = {keys[k]: v for k, v in data.items()}
    renamed["__metadata__"] = dict(
        old_meta,
        format="orinfer.flash_next.weights.v1",
        kind=record["kind"],
        source_shape=json.dumps(info["shape"]),
        first=str(record["first"]),
        count=str(record["count"]),
        keys=json.dumps(keys, sort_keys=True),
    )
    raw = json.dumps(renamed, sort_keys=True, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    temporary = target.with_suffix(".tmp")
    with path.open("rb") as inp, temporary.open("wb") as out:
        inp.seek(begin)
        out.write(struct.pack("<Q", len(raw)))
        out.write(raw)
        shutil.copyfileobj(inp, out, length=4 * 1024**2)
        out.flush()
        os.fsync(out.fileno())
    if payload_hash(temporary) != payload_hash(path):
        raise ValueError("Packaging changed the quantized payload")
    if target.exists() and digest(target) != digest(temporary):
        raise ValueError("Unrecorded destination differs")
    temporary.replace(target)
    return keys


def publish(converted, output, index, config, frontend, *, consume=True):
    converted, output = Path(converted), Path(output)
    if converted.resolve() == output.resolve():
        raise ValueError("Publication must use a separate directory")
    source, contract, records = source_and_records(converted, index)
    component = contract.get("component", "text")
    audit = coverage(source, records, component=component)
    if not audit["complete"]:
        raise ValueError("Conversion is incomplete; refusing publication")
    configuration = json.loads(Path(config).read_text())
    if configuration.get("model_type") != "qwen4_exp":
        raise ValueError("Flash Next configuration required")
    layers = (
        configuration["text_config"].get("mtp_num_hidden_layers")
        if component == "mtp"
        else configuration["text_config"]["num_hidden_layers"]
    )
    if contract["layers"] != [0, layers]:
        raise ValueError("Conversion does not cover configured text layers")
    if not all(
        (Path(frontend) / name).is_file() for name in ("tokenizer.json", "tokenizer_config.json")
    ):
        raise ValueError("Tokenizer files are required before publication")
    frontend_hashes = {
        name: digest(Path(frontend) / name) for name in FRONTEND if (Path(frontend) / name).exists()
    }
    state_path = output / "publication-progress.json"
    if output.exists() and not state_path.exists() and any(output.iterdir()):
        raise ValueError("Refusing a nonempty unrelated publication directory")
    output.mkdir(parents=True, exist_ok=True)
    signature = dict(
        contract=contract,
        records_sha256=hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest(),
        config_sha256=digest(config),
        frontend_sha256=frontend_hashes,
        consume=consume,
    )
    state = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else dict(signature=signature, shards={}, complete=False)
    )
    if state["signature"] != signature:
        raise ValueError("Publication inputs changed")
    weight_map = {}
    total_bytes = 0
    payload_bytes = 0
    for i, r in enumerate(records):
        name = f"model-{i + 1:05d}-of-{len(records):05d}.safetensors"
        target = output / name
        path = converted / r["filename"]
        if name in state["shards"]:
            done = state["shards"][name]
            if digest(target) != done["sha256"]:
                raise ValueError("Published shard changed")
            _, data = header(target)
            keys = {k: k for k in data if k != "__metadata__"}
            if path.exists() and consume:
                if digest(path) != r["sha256"] or payload_hash(target) != payload_hash(path):
                    raise ValueError("Converted/published payload disagreement")
                path.unlink()
        else:
            keys = remap(path, target, r, source)
            state["shards"][name] = {
                "sha256": digest(target),
                "source_filename": r["filename"],
                "source_sha256": r["sha256"],
                "payload_sha256": payload_hash(target),
            }
            atomic_json(state_path, state)
            # Commit the new location before consuming this temporary source.
            if consume:
                path.unlink()
        for key in keys.values():
            if key in weight_map:
                raise ValueError("Duplicate published tensor key")
            weight_map[key] = name
        total_bytes += target.stat().st_size
        begin, _ = header(target)
        payload_bytes += target.stat().st_size - begin
        if (i + 1) % 100 == 0:
            print(json.dumps({"packaged": i + 1, "total": len(records)}), flush=True)
    configuration["language_model_only"] = True
    configuration["quantization_config"] = {
        "quant_method": "orinfer_e8p_int8",
        "version": 1,
        "basis": contract["basis"],
        "expert_rotation": contract["rotation"],
        "embedding_rotation": "paley20-walsh8",
        "calibration": "weight-only",
        "compute_dtype": "int8_quality",
        "source": contract["source"],
        "source_revision": contract["revision"],
        "seed": contract["seed"],
    }
    if component == "mtp":
        configuration["quantization_config"]["component"] = "mtp"
    atomic_json(output / "config.json", configuration)
    for name in FRONTEND:
        path = Path(frontend) / name
        if path.exists():
            target = output / name
            if target.exists() and digest(target) != digest(path):
                raise ValueError("Frontend file changed")
            shutil.copyfile(path, target)
    if not (output / "tokenizer.json").exists() or not (output / "tokenizer_config.json").exists():
        raise ValueError("Tokenizer files are required")
    atomic_json(
        output / "model.safetensors.index.json",
        {
            "metadata": {
                "total_size": payload_bytes,
                "orinfer.sha256": {name: r["sha256"] for name, r in state["shards"].items()},
            },
            "weight_map": weight_map,
        },
    )
    state.update(
        complete=True,
        weight_bytes=total_bytes,
        bits_per_weight=8 * total_bytes / audit["parameters"],
        original_parameters=audit["parameters"],
        full_model_quality_verified=False,
    )
    atomic_json(state_path, state)
    return state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--converted", type=Path, required=True)
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--config", type=Path)
    p.add_argument("--frontend", type=Path)
    p.add_argument("--keep-temporary-shards", action="store_true")
    a = p.parse_args()
    if a.output:
        if not a.config or not a.frontend:
            p.error("Publication requires --config and --frontend")
        state = publish(
            a.converted,
            a.output,
            a.index,
            a.config,
            a.frontend,
            consume=not a.keep_temporary_shards,
        )
        print(
            json.dumps({k: v for k, v in state.items() if k not in ("shards", "signature")}),
            flush=True,
        )
    else:
        source, contract, records = source_and_records(a.converted, a.index)
        audit = coverage(source, records, component=contract.get("component", "text"))
        for r in records:
            check_piece(a.converted / r["filename"], r, source)
        audit.update(
            verified_shards=len(records),
            weight_bytes=sum((a.converted / r["filename"]).stat().st_size for r in records),
        )
        audit["missing_spans"] = len(audit.pop("missing"))
        print(json.dumps(audit), flush=True)


if __name__ == "__main__":
    main()
