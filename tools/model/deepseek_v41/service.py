"""Publish an HF-style native A2 text-service directory from a compiled rows plan."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile

from tools.build.files import atomic_json, file_hash
from tools.model.deepseek_v41.assembly import publish_package, validate_rows


def state_layout(model, swa_storage_rows=0):
    """Declare checkpoint state separately from transient owner workspaces."""
    buffers = {b["name"]: b for b in model["ranks"][0]["buffers"]}
    result = []
    for name in model["batch"]["private"]:
        if not name.startswith("cache.") or buffers[name].get("view"):
            continue
        size = buffers[name]["bytes"]
        if name.startswith("cache.ring."):
            result.append(dict(buffer=name, row_bytes=size, tokens_per_row=0, window=None))
            continue
        swa = name.startswith("cache.swa.")
        row_bytes = (
            (16 if name.endswith(".scale") else 512)
            if name.startswith(("cache.swa.", "cache.long."))
            else (2 if name.startswith("cache.key_scale.") else 128)
        )
        if size % row_bytes:
            raise ValueError("Persistent state row extent differs: " + name)
        rows = size // row_bytes
        if swa:
            ratio = 1
            if rows != (swa_storage_rows or model["capacity"]):
                raise ValueError("SWA storage contract differs")
        else:
            if model["capacity"] % rows:
                raise ValueError("Compressed state ratio differs")
            ratio = model["capacity"] // rows
            if ratio not in (1, 2):
                raise ValueError("Unsupported compression ratio")
        result.append(
            dict(
                buffer=name,
                row_bytes=row_bytes,
                tokens_per_row=ratio,
                window=128 if swa else None,
                circular=bool(swa and swa_storage_rows),
            )
        )
    if not result:
        raise ValueError("Native A2 package has no declared persistent state")
    return result


def enable_prefix_sharing(model):
    for state in model["batch"]["state"]:
        if (
            state["buffer"].startswith(("cache.long.", "cache.key.", "cache.key_scale."))
            and state["tokens_per_row"] in (1, 2)
            and state["window"] is None
            and not state.get("circular", False)
        ):
            # These writers commit a unique absolute row, with C2 slots
            # published only after both tokens complete. Rings stay copied.
            state["append_only"] = True


def publish(plan, checkpoint, output, share_prefix_banks=False):
    model = deepcopy(json.loads(plan.read_text()))
    if not model.get("batch"):
        raise ValueError("Native A2 service requires shared rows and owner-private state")
    if not model["batch"].get("state"):
        model["batch"]["state"] = state_layout(model)
    if share_prefix_banks:
        enable_prefix_sharing(model)
    validate_rows(model, plan.parent)
    if output.exists():
        raise ValueError("Preserve existing model directories: " + str(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output.name + ".", dir=output.parent))
    try:
        cache = temporary / "cache"
        native = cache / "native-build"
        publish_package(model, plan.parent, native)
        package = dict(
            schema_version=1,
            backend="ascend",
            plan=dict(file="model.json", sha256=file_hash(native / "model.json")),
        )
        atomic_json(native / "package.json", package)
        digest = file_hash(native / "package.json")
        destination = cache / "packages" / digest
        destination.parent.mkdir(parents=True)
        native.rename(destination)
        for name in (
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            model["index"]["file"],
        ):
            shutil.copyfile(checkpoint / name, temporary / name)
        index = json.loads((checkpoint / model["index"]["file"]).read_text())
        for name in sorted(set(index["weight_map"].values())):
            target = temporary / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.hardlink_to((checkpoint / name).resolve())
        tokenizer = json.loads((checkpoint / "tokenizer.json").read_text())
        added = {token["content"]: token["id"] for token in tokenizer["added_tokens"]}
        generation = dict(
            eos_token_id=added["<｜end▁of▁sentence｜>"],
            bos_token_id=added["<｜begin▁of▁sentence｜>"],
            pad_token_id=added["<｜end▁of▁sentence｜>"],
            temperature=0.0,
            top_p=1.0,
        )
        atomic_json(temporary / "generation_config.json", generation)
        shutil.copyfile(
            Path(__file__).with_name("templates") / "chat.jinja", temporary / "chat_template.jinja"
        )
        assets = {
            name: file_hash(temporary / name)
            for name in ("tokenizer.json", "chat_template.jinja", "generation_config.json")
        }
        atomic_json(
            cache / "model.json",
            dict(
                schema_version=1,
                architecture="deepseek_v41",
                compute_policy="w4a8",
                execution_package=digest,
                frontend_assets=assets,
                metadata=dict(target="ascend910b", max_context=model["capacity"], vocab=129280),
            ),
        )
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--share-prefix-banks", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            dict(
                output=str(args.output),
                execution_package=publish(
                    args.plan, args.model_dir, args.output, args.share_prefix_banks
                ),
            )
        )
    )


if __name__ == "__main__":
    main()
