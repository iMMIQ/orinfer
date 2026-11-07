"""Independent Transformers BF16 teacher-forced prefill with streamed weights.

Only one text component's weights reside on CUDA at a time. Checkpoint tensors
retain their stored dtypes; no W4 reconstruction or candidate kernel is used.
This freezes a full-sequence reference, not a BF16 autoregressive timing test.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import json
import math
from pathlib import Path
import time

import torch
from safetensors import safe_open
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig
from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling
from orinfer_build.validation.scoring_common import (
    SEED,
    context_hash,
    image_fingerprints,
    image_patches,
)
from orinfer_build.weights.safetensors_source import Source, SIZES


def configure_torch_reference(backend=modeling):
    """Freeze the official Torch equations before optional kernel dispatch."""
    functions = {}
    for name in (
        "causal_conv1d_fn",
        "causal_conv1d_update",
        "torch_chunk_gated_delta_rule",
        "torch_recurrent_gated_delta_rule",
    ):
        function = inspect.unwrap(getattr(backend, name))
        setattr(backend, name, function)
        functions[name] = hashlib.sha256(inspect.getsource(function).encode()).hexdigest()
    return functions


class PinnedTensors:
    """Bounded concurrent reads of original tensors, without a full checkpoint."""

    def __init__(self, checkpoint, repo, revision, output, workers=8, cache=None):
        self.source = Source(
            checkpoint / "model.safetensors.index.json",
            repo=repo,
            revision=revision,
            cache=output.parent / "headers",
        )
        self.workers, self.cache = workers, cache
        self.progress = output.with_suffix(".weights.json")
        self.records = {}

    def raw(self, name):
        filename, begin, info = self.source.tensor(name)
        size = math.prod(info["shape"]) * SIZES[info["dtype"]]
        raw = bytearray(size)
        chunk = 32 * 1024**2

        def read(offset):
            return offset, self.source.read(
                filename, begin + info["data_offsets"][0] + offset, min(chunk, size - offset)
            )

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for offset, data in pool.map(read, range(0, size, chunk)):
                raw[offset : offset + len(data)] = data
        return raw, info

    def tensor(self, name):
        cache = (
            self.cache / (hashlib.sha256(name.encode()).hexdigest() + ".safetensors")
            if self.cache
            else None
        )
        if cache and cache.exists():
            with safe_open(cache, framework="pt", device="cpu") as reader:
                identity = reader.metadata()
                if (
                    identity["repo"] != self.source.repo
                    or identity["revision"] != self.source.revision
                ):
                    raise ValueError("Prefetched original tensor identity mismatch")
                cpu = reader.get_tensor(name)
                raw = cpu.view(torch.uint8).numpy().tobytes()
                digest = hashlib.sha256(raw).hexdigest()
                if digest != identity["sha256"]:
                    raise ValueError("Prefetched original tensor payload changed")
                result = cpu.to("cuda")
            size = len(raw)
            cache.unlink()
        else:
            raw, info = self.raw(name)
            dtype = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}[
                info["dtype"]
            ]
            result = torch.frombuffer(raw, dtype=dtype).reshape(info["shape"]).to("cuda")
            digest, size = hashlib.sha256(raw).hexdigest(), len(raw)
        self.records[name] = dict(bytes=size, sha256=digest)
        self.progress.write_text(json.dumps(self.records) + "\n")
        print("BF16 tensor", name, size, flush=True)
        return result

    def tensors(self, names):
        """Pool byte ranges across one component, including its small tensors."""
        result, pending = {}, {}
        for name in names:
            path = (
                self.cache / (hashlib.sha256(name.encode()).hexdigest() + ".safetensors")
                if self.cache
                else None
            )
            if path and path.exists():
                result[name] = self.tensor(name)
                continue
            filename, begin, info = self.source.tensor(name)
            size = math.prod(info["shape"]) * SIZES[info["dtype"]]
            pending[name] = (filename, begin + info["data_offsets"][0], info, bytearray(size))
        chunk = 32 * 1024**2
        jobs = [
            (name, offset)
            for name, (_, _, _, data) in pending.items()
            for offset in range(0, len(data), chunk)
        ]

        def read(job):
            name, offset = job
            filename, start, _, raw = pending[name]
            return (
                name,
                offset,
                self.source.read(filename, start + offset, min(chunk, len(raw) - offset)),
            )

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for name, offset, data in pool.map(read, jobs):
                pending[name][3][offset : offset + len(data)] = data
        for name in list(pending):
            _, _, info, raw = pending.pop(name)
            dtype = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}[
                info["dtype"]
            ]
            result[name] = torch.frombuffer(raw, dtype=dtype).reshape(info["shape"]).to("cuda")
            self.records[name] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
            print("BF16 tensor", name, len(raw), flush=True)
        self.progress.write_text(json.dumps(self.records) + "\n")
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--requests", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--remote-repo", help="Read original tensors from this pinned HF repository")
    p.add_argument("--read-workers", type=int, default=8)
    p.add_argument("--tensor-cache", type=Path, help="Consume prefetched original safetensors")
    p.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Right-pad independent histories to stream weights once per batch",
    )
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    requests = json.loads(a.requests.read_text())
    if requests["seed"] != SEED or not requests["cases"]:
        raise ValueError("Reference requires fixed seed and nonempty cases")
    for case in requests["cases"]:
        context_hash(case["prompt_ids"], case.get("images", ()))
        image_fingerprints(case.get("images", ()))
        if not case["target_ids"]:
            raise ValueError("Reference requires explicit teacher-forced target IDs")
    if not 1 <= a.batch_size <= 128:
        raise ValueError("batch-size must be in 1..128")
    if not 1 <= a.read_workers <= 16:
        raise ValueError("read-workers must be in 1..16")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(requests["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    # Use the official Torch equations even when optional FLA/conv packages are
    # installed. Their binary kernels may not support this Orin's SM87 target.
    config_raw = (a.checkpoint / "config.json").read_bytes()
    config_json = json.loads(config_raw)
    backend, full_type, text_type, class_prefix = (
        modeling,
        Qwen3_5Config,
        Qwen3_5TextConfig,
        "Qwen3_5",
    )
    if config_json.get("model_type") == "qwen3_5_moe":
        from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe
        from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
            Qwen3_5MoeConfig,
            Qwen3_5MoeTextConfig,
        )

        backend, full_type, text_type, class_prefix = (
            modeling_qwen3_5_moe,
            Qwen3_5MoeConfig,
            Qwen3_5MoeTextConfig,
            "Qwen3_5Moe",
        )
    reference_functions = configure_torch_reference(backend)
    config = text_type(**config_json.get("text_config", config_json))
    config._attn_implementation = "sdpa"
    multimodal = any(c.get("images") for c in requests["cases"])
    with torch.device("meta"):
        if multimodal:
            full_config = full_type(**config_json)
            full_config.text_config._attn_implementation = "sdpa"
            full_config.vision_config._attn_implementation = "sdpa"
            model = (
                getattr(backend, class_prefix + "Model")(full_config)
                .to(dtype=torch.bfloat16)
                .eval()
            )
            text_model = model.language_model
        else:
            model = (
                getattr(backend, class_prefix + "TextModel")(config).to(dtype=torch.bfloat16).eval()
            )
            text_model = model
    text_model.rotary_emb = getattr(backend, class_prefix + "TextRotaryEmbedding")(config).to(
        "cuda"
    )
    index_raw = (a.checkpoint / "model.safetensors.index.json").read_bytes()
    index = json.loads(index_raw)["weight_map"]
    sources = (
        {}
        if a.remote_repo
        else {
            filename: safe_open(a.checkpoint / filename, framework="pt", device="cpu")
            for filename in set(index.values())
        }
    )
    remote = (
        PinnedTensors(
            a.checkpoint, a.remote_repo, a.revision, a.output, a.read_workers, a.tensor_cache
        )
        if a.remote_repo
        else None
    )

    def tensor(name):
        if name not in index and name.startswith("model.visual."):
            name = name.removeprefix("model.")
        if remote:
            return remote.tensor(name)
        return sources[index[name]].get_tensor(name).to("cuda")

    def attach(module, prefix):
        def load(current, inputs):
            names = list(current.state_dict())
            source_names = {name: prefix + name for name in names}
            if prefix == "model.visual." and all("visual." + name in index for name in names):
                source_names = {name: "visual." + name for name in names}
            if remote:
                loaded = remote.tensors(list(source_names.values()))
                state = {name: loaded[source_names[name]] for name in names}
            else:
                state = {name: tensor(source_names[name]) for name in names}
            current.load_state_dict(state, strict=True, assign=True)
            if prefix == "model.visual.":
                # The nonpersistent buffer starts on meta and is unloaded with
                # the vision weights. Recreate the official formula per call.
                current.rotary_pos_emb = getattr(backend, class_prefix + "VisionRotaryEmbedding")(
                    current.config
                ).to("cuda")

        def unload(current, inputs, result):
            current.to_empty(device="meta")

        module.register_forward_pre_hook(load)
        module.register_forward_hook(unload)

    attach(text_model.embed_tokens, "model.language_model.embed_tokens.")
    for i, layer in enumerate(text_model.layers):
        attach(layer, f"model.language_model.layers.{i}.")
    attach(text_model.norm, "model.language_model.norm.")
    if multimodal:
        attach(model.visual, "model.visual.")
    head = torch.nn.Linear(
        config.hidden_size, config.vocab_size, bias=False, device="meta", dtype=torch.bfloat16
    )
    attach(head, "model.language_model.embed_tokens." if config.tie_word_embeddings else "lm_head.")
    started = time.monotonic()
    report = dict(
        seed=requests["seed"],
        reference_kind="BF16 source checkpoint",
        revision=a.revision,
        transformers=__import__("transformers").__version__,
        reference_functions=reference_functions,
        execution="teacher-forced full-sequence prefill",
        weights="stored checkpoint dtypes; one component streamed onto CUDA",
        config_sha256=hashlib.sha256(config_raw).hexdigest(),
        index_sha256=hashlib.sha256(index_raw).hexdigest(),
        requests_sha256=hashlib.sha256(a.requests.read_bytes()).hexdigest(),
        probes=[],
        complete=False,
    )
    report["batch_size"] = a.batch_size
    if remote:
        report["remote_source"] = dict(
            repo=a.remote_repo, revision=a.revision, read_workers=a.read_workers
        )
    report["padding"] = "right padding masked; only real history positions scored"
    report["modalities"] = "text, images and multiple images" if multimodal else "text"
    report["multimodal_scope"] = (
        "Identical normalized FP32 patch pixels; official BF16 encoder, embedding injection and MRoPE; no video or candidate features"
    )
    with torch.inference_mode():
        for start in range(0, len(requests["cases"]), a.batch_size):
            cases = requests["cases"][start : start + a.batch_size]
            histories = [c["prompt_ids"] + c["target_ids"][:-1] for c in cases]
            width = max(map(len, histories))
            input_ids = torch.zeros((len(cases), width), device="cuda", dtype=torch.long)
            attention_mask = torch.zeros_like(input_ids)
            for row, history in enumerate(histories):
                input_ids[row, : len(history)] = torch.tensor(history, device="cuda")
                attention_mask[row, : len(history)] = 1
            kwargs = {}
            if multimodal:
                images = [image for case in cases for image in case.get("images", ())]
                if images:
                    kwargs["pixel_values"] = torch.cat(
                        [
                            torch.tensor(
                                image_patches(image), device="cuda", dtype=torch.bfloat16
                            ).reshape(image["grid_height"] * image["grid_width"], 1536)
                            for image in images
                        ]
                    )
                    kwargs["image_grid_thw"] = torch.tensor(
                        [[1, image["grid_height"], image["grid_width"]] for image in images],
                        device="cuda",
                    )
                # New Transformers versions require explicit modality IDs.
                # Delimiters remain text; only expanded image pads are visual.
                kwargs["mm_token_type_ids"] = (input_ids == full_config.image_token_id).int()
            hidden = model(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=False, **kwargs
            ).last_hidden_state
            selected = torch.cat(
                [
                    hidden[
                        row,
                        len(c["prompt_ids"]) - 1 : len(c["prompt_ids"]) - 1 + len(c["target_ids"]),
                    ]
                    for row, c in enumerate(cases)
                ],
                dim=0,
            )
            # Match the official BF16 head GEMM, then normalize in FP64 over the
            # complete vocabulary. Top-3 ties prefer the smaller token ID.
            logits = head(selected).double()
            logprobs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
            assert bool(torch.isfinite(logprobs).all()), "Nonfinite reference probabilities"
            top_ids = (
                torch.argsort(logits, dim=-1, descending=True, stable=True)[:, :3].cpu().tolist()
            )
            probabilities = logprobs.cpu()
            offset = 0
            for case in cases:
                prompt, targets = case["prompt_ids"], case["target_ids"]
                for position, target in enumerate(targets):
                    ids = top_ids[offset + position]
                    queries = set(
                        ids + [target] + (case.get("query_ids") or [[]] * len(targets))[position]
                    )
                    report["probes"].append(
                        dict(
                            case_id=case["id"],
                            execution_mode="prefill",
                            position=position,
                            seed=requests["seed"],
                            context_sha256=context_hash(
                                prompt + targets[:position], case.get("images", ())
                            ),
                            reference_token_id=target,
                            reference_logprob=float(probabilities[offset + position, target]),
                            top3=[
                                dict(token_id=i, logprob=float(probabilities[offset + position, i]))
                                for i in ids
                            ],
                            queried_logprobs={
                                str(i): float(probabilities[offset + position, i]) for i in queries
                            },
                        )
                    )
                offset += len(targets)
            report["elapsed_s"] = time.monotonic() - started
            report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
            a.output.parent.mkdir(parents=True, exist_ok=True)
            a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            print(
                "scored",
                [c["id"] for c in cases],
                "elapsed_s",
                round(report["elapsed_s"], 2),
                flush=True,
            )
            del hidden, selected, logits, logprobs, probabilities
    report["complete"] = True
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
