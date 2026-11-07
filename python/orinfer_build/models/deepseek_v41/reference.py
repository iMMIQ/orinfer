"""Text-only teacher-forced reference using the official FP4/FP8 checkpoint.

The checkpoint's own model.py supplies architecture/state semantics. Bounded
CPU decoding of its compressed weights and activation formats lets CPU/BF16
devices evaluate it without native FP4/FP8 GEMM support. This is an offline
numerical reference, not a serving or performance execution path.
"""

import argparse
from collections import OrderedDict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import types

import numpy as np
import torch
import torch.nn.functional as F

from orinfer_build.common.files import atomic_json, file_hash
from orinfer_build.models.deepseek_v41.official import Checkpoint, Decoder


def quantize_activation(x, group=32, fp4=False, e4m3_scale=False):
    cpu = x.detach().float().cpu().reshape(-1, group)
    maximum = cpu.abs().amax(-1, keepdim=True)
    if fp4 and e4m3_scale:
        scale = (maximum.clamp_min(6 * 2**-9) / 6).to(torch.float8_e4m3fn).float()
    else:
        maximum = maximum.clamp_min(6 * 2**-126 if fp4 else 1e-4)
        raw = maximum * (1 / 6 if fp4 else 1 / 448)
        bits = raw.view(torch.int32)
        exponent = ((bits >> 23) & 255) - 127 + ((bits & 0x7FFFFF) != 0)
        scale = torch.pow(2.0, exponent)
    value = cpu / scale
    if fp4:
        levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device="cpu")
        boundaries = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5], device="cpu")
        magnitude = value.abs().clamp_max(6)
        codes = torch.bucketize(magnitude.contiguous(), boundaries)
        tie = (codes < 7) & (magnitude == boundaries[codes.clamp_max(6)]) & ((codes & 1) != 0)
        codes += tie
        value = torch.where(value.signbit(), -levels[codes], levels[codes])
    else:
        value = value.clamp(-448, 448).to(torch.float8_e4m3fn).float()
    return (value * scale).reshape(x.shape).to(device=x.device, dtype=x.dtype)


def act_quant(x, block_size=32, scale_fmt=None, scale_dtype=None, inplace=False):
    result = quantize_activation(x, block_size)
    if inplace:
        x.copy_(result)
        return x
    return result, None


def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=None):
    result = quantize_activation(
        x, block_size, fp4=True, e4m3_scale=scale_dtype == torch.float8_e4m3fn
    )
    if inplace:
        x.copy_(result)
        return x
    return result, None


def hc_split_sinkhorn(mixes, scale, base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    h = hc_mult
    pre = torch.sigmoid(mixes[..., :h] * scale[0] + base[:h]) + eps
    post = 2 * torch.sigmoid(mixes[..., h : 2 * h] * scale[1] + base[h : 2 * h])
    comb = (mixes[..., 2 * h :] * scale[2] + base[2 * h :]).unflatten(-1, (h, h))
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def sparse_attn(q, kv, sink, indices, softmax_scale):
    # Match the official 64-key online update and BF16 unnormalized probability
    # boundary. Chunk query rows only to bound temporary reference storage.
    batches, tokens, heads, dimension = q.shape
    output = torch.empty_like(q)
    for batch in range(batches):
        for first in range(0, tokens, 16):
            end = min(first + 16, tokens)
            query = q[batch, first:end].float()
            maximum = torch.full(
                (end - first, heads, 1), -1e30, dtype=torch.float32, device=q.device
            )
            denominator = torch.zeros_like(maximum)
            accumulator = torch.zeros(
                end - first, heads, dimension, dtype=torch.float32, device=q.device
            )
            for start in range(0, indices.shape[-1], 64):
                selected = indices[batch, first:end, start : start + 64].long()
                values = kv[batch][selected.clamp_min(0)].float()
                scores = torch.einsum("shd,skd->shk", query, values) * softmax_scale
                scores.masked_fill_(selected[:, None, :] < 0, -torch.inf)
                updated = torch.maximum(maximum, scores.amax(-1, keepdim=True))
                alpha = (maximum - updated).exp()
                probabilities = (scores - updated).exp()
                denominator = denominator * alpha + probabilities.sum(-1, keepdim=True)
                accumulator = accumulator * alpha + torch.einsum(
                    "shk,skd->shd", probabilities.bfloat16().float(), values
                )
                maximum = updated
            denominator += (sink[None, :, None] - maximum).exp()
            output[batch, first:end] = (accumulator / denominator).to(q.dtype)
    return output


def apply_rotary_emb(x, frequencies, inverse=False):
    values = x.float().unflatten(-1, (-1, 2))
    cosine = frequencies.real.to(x.device).float().unsqueeze(0)
    sine = frequencies.imag.to(x.device).float().unsqueeze(0)
    while cosine.ndim < values.ndim - 1:
        cosine, sine = cosine.unsqueeze(-2), sine.unsqueeze(-2)
    if inverse:
        sine = -sine
    real, imaginary = values[..., 0], values[..., 1]
    result = torch.stack(
        (real * cosine - imaginary * sine, real * sine + imaginary * cosine), dim=-1
    )
    x.copy_(result.flatten(-2).to(x.dtype))
    return x


class Weights:
    def __init__(self, checkpoint, device, budget):
        self.checkpoint, self.device, self.budget = checkpoint, device, budget
        self.cache, self.bytes = OrderedDict(), 0
        self.loads = 0

    def get(self, name, dtype):
        key = name, dtype
        if key in self.cache:
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        array = (
            self.checkpoint.matrix(name) if name.endswith(".weight") else self.checkpoint.read(name)
        )
        value = torch.from_numpy(np.array(array, copy=True)).to(dtype=dtype, device=self.device)
        size = value.numel() * value.element_size()
        while self.cache and self.bytes + size > self.budget:
            _, old = self.cache.popitem(last=False)
            self.bytes -= old.numel() * old.element_size()
        self.cache[key] = value
        self.bytes += size
        self.loads += 1
        return value


def load_model(directory, device, capacity, budget, decoder=None):
    from transformers import AutoTokenizer

    directory = Path(directory)
    kernel = types.ModuleType("kernel")
    for name in ("act_quant", "fp4_act_quant", "hc_split_sinkhorn", "sparse_attn"):
        setattr(kernel, name, globals()[name])

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("Reference quantized linears must use the bounded weight adapter")

    kernel.fp4_gemm = kernel.fp8_gemm = unavailable
    sys.modules["kernel"] = kernel
    vision = types.ModuleType("vision")
    vision.Aligner = vision.ViT = unavailable
    sys.modules["vision"] = vision
    sys.path.insert(0, str(directory / "inference"))
    spec = importlib.util.spec_from_file_location(
        "official_deepseek_v41", directory / "inference/model.py"
    )
    official = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = official
    spec.loader.exec_module(official)
    args = official.ModelArgs(**json.loads((directory / "inference/config.json").read_text()))
    args.max_batch_size, args.max_seq_len = 1, capacity
    args.dspark_block_size, args.n_mtp_layers, args.vision_n_layers, args.temperature = 0, 0, 0, 0
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    hash_type = official.NgramHashState
    official.NgramHashState = lambda *_: None
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("meta"):
        model = official.Transformer(args, tokenizer)
    official.NgramHashState = hash_type
    official.precompute_freqs_cis.cache_clear()
    model.engram_hash = hash_type(args, model.engram_layout, tokenizer).to(device)
    checkpoint = Checkpoint(directory, decoder)
    weights = Weights(checkpoint, device, budget)
    for name, parameter in model.named_parameters():
        if name not in checkpoint.index:
            raise ValueError(f"Official parameter mapping is missing: {name}")
        metadata = checkpoint.metadata(name)
        if list(parameter.shape) != metadata["shape"]:
            raise ValueError(f"Official parameter shape differs: {name}")

    def linear(module, x):
        name = module._reference_name + ".weight"
        quantized = module.weight.dtype in (torch.float8_e4m3fn, torch.float4_e2m1fn_x2)
        dtype = x.dtype
        if quantized:
            x = quantize_activation(x)
        weight = weights.get(
            name, torch.float32 if dtype == torch.float32 or device == "cpu" else torch.bfloat16
        )
        result = F.linear(x.float() if weight.dtype == torch.float32 else x, weight)
        return result.to(dtype)

    for kind in (official.Linear, official.ColumnParallelLinear, official.RowParallelLinear):
        kind.forward = linear

    def embedding(module, ids):
        data = checkpoint.rows(module._reference_name + ".weight", ids.detach().cpu().numpy())
        return torch.from_numpy(np.array(data, copy=True)).to(device=device, dtype=torch.bfloat16)

    official.ParallelEmbedding.forward = embedding
    official.ParallelEngramEmbedding.forward = embedding
    official.apply_rotary_emb = apply_rotary_emb

    def prepare(module, _inputs):
        if isinstance(
            module, (official.Linear, official.ParallelEmbedding, official.ParallelEngramEmbedding)
        ):
            return
        for key, parameter in list(module._parameters.items()):
            if parameter is not None and parameter.is_meta:
                value = weights.get(module._reference_name + "." + key, parameter.dtype)
                module._parameters[key] = torch.nn.Parameter(value, requires_grad=False)
        if isinstance(module, official.Attention):
            module.wo_a.weight = torch.nn.Parameter(
                weights.get(module._reference_name + ".wo_a.weight", torch.bfloat16),
                requires_grad=False,
            )

    for name, module in model.named_modules():
        module._reference_name = name
        for key, buffer in list(module._buffers.items()):
            if buffer is None or not buffer.is_meta:
                continue
            if key == "freqs_cis":
                compressed = module.compress_ratio != 0
                buffer = official.precompute_freqs_cis(
                    args.rope_head_dim,
                    capacity,
                    args.original_seq_len if compressed else 0,
                    args.compress_rope_theta if compressed else args.rope_theta,
                    args.rope_factor,
                    args.beta_fast,
                    args.beta_slow,
                )
            else:
                buffer = torch.full(
                    buffer.shape,
                    -torch.inf if key == "score_state" else 0,
                    dtype=buffer.dtype,
                    device=device,
                )
            module._buffers[key] = buffer
        module.register_forward_pre_hook(prepare)
    torch.set_default_device(device)
    return model, checkpoint, weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--case", default="p1024")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--cache-gib", type=int, default=30)
    parser.add_argument("--positions", type=int, default=129)
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    if args.positions < 1 or args.cache_gib < 1:
        parser.error("positive positions and cache budget are required")
    torch.set_num_threads(8)
    torch.manual_seed(20261002)
    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401 - registers the NPU backend with Torch

        torch.npu.set_device(args.device)
        torch.npu.matmul.allow_hf32 = False
    inputs = json.loads(args.inputs.read_text())["cases"]
    case = next(x for x in inputs if x["name"] == args.case)
    teacher = json.loads(args.teacher.read_text())["requests"]
    target = next(x["output_token_ids"] for x in teacher if x["name"] == args.case)[
        : args.positions
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    model, checkpoint, weights = load_model(
        args.model_dir,
        args.device,
        len(case["tokens"]) + len(target) + 1,
        args.cache_gib << 30,
        Decoder(args.output / "decoder"),
    )
    metadata = {
        "reference": "Official checkpoint FP4 experts + FP8 backbone, canonical model.py architecture and activation/cache quantization; software-decoded values with FP32 accumulation",
        "model_dir": str(args.model_dir),
        "model_source_sha256": file_hash(args.model_dir / "inference/model.py"),
        "kernel_source_sha256": file_hash(args.model_dir / "inference/kernel.py"),
        "config_sha256": file_hash(args.model_dir / "config.json"),
        "index_sha256": file_hash(args.model_dir / "model.safetensors.index.json"),
        "reference_source_sha256": file_hash(Path(__file__)),
        "decode_source_sha256": file_hash(Path(__file__).with_name("official.py")),
        "native_decode_source_sha256": file_hash(Path(__file__).with_name("official_decode.c")),
        "input_tokens": len(case["tokens"]),
        "input_tokens_sha256": hashlib.sha256(json.dumps(case["tokens"]).encode()).hexdigest(),
        "teacher_tokens_sha256": hashlib.sha256(json.dumps(target).encode()).hexdigest(),
        "same_teacher_history": True,
        "seed": 20261002,
        "native_fp4_fp8_gemm": False,
        "device": args.device,
    }
    atomic_json(args.output / "identity.json", metadata)
    print("mapped official model", len(checkpoint.index), "tensor names", flush=True)
    if args.inspect_only:
        checkpoint.close()
        return
    logits_file = np.lib.format.open_memmap(
        args.output / "logits.npy", mode="w+", dtype=np.float32, shape=(len(target), 129280)
    )
    report = []
    with torch.inference_mode():
        for position, token in enumerate(target):
            incoming = case["tokens"] if position == 0 else [target[position - 1]]
            base = 0 if position == 0 else len(case["tokens"]) + position - 1
            _, logits, _ = model(
                torch.tensor([incoming], dtype=torch.long, device=args.device), base
            )
            cpu = logits[0].float().cpu()
            logits_file[position] = cpu.numpy()
            logits_file.flush()
            nll = float(torch.logsumexp(cpu.double(), dim=0) - cpu[token].double())
            item = {
                "position": position,
                "target": token,
                "argmax": int(cpu.argmax()),
                "target_nll": nll,
                "loaded_matrices": weights.loads,
                "elapsed_s": time.monotonic() - started,
            }
            report.append(item)
            atomic_json(args.output / "progress.json", {"identity": metadata, "positions": report})
            print(json.dumps(item), flush=True)
    atomic_json(
        args.output / "report.json",
        {
            "identity": metadata,
            "positions": report,
            "mean_target_nll": sum(x["target_nll"] for x in report) / len(report),
        },
    )
    checkpoint.close()


if __name__ == "__main__":
    main()
