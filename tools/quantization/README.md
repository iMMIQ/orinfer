# Offline Quantization Tools

Offline quantization encodes official BF16 (or pinned FP8) checkpoints into single-resident compressed representations before execution package compilation. For the mathematical derivation of E8P Q2/Q4 and INT4 bit-plane Tensor Core execution, see [E8P Quantization Technical Note](../../docs/quantization.md).

---

## Weight Representations by Model

| Model Family | Primary Format | Scale & Exceptions |
| --- | --- | --- |
| **Qwen3.8-27B / Qwen3.5-4B / Index-Translate-2B, 9B, 35B** | Two-stage E8P Q4 (`U32` / 8 weights, `[4, 1]`, beam-8) | Signed-block128 rotation, FP16 row scale, A8; W8 for sensitive output/MTP projections |
| **Qwen3-ASR-1.7B** | Two-stage E8P Q4 (`[4, 1]`, beam-8) | Layers 0–3 Q/K/V & FFN + all attn output in W8; FP16 audio encoder & projector |
| **Nemotron-3.5-Embed-1B** | Two-stage E8P Q4 (`[4, 1]`, beam-8) | Text layer 2 + attn output + final vision projector in W8 ([Embeddings Guide](../../docs/embeddings.md)) |
| **Qwen3-Reranker-0.6B** | Two-stage E8P Q4 (`[4, 1]`, beam-8) | Layers 0/1/2/3/8 + attn output in W8; FP16 norms & 2-row yes/no head ([Reranking Guide](../../docs/reranking.md)) |
| **Qwen3.8-Flash-Next** | E8P Q2 (`U16` / 8 weights) experts & PLE + rotated scalar-LUT Q4 | FP16 row scale, A8; W8 output projections & shared FFN; BF16 HC |

---

## Usage

Source directories must include original HF config/tokenizer/safetensors index plus `source.json` recording pinned `repo` and 40-char `revision`. Completed shards are checksummed and atomically committed.

```bash
# Quantize Qwen3.5 / Index-Translate / Qwen3-ASR to two-stage E8P Q4A8
bash tools/quantization/run.sh qwen_e8p \
  --source-dir /path/to/original \
  --output artifacts/quantization/index-translate-2b-e8p-q4a8

# Quantize Flash Next (see tools/model/flash_next/README.md for weight assembly)
bash tools/quantization/run_flash_next.sh \
  --index /path/to/original/model.safetensors.index.json \
  --revision SOURCE_COMMIT_SHA --stage all \
  --output artifacts/quantization/flash-next/converted
```

## Verification

```bash
# CPU codebook & reference unit tests
uv run --locked python -m unittest orinfer_build.quantization.test_e8p orinfer_build.quantization.test_e8p_q4

# GPU encoder, beam-8 search, and direct INT4 bit-plane checks
bash tools/environments/cuda/run.sh python/orinfer_build/quantization/check_e8p_q4.py artifacts/check-e8p-q4
bash tools/environments/cuda/run.sh python/orinfer_build/quantization/check_e8p_search.py artifacts/check-e8p-search
bash tools/environments/cuda/run.sh python/orinfer_build/quantization/check_e8p_direct.py artifacts/check-e8p-direct
```

Always validate end-to-end quantization quality against official BF16 using [Quality Evaluation Tools](../eval/README.md).
