# Multimodal Document Embeddings (Nemotron-3.5-Embed-1B)

Orinfer runs **Nemotron-3.5-Embed-1B** (Ministral3 bidirectional text encoder + Pixtral vision encoder) natively on CUDA SM87, producing 2048-dimensional L2-normalized embeddings for text, single-image, and multi-image documents in a shared vector space.

---

## 1. Quick Start

Download the pre-quantized deployment model from [Hugging Face](https://huggingface.co/immiq/Nemotron-3.5-Embed-1B-Orinfer-E8P-Q4A8) or [ModelScope](https://modelscope.cn/models/iammiq/Nemotron-3.5-Embed-1B-Orinfer-E8P-Q4A8), validate, and serve:

```bash
hf download immiq/Nemotron-3.5-Embed-1B-Orinfer-E8P-Q4A8 --local-dir artifacts/nemotron-embed
orinfer validate-model artifacts/nemotron-embed

orinfer serve artifacts/nemotron-embed \
  --model nemotron-3.5-embed-1b \
  --listen 127.0.0.1:8088 \
  --cuda-graph full
```

### HTTP API (`/v1/embeddings`)

```bash
curl http://127.0.0.1:8088/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "nemotron-3.5-embed-1b",
    "input_type": "query",
    "input": ["How much unified memory does Jetson AGX Orin have?"]
  }'
```

- **`input_type`**: `"document"` (default, adds `passage: ` prefix) or `"query"` (adds `query: ` prefix).
- **Multimodal items**: Pass `{"text": "optional caption", "images": ["data:image/png;base64,...", "https://..."]}` (up to **8 images** per item, **128 items** per request).
- **Options**: `dimensions` (Matryoshka prefix truncation + re-normalization), `encoding_format` (`float` or little-endian FP32 `base64`). Over-length inputs (>32,768 tokens) return an error rather than silently truncating.

### Offline CLI

Supports local image file paths in addition to URLs and data URIs:

```bash
orinfer embed-model artifacts/nemotron-embed requests.json --cuda-graph full
```

---

## 2. Precision & Architecture

- **Architecture**: 16-layer bidirectional text encoder (2048 hidden, 24/8 GQA) + 24-layer Pixtral vision encoder (1024 hidden, 14×14 patches; default max side 1120 px / 6,400 patches, configurable up to 12,100 via `--max-patches`).
- **Quantization**: Two-stage E8P Q4A8 (`[4,1]`, signed-block128, beam8) for text/vision matrices and embeddings; W8 for attention output projections, final vision projector, and text layer 2 Q/K/V/FFN (protects short-query quality); FP16 patch conv and norms.
- **Batched Execution**: Multiple inputs in one request share text projections on 32-row aligned packets with isolated bidirectional attention, RoPE, and mean-pooling boundaries; oversized batches split cleanly on input boundaries.

---

## 3. Building & Validation

Input directory must contain the official BF16 checkpoint plus `source.json` (`repo` + 40-char `revision`):

```bash
# 1. Quantize to E8P Q4A8
ORINFER_CHECKPOINT_DIR="$PWD/artifacts/nemotron-source" \
  bash tools/environments/cuda/run.sh python/orinfer_build/quantization/nemotron_embed.py \
  artifacts/nemotron-converted \
  --source-dir "$PWD/artifacts/nemotron-source" \
  --gpu-lock /tmp/nemotron-encoder.lock

# 2. Compile Schema 2 execution package
make build
ORINFER_CHECKPOINT_DIR="$PWD/artifacts/nemotron-converted" \
  bash tools/environments/cuda/run.sh python/orinfer_build/models/nemotron_embed/build.py \
  artifacts/nemotron-build \
  --source "$PWD/artifacts/nemotron-converted" \
  --model-output "$PWD/artifacts/nemotron-build/model" \
  --gpu-lock-held

# 3. Validate kernels & end-to-end cosine/ranking vs. BF16
bash tools/environments/cuda/run.sh python/orinfer_build/models/nemotron_embed/check_kernels.py \
  artifacts/nemotron-kernels

ORINFER_REFERENCE_CHECKPOINT="$PWD/artifacts/nemotron-source" \
ORINFER_CHECKPOINT_DIR="$PWD/artifacts/nemotron-build/model" \
  bash tools/environments/cuda/run.sh python/orinfer_build/models/nemotron_embed/check_model.py \
  artifacts/nemotron-quality \
  --original "$PWD/artifacts/nemotron-source" \
  --model "$PWD/artifacts/nemotron-build/model" \
  --engine "$PWD/target/release/orinfer"
```

- **Benchmarking**: Compare complete HTTP throughput against vLLM BF16 using `tools/bench/encoders.py` ([Benchmarking Toolkit](../tools/bench/README.md)).
- **Licensing**: Model weights follow [OpenMDW-1.1](https://openmdw.ai/) (Ministral base: Apache-2.0); engine and kernels are LGPL-3.0-or-later.
