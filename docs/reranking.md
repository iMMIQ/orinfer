# Query-Document Reranking (Qwen3-Reranker-0.6B)

Orinfer runs **Qwen3-Reranker-0.6B** natively on CUDA SM87, scoring query-document pairs via causal `sigmoid(yes_logit - no_logit)` with exact logit-difference sorting (preserving ranking precision even when sigmoid saturates).

---

## 1. Quick Start

```bash
orinfer serve artifacts/reranker-build/model \
  --model qwen3-reranker-0.6b \
  --listen 127.0.0.1:8089 \
  --cuda-graph full
```

### HTTP API (`/v1/rerank` and `/rerank`)

```bash
curl http://127.0.0.1:8089/v1/rerank \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-reranker-0.6b",
    "query": "How much memory does Jetson AGX Orin 64GB have?",
    "documents": [
      "Jetson AGX Orin 64GB ships with 64 GB of unified memory.",
      "Apple trees blossom in spring."
    ],
    "top_n": 2,
    "return_documents": true
  }'
```

- **Inputs**: `documents` accepts up to **128** strings or objects with a `text` field. `top_n` (`1..128`) defaults to all documents; `return_documents` (default `false`) preserves extra fields on document objects; optional `instruction` overrides the default retrieval prompt.
- **Limits**: Query/instruction ≤ 1 MiB each, document ≤ 4 MiB each, HTTP body ≤ 32 MiB, max context 32,768 tokens (over-length inputs return an error).

### Offline CLI

Also outputs raw `yes`/`no` logits in input order for quality diagnostics:

```bash
orinfer rerank-model artifacts/reranker-build/model requests.json --cuda-graph full
```

---

## 2. Precision & Architecture

- **Architecture**: 28-layer causal Qwen3 (1024 hidden, 16/8 GQA heads, head dim 128, per-head Q/K RMSNorm).
- **Quantization**: Two-stage E8P Q4A8 (`[4,1]`, signed-block128, beam8) for main matrices and embeddings; W8 for attention output projections and layers `0, 1, 2, 3, 8` Q/K/V & FFN; FP16 norms and 2-row (`no`, `yes`) classification head (4,096 bytes).
- **Batched Execution**: Multiple documents in a request share projections on 32-row aligned packets while keeping independent causal attention, RoPE, and final-position scoring boundaries.

---

## 3. Building & Validation

Input directory must contain official `Qwen/Qwen3-Reranker-0.6B` BF16 weights plus `source.json`:

```bash
# 1. Quantize to E8P Q4A8
ORINFER_CHECKPOINT_DIR="$PWD/artifacts/reranker-source" \
  bash tools/environments/cuda/run.sh python/orinfer_build/quantization/qwen_reranker.py \
  artifacts/reranker-converted \
  --source-dir "$PWD/artifacts/reranker-source" \
  --gpu-lock /tmp/qwen-reranker-encoder.lock

# 2. Compile Schema 2 execution package
make build
ORINFER_CHECKPOINT_DIR="$PWD/artifacts/reranker-converted" \
  bash tools/environments/cuda/run.sh python/orinfer_build/models/qwen_reranker/build.py \
  artifacts/reranker-build \
  --source "$PWD/artifacts/reranker-converted" \
  --model-output "$PWD/artifacts/reranker-build/model" \
  --gpu-lock-held

# 3. Validate kernels & end-to-end probabilities/ranking vs. BF16
ORINFER_CHECKPOINT_DIR="$PWD/artifacts/reranker-source" \
  bash tools/environments/cuda/run.sh python/orinfer_build/models/qwen_reranker/check_kernels.py \
  artifacts/reranker-kernels --original "$PWD/artifacts/reranker-source"

ORINFER_REFERENCE_CHECKPOINT="$PWD/artifacts/reranker-source" \
ORINFER_CHECKPOINT_DIR="$PWD/artifacts/reranker-build/model" \
  bash tools/environments/cuda/run.sh python/orinfer_build/models/qwen_reranker/check_model.py \
  artifacts/reranker-quality \
  --original "$PWD/artifacts/reranker-source" \
  --model "$PWD/artifacts/reranker-build/model" \
  --engine "$PWD/target/release/orinfer"
```

- **Benchmarking**: Compare HTTP throughput against vLLM BF16 (`--chat-template tools/bench/templates/qwen3_reranker.jinja`) via `tools/bench/encoders.py` ([Benchmarking Toolkit](../tools/bench/README.md)).
- **Licensing**: Model weights are Apache-2.0; engine and kernels are LGPL-3.0-or-later.
