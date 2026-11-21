# Serving Guide

Orinfer provides a native Rust HTTP server with OpenAI-compatible and Ollama-compatible endpoints, continuous batching, prefix caching, speculative decoding (MTP / DSpark), and multi-model management.

---

## 1. Starting the Server

### Single-Model Mode

Pass a prepared model directory directly to `orinfer serve`:

```bash
orinfer validate-model /path/to/prepared-model
orinfer serve /path/to/prepared-model \
  --model qwen3.8-27b \
  --listen '[::]:8088' \
  --cuda-graph decode_only
```

- **Startup flow**: Binds the HTTP listener immediately (exiting fast on port conflict), prepares the chat frontend and GPU/NPU model in parallel, runs a 1-token warmup (`WARMUP READY`), and then begins serving traffic (`API READY`).
- **Execution package lookup**: Searches `$ORINFER_EXECUTION_CACHE` (default `$XDG_CACHE_HOME/orinfer/packages` or `~/.cache/orinfer/packages`) by package digest first, then falls back to `MODEL_DIR/cache/packages/<digest>`.
- **Weight verification**: Normal startup checks config signatures, ABI, dtype/shape/layout, and file bounds without hashing payloads. Pass `--verify-weights` for full content verification on load, or run `orinfer validate-model` beforehand.
- **Authentication**: Set `ORINFER_API_KEY` to require `Authorization: Bearer <key>` on all `/v1` and `/api` endpoints.

### Managed Multi-Model Mode

Omit `MODEL_DIR` to start the model manager, which loads registered models on demand and evicts idle models under memory pressure or `keep_alive` expiration:

```bash
orinfer serve --listen '[::]:8088' --keep-alive 5m --max-loaded-models auto

# Register or download models
orinfer import qwen3.8-27b /path/to/prepared-model
orinfer pull hf://immiq/Nemotron-3.5-Embed-1B-Orinfer-E8P-Q4A8 --name nemotron-embed

# Inspect and control loaded workers
orinfer list
orinfer ps
orinfer show qwen3.8-27b
orinfer load qwen3.8-27b --keep-alive 30m
orinfer run qwen3.8-27b "Summarize Orinfer in one sentence."
orinfer stop qwen3.8-27b
orinfer rm qwen3.8-27b
```

---

## 2. OpenAI-Compatible API

| Endpoint | Description |
| --- | --- |
| `GET /health` | Service status (`starting`, `ready`, `draining`, `failed`), capabilities, scheduler & cache telemetry |
| `GET /metrics` | Prometheus-style histograms and counters for queuing, TTFT, decode, and batch execution |
| `GET /v1/models` | List served / registered models |
| `POST /v1/chat/completions` | Streaming & non-streaming chat, thinking, tools, JSON Schema, images, audio, logprobs |
| `POST /v1/chat/completions/render` | Render chat template, tokenize, and count tokens/multimodal placeholders without GPU execution |
| `POST /v1/audio/transcriptions` | Speech transcription ([Audio Guide](audio.md)) |
| `POST /v1/embeddings` | Text & image document embeddings ([Embeddings Guide](embeddings.md)) |
| `POST /v1/rerank` (alias `/rerank`) | Query-document relevance reranking ([Reranking Guide](reranking.md)) |

### Chat Completions Example

```bash
curl http://127.0.0.1:8088/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "What is 37 * 24 - 19 * 13?"}],
    "temperature": 0,
    "max_completion_tokens": 512,
    "enable_thinking": true,
    "stream": true,
    "stream_options": {"include_usage": true}
  }'
```

### Model-Specific Sampling Defaults

Priority order: **explicit request fields** > **`--default-request-params`** > **model defaults**.

| Model Family | Default Thinking | `temperature` / `top_p` / `top_k` / `presence_penalty` | Default Output Budget |
| --- | --- | --- | --- |
| **Qwen3.8-27B / Flash-Next** | On | `1.0 / 0.95 / 20 / 0` (off: `0.7 / 0.8 / 20 / 1.5`) | 8,192 |
| **Qwen3.5-4B** | On | `1.0 / 0.95 / 20 / 1.5` (off: `0.7 / 0.8 / 20 / 1.5`) | 8,192 |
| **Index-Translate-2B / 9B / 35B** | Off | `0 / 1.0 / 0 / 0` (greedy) | 1,024 |
| **Qwen3-ASR-1.7B** | Off | `0 / 1.0 / 0 / 0` (greedy) | 8,192 |
| **DSv4.1 (Ascend A2)** | Off | `1.0 / 1.0 / 0 / 0` | 8,192 (thinking: 65,536; `max`: 131,072) |

- **Request defaults**: Pass `--default-request-params '{"enable_thinking":true,"temperature":0.7}'` to set deployment defaults. Explicit `null` in a request clears the deployment default; grouped fields (`enable_thinking` / `reasoning_effort`, `max_tokens` / `max_completion_tokens`) override as a group.
- **Seeds & sampling**: Supports signed 64-bit `seed`, `top_k` (`0` or `-1` disables), `top_p`, `presence_penalty`, `frequency_penalty`, `repetition_penalty`, `logit_bias` (`-100..100` by token ID), `stop`, and `ignore_eos`. Unspecified seeds get a unique per-request 64-bit random seed.
- **Context overflow**: Returns HTTP 400 with `error.code = "context_length_exceeded"`.

### Thinking & `chat_template_kwargs`

- Enable via `enable_thinking: true` or non-`none` `reasoning_effort` (`minimal`/`low` → `low`, `medium`, `high`/`xhigh`/`max` → `xhigh`; on DSv4.1: `low=50`, `high=75`, `max/xhigh=100`). Reasoning text is emitted in `reasoning_content`.
- **`thinking_token_budget`** (alias `chat_template_kwargs.thinking_budget`): Hard cap on initial thinking tokens (`>= 0`). When exhausted, decoding forces `</think>` and reserves room for at least one answer token.
- **`chat_template_kwargs`**: Accepts template parameters (`enable_thinking`, `thinking`, `reasoning_effort`, `thinking_token_budget`, `preserve_thinking`, and custom variables) for compatibility with vLLM, DeepSeek Harness, and [Estuary](https://github.com/iMMIQ/estuary). Top-level fields take precedence.

### Structured JSON Output (`response_format`)

Supports `text`, `json_object`, and `json_schema` using Rust `llguidance` token-level grammar constraints:

```json
{
  "model": "qwen3.8-27b",
  "messages": [{"role": "user", "content": "Return 2+3 as JSON."}],
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "answer",
      "strict": true,
      "schema": {
        "type": "object",
        "properties": {"answer": {"type": "integer"}},
        "required": ["answer"],
        "additionalProperties": false
      }
    }
  },
  "max_completion_tokens": 128
}
```

- Compatible with `enable_thinking` and MTP (grammar state only commits accepted tokens).
- Omit `stop` when using grammar constraints; always check `finish_reason` (`length` indicates a truncated JSON prefix).

### Tool Calling

- Supports `tools`, `tool_choice` (`auto`, `none`, `required`, or a named function), and `parallel_tool_calls`.
- Qwen models use native XML tool calls; DSv4.1 uses native DSML (`chat_format=deepseek_v41`). Both are translated to standard OpenAI `tool_calls` with incremental `function.arguments` deltas in SSE.
- `strict: true`, `tool_choice: "required"`, named function choice, or `parallel_tool_calls: false` engage grammar-constrained decoding.

### Logprobs & Gateway Rendering

- **`logprobs: true`** with `top_logprobs: 0..20` returns normalized target probabilities (after penalties/bias/grammar, before top-k/top-p truncation) and UTF-8 `bytes` for response content tokens.
- **`/v1/chat/completions/render`** runs template rendering, tokenization, and image placeholder expansion on CPU without consuming GPU slots—useful for gateways such as [Estuary](https://github.com/iMMIQ/estuary).

---

## 3. Images & Multi-Image Input

Vision-capable packages accept interleaved text and multiple images (PNG, JPEG, WebP via HTTP(S) URL or base64 data URI) in `image_url` content parts:

```json
{
  "model": "qwen3.8-27b",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "Compare these two images."},
      {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    ]
  }],
  "image_processing": {
    "max_pixels": 1048576,
    "max_tokens": 16384,
    "max_tokens_per_image": 1024,
    "max_images": 20,
    "auto_resize": false
  }
}
```

- **Limits**: Max 24 MiB compressed image size, 20 s URL fetch timeout. `detail: "low"` targets ~256×256 px; `auto`/`high` uses checkpoint defaults bounded by package capacity.
- **UUID Image References**: To avoid re-uploading historical images across multi-turn chats, attach a unique `"uuid"` on the first send and pass `"image_url": null` with the same `"uuid"` on subsequent turns. Cache budget is controlled by `--image-cache-mib` (default `256` MiB; `0` disables). On eviction or restart, missing UUIDs return HTTP 400 instructing the client to resend the full image.
- **Upgrading Vision Capacity (CUDA)**: Existing Qwen3.5 / Flash Next packages can upgrade total image feature capacity to match full context without re-quantizing weights:
  ```bash
  bash tools/environments/cuda/run.sh python/orinfer_build/packaging/vision_capacity.py \
    artifacts/vision-capacity-build --model /path/to/model --model-output /path/to/upgraded-model
  ```

---

## 4. Ollama-Compatible API

Managed `orinfer serve` exposes Ollama native endpoints on the same listener (`[::]:8088`):

| Endpoint | Purpose |
| --- | --- |
| `GET /api/version`, `GET /api/tags`, `POST /api/show`, `GET /api/ps` | Version, registered models, details, and loaded models |
| `POST /api/chat`, `POST /api/generate` | Streaming NDJSON (`stream=true` default) or single JSON generation |
| `POST /api/copy`, `DELETE /api/delete` | Alias and unregister models (user-owned files on disk are preserved) |

```bash
curl http://127.0.0.1:8088/api/chat -d '{
  "model": "index-translate-2b",
  "messages": [{"role": "user", "content": "Translate to English: 你好"}],
  "think": false,
  "options": {"num_predict": 256},
  "keep_alive": "60s"
}'
```

- Supports `think`, base64 images, tools, `format: "json"` or JSON Schema, `logprobs`, and `options` (`temperature`, `top_p`, `top_k`, `seed`, `stop`, penalties, `num_predict`).
- `keep_alive` accepts seconds, duration strings (`"1h30m"`), `-1` (keep until memory pressure), or `0` (unload immediately). Empty requests preload or unload.
- Unsupported Ollama registry operations (`pull`/`push`/`create`, raw prompts, Go template overrides, sliding-window truncation) return explicit errors.

---

## 5. Runtime Execution Controls

### CUDA / ACL Graph (`--cuda-graph` / `--graph`)
- **`decode_only`** *(default)*: Captures graphs for decode, MTP draft/verify/commit, and short refreshes; submits prefill and vision directly.
- **`full`**: Captures all eligible plans including chunked prefill.
- **`off`**: Direct kernel/copy submission without graph capture (useful for debugging or required automatically when `--resident-experts` is bounded).

### Adaptive Speculative Decoding (MTP / DSpark)
- Enabled automatically on models with native MTP/DSpark plans (`--mtp-drafts auto`). Set `--mtp-drafts 0` to disable (also skips loading draft-only weights) or `1..31` to cap draft depth within package capacity.
- Dynamically selects draft depth per cohort using measured hot-path step costs, per-request conditional acceptance EMAs, and a bounded model-level exploration budget.
- **`--prompt-lookup`**: Optional greedy token-history lookup drafts (off by default); helpful for code edits or repetitive templates.

### Prefix Cache & Tiered CPU KV Offload
- **`--prefix-cache-mib`** *(default `4096`)*: Per-device radix prefix cache storing immutable VMM-shared KV pages plus recurrent state checkpoints (FP32 GDN, conv, PLE, QSA pending/index, MTP state). Set `0` to disable the device tier.
- **`--kv-offload-mib`** *(default `0`)*: Global CPU prefix-state cache shared across TP ranks. Evicted device snapshots spill to CPU and restore on hit with bounded pinned DMA.
- Accepts `prompt_cache_key` (1–64 chars) and `prompt_cache_retention` (`in-memory` | `24h`) as compatibility hints.

### Bounded Resident Experts (Flash Next)
- `--resident-experts full` *(default)* keeps all experts resident.
- `--resident-experts N` (`1..512`) maintains a global `48 × N` SLRU cache paged on demand from SSD into stable CUDA VMM addresses (MTP experts stay fully resident; CUDA Graph is automatically disabled).

### Continuous Batching & Scheduling Policies
- **`--scheduler-policy latency`** *(default for `serve`)*: Balances decode TPOT (`--target-tpot-ms`) and mixed prefill (`--prefill-budget-ms`), reserving decode bursts and prioritizing aging queued requests (>30 s).
- **`--scheduler-policy throughput`** *(default for `batch`)*: Maximizes aggregate completion rate by relaxing per-step latency caps while preserving memory admission and request isolation.

---

## 6. CLI Options Reference

| Option | Default | Description |
| --- | --- | --- |
| `--listen` | `[::]:8088` | HTTP bind address |
| `--model` | `qwen3.8-27b` | API model ID (single-model mode) |
| `--default-request-params` | `{}` | JSON default Chat request fields |
| `--scheduler-policy` | `latency` | `latency` (`serve` default) or `throughput` (`batch` default) |
| `--cuda-graph` / `--graph` | `decode_only` | Graph mode (`decode_only`, `full`, `off`) |
| `--mtp-drafts` | `auto` | `auto`, `0` (off), or `1..31` |
| `--prompt-lookup` | off | Enable greedy prompt-history lookup drafts |
| `--resident-experts` | `full` | `full` or per-layer SLRU expert capacity `N` |
| `--verify-weights` | off | Verify full weight & CPU table hashes on load |
| `--load-workers` | auto (≤4) | Parallel weight reader threads (`1..32`) |
| `--prefix-cache-mib` | `4096` | Device prefix cache budget per device in MiB |
| `--kv-offload-mib` | `0` | Global CPU prefix cache budget in MiB |
| `--max-active-requests` | `32` | Max concurrent active requests (`1..128`) |
| `--max-batch-tokens` | `128` | Ordinary decode row budget (`1..128`) |
| `--max-speculative-tokens` | `0` | MTP verification row budget (`0` = package capacity) |
| `--prefill-budget-ms` | `200` | Mixed-prefill time budget (`latency` policy) |
| `--target-tpot-ms` | `400` | Soft decode interval target (`latency` policy) |
| `--memory-reserve-mib` | `1024` | Extra memory reserve beyond OS margin |
| `--preprocess-workers` | `auto` | Preprocessing tasks (`auto` 1–8, or explicit `1..32`) |
| `--preprocess-memory-mib` | `2048` | Preprocessing & queued image tensor budget |
| `--image-cache-mib` | `256` | Image payload/preprocess LRU cache (`0` disables) |
| `--request-body-mib` | `32` | Max HTTP body size (≤ 1/8 of preprocess budget) |
| `--queue-timeout-ms` | `0` | Queue wait deadline (`0` = unlimited) |
| `--output-timeout-ms` | `60000` | Output stall timeout |
| `--drain-timeout-ms` | `30000` | Graceful shutdown drain timeout on SIGINT/SIGTERM |
| `--keep-alive` | `5m` | Idle model lifetime in managed mode |
| `--max-loaded-models` | `auto` | Max concurrently loaded models in managed mode |
| `--load-timeout-s` | `120` | Worker load timeout in managed mode |
| `--gpu-lock` | `artifacts/gpu-experiment.lock` | Shared device lock path |

---

## 7. Environment Variables Reference

| Variable | Scope | Default | Purpose |
| --- | --- | --- | --- |
| `ORINFER_API_KEY` | All | unset | Enables Bearer auth on `/v1` and `/api` |
| `ORINFER_EXECUTION_CACHE` | All | `~/.cache/orinfer/packages` | Execution package cache directory |
| `ORINFER_ASYNC_OUTPUT` | All | `1` | Set `0` to disable cross-step async output pipeline for debugging |
| `ORINFER_SINGLE_MTP_POLICY` | CUDA | `auto` | Diagnostic override (`auto`, `fixed0`..`fixed3`) |
| `ORINFER_MTP_TRACE` | CUDA | `0` | Set `1` to log per-step MTP cost & budget decisions |
| `ORINFER_ASCEND_LIBRARY` | Ascend | auto | Explicit path to `libascendcl.so` |
| `ORINFER_A2_HOST_TABLE_MIB` | Ascend | `0` | Resident RAM budget for DSv4.1 Engram tables (e.g. `212000`) |
| `ORINFER_A2_HOST_ROW_CACHE_MIB` | Ascend | `2048` | Cross-packet Engram host row cache (`0` disables) |
| `ORINFER_VISION_CACHE_MIB` | Ascend | `128` | Retained CPU vision feature cache (`0` disables) |
| `ORINFER_A2_COMMAND_TIMEOUT_MS` | Ascend | `30000` | Per-command TP rank timeout for fast commands (`1..1800000` ms) |
| `ORINFER_A2_HEAVY_COMMAND_TIMEOUT_MS` | Ascend | `max(4×base, 120000)` | Timeout for heavy capture/offload/host-restore/vision commands (`1..1800000` ms) |
| `ORINFER_A2_WORKER_CPUS` | Ascend | unset | Semicolon-separated per-rank CPU affinity lists; keep host cores free for Tokio/Rayon |
| `ORINFER_A2_DSPARK_COST_MODEL` | Ascend | auto | Explicit offline DSpark cost profile JSON |
| `ORINFER_A2_DSPARK_SAMPLING` | Ascend | `coupled` | `coupled` (default) or experimental `rejection` |
| `ORINFER_A2_SHARED_GRAPH` | Ascend | `0` | Set `1` to enable shared FFN fragment graph cache |
| `ORINFER_A2_DSPARK_POLICY_TRACE` | Ascend | `0` | Set `1` to log DSpark policy and cost-model trace |
| `ORINFER_A2_BATCH_PROFILE` | Ascend | `0` | Set `1` to log A2 cold-setup and prefill cost profiles |
| `HCCL_OP_EXPANSION_MODE` | Ascend | — | Recommended `AIV` for 910B TP8 |
| `HCCL_NPU_SOCKET_PORT_RANGE` | Ascend | — | Recommended `auto` |
| `HCCL_EXEC_TIMEOUT` | Ascend | `1800` | Recommended `120` s to bound collective failure teardown |
| `HCCL_DETERMINISTIC` | Ascend | unset | Set `strict` for bit-reproducible cross-batch diagnostics |
| `ASCEND_COREDUMP_SIGNAL` | Ascend | `none` (auto) | Prevents CANN trace handler from intercepting SIGINT/SIGTERM |

---

## 8. Coding Agent Integrations

Ready-to-use client templates live in `examples/` (see [examples/README.md](../examples/README.md)):

- **OpenCode** ([examples/opencode.json](../examples/opencode.json)): Copy to your project as `opencode.json` and run `opencode run --model orinfer/qwen-flash-next '...'` or `opencode run --model orinfer/deepseek-v4.1 '...'`.
- **Pi** ([examples/pi-models.json](../examples/pi-models.json)): Merge into `~/.pi/agent/models.json` and run `pi --provider orinfer --model qwen-flash-next --thinking low`.
- **DeepSeek Harness (`dsh`)**: See [DeepSeek Harness Integration](deepseek-harness.md).

---

## 9. Diagnostic CLI Commands

| Command | Runs On | Purpose |
| --- | --- | --- |
| `orinfer plan-model MODEL_DIR` | CPU | Expand static model adapter plan and verify exhaustive dynamic row domains |
| `orinfer validate-model MODEL_DIR` | CPU | Verify frontend assets, execution package, bindings, and all weight SHA256 hashes |
| `orinfer run-model MODEL_DIR REQ.json` | GPU | Fixed-length token-ID execution diagnostic (resets state per request) |
| `orinfer score-model MODEL_DIR REQ.json` | GPU | Teacher-forced token probability scoring against frozen histories ([Quality Eval](../tools/eval/README.md)) |
| `orinfer embed-model` / `rerank-model` | GPU | Offline JSON execution for embedding and reranking models |
