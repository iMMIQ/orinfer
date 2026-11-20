# Orinfer

[English](README.md) · [简体中文](README.zh-CN.md) · [日本語](README.ja.md)

High-performance text, vision-language, speech, embedding, and reranking inference engine written in **Rust** and **TileLang**.

Built primarily for **NVIDIA Jetson AGX Orin 64GB (CUDA SM87)**, with a backend-agnostic runtime that also supports **Huawei Ascend 910B / A2 (CANN 9.2)** for experimental TP8 serving of DeepSeek-V4.1 Flash-1017 W4A8.

---

## Highlights

- **OpenAI & Ollama APIs**: Streaming Chat Completions, Ollama `/api/chat` & `/api/generate`, thinking & `thinking_token_budget`, function calling, strict JSON Schema constraints (`llguidance`), and `logprobs`.
- **Multi-Modal & Multi-Task**:
  - **Text & Vision-Language**: Qwen3.8-27B, Qwen3.8-Flash-Next, Qwen3.5-4B, Index-Translate-2B/9B/35B-A3B-preview, and DSv4.1 Flash-1017 (Ascend 910B TP8).
  - **Speech Recognition**: Qwen3-ASR-1.7B with native Rust audio decoding/resampling and `/v1/audio/transcriptions` + Chat `input_audio` ([Audio Guide](docs/audio.md)).
  - **Embeddings**: Nemotron-3.5-Embed-1B for text, image, and multi-image retrieval via `/v1/embeddings` ([Embeddings Guide](docs/embeddings.md)).
  - **Reranking**: Qwen3-Reranker-0.6B via `/v1/rerank` ([Reranking Guide](docs/reranking.md)).
  - **Offline Batching**: Resumable JSONL inference with prefix-aware scheduling ([Batch Guide](docs/batch.md)).
- **Quantization & Memory Efficiency**: Two-stage E8P Q4A8/W8 for Qwen3.5/Embed/Reranker/ASR, E8P Q2A8 experts + Q4/W8 projections for Flash Next, INT8 KV cache, and up to 256K context ([Quantization Note](docs/quantization.md)).
- **Production Runtime**: Chunked prefill, isolated per-request state, radix prefix cache with optional CPU KV offload, CUDA/ACL Graph capture (`decode_only`, `full`, `off`), and adaptive native MTP / DSpark speculative decoding (`--mtp-drafts 0..31` or `auto`).

---

## Supported Models

| Model | Modalities / Task | Backend | Weight Format | Max Context |
| --- | --- | --- | --- | ---: |
| **Index-Translate-2B / 9B / 35B-A3B** | Text, Image, Multi-Image | CUDA SM87 | E8P Q4A8 / W8 | 256K |
| **Qwen3.5-4B** | Text, Image, Multi-Image | CUDA SM87 | E8P Q4A8 / W8 | 256K |
| **Qwen3.8-27B** | Text, Image, Multi-Image | CUDA SM87 | E8P Q4A8 / W8 | 256K |
| **Qwen3.8-Flash-Next** | Text, Image, Multi-Image | CUDA SM87 | E8P Q2A8 experts + Q4/W8 | 256K |
| **Qwen3-ASR-1.7B** | Speech Transcription | CUDA SM87 | E8P Q4A8 / W8 + FP16 audio | 64K |
| **Nemotron-3.5-Embed-1B** | Text & Image Embeddings | CUDA SM87 | E8P Q4A8 / W8 | 32K |
| **Qwen3-Reranker-0.6B** | Query-Document Reranking | CUDA SM87 | E8P Q4A8 / W8 | 32K |
| **DSv4.1 Flash-1017** *(experimental)* | Text, Image, Tools, DSpark | Ascend 910B TP8 | Native W4A8 + INT8 KV | 256K |

---

## Quick Start (Jetson AGX Orin)

### Prerequisites
- `aarch64` Linux with NVIDIA CUDA Driver
- Precompiled `orinfer` engine archive and matching execution package from [Releases](https://github.com/iMMIQ/orinfer/releases)
- Prepared model directory (config, tokenizer, chat template, and quantized safetensors whose `execution_package` digest matches the installed package)

### Install & Serve

```bash
# 1. Verify release checksums and extract bin/orinfer into PATH
sha256sum --ignore-missing -c SHA256SUMS

# 2. Install the matching execution package into the local cache
orinfer package install /path/to/matching-execution.tar.xz

# 3. Validate model files, execution package, and weight hashes
orinfer validate-model /path/to/prepared-model

# 4. Start the OpenAI / Ollama HTTP server
orinfer serve /path/to/prepared-model --model qwen3.8-27b
```

The server listens on `[::]:8088` by default. Set `ORINFER_API_KEY` to require Bearer authentication on `/v1` and `/api`.

```bash
curl http://127.0.0.1:8088/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Introduce yourself in one sentence."}],
    "max_tokens": 512,
    "stream": true
  }'
```

For multi-model managed serving (`orinfer serve` without a model path), Ollama API, images, and [OpenCode](examples/opencode.json) / [Pi](examples/pi-models.json) / [DeepSeek Harness](docs/deepseek-harness.md) integration, see the [Serving Guide](docs/serving.md).

For Ascend 910B / A2 deployment, run `orinfer devices --backend ascend` and `orinfer validate-ascend --device 0`, and see the [DSv4.1 A2 Production Guide](docs/deepseek-v41-production.md).

---

## Performance Snapshot (Jetson AGX Orin 64GB)

Fixed English-to-Chinese translation benchmark (2026-10-21, `decode_only`, greedy, seed `20261002`, thinking off, prefix cache off; single-request uses adaptive MTP up to 3 drafts, multi-request disables MTP). Throughput is in **tok/s**; startup times are medians of 3 fresh processes in seconds.

| Model | Prefill 512 | Prefill 8K | Single Decode (MTP) | 8-Req Decode | Max Concurrency / Decode | Serve → API (s) | Serve → First Output (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Index-Translate-2B | 8,121.1 | 7,507.7 | 171.5 | 757.4 | 128 / 2,361.1 | 1.12 | 1.26 |
| Qwen3.5-4B | 3,592.6 | 3,417.0 | 90.7 | 360.5 | 128 / 1,071.3 | 1.68 | 1.83 |
| Index-Translate-9B | 2,213.5 | 2,223.3 | 56.5 | 233.9 | 128 / 890.7 | 2.92 | 3.10 |
| Qwen3.8-27B | 658.9 | 598.5 | 33.6 | 67.6 | 64 / 135.9 | 6.68 | 7.05 |
| Index-Translate-35B-A3B | 2,233.2 | 3,222.2 | 83.0 | 338.7 | 128 / 825.0 | 8.01 | 8.18 |
| Qwen3.8-Flash-Next | 925.6 | 858.1 | 56.4 | 157.3 | 32 / 282.9 | 15.79 | 15.96 |

See [Performance Reference](docs/performance.md) for the latest **Schema 2 + P1–P6 Mini-Weka end-to-end baselines (2026-11-20)**, vLLM comparisons, JSONL batch throughput, and Ascend 910B3 TP8 results.

---

## Common CLI Options

| Option | Default | Description |
| --- | --- | --- |
| `--listen` | `[::]:8088` | HTTP listen address |
| `--default-request-params` | `{}` | JSON object of default Chat request fields |
| `--cuda-graph` / `--graph` | `decode_only` | Graph capture mode (`decode_only`, `full`, `off`) |
| `--mtp-drafts` | `auto` | Adaptive MTP cap (`auto`, `0` off, or `1..31`) |
| `--resident-experts` | `full` | Keep all experts resident, or set per-layer SLRU budget `N` (Flash Next) |
| `--prefix-cache-mib` | `4096` | Device prefix-cache budget per device in MiB (`0` disables) |
| `--kv-offload-mib` | `0` | Global CPU prefix-state cache in MiB (`0` disables) |
| `--max-active-requests` | `32` | Maximum concurrent active requests (`1..128`) |
| `--max-batch-tokens` | `128` | Ordinary decode row budget (`1..128`) |
| `--max-speculative-tokens` | `0` | MTP verification row budget (`0` uses package capacity) |
| `--prefill-budget-ms` | `200` | Predicted mixed-prefill budget under `latency` policy |

Execution packages are stored in `~/.cache/orinfer/packages` by default (override via `ORINFER_EXECUTION_CACHE`). Run `orinfer serve --help` for the full option list.

---

## Build from Source

Rust builds the online engine and CLI; Python 3.12+ (`uv`) handles offline quantization, TileLang compilation, and validation.

```bash
git clone https://github.com/iMMIQ/orinfer.git
cd orinfer
cargo fetch --locked
make build
make python-env
make check
```

---

## Documentation

Browse the full [Documentation Hub](docs/README.md) (all detailed guides are in English):

- **Serving & Clients**: [Serving Guide](docs/serving.md) · [Offline JSONL Batch](docs/batch.md) · [DeepSeek Harness](docs/deepseek-harness.md) · [DSv4.1 A2 Production](docs/deepseek-v41-production.md)
- **Models & Modalities**: [Speech Recognition](docs/audio.md) · [Embeddings](docs/embeddings.md) · [Reranking](docs/reranking.md)
- **Architecture & Packages**: [Model Execution Packages](docs/model-packages.md) · [Ascend A2 Build Pipeline](docs/ascend-a2-build.md) · [E8P Quantization Note](docs/quantization.md)
- **Benchmarks & Development**: [Performance Reference](docs/performance.md) · [Development Guide](docs/development.md) · [Build & Release](tools/build/README.md) · [GPU/NPU Kernels](kernels/README.md)

---

## License

**LGPL-3.0-or-later**. See [LICENSE](LICENSE), [COPYING](COPYING), and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Model weights and third-party components retain their respective licenses.
