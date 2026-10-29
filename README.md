# Orinfer

[English](README.md) · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · [한국어](README.ko.md)

Text, vision-language and speech inference on NVIDIA Jetson, powered by Rust and TileLang.

Built for **Jetson AGX Orin 64GB / CUDA SM87**. Rust runs the engine and APIs; TileLang compiles GPU kernels into execution packages ahead of deployment. Models include Qwen3.8-27B, Qwen3.8-Flash-Next, Qwen3.5-4B, and Index-Translate-2B/9B/35B-A3B-preview. Qwen3-ASR-1.7B provides speech transcription. Each model package declares its input capabilities.

The shared runtime separates device execution from scheduling, model semantics and prefix state. The Ascend 910B / A2 backend provides native ACL resources, asynchronous transfers, grouped W4A8, TileLang AOT operators and ACL Graph validation. An experimental native provider executes the complete 40-layer DSv4.1 Flash-1017 W4A8 model through the shared HTTP/SSE service, continuous worker and prefix cache. Run `orinfer devices --backend ascend` to inspect devices or `orinfer validate-ascend --device 0` for bounded native checks in a configured CANN environment. Existing CUDA packages and the model C ABI remain compatible. See [A2 text serving](docs/model-packages.md#a2-文本服务与长上下文).

## Features

- OpenAI Chat Completions and Ollama chat/generate APIs: streaming, thinking, sampling, function calling, JSON Schema constraints and token probabilities.
- [Speech recognition](docs/audio.md): Qwen3-ASR-1.7B, file transcription and Chat audio input through the Rust runtime.
- [Embeddings](docs/embeddings.md): Nemotron-3.5-Embed-1B with E8P Q4A8 text, image and multi-image document retrieval, an embeddings API and offline CLI.
- [Reranking](docs/reranking.md): Qwen3-Reranker-0.6B with E8P Q4A8, causal yes/no scoring, a document reranking API and offline CLI.
- [JSONL offline inference](docs/batch.md): continuous batching, incremental output and resumable jobs through the Rust CLI.
- E8P Q4A8/W8 mixed weights for Qwen3_5 models; E8P Q2 experts and Q4/W8 projections for Flash Next. Sensitive paths retain BF16/FP16/FP32.
- Chunked prefill, isolated request state, prefix cache, INT8 KV and model-configured context up to 256K tokens.
- Adaptive native MTP chooses draft depth from measured cost and acceptance. `--mtp-drafts 1..7` sets the upper bound, `0` disables it, and `auto` uses the package limit.
- CUDA Graph modes: `decode_only` (default), `full` and `off`.

## Quick start

Deployment uses **aarch64 Linux, the NVIDIA CUDA Driver, a precompiled engine and a matching execution package**.

Download the engine archive, matching execution package and `SHA256SUMS` from [Releases](https://github.com/iMMIQ/orinfer/releases). Verify the files, extract the engine and add `bin/orinfer` to `PATH`. Download the prepared model separately; its directory contains configuration, tokenizer, chat template and quantized safetensors. Its `execution_package` digest must match the installed package.

```bash
sha256sum --ignore-missing -c SHA256SUMS
orinfer package install /path/to/matching-execution.tar.xz
orinfer validate-model /path/to/prepared-model
orinfer serve /path/to/prepared-model --model qwen3.8-27b
```

The server listens on `[::]:8088`. Set `ORINFER_API_KEY` to enable Bearer authentication for `/v1` and `/api`.

```bash
curl http://127.0.0.1:8088/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Introduce yourself in one sentence."}],"max_tokens":512,"stream":true}'
```

Endpoints include `/health`, `/v1/models` and `/v1/chat/completions`. The [serving guide](docs/serving.md) covers Ollama deployment, images, request defaults and [OpenCode](examples/opencode.json) / [Pi](examples/pi-models.json) integration.

## Performance

Text and vision-language models, Jetson AGX Orin 64GB, development execution packages, measured **2026-10-21**. Inputs are fixed English-to-Chinese translation prompts. The benchmark enables adaptive MTP (up to 3 drafts) for single-request runs and automatically disables MTP for multi-request runs.

| Model | Prefill, 512 tokens | Prefill, 8K tokens | Single decode, MTP | 8-request decode | Max requests / decode | Serve → API (s) | Serve → first output (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Index-Translate-2B | 8121.13 | 7507.67 | 171.47 | 757.39 | 128 / 2361.11 | 1.12 | 1.26 |
| Qwen3.5-4B | 3592.58 | 3416.98 | 90.71 | 360.49 | 128 / 1071.29 | 1.68 | 1.83 |
| Index-Translate-9B | 2213.52 | 2223.34 | 56.49 | 233.87 | 128 / 890.74 | 2.92 | 3.10 |
| Qwen3.8-27B | 658.87 | 598.46 | 33.64 | 67.56 | 64 / 135.89 | 6.68 | 7.05 |
| Index-Translate-35B-A3B-preview | 2233.16 | 3222.22 | 82.98 | 338.65 | 128 / 824.98 | 8.01 | 8.18 |
| Qwen3.8-Flash-Next | 925.57 | 858.11 | 56.35 | 157.25 | 32 / 282.87 | 15.79 | 15.96 |

Throughput is **tok/s**; startup times are seconds; multi-request columns report aggregate throughput. Single-request input is 512 or 8192 tokens, with 257 output tokens; the single-decode column uses the 512-token input. Prefill includes request setup, the output head and first-output selection. Decode excludes the first emitted step. The maximum-concurrency column shows the highest completed tier among 8/16/32/64/128 requests and its aggregate tok/s. Multi-request runs use 256 input tokens and 65 output tokens per request, timing the 64 tokens after every first output is ready.

Each configuration has one warmup followed by three measured requests or cohorts; the table reports medians. All runs use `decode_only`, greedy sampling, seed `20261002`, thinking off and prefix cache off. Fixed-length diagnostics ignore EOS.

Startup columns are medians of three fresh Rust `serve` processes: process launch → `/health` ready, then launch → the first nonempty SSE output of a short English-to-Chinese translation request (32-token output budget). File caches are preserved; model loading, GPU setup and first-request work are included. SSE output can contain multiple tokens.

[Performance reference](docs/performance.md) records execution-package identities, 8K decode, MTP rounds, vLLM comparisons, startup, offline throughput and quality scope. Complex image translation is an active quality-improvement area.

## Common options

| Option | Default | Purpose |
| --- | --- | --- |
| `--listen` | `[::]:8088` | HTTP listener |
| `--default-request-params` | `{}` | JSON request defaults; explicit request fields take priority |
| `--cuda-graph` | `decode_only` | CUDA Graph mode |
| `--resident-experts` | `full` | Keep all experts resident; Flash Next also supports an SSD-backed capacity |
| `--prefix-cache-mib` | `12288` | GPU prefix-cache budget in MiB; 0 disables caching |
| `--max-active-requests` | `32` | Active request limit, 1–128 |
| `--max-batch-tokens` | `128` | Decode row budget; mixed prefill uses the package capacity |
| `--max-speculative-tokens` | `0` | MTP verification row budget; 0 uses the package capacity |
| `--prefill-budget-ms` | `200` | Predicted mixed-prefill time target |

Concurrency and context capacity follow the execution package and available memory. Execution packages are installed in `~/.cache/orinfer/packages`; `ORINFER_EXECUTION_CACHE` selects another location. See `orinfer serve --help` for all options.

## Development and documentation

Rust handles online loading, scheduling, model execution and serving. Python 3.12 or later is used for offline quantization, compilation and validation.

```bash
git clone https://github.com/iMMIQ/orinfer.git
cd orinfer
cargo fetch --locked
make build
make python-env
make check
```

- [Serving](docs/serving.md) · [Offline JSONL](docs/batch.md) · [Performance](docs/performance.md).
- [Build and release](tools/build/README.md) · [Model packages](docs/model-packages.md).
- [Model builders](tools/model/README.md) · [GPU kernels](kernels/README.md) · [Quality evaluation](tools/eval/README.md).

Additional guides are available in Chinese.

## License

**LGPL-3.0-or-later**. See [LICENSE](LICENSE) and [COPYING](COPYING). Third-party components and model weights retain their respective licenses.
