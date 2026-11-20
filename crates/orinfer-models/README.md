# `orinfer-models`

Statically linked Rust model adapters compiled as an `rlib` into `orinfer-engine`.

## Registered Model Families (`int8_quality`)

- **Qwen3.5 Dense & MoE**: Qwen3.8-27B, Qwen3.5-4B, Index-Translate-2B / 9B / 35B-A3B-preview.
- **Qwen3-ASR-1.7B**: Audio encoder + Qwen3 dynamic-row text decoder.
- **Qwen3.8-Flash-Next**: Dynamic prefill (1–4096 rows), shared target (≤1024 rows), draft/warm (≤512 rows), MTP verification (2–8 rows), HC/PLE/QSA, and bounded resident expert banks.
- **Nemotron-3.5-Embed-1B**: Bidirectional text + Pixtral vision embeddings.
- **Qwen3-Reranker-0.6B**: Causal query-document reranking.

Each adapter reads a CUDA **Schema 2** execution package (`package.json`, `bindings.json`, `assets.safetensors`) and assembles verified execution plans, dynamic batches, and request-state bindings.

## Build & Test

```bash
cargo build --release --locked --offline -p orinfer-cli
cargo test --locked --offline -p orinfer-models
```

See [Model Execution Packages](../../docs/model-packages.md) for full contract and validation details.
