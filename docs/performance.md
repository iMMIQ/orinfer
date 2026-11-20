# Performance Reference

This page records current end-to-end performance baselines on **Jetson AGX Orin 64GB (CUDA SM87)** and **Ascend 910B3 TP8 (CANN 9.2)**. Every measurement uses Rust release binaries, fixed `seed=20261002`, and actual committed token counts.

---

## 1. Orin SM87 — Latest Mini-Weka 512-Trajectory Baselines (2026-11-20)

- **Hardware & Runtime**: Jetson AGX Orin 64GB (R36.4.3, CUDA SM87), Schema 2 execution packages, `decode_only`, `--mtp-drafts auto`, `--prompt-lookup`, `--prefix-cache-mib 4096`, `--max-active-requests 4`.
- **Workload**: 512 synthetic multi-turn agent trajectories (`tools.bench.mini_weka_synthetic --profile compact`, cold first turns 2/3/4/6 Ki tokens, concurrency 4, 240 s wall limit, `temperature=1.0`, `ignore_eos=true`).
- **Progression**: `Baseline (58ab647)` → `P1–P5 (Rust scheduler/cache)` → `P6 (Batched GPU sampling packages)`.

| Model (Latency Policy) | Completed Reqs | Full Trajectories | Output TPS | `batch_commit_s` | TTFT P50 / P95 (s, P5) |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Index-Translate-2B** | 288 → 358 → **356** | 39 → 50 → **48** | 124.0 → 156.2 → **155.6** | 29.3 → 45.0 → **43.2** | 0.24 / 0.80 |
| **Qwen3.5-4B** | 133 → 168 → **198** | 17 → 24 → **28** | 57.6 → 73.6 → **86.6** | 24.3 → 36.3 → **8.0** | 0.52 / 1.95 |
| **Index-Translate-9B** | 90 → 125 → **128** | 12 → 16 → **16** | 39.3 → 53.8 → **55.5** | 16.1 → 23.2 → **14.2** | 0.76 / 2.98 |
| **Qwen3.8-27B** | 26 → 34 → **34** | 4 → 4 → **4** | 13.5 → 15.8 → **15.8** | 4.0 → 1.9 → **0.6** | 3.86 / 26.43 |
| **Index-Translate-35B-A3B** | 68 → 124 → **133** | 8 → 16 → **17** | 28.4 → 52.8 → **59.7** | 94.0 → 33.4 → **16.9** | 0.72 / 2.29 |
| **Qwen3.8-Flash-Next** *(P5)* | 72 → **86** | 8 → **12** | 32.3 → **39.8** | 1.6 → **1.1** | 1.53 / 6.70 |
| **Qwen3.8-Flash-Next** *(throughput, P5)* | 92 → **93** | 12 → **12** | 41.5 → **41.8** | 2.1 → **1.6** | 0.99 / 3.93 |

- **Notes**: Prefix cache hit rates reached **77.9%–79.7%** across all six models under P5/P6. Output TPS includes queuing, prefill, generation, and HTTP/SSE over the profiling window (total wall minus warmup). Zero CUDA/runtime errors, panics, or OOMs across all runs.

### Reproduce

```bash
uv run --locked python -m tools.bench.mini_weka_synthetic --profile compact --output artifacts/bench/mini-512
orinfer serve "$MODEL_DIR" --listen 127.0.0.1:18828 --model "$MODEL_ID" \
  --cuda-graph decode_only --mtp-drafts auto --resident-experts full \
  --prefix-cache-mib 4096 --max-active-requests 4 --prompt-lookup
uv run --locked python -m tools.bench.mini_weka_compact \
  --manifest artifacts/bench/mini-512/manifest.json --tokenizer "$MODEL_DIR" \
  --url http://127.0.0.1:18828/v1 --model "$MODEL_ID" --concurrency 4 --wall-limit 240 \
  --output artifacts/bench/out
```

---

## 2. Orin SM87 — 6-Model Throughput & Startup Matrix (2026-10-21)

Fixed English-to-Chinese translation inputs (`decode_only`, greedy, `seed=20261002`, thinking off, prefix cache off; median of 3 measured runs after 1 warmup). Single-request enables adaptive MTP (≤3 drafts); multi-request disables MTP (256 in / 65 out per request). Startup columns are medians of 3 fresh `orinfer serve` processes (warm page cache).

| Model | Prefill 512 | Prefill 8K | Single Decode 512 (MTP) | Single Decode 8K (MTP) | 8-Req Decode | Max Reqs / Decode | Serve → API (s) | Serve → First Output (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Index-Translate-2B** | 8,121.1 | 7,507.7 | 171.5 | 149.9 | 757.4 | 128 / 2,361.1 | 1.12 | 1.26 |
| **Qwen3.5-4B** | 3,592.6 | 3,417.0 | 90.7 | 76.3 | 360.5 | 128 / 1,071.3 | 1.68 | 1.83 |
| **Index-Translate-9B** | 2,213.5 | 2,223.3 | 56.5 | 50.4 | 233.9 | 128 / 890.7 | 2.92 | 3.10 |
| **Qwen3.8-27B** | 658.9 | 598.5 | 33.6 | 30.6 | 67.6 | 64 / 135.9 | 6.68 | 7.05 |
| **Index-Translate-35B-A3B** | 2,233.2 | 3,222.2 | 83.0 | 74.9 | 338.7 | 128 / 825.0 | 8.01 | 8.18 |
| **Qwen3.8-Flash-Next** | 925.6 | 858.1 | 56.4 | 48.0 | 157.3 | 32 / 282.9 | 15.79 | 15.96 |

---

## 3. Orin SM87 — Dense Models vs. vLLM (MTP Off)

Same Jetson AGX Orin 64GB, MTP off, prefix cache off, greedy (`seed=20261002`). vLLM baseline: `0.20.0+cu126` (2B/9B official FP8, 4B community AWQ g32, Marlin kernels, BF16 KV, FP32 GDN).

| Model | Input | Orinfer Prefill | vLLM Prefill | Orinfer Decode | vLLM Decode | 128-Req Decode (Orinfer vs vLLM) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Index-Translate-2B** | 512 / 8K | **8,846** / **8,186** | 5,023 / 5,643 | **119.7** / **110.9** | 63.9 / 61.7 | **2,362.1** vs 2,199.6 |
| **Qwen3.5-4B** | 512 / 8K | **3,801** / **3,627** | 2,637 / 2,658 | **58.1** / **52.8** | 45.0 / 42.2 | **1,071.2** vs 995.5 |
| **Index-Translate-9B** | 512 / 8K | **2,394** / **2,340** | 1,270 / 1,287 | **34.6** / **32.6** | 18.4 / 17.9 | **886.4** vs 687.5 |

---

## 4. Orin SM87 — Offline JSONL Batch Throughput (10K OPUS-100)

10,000 deduplicated English-to-Chinese OPUS-100 sentences on **Index-Translate-2B** (`--max-active-requests 64`, `--mtp-drafts 0`, `--prefix-cache-mib 0`, `decode_only`, natural EOS, 217,196 output tokens):

| Policy | Total CLI Wall Time | Post-Ready Wall Time | End-to-End Output TPS | Post-Ready Output TPS |
| --- | ---: | ---: | ---: | ---: |
| `latency` | 535.89 s | — | 405.3 | — |
| `throughput` + `--batch-order auto` | **314.78 s** (1.70×) | **309.67 s** | **690.0** | **701.4** |

On 1,024 sentences sharing 32 document translation guidelines (`--prefix-cache-mib 128`), global prefix planning reduced total CLI time from **100.63 s → 63.85 s (1.58×)** with **707/1,024** cache hits (median restore 0.70 ms).

For speech recognition (Qwen3-ASR-1.7B vs. vLLM FP8), see [Audio Guide](audio.md#3-validation--performance-jetson-agx-orin-64gb).

---

## 5. Ascend 910B3 TP8 — DSv4.1 Reference Baselines

8× Ascend 910B3, TP8, CANN 9.2, full 40-layer native W4A8, group-64 INT8 KV, default HCCL (`HCCL_OP_EXPANSION_MODE=AIV`), MTP/DSpark off, prefix cache off (2K input, 32 decode steps, median of 3 warm runs):

### Dynamic Batch Decode Scaling (`bench-ascend-batch`)

| Concurrency | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Aggregate Decode TPS** | 49.67 | 79.70 | 113.63 | 153.52 | 185.63 | 202.32 | 210.15 | 211.71 |
| **Per-Request Decode TPS** | 49.67 | 39.85 | 28.41 | 19.19 | 11.60 | 6.32 | 3.28 | 1.65 |

### Prefill Throughput (Full Decoder vs. Bounded Replay-128)

| Scenario | Full Decoder TPS | Bounded Replay-128 TPS | Change |
| --- | ---: | ---: | ---: |
| Single 2,048 tokens | 4,776.4 | **6,866.3** *(7,313.0 w/ source-KV boundary + 4-head softmax)* | +43.8% |
| Single 2,049 tokens (2048 chunk) | 4,498.2 | **6,353.1** | +41.2% |
| Batch 32 × 17 tokens | 3,358.0 | 3,269.8 | −2.6% |

- **DSv4.1 Production Acceptance**: Use `tools.bench.agentx_mini` (paired cold/warm Mini-Weka replay with fixed `--sampling-seed 20261002` and provenance) as the final end-to-end gate.

---

## 6. Quality Validation Scope

- **Teacher-Forced BF16 Reference (`seed=20261002`)**:
  - **Index-Translate-2B / 9B & Qwen3.5-4B**: Text ΔNLL **+0.026 / +0.017 / −0.016 nats/token** vs. official BF16; 100% of high-confidence top-1 positions match BF16.
  - **Qwen3.8-27B**: 98.19% top-1 match (150/150 high-confidence), 100% inside BF16 top-3.
  - **Qwen3.8-Flash-Next**: 96.88% top-1 match, 100% inside BF16 top-3, ΔNLL +0.0224 nats/token.
- **Known Scope Limits**: Complex dense-text / multi-image translation quality remains an active improvement area. On Ascend A2, `HCCL_DETERMINISTIC=strict` is required for bit-exact cross-batch diagnostics; default HCCL trades bit determinism for throughput.
