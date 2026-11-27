# Performance Reference

This page records current end-to-end performance baselines on **Jetson AGX Orin 64GB (CUDA SM87)** and **Ascend 910B3 TP8 (CANN 9.2)**. Every measurement uses Rust release binaries, fixed `seed=20261002`, and actual committed token counts.

---

## 1. Orin SM87 — Latest Mini-Weka 512-Trajectory Baselines (2026-11-23)

- **Hardware & Runtime**: Jetson AGX Orin 64GB (R36.4.3, CUDA SM87), Schema 2 execution packages, `decode_only`, `--mtp-drafts auto`, `--prompt-lookup`, `--prefix-cache-mib 4096`, `--max-active-requests 4`.
- **Workload**: 512 synthetic multi-turn agent trajectories (`tools.bench.mini_weka_synthetic --profile compact`, cold first turns 2/3/4/6 Ki tokens, concurrency 4, 240 s wall limit, `temperature=1.0`, `ignore_eos=true`).
- **Progression**: `Baseline (58ab647)` → `P1–P5 (Rust scheduler/cache)` → `v0.4.0 (P6 / Flash-Next stage1)` → `v0.4.1 (Flash-Next Phase 1–3 kernels, Row-Q4 out-proj, async layered prefill)`.

| Model (Latency Policy) | Completed Reqs | Full Trajectories | Output TPS | `batch_commit_s` | TTFT P50 / P95 (s, latest) |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Index-Translate-2B** | 288 → 358 → **356** | 39 → 50 → **48** | 124.0 → 156.2 → **155.6** | 29.3 → 45.0 → **43.2** | 0.24 / 0.80 |
| **Qwen3.5-4B** | 133 → 168 → **198** | 17 → 24 → **28** | 57.6 → 73.6 → **86.6** | 24.3 → 36.3 → **8.0** | 0.52 / 1.95 |
| **Index-Translate-9B** | 90 → 125 → **128** | 12 → 16 → **16** | 39.3 → 53.8 → **55.5** | 16.1 → 23.2 → **14.2** | 0.76 / 2.98 |
| **Qwen3.8-27B** | 26 → 34 → **34** | 4 → 4 → **4** | 13.5 → 15.8 → **15.8** | 4.0 → 1.9 → **0.6** | 3.86 / 26.43 |
| **Index-Translate-35B-A3B** | 68 → 124 → **133** | 8 → 16 → **17** | 28.4 → 52.8 → **59.7** | 94.0 → 33.4 → **16.9** | 0.72 / 2.29 |
| **Qwen3.8-Flash-Next** | 72 → 86 → 93 → **102** | 8 → 12 → 12 → **12** | 32.3 → 39.8 → 41.6 → **46.5** | 1.6 → 1.1 → 1.1 → **1.0** | 1.00 / 4.08 |
| **Qwen3.8-Flash-Next** *(throughput)* | 92 → 93 → 100 → **102–105** | 12 → 12 → 12 → **13** | 41.5 → 41.8 → 44.9 → **46.3–48.1** | 2.1 → 1.6 → 1.4 → **1.2–1.4** | 0.80 / 3.29 |

- **Notes**: Prefix cache hit rates reached **77.9%–79.7%** across all six models (`79.2%–79.3%` on Flash-Next v0.4.1 with digest `28040d50`). Output TPS includes queuing, prefill, generation, and HTTP/SSE over the profiling window (total wall minus warmup). Zero CUDA/runtime errors, panics, or OOMs across all runs.

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

## 2. Orin SM87 — 6-Model Throughput & Startup Matrix (2026-11-20)

Fixed English-to-Chinese translation inputs using the latest Schema 2 packages (`decode_only`, greedy, `seed=20261002`, thinking off, prefix cache off; median of 3 measured runs after 1 warmup). Single-request enables adaptive MTP (≤3 drafts, 512/8192 input, 257 output); multi-request disables MTP (256 in / 65 out per request). Startup columns are medians of 3 fresh `orinfer serve` processes (warm page cache).

| Model | Prefill 512 | Prefill 8K | Single Decode 512 (MTP) | Single Decode 8K (MTP) | 8-Req Decode | Max Reqs / Decode | Serve → API (s) | Serve → First Output (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Index-Translate-2B** | 8,100.7 | 7,551.6 | 153.8 | 137.6 | 759.2 | 128 / 2,475.0 | 0.95 | 1.05 |
| **Qwen3.5-4B** | 3,582.9 | 3,379.8 | 114.7 | 94.7 | 360.6 | 128 / 1,094.2 | 1.19 | 1.35 |
| **Index-Translate-9B** | 2,223.4 | 2,243.9 | 53.7 | 48.5 | 234.3 | 128 / 909.0 | 1.93 | 2.12 |
| **Qwen3.8-27B** | 655.2 | 596.9 | 24.6 | 22.2 | 61.0\* | 64 / 115.7\* | 6.08 | 6.43 |
| **Index-Translate-35B-A3B** | 2,204.5 | 3,206.6 | 75.7 | 70.1 | 350.1 | 128 / 848.8 | 7.14 | 7.40 |
| **Qwen3.8-Flash-Next** | 979.3 | 1,081.8 | 62.9 | 31.5 | 163.3 | 32 / 284.8 | 15.10 | 15.27 |

\* `Qwen3.8-27B` 8/64-request decode measured end-to-end through `orinfer serve --mtp-drafts auto` (HTTP/SSE, `throughput` policy); the other five models use `benchmark_fixed_requests`. Package digests: 2B `cfc11bc2`, 4B `5ee0cdf3`, 9B `6d283b0c`, 27B `cca4c6fe`, 35B `793b8064`, Flash-Next `fc87e2ca`.

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

### Frozen 240s Mini-Weka Replay (`tools.bench.agentx_mini`, 2026-11-26)

8× Ascend 910B3 TP8, CANN 9.2, `--graph full`, `--mtp-drafts auto`, `--prefix-cache-mib 8192`, `--kv-offload-mib 262144`, `concurrency=16`, `wall_limit=240 s`, fixed `--sampling-seed 20261002` (paired quiet-host runs):

| Release / Package Digest | Completed Profiling Reqs | Warmup Wall (s) | Errors | Notes |
| --- | ---: | ---: | ---: | --- |
| **v0.4.0** (`ce7a1cec`) | 73 | 143.27 | 0 | Baseline |
| **Pre-v0.4.2 (`4968364`)** | 92–93 | 124.28 | 0 | Engram/scheduler + 1-wave/Cube wave |
| **v0.4.2** (`8f58245c`) | 101 / 101 *(+38.4%)* | 117.88 / 117.68 | 0 | FastForward IPC, mixed-packet/bucketed graph reuse, async Layer-14 Engram prefetch, 20-slot 1-wave & Cube upgrades |
| **v0.4.3** (`67099303`) | 108 *(+47.9%)* | 107.33 | 0 | 20-AIC persistent dense/vector/fused indexer kernels, canonical ACL Graph prewarm, admission Engram prefetch, non-Rayon prefetch deadlock fix |
| **v0.4.4** (`3b3961a7`) | 122 *(+67.1%)* | 105.35 | 0 | `overlap_qk` fused decode attention, `small_regroup` single-UB MoE regroup, `FastVerifyForward`, Layer-1 Engram prefetch, auto TP rejection sampling, uniform cross-rank bucketed graph reuse, expanded prewarm & decode/mixed buckets (`24`/`384`), bucket-slack draft fill |
| **Pre-v0.4.6** (`377bd367`) | 129 *(+76.7%)* | 99.80 | 0 | Cross-row pipelined `selected_int8` attention (`rows_per_cta=13`), lossless W4A4 MoE `down_proj` fusion, 20-AIC persistent routing/sampling/model_ops, device-side `SpanCopier`, single-graph TP rejection sampling, fine low-load decode/verify buckets, rejection normalizer stride fix, and `Team::resident_graphs` eviction sync |
| **v0.4.6** (`0a6c09f4`) | **130** *(+78.1%)* | **99.44** | 0 | Persistent 20-AIC `engram.cast_fp32` replacing rotated Engram kernels, paired with activation-calibrated unrotated weights (execution-graph-exact SmoothQuant + 262K routing-weighted Act-Order Hessian-GPTQ + L15/L39 lossless W8 + BF16 KV-source + full W8 DSpark draft); preserves 4.64 GiB device prefix cache (`7.35 GiB`/card free HBM, 99.5% ACL Graph hit rate) |

- **DSv4.1 Production Acceptance**: Use `tools.bench.agentx_mini` (paired cold/warm Mini-Weka replay with fixed `--sampling-seed 20261002` and provenance) as the final end-to-end gate. Held-out 1,556-position teacher-forced evaluation against official MXFP4+FP8 on **v0.4.6** (`0a6c09f4`): `mean KL(Official || Model) = 0.01992` (**−35.2%** vs. legacy rotated `0.03075`, 95% CI `[-0.01333, -0.00832]`), `median KL = 0.00533`, `p95 KL = 0.07837`, `argmax_match = 95.76%`, `top3_cover = 99.55%`; DSpark 256-window acceptance `69.84%` (`4.492 committed/round`, `+0.137/round` vs. legacy rotated; held-out `test` split `81.56%` / `5.078 committed/round`).

---

## 6. Quality Validation Scope

- **Teacher-Forced BF16 Reference (`seed=20261002`)**:
  - **Index-Translate-2B / 9B & Qwen3.5-4B**: Text ΔNLL **+0.026 / +0.017 / −0.016 nats/token** vs. official BF16; 100% of high-confidence top-1 positions match BF16.
  - **Qwen3.8-27B**: 98.19% top-1 match (150/150 high-confidence), 100% inside BF16 top-3.
  - **Qwen3.8-Flash-Next**: 96.88% top-1 match, 100% inside BF16 top-3, ΔNLL +0.0224 nats/token.
- **Known Scope Limits**: Complex dense-text / multi-image translation quality remains an active improvement area. On Ascend A2, `HCCL_DETERMINISTIC=strict` is required for bit-exact cross-batch diagnostics; default HCCL trades bit determinism for throughput.
