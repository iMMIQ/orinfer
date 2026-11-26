# Ascend 910B (DSv4.1) Offline Build Pipeline

This guide summarizes the offline build, weight prepacking, DSpark, vision, and compact release workflow for **DeepSeek-V4.1 Flash-1017 W4A8** on **Ascend 910B / CANN 9.2 (TP8)**.

For production deployment, see [DSv4.1 Production Guide](deepseek-v41-production.md).

---

## 1. Toolchain & Environment

- **Compiler Runtime**: CANN 9.2 (mounted from `/usr/local/Ascend/cann-9.2.0`, override via `ORINFER_ASCEND_CANN_ROOT`) + offline container managed by `tools/environments/ascend/run.sh` (Python 3.12, PyTorch, `torch_npu`, pinned TileLang-Ascend).
- **CPU-Only Compilation**: Pass `--without-devices` to `run.sh` to compile TileLang-Ascend AOT libraries without claiming NPUs (uses explicit NPU storage rewrite `storage_rewrite=explicit-npu-v1` to match on-board compilation).
- **Model Root**: Host `/data/models` is mounted read-only; override via `ORINFER_ASCEND_MODEL_ROOT`.

```bash
# Build the offline Ascend compiler image once
ORINFER_ASCEND_BASE_IMAGE='your-torch-npu-frontend:py312' \
  bash tools/environments/ascend/run.sh build
```

---

## 2. Base Model Build & Weight Prepacking

### Step 1 — Export Engram Metadata & Build Base Plan

```bash
# Export deterministic tokenizer-compression & hash metadata
uv run --locked python -m orinfer_build.models.deepseek_v41.export_engram \
  --model /data/models/DeepSeek-V4.1-Flash-1017-w4a8-Ascend \
  --output artifacts/engram-metadata.json

# Compile base TP8 plan (add --prefill-capacity 2048 for large prefill)
bash tools/environments/ascend/run.sh --without-devices python3 \
  -m orinfer_build.models.deepseek_v41.build \
  --model-dir /data/models/DeepSeek-V4.1-Flash-1017-w4a8-Ascend \
  --engram-metadata artifacts/engram-metadata.json \
  --output artifacts/ascend-model --compile-workers 8

# Materialize physical W4/scale/assist banks on disk (ARM NEON 8x8 transpose)
uv run --locked python -m orinfer_build.models.deepseek_v41.prepare \
  --plan artifacts/ascend-model/model.json \
  --model-dir /data/models/DeepSeek-V4.1-Flash-1017-w4a8-Ascend \
  --output artifacts/ascend-model/model-prepared.json --workers 16
```

### Step 2 — Quantize KV, Extend to 262K & Create Service Directory

```bash
# Convert BF16 KV plan to group-64 INT8 KV + FP16 scales
bash tools/environments/ascend/run.sh --without-devices python3 \
  -m orinfer_build.models.deepseek_v41.quantize_kv \
  --plan artifacts/ascend-model/model-prepared.json \
  --output artifacts/ascend-model/model-kv-int8.json

# Build 262K multi-owner plan (32 owners, length-input causal indexer)
bash tools/environments/ascend/run.sh --without-devices python3 \
  -m orinfer_build.models.deepseek_v41.long_context \
  --plan artifacts/ascend-model/model-kv-int8.json \
  --output artifacts/a2-long \
  --capacity 262144 --owners 32 --length-input --gather-slots 16

# Publish service directory (add --share-prefix-banks for append-only KV bank sharing)
uv run --locked python -m orinfer_build.models.deepseek_v41.service \
  --plan artifacts/a2-long/package/model.json \
  --model-dir /data/models/DeepSeek-V4.1-Flash-1017-w4a8-Ascend \
  --output artifacts/models/dsv41-a2-service --share-prefix-banks
```

### Step 3 — Release Export, Transposed-NZ & Packed Host Tables

For production releases, export physical safetensors (≤4 GiB shards), convert W4 banks to Ascend transposed-NZ `[E, K/64, N/16, 16, 32]`, and pack Engram rows (`i8-f32-g32-v1`) so online startup loads directly without runtime transposes:

```bash
# Single-pass physical safetensors export with transposed-NZ W4 banks & packed Engram host rows
uv run --locked python -m orinfer_build.models.deepseek_v41.export_release \
  --source artifacts/models/dsv41-a2-service \
  --output artifacts/release/packed-weights \
  --execution-output artifacts/release/packed-exec \
  --attribution /data/models/DeepSeek-V4.1-Flash-1017-w4a8-Ascend \
  --w4-layout transposed_nz --pack-host
```

For incremental upgrades of an existing ND release (hardlinking unchanged shards), pass `--mode prepack` or `--mode pack-host` with `--source-execution`.

---

## 3. DSpark Speculative Decoding & Vision

### Adding DSpark (3 Draft Layers + Markov Head)

```bash
ORINFER_ASCEND_MODEL_ROOT=/data bash tools/environments/ascend/run.sh --without-devices \
  python3 -m orinfer_build.models.deepseek_v41.dspark \
  --base-plan artifacts/execution-packages/BASE_DIGEST/model.json \
  --base-model /data/model/TARGET_MODEL \
  --checkpoint /data/models/ASCEND_CHECKPOINT \
  --output artifacts/dspark/build/model.json --workers 16

uv run --locked python -m orinfer_build.models.deepseek_v41.dspark_export \
  --plan artifacts/dspark/build/model.json \
  --base-model /data/model/TARGET_MODEL \
  --checkpoint /data/models/ASCEND_CHECKPOINT \
  --output /data/model/DSPARK_MODEL \
  --execution-output artifacts/execution-packages/dspark
```

- Supports `--block-size 5` (default) or `16` (exposing 5/8/16 aliases sharing the same kernels).
- Device sampling extensions (`processed_local/processed_merge`, `--couple-draft`, `--rejection`) are attached via `python -m orinfer_build.models.deepseek_v41.sampling`.
- Device-side continuation inputs (`device_inputs`) are attached via `python -m orinfer_build.models.deepseek_v41.device_inputs --plan PLAN --output OUT`.

### Adding Vision

Vision weights live in `vision.safetensors` (matching the Ascend language preprocessing basis):

- `orinfer_build.models.deepseek_v41.vision export` — export `vision.safetensors`.
- `orinfer_build.models.deepseek_v41.vision_build` — compile dynamic-patch TileLang-Ascend vision encoder.
- `orinfer_build.models.deepseek_v41.vision attach` / `publish` — attach vision to an existing language plan and publish.
- `orinfer validate-ascend-vision` — validate encoder outputs against checkpoint reference fixtures.

---

## 4. Unrotated Native-Grid Quantization (`quantize_norot`)

To eliminate QuaRot32 rotation distortion against the official MXFP4+FP8 power-of-two weight grid, re-quantize directly from `/data/models/DeepSeek-V4.1-Flash-1017` using `orinfer_build.models.deepseek_v41.quantize_norot`:

- **`L00–L37` routed experts**: Per-channel SSZ signed INT4 `[-8, 7]` (`ssz_w4.c`) packed into Ascend Transposed-NZ (`mean NMSE ≈ 0.0047–0.0049`).
- **`L38–L39` tail routed experts**: Exact-grid lossless INT8 ND with BF16 scales (`0.5 * s_min` power-of-two step), reusing active-row `gather_quantized_active` and `swiglu_packed_quant_active_p20`.
- **Shared KV-source layers (`L02, L08, L14, L20`)**: BF16 `attn.wq_a` and `attn.wkv` with tuned small-M `row_selections`; the other 36 layers stay unrotated W8A8 with fused `norm_quant`.
- **Engram basis restoration**: 20-AIC persistent BF16→FP32 cast (`engram.cast_fp32`), replacing legacy rotated Engram kernels.
- **DSpark draft (`mtp.0/1/2`)**: Exact-grid lossless full W8 routed experts (`--draft-mode full_w8`).
- **Host Engram tables (`weights/host/`, 206 GiB)** and `weights/dspark/embedding.safetensors` are basis-invariant and hardlinked without copying.

```bash
# 1. Compile extra AOT kernels (engram.cast_fp32, tuned BF16 wq_a/wkv, draft A8 gather/swiglu)
bash tools/environments/ascend/run.sh --without-devices python3 \
  -m orinfer_build.models.deepseek_v41.quantize_norot compile-kernels \
  --base-package /path/to/base-installed-cache/BASE_DIGEST \
  --output-dir artifacts/dsv41-norot/kernels --workers 16

# 2. Quantize official checkpoint, export physical shards, and compact RoPE execution package
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 uv run --locked python \
  -m orinfer_build.models.deepseek_v41.quantize_norot export \
  --official-dir /data/models/DeepSeek-V4.1-Flash-1017 \
  --base-model /path/to/base-model \
  --kernels-json artifacts/dsv41-norot/kernels/extra-kernels.json \
  --work-dir artifacts/dsv41-norot \
  --w8-layers 38 39 --bf16-qkv kv-sources --draft-mode full_w8 \
  --compact-engine target/release/orinfer \
  --workers 32 --export-workers 16
```

---

## 5. Plan Optimization & Compact Publication

Key offline plan transforms in `python/orinfer_build/models/deepseek_v41/`:

| Module | Purpose |
| --- | --- |
| `build --rebuild-plan PLAN --rebuild-op OP` | Selectively rebuild specific operators (optional `--rebuild-grid-limit N`, e.g., `20` for 20-Cube devices) |
| `tune` / `tune_auxiliary` / `tune_attention` | On-device real-weight tile & pipeline tuning with changed-input Graph replay verification |
| `fuse_attention` / `fuse_indexer` / `fuse_topk` / `fuse_ffn_collectives` / `expert_fusion` | Offline operator & collective fusion passes preserving numeric contracts |
| `decoder_rows` | Bounded decoder replay (`--keep 128..512`, `--final-prompt-only`, `--source-kv-boundary`) |
| `candidate_blocks` | Checkpoint-declared cross-layer candidate block TopK (layer 20 → layers 24/28/32/36) |
| `floating_head` | Convert FP32 target/draft heads to 3-plane (or 2-plane) BF16 Cube partials + FP32 reduce |
| `publication` | Prune unreferenced AOT libs, strip static symbols, set `--packet-boundary` / `--verification-capacity`, and emit final digest directory |
| `compact` | Emit compact RoPE (`generated_assets`) + `shared-json-v1` plans; `orinfer package install` expands RoPE deterministically on CPU |

```bash
# Publish & compact an execution package
uv run --locked python -m orinfer_build.models.deepseek_v41.publication \
  --source PACKAGE_DIR --output-cache OUTPUT_CACHE --compact-rope --engine target/release/orinfer
```

---

## 6. Benchmarking & Verification

```bash
# Native single-sequence benchmark
HCCL_DETERMINISTIC=true HCCL_OP_EXPANSION_MODE=AIV HCCL_NPU_SOCKET_PORT_RANGE=auto \
  flock -w 5 artifacts/gpu-experiment.lock \
  orinfer bench-ascend --plan PLAN --model-dir MODEL_DIR --inputs artifacts/ascend-inputs.json

# Native shared-row multi-owner benchmark
orinfer bench-ascend-batch --plan PLAN --model-dir MODEL_DIR --inputs inputs-batch.json

# Official FP4/FP8 reference teacher-forced logits comparison
bash tools/environments/ascend/run.sh python3 -m orinfer_build.models.deepseek_v41.reference \
  --model-dir /data/models/DeepSeek-V4.1-Flash-1017 \
  --inputs artifacts/ascend-inputs.json --teacher artifacts/ascend-result.json \
  --case p1024 --positions 129 --device npu:0 --output artifacts/ascend-official-reference
```

- **Determinism**: Default serving uses high-performance HCCL; set `HCCL_DETERMINISTIC=strict` for bit-exact cross-batch diagnostics.
- **Operator tests**: Live under `tests/ascend/operators/` (`validate`, `validate_rows`, `validate_private_rows`, `validate_kv_int8`, `validate_expert_fusion`, `validate_frontier`, etc.).
