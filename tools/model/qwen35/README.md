# Qwen3.5 Dense & Shared-Architecture Build

Unified offline builder for **Qwen3.8-27B**, **Qwen3.5-4B**, **Index-Translate-2B / 9B / 35B-A3B**, and **Qwen3-ASR-1.7B** (`--context 65536`), producing CUDA **Schema 2** execution packages (`package.json`, `bindings.json`, `assets.safetensors`).

---

## 1. Build Command

Input must be a completed two-stage E8P Q4A8 quantized checkpoint (from `python/orinfer_build/quantization/qwen_e8p.py`; current 27B baseline: `artifacts/quantization/qwen35-e8p-beam8-20261016/converted`).

```bash
make build
bash tools/environments/cuda/run.sh python/orinfer_build/models/qwen35/build.py \
  artifacts/qwen35-build \
  --source /path/to/converted \
  --model-output artifacts/models/qwen35-q4a8 \
  --gpu-lock-held
```

- **Context**: Defaults to `262144`; pass `--context N` to lower (use `--context 65536` for Qwen3-ASR-1.7B).
- **Frontend Pinning**: Automatically merges HF `tokenizer_config.json` added tokens into `tokenizer.json` (verifying ID parity), pins `chat_template.jinja`, and fills EOS/padding defaults if `generation_config.json` is absent.
- **Cache & Resume**: Default cache is `artifacts/build-cache/qwen35`; pass `--resume` to `run.sh` to continue after interruption.

---

## 2. Precision & Execution Design

- **Two-Stage E8P Q4A8**: `coefficients=[4, 1]`, signed-block128 rotation, beam-8 search, losslessly repacked to `[K/128, N, 16]` with FP16 row scales. Offline autotuning selects between direct INT4 bit-plane MMA and INT8 decode paths over the same compressed bank.
- **Weight Sharing**: 2B and 4B share one compressed E8P Q4 bank for tied embedding/head; 9B keeps separate banks; 27B keeps W8 for embeddings, mixer output projections, and MTP draft matrices.
- **Dynamic Rows & Layer-Boundary Prefill**: Exact AOT specializations for `1/2/3/4/8/16/32` rows + one tuned dynamic fallback per matrix geometry; bounded dynamic specializations for intermediate rows ≤128. Prefill pauses cleanly at layer boundaries keeping only `Hidden` and FP32 residual live.
- **Address-Table Decode & MTP**: 1–128 active decode rows wrap single-row GDN/attention via `independent_rows`; multi-request MTP shares target/draft/commit/warm packets with explicit `MixedRows` role boundaries.
- **Vision**: Shared ViT with one dynamic patch template (`4–32768` patches).

---

## 3. Selective Rebuilds & Kernel Checks

```bash
# Rebuild only short-segment attention (preserving weights & decode geometry)
bash tools/environments/cuda/run.sh python/orinfer_build/models/qwen35/rebuild_attention.py \
  artifacts/qwen35-attention-rebuild --model /path/to/model --target /path/to/out

# Incrementally add MTP packet address tables to an existing dense package
bash tools/environments/cuda/run.sh python/orinfer_build/models/qwen35/rebuild_mtp.py \
  artifacts/qwen35-mtp-rebuild --model /path/to/model --target /path/to/out

# Operator numerical, tail, and Graph replay checks
bash tools/environments/cuda/run.sh python/orinfer_build/models/qwen35/check_kernels.py artifacts/qwen35-kernels --gpu-lock-held
bash tools/environments/cuda/run.sh python/orinfer_build/models/qwen35/check_rows.py artifacts/qwen35-rows
bash tools/environments/cuda/run.sh python/orinfer_build/models/qwen35/check_decode.py artifacts/qwen35-decode
bash tools/environments/cuda/run.sh python/orinfer_build/models/qwen35/check_causal_attention.py artifacts/qwen35-causal-attention
```

For E8P Q4 integer & direct INT4 MMA verification, run `check_e8p_q4.py` and `check_e8p_direct.py` under `python/orinfer_build/quantization/`.
