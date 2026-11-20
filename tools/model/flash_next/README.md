# Flash Next Offline Build

Offline compilation, tile tuning, and packaging for **Qwen3.8-Flash-Next** on CUDA SM87. Published packages use **Schema 2** (`package.json`, `bindings.json`, `assets.safetensors`) with the statically linked `orinfer-models` adapter.

---

## 1. Rebuild & Tune Native Model

```bash
make build
bash tools/environments/cuda/run.sh python/orinfer_build/models/flash_next/build.py \
  artifacts/flash-build \
  --model /path/to/native-model \
  --model-output artifacts/models/flash-ready
```

- **What it tunes**: Reads Q4, W8, and fused SwiGLU geometries from the input package, measures real-weight candidates for exact AOT row sizes (`1/2/3/4/8/16/32`) and dynamic row fallbacks (default prefill tuning rows `512/2048/4096`, override via `--prefill-tuning-rows`), verifies unaligned tails + changed-input Graph replay, and publishes only selected cubins.
- **Resume on interruption**: Add `--resume` before the script path (`bash tools/environments/cuda/run.sh --resume ...`); validated compile objects in `cache/build/flash-next` are reused.
- **Selective rebuild flags**:
  - `--prefill-only`: Rebuild only prefill projections.
  - `--projection-kinds w8 --projection-shape 2560 640`: Limit rebuild to specific kinds/geometries.
  - `--hc-combine-norm`: Fuse adjacent decode HC combine + norm pairs.
  - `--plan-only`: Inspect build contracts on CPU without GPU execution.

---

## 2. Precision & Execution Contracts

- **Weights**: Single-resident E8P Q2A8 experts (dual INT4 MMA bit-plane decomposition), rotated Q4A8 for GDN QKV/Z, QSA Q/gate, and shared target/MTP head; W8 for GDN/QSA output projections, shared FFN, and draft Q/O; BF16 HC; bounded CPU PLE cache.
- **Dynamic Row Capacities**: Prefill `1–4096` rows, shared target `1–1024` rows, draft/warm `1–512` rows, MTP verification `2–8` rows.
- **Request-Private State**: Group-64 INT8 QSA KV + pending ring + compressed index, FP32 GDN state, conv/PLE history, positions, and MTP hidden ring.

---

## 3. Quantizing from Official BF16

Raw BF16 weights are streamed per matrix/expert with a pinned source revision:

```bash
bash tools/quantization/run_flash_next.sh \
  --index /path/to/original/model.safetensors.index.json \
  --revision SOURCE_COMMIT_SHA --stage all \
  --output artifacts/quantization/flash-next/converted

uv run --locked --project tools/environments/cuda python -m orinfer_build.models.flash_next.weights \
  --converted artifacts/quantization/flash-next/converted \
  --index /path/to/original/model.safetensors.index.json \
  --config /path/to/original/config.json --frontend /path/to/original \
  --output artifacts/models/flash-next-quantized
```

---

## 4. Verification & Serving

- **Serving**: `orinfer serve artifacts/models/flash-ready --model qwen-flash-next --cuda-graph decode_only --mtp-drafts auto --prefix-cache-mib 4096` (see [Serving Guide](../../../docs/serving.md)).
- **Checks**: Operator & replay tests live in `tests/cuda/operators/` and `python/orinfer_build/models/flash_next/checks/`; BF16 reference math in `python/orinfer_build/models/flash_next/reference/`. Always validate quality (`seed=20261002`), state isolation, and complete-request TPS after rebuilding.
