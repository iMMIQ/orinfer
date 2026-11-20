# Vision & Multi-Image Validation Tools

Qwen3.5 and Flash Next share the dynamic-patch ViT encoder in `kernels/cuda/vision/encoder.py` (default up to 32,768 patches / 8,192 visual tokens per image; total history capacity can be upgraded via `vision_capacity.py`). Online image preprocessing, MRoPE positioning, and per-request feature storage run in Rust.

---

## Kernel & Preprocessing Verification

```bash
# ViT kernels (FP16/BF16, K/N tails, changed-input Graph replay)
bash tools/environments/cuda/run.sh tools/vision/validate_kernels.py artifacts/vision/kernels-check

# MRoPE & feature injection Graph replay
bash tools/environments/cuda/run.sh tools/vision/validate_mrope.py artifacts/vision/mrope-check

# Image preprocessing parity against Transformers Qwen3VLProcessor
bash tools/environments/cuda/run.sh tools/vision/preprocess_reference.py \
  artifacts/vision/preprocessor-check --model /path/to/prepared-model
ORINFER_IMAGE_REFERENCE="$PWD/artifacts/vision/preprocessor-check" \
  cargo test --offline checkpoint_image_processor_reference -- --ignored --nocapture
```

Use `official_reference.py` to export independent BF16/FP16 reference features from Transformers, and `make check-gpu` for end-to-end prefix/isolation checks with images.

---

## Live API Acceptance

Run against a running vision-enabled `orinfer serve` instance (`--base-url` and `--model` supported):

```bash
python3 tools/vision/smoke.py --output artifacts/vision/smoke.json
python3 tools/vision/scenarios.py --output artifacts/vision/scenarios.json
python3 tools/vision/limits.py --manifest /path/to/prepared-model --output artifacts/vision/limits.json
```

Covers single-image, ordered multi-image, SSE, image history, tool calling with images, capacity-limit error recovery, and concurrent queuing (`seed=20261002`).
