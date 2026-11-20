# Quality Evaluation Tools

Evaluate quantization and runtime quality against official **BF16** (or clearly labeled **FP8**) references using fixed `seed=20261002` and identical teacher-forced token histories. Judge quality by **task results, target-token NLL, full-vocabulary probabilities, and long-context behavior** (top-1 agreement is diagnostic only).

---

## Offline BF16 Teacher-Forced Comparison

`reference_bf16.py` streams official BF16 safetensors layer-by-layer (supporting local directories or `--remote-repo REPO --revision 40_CHAR_SHA` via HTTP Range reads), while `orinfer score-model` scores the exact same frozen histories through real prefill/decode kernels:

```bash
# 1. Prepare frozen prompts & reference histories
python3 -m tools.eval.offline_quality prepare \
  --checkpoint MODEL_DIR \
  --output artifacts/quality/requests.json

# 2. Score with official BF16 reference
flock artifacts/gpu-experiment.lock \
  python3 -m tools.eval.reference_bf16 \
  --checkpoint BF16_DIR --revision SOURCE_COMMIT \
  --requests artifacts/quality/requests.json \
  --output artifacts/quality/bf16.json

# 3. Extract top-3 probability query list
python3 -m tools.eval.offline_quality queries \
  --requests artifacts/quality/requests.json \
  --baseline artifacts/quality/bf16.json \
  --output artifacts/quality/queries.json

# 4. Score candidate model with Rust CLI
flock artifacts/gpu-experiment.lock \
  target/release/orinfer score-model MODEL_DIR artifacts/quality/queries.json \
  > artifacts/quality/candidate.json

# 5. Compare NLL, KL, top-1/top-3 agreement
python3 -m tools.eval.offline_quality compare \
  --requests artifacts/quality/queries.json \
  --baseline artifacts/quality/bf16.json \
  --candidate artifacts/quality/candidate.json \
  --output artifacts/quality/comparison.json
```

### Options & Scenarios
- **Translation fixtures**: Pass `--fixtures fixtures/translation-quality-scenarios.json` to `prepare`.
- **Long-context needle retrieval**: Pass `--long-context-tokens 512 2048 8192` to `prepare`.
- **Vision scoring**: Cases may include `images` (`grid_height`, `grid_width`, normalized FP32 `pixels`); Transformers reference explicitly passes `mm_token_type_ids` so MRoPE matches Orinfer.
- **CPU unit tests**: `python3 -m unittest discover -s tools/eval -p 'test_*.py' -v`.
