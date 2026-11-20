# Offline Model Builders

Offline weight preparation, TileLang compilation, tile tuning, and Schema 2 / Ascend package assembly live in `python/orinfer_build/` and are invoked via `python -m orinfer_build...`.

## Supported Model Builders

| Model | Input | Builder & Guide |
| --- | --- | --- |
| **Qwen3.8-27B / Qwen3.5-4B / Index-Translate-2B, 9B, 35B** | E8P Q4A8 quantized checkpoint | `python/orinfer_build/models/qwen35/build.py` ([Qwen3.5 Guide](qwen35/README.md)) |
| **Qwen3-ASR-1.7B** | Text E8P Q4A8 + FP16 audio checkpoint | `qwen35/build.py --context 65536` ([Audio Guide](../../docs/audio.md)) |
| **Nemotron-3.5-Embed-1B** | Text + Vision E8P Q4A8 checkpoint | `python/orinfer_build/models/nemotron_embed/build.py` ([Embeddings Guide](../../docs/embeddings.md)) |
| **Qwen3-Reranker-0.6B** | Causal E8P Q4A8 + FP16 yes/no head | `python/orinfer_build/models/qwen_reranker/build.py` ([Reranking Guide](../../docs/reranking.md)) |
| **Qwen3.8-Flash-Next** | Native Flash Next model + package | `python/orinfer_build/models/flash_next/build.py` ([Flash Next Guide](flash_next/README.md)) |
| **DSv4.1 Flash-1017 (Ascend A2)** | Native W4A8 checkpoint + CANN 9.2 | `orinfer_build.models.deepseek_v41.build` ([Ascend A2 Guide](../../docs/ascend-a2-build.md)) |

## Quick Examples (CUDA SM87)

```bash
make python-env && make build

# Build Qwen3.5 / Index-Translate Schema 2 package
bash tools/environments/cuda/run.sh python/orinfer_build/models/qwen35/build.py \
  artifacts/qwen35-build --source /path/to/converted \
  --model-output artifacts/models/qwen35-ready --gpu-lock-held

# Rebuild Flash Next integer projections & Schema 2 package
bash tools/environments/cuda/run.sh python/orinfer_build/models/flash_next/build.py \
  artifacts/flash-build --model /path/to/native-flash-model \
  --model-output artifacts/models/flash-ready
```

## Selective Upgrade Entrypoints (Existing Packages)

- **Batched GPU Sampling** (`sampling.py`): Rebuilds single-row + 1–128 row address-table GPU sampling programs while reusing existing weights:
  ```bash
  bash tools/environments/cuda/run.sh python/orinfer_build/models/sampling.py \
    artifacts/sampling-build --model /path/to/source-model --model-output /path/to/upgraded-model
  ```
- **Packet Address-Table Graphs** (`packet_graph.py`): Extracts private operands via `plan-model --packet-probes` and wraps decode/mixed/MTP packets into fixed address tables:
  ```bash
  bash tools/environments/cuda/run.sh python/orinfer_build/compiler/packet_graph.py \
    artifacts/packet-build --model /path/to/prepared-model --target /path/to/packet-ready
  ```
- **Auxiliary Kernel Tuning** (`orinfer_build.tuning.auxiliary_tuning`): Tunes MoE top-k router (`--families router`), QSA selection/prefill (`--families qsa-prefill`), and FP16 PLE norm/gate fusion.

## Verification & Profiling

```bash
./target/release/orinfer plan-model /path/to/prepared-model
./target/release/orinfer validate-model /path/to/prepared-model
./target/release/orinfer run-model /path/to/prepared-model examples/requests.json
make check && make check-offline
make check-gpu MODEL=/path/to/prepared-model OUTPUT=artifacts/check-gpu/run1
```

Use `tools/model/run.sh` for logged CLI runs with machine telemetry and `tools/model/profile.sh` for Nsight Systems profiling.
