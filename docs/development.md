# Development Guide

Orinfer splits cleanly into a **Rust workspace** for online execution/serving and a **Python + TileLang offline toolchain** for quantization, kernel compilation, tuning, and validation.

---

## 1. Repository Layout

### Rust Workspace (`crates/`)

| Crate | Responsibility |
| --- | --- |
| `orinfer-cli` | CLI binary (`serve`, `batch`, `package`, `validate-model`, `plan-model`, manager & diagnostic commands) |
| `orinfer-api` | OpenAI / Ollama HTTP & SSE server, request preprocessing, JSONL batch runner, multi-model manager |
| `orinfer-engine` | Core runtime: continuous scheduler, tiered prefix cache, adaptive MTP/DSpark, CUDA & Ascend backends, native DSv4.1 provider |
| `orinfer-device` | Driver-free device, target ABI, graph policy, and memory interval traits |
| `orinfer-model-sdk` | Shared Rust plan, state, packet, and Schema 2 package data contracts ([README](../crates/orinfer-model-sdk/README.md)) |
| `orinfer-models` | Statically linked CUDA model adapters (Qwen3.5, Qwen3-ASR, Flash Next, Nemotron Embed, Qwen Reranker) ([README](../crates/orinfer-models/README.md)) |

### Offline Python, Kernels & Tests

| Directory | Responsibility |
| --- | --- |
| `kernels/cuda` | Production CUDA SM87 TileLang kernels (`operators`, `model`, `audio`, `vision`) |
| `kernels/ascend` | Production Ascend 910B TileLang kernels + third-party licenses |
| `kernels/shared` | Shared dynamic-row contracts used by both backends |
| `python/orinfer_build/contracts` | Geometry, layout, host ABI, and binding contracts |
| `python/orinfer_build/compiler` | AOT compilation, native binding generation, packet graphs, content cache |
| `python/orinfer_build/models` | Model-specific offline builders & pipelines |
| `python/orinfer_build/tuning` | Candidate selection & Graph benchmarking |
| `python/orinfer_build/packaging` | Frontend pinning, execution package archiving, shard compaction |
| `python/orinfer_build/quantization` | E8P Q2 / Q4 encoders & activation calibration helpers |
| `python/orinfer_build/weights` | Shared `pread` / safetensors weight readers |
| `python/orinfer_build/common` | Paths, file identity, shared GPU locks, test discovery |
| `tests/cuda`, `tests/ascend`, `tests/support` | On-device operator & model validation suites |
| `tools/` | Benchmarks, API/eval harnesses, release packaging, environment wrappers, workspace CLI |

Invoke offline modules via `python -m orinfer_build...` or `python -m tests...`.

---

## 2. Worktree & Data Layout

Keep development worktrees outside the main clone, and separate mutable builds from shared caches, experiment runs, and immutable releases:

```bash
git worktree add /data/worktrees/orinfer/my-task -b dev/my-task
cd /data/worktrees/orinfer/my-task
make python-env
uv run --locked python -m tools.dev.workspace init --work-root /data/orinfer --name my-task
uv run --locked python -m tools.dev.workspace inventory
```

- `workspace init` creates ignored `artifacts/layout.json` and symlinks (`builds` per worktree, shared `cache`, `runs`, `releases`, `locks`, and per-worktree `target`). It never overwrites existing data.
- Override locations via `ORINFER_WORK_ROOT` and `ORINFER_GPU_LOCK`. All GPU/NPU jobs serialize on the shared lock (`artifacts/gpu-experiment.lock`); do not kill unrelated services or alter host clock/power settings.

---

## 3. Toolchains & 7-Day Dependency Maturity Audit

- **Rust**: Pinned to stable **1.98.1** in `rust-toolchain.toml` and `Cargo.toml`.
- **Python**: Managed by `uv` (`.python-version`):
  - CPU env: root `pyproject.toml` + `uv.lock` (`make python-env`).
  - CUDA offline env: `tools/environments/cuda/pyproject.toml` + `uv.lock` (`make cuda-env`).
  - Ascend offline env: CANN 9.2 container wrapper `tools/environments/ascend/run.sh`.
- **7-Day Maturity Policy**: Both `uv` projects enforce `exclude-newer = "7 days"` and `prerelease = "disallow"`. Cargo lockfile and wheel/CI release dates are audited against `configs/dependency-releases.json`.

When updating dependencies:

```bash
uv lock --upgrade
uv lock --upgrade --project tools/environments/cuda
python3 -m tools.dev.dependencies refresh
make python-env
cargo fetch --locked
make check
```

If a transitive Cargo crate is younger than 7 days, pin an older release with `cargo update -p NAME --precise VERSION` and re-run `refresh`. Normal `make` targets verify the audit offline without network access.

---

## 4. Standard Checks & Tests

| Command | What It Runs |
| --- | --- |
| `make fmt` | `cargo fmt` + `ruff format` |
| `make lint` | Rust fmt/clippy (`-D warnings`) + Ruff format/check |
| `make check` | Dependency age & lock check, lint, `cargo check`, `cargo test`, CPU Python tests (`test_*.py`), benchmark plan check |
| `make check-offline` | CUDA offline import checks & CPU-side Torch/ABI tests (`TEST_ENVIRONMENT = "cuda"`) |
| `make check-ascend-offline` | Device-free Ascend import checks inside the CANN container |
| `make check-gpu MODEL=... OUTPUT=...` | Real-GPU end-to-end check (off/decode_only/full graph modes, dynamic batch, isolation, prefix, vision) |

Offline compile caches hash production component sources, inlined kernel macros, compiler versions, and contracts—unrelated tests or other model directories do not invalidate cached artifacts.
