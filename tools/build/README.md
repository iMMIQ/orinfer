# Build, Checks & Release

Online inference uses the Rust binary, device drivers, prepared model directories, and precompiled execution packages. Python, PyTorch, and TileLang are strictly offline.

---

## 1. Python Environments (`uv`)

| Environment | Manifest & Lockfile | Purpose |
| --- | --- | --- |
| **CPU** | Root `pyproject.toml`, `uv.lock` | `orinfer_build` (editable), NumPy, Safetensors, Tokenizers, Pillow, Ruff |
| **CUDA Offline** | `tools/environments/cuda/pyproject.toml`, `uv.lock` | Adds Jetson SM87 PyTorch/TorchVision wheels, TileLang, CUDA Python, Transformers |
| **Ascend Offline** | `tools/environments/ascend/run.sh` | CANN 9.2 + pinned TileLang-Ascend container |

```bash
make python-env
make cuda-env
uv run --locked python -m orinfer_build.contracts.catalog --help
```

All dependencies follow the **7-day release maturity policy** (`exclude-newer = "7 days"` in uv; audited in `configs/dependency-releases.json`). See [Development Guide](../../docs/development.md#3-toolchains--7-day-dependency-maturity-audit) for the upgrade workflow.

---

## 2. Host CUDA 12.6 Toolchain

CUDA builds run directly against the host Jetson AGX Orin CUDA 12.6 toolchain (`nvcc`, Nsight, Compute Sanitizer) and isolated CUDA uv environment via `tools/environments/cuda/run.sh`, which syncs locked dependencies and serializes on `artifacts/gpu-experiment.lock`.

Jetson PyTorch/TorchVision wheels are built from [iMMIQ/pytorch](https://github.com/iMMIQ/pytorch/tree/jetson-r36.4-cu126-py312-v2.14/jetson) and pinned with SHA256 hashes in `tools/build/jetson_wheels.json`.

Run the minimal compiler smoke test on a fresh host:

```bash
bash tools/environments/cuda/run.sh \
  python/orinfer_build/common/smoke.py artifacts/build/compiler-smoke
```

---

## 3. Automated Checks

```bash
make fmt                    # Format Rust + Python
make lint                   # Check Rust fmt/clippy + Ruff
make check                  # Full CPU CI: dependency age, locks, lint, cargo test, CPU Python tests
make check-offline          # CUDA offline imports + CPU Torch/ABI tests
make check-ascend-offline   # Device-free Ascend import checks
make check-gpu MODEL=/path/to/model OUTPUT=artifacts/check-gpu/run1
```

---

## 4. Release Packaging

Ensure the Git working tree is clean, workspace versions are updated, and an annotated tag points to `HEAD`:

```bash
uv run --locked python -m tools.release.package \
  --version v0.4.3 \
  --operators /path/to/qwen35-execution.tar.xz \
  --operators /path/to/flash-next-execution.tar.xz \
  --output artifacts/releases/v0.4.3
```

- Verifies `aarch64` ELF target, binary `--version`, clean Git revision, and every execution package's ABI/digest/architecture contract (`configs/architecture-contract.json`).
- Produces `.tar.xz` (preset 6) archives for the engine, execution packages, and source tarball plus `SHA256SUMS` and `RELEASE.json`.
- For Ascend A2, `--operators` also accepts a digest directory published by `orinfer_build.models.deepseek_v41.publication` (including compact RoPE `generated_assets` + `shared-json-v1` plans created via `--compact-rope` or `orinfer_build.models.deepseek_v41.compact`).
- Pass `--publish --notes /path/to/release-notes.md` to push the tag, create the GitHub Release, and re-download/verify uploaded asset SHA256 checksums.
