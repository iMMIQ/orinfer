# Engineering Conventions (`AGENTS.md`)

These conventions apply across the repository. Explicit user instructions take precedence.

---

## 1. Scope & Architecture

- **Primary Target**: Jetson AGX Orin 64GB (`aarch64` / CUDA SM87).
- **Secondary Target**: Ascend 910B / A2 (CANN 9.2) via the shared backend-agnostic runtime (native ACL resources, async transfers, ACLNN W4A8 grouped matmul, TileLang-Ascend AOT pointer-table C ABI, ACL Graph, HCCL collectives). DSv4.1 Flash-1017 W4A8 (40 layers, TP8) runs through an experimental native provider integrated with HTTP/SSE, continuous batching, DSpark, and prefix cache while retaining native bench entrypoints.
- **Online vs Offline Split**:
  - **Rust** handles online loading, scheduling, APIs (`orinfer-api`), CLI (`orinfer-cli`), device execution (`orinfer-engine`, `orinfer-device`), and statically linked model adapters (`orinfer-models`, `orinfer-model-sdk`).
  - **TileLang** (`kernels/cuda`, `kernels/ascend`, `kernels/shared`) is the sole source of production GPU/NPU kernels.
  - **Python** (`python/orinfer_build`, `tools/`, `tests/`) is strictly offline: quantization, compilation, tuning, reference checks, and validation.
- **Execution Packages**:
  - **CUDA**: Schema 2 (`package.json`, `bindings.json`, `assets.safetensors` with deduplicated cubin/source/ABI). Dynamic launch exhaustive row-domain checks run offline (`plan-model`, `validate-model`); online loading validates config signature, ABI, dtype/shape/layout, and bounds (skipping content SHA256 unless `--verify-weights` is set).
  - **Ascend A2**: Schema 1 native plan + AOT libraries. Offline `prepare` writes physical W4/scale/assist banks; online loading reads physical layouts directly without transposing or payload hashing.
  - Distribution archives use `.tar.xz` (installer also accepts `.tar.gz`).
- **License**: LGPL-3.0-or-later. Preserve all third-party copyrights and notices in `THIRD_PARTY_NOTICES.md`.

---

## 2. Models, Precision & State Contracts

- **Checkpoint Fidelity**: Always verify GDN, GQA/QSA, RoPE/MRoPE, zero-centered RMSNorm, embedding/head tying, and added-token IDs against real checkpoint configs and Transformers outputs.
- **Request Isolation & State**:
  - GDN recurrent state is **FP32**. KV cache is group-64 **INT8** + FP16 scales.
  - KV, convolution, GDN, QSA pending/index, PLE n-gram/conv history, audio features, positions, and MTP hidden rings are **request-private state** and must be preserved across chunking, prefix restoration, cancellation, and owner reuse.
  - Any change to prefill/decode, restoration, or branching must validate all request-private states.
- **Single-Resident Quantized Weights**:
  - Keep a single resident compressed bank (W4/Q2/Q4). Scales, zeros, LUTs, padding, and persistent duplicates count toward average bits/weight. Temporary W8 expansion is restricted to compute tiles.
  - **Qwen3.5 / Index-Translate-2B/9B/35B / Qwen3.5-4B / Qwen3-ASR-1.7B / Nemotron-Embed-1B / Qwen3-Reranker-0.6B**: Two-stage E8P Q4A8 (`coefficients=[4,1]`, signed-block128 rotation, beam8) with W8/FP16 sensitive exceptions.
  - **Qwen3.8-Flash-Next**: E8P Q2A8 experts + Q4A8/W8 projections, BF16 HC, bounded CPU PLE cache. `--resident-experts N` pages experts via CUDA VMM + global SLRU (CUDA Graph off in bounded mode; MTP experts stay resident).
  - **27B Baseline**: `artifacts/models/qwen3.8-27b-e8p-q4a8-current` (quantized from `artifacts/quantization/qwen35-e8p-beam8-20261016/converted`).
- **Dynamic Row Execution & Graphs**:
  - Reuse shared `Rows`, address-table packet boundaries (`independent_rows`, `packet_graph`), and dynamic cubin/AOT templates across ordinary decode, mixed prefill/decode, verification, draft, and warm packets.
  - CUDA/ACL Graphs require stable virtual addresses and pre-allocated workspaces; always validate changed-input replay.
  - Cross-block sync requires explicit `cooperative` launch with occupancy checks.
  - A2 multi-rank Graph capture probes all ranks first; on any rank miss, invalidate and recapture the same shape across all ranks to keep HCCL sequences identical. Default communication uses high-performance HCCL; `HCCL_DETERMINISTIC=strict` is an explicit reproducibility mode.
- **Shared Projection Cores**: Compose projections from `kernels/cuda/model/{integer_core,floating_core,e8p_core}.py`. Inline component source identities must be included in tuning/compile cache keys.
- **Pinned Transfers & Memory Accounting**: Host inputs upload asynchronously in stream order via bounded pinned slabs; caller buffers may be freed on return; slab reuse and teardown must await DMA completion. Weight loading pipelines parallel reads with async DMA; read-only weight arenas keep per-tensor alignment and single-owner release.

---

## 3. Quality, Performance & Verification Gates

- **Quality Reference**: Compare against official BF16 (or clearly labeled FP8) using fixed `seed=20261002` and identical teacher-forced token histories. Judge quality by task outcomes, target NLL, token probabilities, and long-context behavior (top-1 agreement is diagnostic only).
- **Performance Metric**: Accept performance only on **real-model complete-request TPS** (and Mini-Weka end-to-end replay for DSv4.1 A2). Synthetic probes, microbenchmarks, and profiles are for bottleneck localization only.
- **Staged Iteration Discipline**:
  1. **Candidate stage**: Run only relevant operator numerical + changed-input Graph replay checks, 2K/2049, one representative batch, and a fixed-history decode short check. Reuse passed validations when contracts have not changed.
  2. **Confirmed gain**: Run one 240s Mini benchmark against the frozen baseline; add paired re-runs only on noise or suspected >10% regression.
  3. **Pre-commit**: Run `make check` once. Scope state/prefix/isolation checks to actually modified contracts—do not repeat full architectural acceptance on every tile candidate.
- **GPU Test Binaries**: Always resolve and execute the actual test binary built in the current round and verify tests actually ran.

---

## 4. Development Workflow & Hygiene

- **Code Search First**: Use `rg` to inspect callers and dependencies before editing; keep changes small and complete. Only spawn subagents when explicitly requested.
- **7-Day Dependency Maturity Policy**: Both uv projects (`pyproject.toml`, `tools/environments/cuda/pyproject.toml`) keep `exclude-newer = "7 days"`. After updating Cargo/uv locks, CI, or wheels, run `python3 -m tools.dev.dependencies refresh` to update `configs/dependency-releases.json` and pass the audit before fetching/building.
- **Environments & Commands**:
  - CPU Python: `make python-env` (`uv run --locked`).
  - CUDA offline: `make cuda-env` and `bash tools/environments/cuda/run.sh` (uses host CUDA toolchain + isolated uv env).
  - Ascend offline: `bash tools/environments/ascend/run.sh` (`--without-devices` for CPU-only compilation).
  - Standard checks: `make check` (Rust fmt/clippy/test + CPU Python tests), `make check-offline` (CUDA offline), `make check-ascend-offline`.
- **Worktrees & Locks**: Keep dev worktrees outside the repo and initialize via `python -m tools.dev.workspace init`. Serialize GPU/NPU work with the shared lock (`artifacts/gpu-experiment.lock` / `common.paths`). Do not kill unrelated services or alter global clocks, power modes, or caches.
- **Repository Cleanliness**: Keep source, build tools, tests, examples, and user/architecture docs in Git. Keep research logs, experiment reports, weights, binaries, cubins, caches, and traces in ignored `artifacts/` or `target/`. Secrets/tokens live outside the repo.
- **Documentation Style**: State capabilities, usage, preconditions, and validation scope directly in clear affirmative sentences; describe limitations by exact supported bounds and error behavior.
