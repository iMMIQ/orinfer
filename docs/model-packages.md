# Model Execution Packages & Device Backends

Orinfer separates **model data** (config, tokenizer, weights) from **precompiled execution packages** (compiled device kernels, ABI definitions, and execution bindings). Both are linked by an immutable SHA256 digest (`execution_package`) in `cache/model.json`.

---

## 1. Package Formats

| Backend | Package Schema | Model Adapter | Device Assets |
| --- | --- | --- | --- |
| **CUDA SM87** | **Schema 2** | Statically linked Rust adapter in `orinfer-models` | `package.json`, `bindings.json`, `assets.safetensors` (deduplicated cubin, source, host ABI) |
| **Ascend 910B / A2** | **Schema 1** (native plan) | Built-in native provider in `orinfer-engine` | Native plan JSON (`model.json` / `shared-json-v1`), TileLang-Ascend `.so` AOT libraries, optional compact RoPE recipes |

### Model Directory Layout (CUDA Schema 2)

```text
MODEL_DIR/
├── config.json, tokenizer.json, chat_template.jinja, generation_config.json
└── cache/
    ├── model.json                  # Schema 1 data manifest + execution_package SHA256
    ├── weights/*.safetensors       # Quantized weights (standard safetensors shards)
    ├── cpu/                        # Optional CPU lookup tables (e.g. Flash Next PLE / W8 embed)
    └── packages/<sha256>/          # Optional bundled execution package (fallback)
        ├── package.json            # Schema 2 manifest, config signature, buffer contracts, catalog
        ├── bindings.json           # Factorized tensor roles, layer params, specializations
        ├── assets.safetensors      # Named U8 tensors for cubin, generated source, host ABI
        └── LICENSE
```

- **Catalog & Bindings**: `package.json` stores the shared compilation catalog (cubin, host ABI, block/shared/cooperative metadata, dynamic row launch templates). `bindings.json` stores factorized per-layer tensor roles and row-selection tables.
- **Validation Split**:
  - **Offline (`plan-model`, `validate-model`, `package install`)**: Exhaustively verifies dynamic row domains, graph branches, bindings, and full SHA256 hashes.
  - **Online (`serve`, `batch`)**: Verifies config signature, kernel ABI, tensor dtype/shape/layout, and container bounds—skipping payload SHA256 scans unless `--verify-weights` is passed.
- **Address-Table Packet Graphs**: Qwen3.5, Flash Next, and ASR packages bind request-private operands through fixed address tables (`mtp.packet_table` / `packet_graph`), allowing captured CUDA Graphs to be reused across changing owners and requests by updating only the address table.

---

## 2. Installing, Validating & Compaction

### Install an Execution Package

```bash
sha256sum --ignore-missing -c SHA256SUMS
orinfer package install /path/to/execution-package.tar.xz
orinfer plan-model /path/to/prepared-model
orinfer validate-model /path/to/prepared-model
```

- **Cache path**: Resolves `$ORINFER_EXECUTION_CACHE`, then `$XDG_CACHE_HOME/orinfer/packages`, then `~/.cache/orinfer/packages`. Pass `--cache DIR` to install to a custom directory (set `ORINFER_EXECUTION_CACHE` to the same path at runtime).
- **Archive formats**: Detects `.tar.xz` and `.tar.gz` by magic bytes; rejects path traversal, symlinks, and duplicate entries; installs atomically.

### Offline Packaging Utilities

| Task | Command |
| --- | --- |
| Archive a package directory as `.tar.xz` | `uv run --locked python -m orinfer_build.packaging.package archive PKG_DIR OUT.tar.xz` |
| Re-pin modified frontend assets (`tokenizer.json`, template) | `uv run --locked python -m orinfer_build.packaging.package pin-assets MODEL_DIR` |
| Attach a replacement Schema 2 package to model data | `uv run --locked python -m orinfer_build.packaging.attach_execution --model MODEL --package PKG_DIR --output NEW_MODEL` |
| Rebuild catalog & bindings (hardlinking immutable weights) | `uv run --locked python -m orinfer_build.contracts.catalog --model MODEL --output NEW_MODEL` |
| Compact GPU shards (~2 GiB) & split execution package to cache | `uv run --locked python -m orinfer_build.packaging.compact --model MODEL --output COMPACT --package-cache ~/.cache/orinfer/packages` |
| Repack Flash Next into canonical release shards | `uv run --locked python -m orinfer_build.models.flash_next.repack --model MODEL --output PUBLISHED` |

---

## 3. Device Backend Boundary

`orinfer-device` defines backend-agnostic device, target ABI, graph policy, logical memory interval, and session ownership traits. Concrete backends live in `crates/orinfer-engine/src/backend/{cuda,ascend}/`.

- **Targets**: `sm_87` (`aarch64` CUDA), `ascend910b` (`aarch64` Ascend), `ascend910b-x86_64` (`x86_64` Ascend).
- **Tiered Prefix State**: Logical radix indexing, physical bank deduplication, copy-on-write, and device↔CPU eviction are shared across CUDA and Ascend. Snapshots tag backend, device, and session generation to prevent cross-session pointer reuse.

### Inspecting & Validating Backends

```bash
# Inspect available devices
orinfer devices --backend cuda
orinfer devices --backend ascend

# Ascend 910B bounded native checks (CANN 9.2 environment required)
orinfer validate-ascend --device 0
HCCL_NPU_SOCKET_PORT_RANGE=auto orinfer validate-hccl --devices 0,1,2,3,4,5,6,7
orinfer validate-engram --metadata artifacts/engram-metadata.json --model-dir /path/to/dsv41-checkpoint
```

- **`validate-ascend`**: Verifies async DMA, ACLNN `aclnnGroupedMatmulV5` W4A8 arithmetic, empty expert groups, and changed-input ACL Graph replay (512 MiB budget).
- **`validate-hccl`**: Verifies multi-rank AllReduce, AllGather, ReduceScatter, and uneven AlltoAllV on 2–8 NPUs with changed-input Graph replay.
- **CANN HCOMM Graph Registry**: Automatically loads an in-memory per-device HCOMM 9.2.0 callback registry before CANN initialization so multi-rank single-process workers can independently evict and rebuild ACL Graphs (`glibc >= 2.34`).

For the full Ascend 910B / DSv4.1 offline build, prepacking, DSpark, and vision pipeline, see **[Ascend 910B Offline Build Pipeline](ascend-a2-build.md)**. For production serving on A2, see **[DSv4.1 Production Guide](deepseek-v41-production.md)**.

---

## 4. Verification

```bash
# CPU workspace checks (package validation, binding expansion, install atomicity)
make check

# Standalone CUDA package-loader GPU fixture (off / decode_only / full graph modes)
bash tools/environments/cuda/run.sh \
  python/orinfer_build/validation/validate_model_package_gpu.py artifacts/model-package/check01
```
