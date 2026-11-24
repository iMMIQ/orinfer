# Generic Non-Orin CUDA Hardware Support

Orinfer's primary edge target remains **Jetson AGX Orin 64GB (`sm_87`, `aarch64`)** with its unified-memory execution path and SM87-tuned kernels. In addition, Orinfer provides a portable **generic CUDA hardware path** for discrete-VRAM NVIDIA GPUs across **Ampere, Ada, Hopper, and Blackwell** (`sm_80`, `sm_86`, `sm_89`, `sm_90`, `sm_100`, `sm_103`, `sm_120`) on `x86_64` and `aarch64` hosts.

---

## 1. Design Philosophy

Rather than maintaining hand-tuned PTX micro-kernels per GPU generation (such as SM87 dual-INT4 `mma.s4` or architecture-specific shared-memory bank swizzles that regress or break on Hopper/Blackwell), the generic CUDA path relies exclusively on **portable high-level TileLang primitives** (`T.copy`, `T.gemm`, `T.Pipelined`, `T.Parallel`, `T.reduce_max`):

- **Zero impact on Jetson AGX Orin (`sm_87`)**: When running on SM87 (`is_orin_arch() == True`), Orinfer retains all existing SM87 target contracts, kernels, tuning tables, and unified-memory policies unchanged.
- **Bit-exact checkpoint compatibility**: Generic CUDA kernels consume the exact same `orinfer_e8p_int8` (`orinfer.flash_next.weights.v1`) physical layouts (`P: [E, K//128, N, 16]` / `packed_k=64` U16 banks, `integer-e8p-spread29-v1` INT8 basis, `int8-row` dense W8 weights, and FP32 GDN state) with zero persistent FP16 weight expansion.
- **Native Tensor Core lowering across architectures**: On-chip E8P U16 decoding feeds portable `int8 x int8 -> int32` `T.gemm`, which TileLang lowers to native INT8 Tensor Core instructions on Ampere, Ada, Hopper, and Blackwell while matching Orinfer's exact integer dot-product and FP32 rescaling order.
- **Discrete-VRAM awareness**: The Rust CUDA executor queries `CU_DEVICE_ATTRIBUTE_INTEGRATED`. Unified-memory Tegra devices keep the existing reclaimable-page-cache allocation policy, while discrete-VRAM GPUs validate free VRAM directly against `cuMemGetInfo_v2`.

---

## 2. Supported Targets & Contracts

| Package Target String | Backend | Device Arch | Host Arch | Typical Hardware |
| --- | --- | --- | --- | --- |
| `sm_87` | `cuda` | `sm_87` | `aarch64` | Jetson AGX Orin 64GB (legacy/primary path) |
| `sm_80` | `cuda` | `sm_80` | `x86_64` | NVIDIA A100 40GB / 80GB |
| `sm_86` | `cuda` | `sm_86` | `x86_64` | RTX 3090 / A40 / A10 |
| `sm_89` | `cuda` | `sm_89` | `x86_64` | RTX 4090 / L4 / L40S |
| `sm_90`, `sm_90-aarch64` | `cuda` | `sm_90` | `x86_64` / `aarch64` | NVIDIA H100 / H200 / GH200 |
| `sm_100`, `sm_100-aarch64` | `cuda` | `sm_100` | `x86_64` / `aarch64` | NVIDIA B200 / GB200 |
| `sm_103`, `sm_120` | `cuda` | `sm_103` / `sm_120` | `x86_64` | Blackwell Ultra / RTX 50 series |

Supported host-qualified forms `<sm>-x86_64` and `<sm>-aarch64` are accepted by `orinfer_model_sdk::device::Target::from_package` and `orinfer_build.compiler.cuda.target.package_target`. Unsupported capabilities (such as SM7x) are rejected at both compile time and runtime.

---

## 3. Portable TileLang Operator Suite (`kernels/cuda/generic/`)

| Module | Operators | Highlights |
| --- | --- | --- |
| `kernels/cuda/generic/e8p_gemm.py` | `e8p_grouped_gemm`, `e8p_dense_gemm` | Direct compressed Orinfer U16 (`packed_k=128` and `packed_k=64`) -> on-chip INT8 decode -> portable `T.gemm(int8 -> int32)` |
| `kernels/cuda/generic/rotation_a8.py` | `rotate_activation_a8`, `plain_activation_a8` | Pure TileLang shared-memory 7-stage block-128 Hadamard + optional SwiGLU + exact FP16 materialization boundary + row-A8 |
| `kernels/cuda/generic/w8a8_gemm.py` | `w8a8_gemm` | Portable dense INT8 Tensor Core GEMM with exact `(float32(acc) * WS) * AS` rescaling |
| `kernels/cuda/generic/gdn_recurrent.py` | `gdn_recurrent` | Single-launch FP32 Gated DeltaNet recurrence keeping state columns resident in on-chip registers for both prefill and stateful `L=1` decode |

---

## 4. Usage & Verification

### Target Selection

By default, `orinfer_build.compiler.cuda.target.cuda_arch()` auto-detects the live GPU capability (`torch.cuda.get_device_capability()`). Override explicitly when cross-compiling:

```bash
export ORINFER_CUDA_ARCH=sm_80   # or sm_86, sm_89, sm_90, sm_100, sm_120
```

Set `ORINFER_REQUIRE_SM87=1` to enforce strict Jetson Orin SM87 checking during `configure()`.

### Device Discovery & Operator Validation

```bash
# Inspect live CUDA device compatibility via Rust CLI
orinfer devices --backend cuda

# Validate portable TileLang kernels + CUDA Graph replay + export a Rust AOT fixture
python3 -m tests.cuda.operators.generic_cuda --fixture-output /tmp/orinfer-generic-fixture

# Verify hashes without CUDA, then execute and validate graph replay through the Rust CUDA Driver runtime
orinfer validate-artifact /tmp/orinfer-generic-fixture/manifest.json
orinfer run-artifact /tmp/orinfer-generic-fixture/manifest.json
```
