# TileLang Kernels

Production GPU and NPU kernels authored in TileLang, compiled offline into cubins (CUDA SM87) or AOT `.so` libraries (Ascend 910B / CANN 9.2), and executed online by Rust with zero Python/PyTorch dependency.

## Directory Layout

| Directory | Target | Contents |
| --- | --- | --- |
| `cuda/operators/` | CUDA SM87 | Core building blocks (activation quantization, norms, reductions) |
| `cuda/model/` | CUDA SM87 | Projections, E8P, GDN, GQA/QSA, MoE, HC, PLE, MTP, GPU sampling |
| `cuda/audio/` | CUDA SM87 | Qwen3-ASR NHWC chunk CNN & window encoder |
| `cuda/vision/` | CUDA SM87 | Shared ViT encoder, Conv3D, axial/MRoPE, BF16 PTX RNE epilogue |
| `ascend/` | Ascend 910B | DSv4.1 TileLang-Ascend operators + third-party licenses |
| `shared/` | Both | Shared dynamic-row contracts |

Device validation suites live in `tests/cuda/` and `tests/ascend/`; offline compilation and tuning live in `python/orinfer_build/`.

## Shared CUDA Compute Cores

Production projections are composed via compile-time TileLang macros so prefill, decode, and MTP share identical numerical and rounding contracts:

- **`cuda/model/integer_core.py`**: Shared DP4A / Tensor Core loops for W8, scalar-LUT Q4, and fused W8 SwiGLU (gate/up round to FP16 before SwiGLU).
- **`cuda/model/floating_core.py`**: Shared BF16/FP16 MMA loops for floating projections, HC (strict K16 accumulation order), and vision linear layers.
- **`cuda/model/e8p_core.py`**: Full-A8 bit-plane decomposition, dual INT4 Tensor Core MMAs, and output rescaling shared by Flash Next Q2 experts and Qwen3.5 two-stage Q4A8.

Inlined component source hashes are automatically included in compile and tuning cache keys.

## Validation Requirements

Every new or modified kernel must pass:
1. Exact numerical comparison against an independent reference (CPU FP32/FP64 or official BF16).
2. Unaligned tails, dynamic row bounds, and empty/invalid routing edges.
3. Changed-input CUDA / ACL Graph replay.

See [Offline Model Builders](../tools/model/README.md) and [Ascend A2 Build Pipeline](../docs/ascend-a2-build.md) for build and test entrypoints.
