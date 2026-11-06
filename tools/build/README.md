# 构建、检查与发布

在线使用 Rust 原生运行时、设备驱动、准备好的模型目录和算子包；Ascend 原生后端同时使用 CANN 运行库。Python、PyTorch、TileLang 均属于离线环境。

## Python 环境与依赖

Python 使用 uv 管理，解释器版本由根目录 `.python-version` 固定。依赖按实际工具用途声明，完整解析结果与 wheel 哈希保存在提交的 `uv.lock`：

| 环境 | 声明和锁文件 | 直接依赖用途 |
| --- | --- | --- |
| CPU | 根目录 `pyproject.toml`、`uv.lock` | NumPy、Safetensors、Tokenizers、Pillow；dev 组的 Ruff |
| CUDA | `tools/environments/cuda/pyproject.toml`、`uv.lock` | 共用 CPU 工具依赖；Torch、TileLang、NVRTC 的 CUDA Python、Torch 原生库需要的 NCCL、Transformers 的视觉和 Chat 模板能力 |

CUDA 项目通过本地虚拟依赖引用根项目，使用自己的独立环境。TVM 来自 TileLang 发布包；FFI、Z3 和其他传递依赖由 uv 按上游约束解析。项目使用的 Jetson Torch wheel 不要求 Triton，当前工具也没有 Triton 调用。

```bash
make python-env
make cuda-env
uv run --locked python -m tools.model.catalog --help
uv run --locked --project tools/environments/cuda python -c 'import torch; print(torch.__version__)'
```

`uv sync --locked` 按提交的锁文件同步，`uv run --locked` 执行工具。新增依赖时修改实际使用它的项目；更新依赖使用对应项目的 `uv lock --upgrade`，随后执行该环境的检查。常规安装和验收保持锁文件不变。

## 物理机 CUDA 工具链

CUDA 编译与验证直接使用 Jetson AGX Orin 物理机的 CUDA 12.6、驱动及原生 C/C++ 和 Rust 工具链。`nvcc`、Nsight 或 Compute Sanitizer 从机器的 `PATH` 读取；保持设备原有功耗、频率和缓存设置。Python 环境由 uv 单独管理。

Torch 2.14.1+orinfer1、torchvision 0.29.1+orinfer1 使用我们发布的 CPython 3.12 / SM87 / CUDA 12.6 wheel，下载 URL 声明在 CUDA 项目中，内容哈希由锁文件固定。PyPI 通用 CUDA 13 Torch wheel 不适用于这个部署目标。`jetson_wheels.json` 保留源码提交、构建目标和发布资产身份，供更新来源核对。

Torch 与 torchvision 的源码、构建配置和验收入口在 [iMMIQ/pytorch 的 Jetson 分支](https://github.com/iMMIQ/pytorch/tree/jetson-r36.4-cu126-py312-v2.14/jetson) 维护。wheel 验收覆盖 CPU 三角求解/SVD、CUDA 矩阵乘法、torchvision CUDA NMS 和改变输入的 Graph 回放。

```bash
ORINFER_CHECKPOINT_DIR=/path/to/converted \
  bash tools/operators/run.sh tools/model/qwen35/build.py artifacts/build/qwen35 \
  --source /path/to/converted --model-output artifacts/models/qwen35-ready \
  --gpu-lock-held
```

入口先同步锁定的 CPU/CUDA 环境，再获取 `artifacts/gpu-experiment.lock`。模型、参考与编译缓存路径直接引用物理机文件；运行日志、依赖声明与锁文件副本和机器采样写入输出目录。中断时先回收本次工作进程，再释放设备锁；已有服务保持运行。

新环境使用新输出目录执行最小 GPU 编译验收：

```bash
bash tools/operators/run.sh tools/build/smoke.py artifacts/build/compiler-smoke
```

它从空缓存编译生产 norm 与动态行数 greedy 归约，检查非对齐尾部、3/7 行共用 cubin、标量 ABI、输出 guard 和改变输入的 Graph replay，并导出 AOT。算子数值和状态专项检查分别使用 `tools/operators` 中当前的验收入口。Ascend 使用独立的 CANN 工具链，入口与支持范围见[模型包文档](../../docs/model-packages.md)。

## 自动检查

`make check` 包含 Rust/Ruff 格式、Python 静态检查、编译、clippy、Rust/CPU Python 测试和验收配置校验。GitHub Actions 在 x86_64 和 aarch64 runner 使用相同入口，Python 与 uv 版本固定；Rust 版本由 `rust-toolchain.toml` 固定。

`make fmt` 统一格式，`make lint` 执行格式和静态检查。`make check-offline` 通过 CUDA uv 环境检查离线模块导入，并在 CPU 上执行需要 Torch 的参考数学、前端和动态行数 ABI 测试。

```bash
make check-gpu MODEL=/path/to/prepared-model OUTPUT=artifacts/check-gpu/new-run
```

该入口生成原生 Chat 模板的文本、图片、多图夹具，固定 seed20261002，依次验证 off/decode_only/full 模式的动态 batch、取消/重排/复用隔离、请求丢弃清理和同历史概率。独立 BF16 质量验收另行执行。

## 发布

提交源码、更新 workspace 版本并建立指向该提交的 annotated tag 后，运行：

```bash
uv run --locked python -m tools.release.package --version v0.3.0 \
  --operators /path/to/qwen3_5-execution.tar.xz \
  --operators /path/to/flash-next-execution.tar.xz \
  --output artifacts/releases/v0.3.0
```

默认仅生成和校验产物。工具要求工作区干净，使用 `Cargo.lock` 构建，验证 aarch64/版本，打包完整源码、许可证、依赖声明、模型打包工具和架构契约，并逐个验证原生执行包及实际库身份。引擎、执行包和随附的源码归档统一使用 XZ 压缩（`.tar.xz`，preset 6）；输入旧 `.tar.gz` 执行包会经过验证并重新执行 XZ 压缩。资产名称取自架构规格 ID、计算策略、GPU 目标和版本，例如 `orinfer-exec-qwen35-2b-a8-sm87-v0.3.0.tar.xz`。规格 ID 由 `configs/architecture-contract.json` 中的配置签名选择器识别；`a8` 对应 `int8_quality` 混合精度策略。每种规格分别发布，重复规格与策略会阻止发布；`RELEASE.json` 保留完整包 digest 和架构规格 ID；完整离线工具及 Flash Next 文档随包提供。权重通过模型仓库单独发布。部署使用 Rust CLI 的 `orinfer package install` 安装执行包；详见[安装说明](../../docs/model-packages.md#安装执行包)。

A2 的 `--operators` 直接接受经 `tools.model.deepseek_v41.publication` 整理的 digest 目录；发布器生成原生 A2 `.tar.xz`，用本轮构建的 Rust CLI 执行安装验收后再发布。该过程不加载模型权重或创建 NPU context。

从 0.1.2 起，引擎使用包含 `.so` 与 cubin 的 schema 1 原生执行包。0.1.1 及更早格式的模型需通过当前构建工具重新发布。当前原生格式的模型可以用[执行库更新工具](../../docs/model-packages.md)生成新目录，保留权重和 kernel 资产。

实际发布时添加 `--publish --notes /path/to/release-notes.md`。工具核对 tag 与源码 revision，推送 tag，创建 GitHub Release，再重新下载并核对所有上传资产的 SHA256。每个 release 以独立版本发布，已发布资产保持原样。

### A2 派生常量的紧凑发布

已发布的 A2 包可保持全部算子、执行计划和运行时 RoPE 字节，改为安装时生成表：

```bash
uv run --locked python -m tools.model.deepseek_v41.compact \
  --source /path/to/published-digest-directory \
  --output-cache artifacts/compact-a2 \
  --engine /path/to/new/orinfer
```

现有 `tools.model.deepseek_v41.publication` 也接受 `--compact-rope --engine PATH`。发布包仅携带 FP32 频率位表示和压缩 XOR 校正流；数学近似与校正共同组成版本固定的派生资产编码。`orinfer package install` 在私有临时目录内生成原始 FP32 文件，逐项校验原内容身份，通过全部包校验后原子发布到 cache。生成失败不会发布半成品，也不会覆盖已有包。安装不需要 Python、PyTorch、GPU 或 NPU；正常加载读取已安装文件，推理路径沿用原数值和 Graph 地址契约。

新包保留 schema 1，增加可选 `generated_assets` 契约，需使用支持该契约的新引擎安装。新引擎继续支持原有 `.tar.xz` 和 `.tar.gz` 包。由 `package encode-rope` 提供离线编码入口，Python 构建器仅负责克隆、压缩及内容寻址发布；该命令不进入服务加载流程。完整内容校验仍在安装和独立 `validate-model` 中执行。

紧凑发布同时以 `shared-json-v1` 结构表保存最终执行计划，复用 rank、layer、packet 间完全相同的值；数组顺序、数字类型、参数与 launch 表达式保持不变。安装后的计划仍保持紧凑，加载器按有界无环引用展开一次后交给原模型契约。旧的普通 JSON 计划继续直接读取。离线重建使用保留的原始构建计划；需要检查分发计划时，可通过 `python -m tools.build.shared_json --input PLAN --output EXPANDED --expand` 导出语义相同的普通 JSON，相对资产路径仍按原包目录解释。发布与紧凑工具直接识别这两种计划。

模型归档器与发行归档器从已安装 cache 重新打包时，自动省略 `generated_assets` 声明的生成结果，保留生成配方及校正流，避免把展开的常量表重新放回分发包。
