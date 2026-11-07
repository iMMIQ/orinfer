# Flash Next 离线构建

在线推理由 Rust 执行，TileLang 只参与离线编译。部署目录使用 HF 配置与 tokenizer、`cache/model.json`、标准 safetensors 权重，以及由 digest 固定的执行包。数据格式与 C ABI 均为版本 1。

## 重建当前原生模型

已有当前格式的模型，用一个入口重新调优整数投影并发布匹配的执行包：

```bash
make build
bash tools/environments/cuda/run.sh python/orinfer_build/models/flash_next/build.py artifacts/flash-build \
  --model /path/to/native-model --model-output artifacts/models/flash-ready
```

输入必须包含匹配的执行包：它提供架构、请求状态、程序与绑定契约。构建器以当前原生模型为输入，保留其图片、MTP、上下文容量和请求状态契约。量化中间产物需先完成原生模型组装。功能按执行契约启用。

构建直接从绑定的权重形状、dtype 和 host ABI 读取 Q4、W8、融合 SwiGLU 的构造参数。相同矩阵几何共享编译结果，使用一组确定的真实权重与 scale 测量候选。固定行数保留 1/2/3/4/8/16/32 特化，其余使用动态行参数。decode 动态候选按几何分布测量多个实际行数；prefill 根据模型声明的程序识别，默认针对 512/2048/4096 行测量，可用 `--prefill-tuning-rows` 调整。所有动态实现仍检查短行、非对齐尾部与容量上限；输出必须精确保持整数点积、scale 与舍入边界，并通过修改输入后的 CUDA Graph replay。

专家的路由、上投影和下投影共同使用输入中已验证的联合 tile 选择表。构建另外读取真实专家 bank，在完整 E/N/K 几何上验证两种合法行 tile、2/4 级流水线、短尾与最大容量。整数投影重建保留 HC、QSA、视觉和因果状态实现的固定身份。

缓存默认在工作数据根目录的 `cache/build/flash-next`，可用 `--cache` 指定。键包含构造参数、真实权重身份、源码、编译器、CUDA 和设备；内容校验失败自动重建该项。单项完成后原子发布，整个模型在 Rust 计划检查通过后发布。失败可重跑同一条命令，只需在包装器上增加 `--resume`：

```bash
bash tools/environments/cuda/run.sh --resume python/orinfer_build/models/flash_next/build.py artifacts/flash-build \
  --model /path/to/native-model --model-output artifacts/models/flash-ready
```

对已经验收的原生包，可加 `--prefill-only` 只重建其声明的 prefill 投影，沿用其他固定身份的算子与专家组合。它只限制离线构建范围，运行时仍使用同一动态行模板。

`--projection-kinds w8` 可只重建 W8 投影；`--projection-shape N K` 按输出、输入维度进一步限定范围，可重复指定多个几何，并可与 `--prefill-only` 组合。其他投影、专家和辅助绑定沿用输入包。例如，只重建 640→2560 的 W8 投影：

```bash
bash tools/environments/cuda/run.sh python/orinfer_build/models/flash_next/build.py artifacts/flash-w8-build \
  --model /path/to/native-model --model-output artifacts/models/flash-w8-ready \
  --projection-kinds w8 --projection-shape 2560 640
```

W8 调优包含寄存器 scale 复用和按行 tile 遍历的候选；它们读取相同的原生 W8 权重，保持完整 INT32 点积、先权重 scale 后 token scale 的 FP32 乘法顺序，以及最终输出舍入。候选由真实权重和多个行数的测量选择，发布包只保留选中的实现，Rust 在线路径没有新增依赖。

`--hc-combine-norm` 将单行 decode 中相邻的 HC combine 与 norm 合并，保留 combine 的 FP16 舍入和 norm 的 FP32 归约。Rust 适配器校验融合绑定并移除相邻的独立 norm；构建仍通过同一入口原子发布。GDN 持续状态统一使用 FP32。

在 CPU 上清点构造契约：

```bash
uv run --locked --project tools/environments/cuda python -m orinfer_build.models.flash_next.build \
  --model /path/to/native-model --model-output artifacts/models/flash-ready \
  --output artifacts/flash-plan --plan-only
```

构建记录作为离线诊断保存在指定输出目录。最终包合并资产、去重绑定，只包含所需 cubin。重建完成仍须进行真实请求的精度、完整状态及性能验收；局部算子和端到端请求分别验收。

## 计算与状态契约

专家使用自有整数 E8P Q2A8；GDN QKV/Z、QSA Q/gate 和 target/MTP 共享输出头采用 signed-block128 旋转、输出行 scale 的码表 Q4A8。GDN/QSA 输出投影、共享 FFN 与草稿 Q/O 保留 W8，其余敏感部分维持混合精度。E8P 的完整 A8 位平面分解用两个 INT4 Tensor Core 点积恢复完整 A8 的 INT32 点积。权重只保留一份压缩 bank。

prefill 模板接受 1–4096 行，更长输入分块。普通 decode、混合 batch 与 MTP 验证共享最大 1024 行的 target 计算；draft/warm 共享最大 512 行。完整模型共用执行流程，局部 launch 按真实行数执行。局部实现由离线测量选择。辅助算子统一动态模板，HC 无损指数、QSA radix shift 等标量参数在运行时传入，保持同一计算几何。

QSA KV 为 group-64 INT8，GDN 持续状态为 FP32。KV、pending、压缩索引、卷积/PLE 历史、位置与 MTP 隐藏环均为请求私有状态。改变分块、恢复、捕获或提交时必须比较全部状态。混合 packet 显式记录 prefill/验证角色，角色边界也参与 Graph 身份。

## 部署

```bash
./target/release/orinfer serve artifacts/models/flash-ready \
  --model qwen-flash-next --listen 0.0.0.0:8088 \
  --cuda-graph decode_only --mtp-drafts 3 --prefix-cache-mib 512
```

MTP 包支持 1–7 个草稿；`0` 关闭，`auto` 沿用包默认。随机采样使用 p/q 接受与残差分布修正。短尾与调度预算可缩小验证宽度；prefix 恢复保留完整 target/draft 状态。历史续写候选仅通过显式 `--prompt-lookup` 启用。

服务、请求默认参数、图片和多图输入见[服务说明](../../../docs/serving.md)。文本重建保留输入模型声明的视觉与上下文容量。

## 原始权重与参考工具

CPU 依赖由 `make python-env` 创建的 uv 环境管理。GPU 工具使用 CUDA uv 环境、物理机工具链和共享 GPU 锁；原始 BF16 固定 revision，按矩阵或专家分块读取和释放：

```bash
bash tools/quantization/run_flash_next.sh \
  --index /path/to/original/model.safetensors.index.json \
  --revision SOURCE_COMMIT_SHA --stage all \
  --output artifacts/quantization/flash-next/converted

uv run --locked --project tools/environments/cuda python -m orinfer_build.models.flash_next.weights \
  --converted artifacts/quantization/flash-next/converted \
  --index /path/to/original/model.safetensors.index.json \
  --config /path/to/original/config.json --frontend /path/to/original \
  --output artifacts/models/flash-next-quantized
```

这些命令生成量化中间权重，后续需完成原生模型组装。`native.py`、`mtp.py` 与 `reference/` 提供离线数学对照；正式执行包重建统一使用上方 `build.py`，共用 `compiler.py` 的 host ABI 绑定与编译产物导出。构建输入统一为当前原生模型及其匹配执行包。

数值验证固定 seed 20261002，对同一 teacher-forced 历史比较 top3 概率、KL/NLL 和任务结果；token 一致率用作诊断，质量结合概率与任务结果判断。`reference/` 提供完整 BF16 参考，整模型基线需执行全部层。`checks/` 提供 kernel、尾部和 replay 验证，Rust GPU 测试覆盖状态、prefix、MTP 和混合请求。完整请求性能通过 Rust CLI 测量，算子微测用于定位局部成本。

源码职责：`construction.py`/`build.py` 管理正式重建，公共 `../projection_tuning.py` 管理整数候选；`weights.py`/`checkpoint.py` 读取离线权重，`native.py`/`mtp.py` 保留数学参考。公共算子在 `kernels/model/`，在线架构适配器在 `crates/orinfer-models/src/flash_next/`。
