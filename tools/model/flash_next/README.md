# Flash Next 模型准备

提供自有 Q2A8 权重、Rust 在线适配器，以及离线完整推理与原始 BF16 参考。权重转换产物先通过下述部署构建生成 `cache/model.json` 和匹配执行包，再交给 CLI。

执行器固定使用已采用的 SM87 路径：专家 E8P＋block128 旋转、普通投影 W8、关键系数 BF16/FP32、group-64 INT8 KV、FP32 GDN 状态。prefill 按长度选择专家 tile；MTP 支持 1..7 个草稿和完整接受前缀提交。架构与编码约束位于 [`architecture-contract.json`](../../../configs/architecture-contract.json)，加载前拒绝不匹配的配置。

## 权重准备

CPU 工具使用 `make python-env` 创建的 uv 环境，GPU 转换与验证使用固定编译镜像及 GPU 锁。固定原始 checkpoint revision，流式读取 BF16，不需要保存完整原始模型：

```bash
bash tools/quantization/run_flash_next.sh \
  --index /path/to/original/model.safetensors.index.json \
  --revision SOURCE_COMMIT_SHA --stage all \
  --output artifacts/quantization/flash-next/converted

PYTHONPATH=. .venv/bin/python -m tools.model.flash_next.weights \
  --converted artifacts/quantization/flash-next/converted \
  --index /path/to/original/model.safetensors.index.json \
  --config /path/to/original/config.json --frontend /path/to/original \
  --output artifacts/models/flash-next-e8p-a8
```

MTP 另用 `--component mtp --stage all` 转换，再发布到独立目录。target/draft 必须来自同一源、revision、编码与文本配置；共享 embedding 和输出头，分别保有 KV/index 状态。

发布目录保留 HF config、tokenizer、chat template 与标准 safetensors index。物理 tensor layout 存在 shard metadata 中；发布逐字节保留已转换权重，默认在验证后消费临时 shard，`--keep-temporary-shards` 可保留。`checkpoint.py` 读取发布目录，不依赖转换进度文件。模型格式和权重编码未因目录整理而改变。

## Rust 部署

在线包含 W8 embedding、CPU E8P PLE 查表、48 层 HC/GDN/QSA/MoE、256K INT8 KV，以及完整请求状态和 prefix cache。prefill 提供 4096／2048／512／128／16 token 档位，尾部用真实 M=1 执行；不填充虚假 token。基础包支持单个活跃请求、其余排队；下述 batch 构建可启用多请求 decode。可选原生 MTP 支持贪心和带惩罚项的随机采样。

```bash
make build
bash tools/operators/run.sh tools/model/flash_next/prepare.py artifacts/flash-build \
  --checkpoint /path/to/flash-target --model-output artifacts/models/flash-serving
PYTHONPATH=. .venv/bin/python -m tools.model.flash_next.package artifacts/models/flash-serving
./target/release/orinfer serve artifacts/models/flash-serving \
  --model qwen-flash-next --max-active-requests 1 \
  --cuda-graph decode_only --prefix-cache-mib 512
```

可选 MTP 执行包共享主模型 embedding 和完整输出头，增加单层 draft、512 行 HC 环和验证/提交程序。在线支持每轮 1–7 个草稿，默认最多 3 个；用 `serve --mtp-drafts 7` 指定上限，`auto` 沿用包默认值，`0` 关闭 MTP 执行。短尾、上下文边界和调度 token 预算不足时自动缩小验证批次或使用普通 decode；prefix checkpoint 保留 P−1 的 draft 状态，恢复时用真实 continuation 连接。关闭 MTP 不移除包中的 draft 权重或 workspace；所指定的验证档位必须存在于执行包中。

```bash
bash tools/operators/run.sh tools/model/flash_next/prepare_mtp.py artifacts/flash-mtp-build \
  --checkpoint /path/to/flash-target --mtp-checkpoint /path/to/flash-draft \
  --base-model artifacts/models/flash-serving \
  --model-output artifacts/models/flash-serving-mtp
PYTHONPATH=. .venv/bin/python -m tools.model.flash_next.package artifacts/models/flash-serving-mtp
# serve 使用上面的相同参数，改为 flash-serving-mtp 目录；包含 MTP 的包自动启用。
```

验证阶段保存 GDN 的 FP32 紧凑更新、卷积/PLE 历史和每个 QSA pending 前缀；全部接受和部分接受均执行提交。Draft 分支会恢复其原始 pending 块，再以 target HC 重算已提交位置。随机采样使用 p/q 接受和残差分布修正。

构建会按实际 host ABI 导出并绑定 cubin，保存同权重的固定历史 logits；打包时逐项比较 Rust 注册计划与离线计划。CPU 表按行读取，embedding／PLE 解码缓存分别限制为 8／32 MiB，部署不用 Python。中断构建可用 `--reuse-data` 校验并复用已写入的 GPU 权重；CPU 表通过 hardlink 复用。

可用 `score-model MODEL_DIR artifacts/flash-build/score-requests.json` 核对部署输出。这个对照验证执行迁移，独立 BF16 质量评估仍按下文进行。服务兼容 OpenCode 的 `@ai-sdk/openai-compatible` provider，模型 ID 为 `qwen-flash-next`；建议输出预算至少 4096 tokens。

## 图片与多图

在已准备的文本目录上附加视觉编码器；如需 MTP，先准备上述 MTP 文本目录；沿用同源 checkpoint revision，仅下载 `model.visual.*` 原始 BF16 张量，不下载整份模型。文本权重和已有 cubin 通过 hardlink 复用，视觉权重以 FP16 保存，计算使用 FP16/FP32。原始权重转换仅允许 FP16 subnormal 范围内的舍入，拒绝溢出或更大的改动。

```bash
PYTHONPATH=. .venv/bin/python -m tools.vision.source \
  --model artifacts/models/flash-serving-mtp --output artifacts/checkpoints/flash-vision
bash tools/operators/run.sh tools/model/flash_next/prepare_vision.py artifacts/flash-vision-build \
  --base-model artifacts/models/flash-serving-mtp \
  --checkpoint artifacts/checkpoints/flash-vision \
  --model-output artifacts/models/flash-serving-mm
PYTHONPATH=. .venv/bin/python -m tools.model.flash_next.package artifacts/models/flash-serving-mm
./target/release/orinfer serve artifacts/models/flash-serving-mm \
  --model qwen-flash-next --max-active-requests 1 \
  --cuda-graph decode_only --prefix-cache-mib 512 --mtp-drafts 7
```

视觉编码器与 27B 共享 TileLang 算子和执行组件，merger 输出为 2560 维；QSA 查询、KV 和压缩索引使用交错三轴 MRoPE，MTP 读取移位后的图像特征。PLE 保留原始 image token ID。多图按消息次序独立编码，特征和位置索引属于请求私有状态；图片身份参与 prefix 匹配。

默认单图最多 8192 patches（2048 image tokens），多图合计最多 16384 image tokens；可用 `--max-patches`、`--max-features` 调整构建容量。`--compile-cache` 可复用已有 TileLang 编译缓存。上下文仍为 256K、KV 仍为 INT8；视觉编码不被文本 prefix cache 省略。请求格式见[服务使用](../../../docs/serving.md)，独立编码器、chat template 和 MRoPE 的验证入口见[图文工具](../../vision/README.md)。

## 多请求 decode

在已发布的文本或多模态包上离线增加 batch 算子。先执行 `make build`，使用新的目标目录；权重逐字节复用，shape-only 编译不需要加载第二份模型：

```bash
bash tools/operators/run.sh tools/model/flash_next/batching.py artifacts/flash-batch-build \
  --model artifacts/models/flash-serving-mm \
  --model-output artifacts/models/flash-serving-batch
./target/release/orinfer serve artifacts/models/flash-serving-batch \
  --model qwen-flash-next --max-active-requests 32 \
  --cuda-graph decode_only --prefix-cache-mib 512 --mtp-drafts 7
```

不需要再次执行 `package.py`。可用 `--compile-cache` 复用完成的 TileLang 缓存。2/4/8/16/32/64/128 档共享 HC、普通投影、路由、专家、共享专家和输出头；其余请求数补零到下一档。GDN、PLE 卷积和 QSA 沿用已验证的单序列状态 kernel，只推进真实 lane。MTP 捕获每个请求的 target HC，调度器根据收益在 target batch 和逐请求 MTP 之间选择。CPU PLE 的行缓存共享，历史独立。

Prefill 使用原有大块，与 decode 交替调度；每请求 KV/context 预留参与内存准入，超出活跃或内存预算的请求排队。128 个提交请求不要求同时在显存中驻留 128 份满上下文状态。Graph 按槽位和形状缓存；取消、槽位复用和 prefix 恢复保持请求隔离。

## 离线执行与验证

所有输出放在新的 artifact 目录。场景包含实际 chat template、thinking、固定历史 top3 概率和 NLL；seed 为 20261002：

```bash
bash tools/operators/run.sh tools/model/flash_next/native.py artifacts/flash-check \
  --checkpoint /path/to/flash-target --context 262144 --chunk 4096 --graph on \
  --max-new-tokens 64

# 可选 MTP：在上述入口加 --mtp-checkpoint /path/to/flash-draft --mtp-drafts 3
# --mtp-vocab-size 0 使用完整草稿头；默认 65536，只限制草稿，不限制 target。
# --mtp-adaptive 可选启用 1/3/7 动态深度。

bash tools/operators/run.sh tools/model/flash_next/checks/mtp.py artifacts/flash-mtp-check \
  --checkpoint /path/to/flash-target --mtp-checkpoint /path/to/flash-draft \
  --lengths 512 2048 8192 --drafts 3 7

bash tools/operators/run.sh tools/model/flash_next/long_context.py artifacts/flash-long-check \
  --checkpoint /path/to/flash-target --context 262144

bash tools/operators/run.sh tools/model/flash_next/checks/w8_kernels.py artifacts/flash-w8-kernels
bash tools/operators/run.sh tools/model/flash_next/checks/w8.py artifacts/flash-w8-requests \
  --checkpoint /path/to/flash-target --draft /path/to/flash-draft
```

W8 对照保留朴素路径用于数值回归，逐位置比较 logits 和 target/draft 状态；它不作为部署策略。可设 `ORINFER_OPERATOR_NSYS=1` 并给 `checks/w8.py` 加 `--trace-only`，单独采集诊断 trace。生成/接受率测试与独立量化质量评估分别进行。

独立 BF16 对照使用 `reference/`：先在编译镜像中用 `scenes.py` 固定请求，再在 CPU 环境读取被选中的原始 embedding/PLE，最后执行分层 teacher：

```bash
PYTHONPATH=. .venv/bin/python -m tools.model.flash_next.reference.inputs \
  --index /path/to/original/model.safetensors.index.json --config /path/to/original/config.json \
  --scenes /path/to/scenes.json --revision SOURCE_COMMIT_SHA --output artifacts/flash-inputs

bash tools/model/run_flash_teacher.sh \
  --index /path/to/original/model.safetensors.index.json --config /path/to/original/config.json \
  --scenes /path/to/scenes.json --revision SOURCE_COMMIT_SHA \
  --inputs artifacts/flash-inputs --output artifacts/flash-bf16
```

把完整 teacher 的 `results.json` 传给 `native.py --baseline`，比较相同历史的 token 质量和概率。仅运行部分层不能作为整模型质量基线。

## 源码边界

`weights.py`/`checkpoint.py` 管理权重，`native.py`/`mtp.py` 是离线执行参考，`reference/` 保留独立数学与原始权重读取，`checks/` 与 `tests/` 管理回归验证。算子留在公共 `kernels/model/`，不按模型复制。

`orinfer-models/src/flash_next` 提供在线适配器和 CPU 查表，复用 SDK 和 CUDA 执行器；QSA KV/scale/index/pending、GDN 与卷积均为请求私有状态。Python 执行器用于离线计划、数值参考和 MTP 研究。
