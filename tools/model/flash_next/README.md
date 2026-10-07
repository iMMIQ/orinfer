# Flash Next 模型准备

当前提供自有 Q2A8 权重的离线完整推理、原始 BF16 参考与验证。Rust 在线适配器尚未注册，以下产物不能直接传给 `orinfer serve`。

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

## 执行与验证

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

正式在线接入将在 `orinfer-models` 增加适配器，复用 SDK 和 CUDA 执行器；必须覆盖 PLE 的有预算 CPU 查表/上传、请求历史、QSA KV/scale/index/pending、GDN/卷积与 MTP 接受前缀。Python 执行器只作为离线计划和数值参考。
