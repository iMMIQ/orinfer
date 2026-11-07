# 图片与多图

27B 与 Flash Next 使用 `kernels/cuda/vision/encoder.py` 的共享 ViT 实现，文本模型分别声明 merger 输出维度、特征注入和 MRoPE 状态。27B 通过 [原生构建器](../model/qwen35/README.md)生成包含视觉功能的模型；Flash Next [原生重建](../model/flash_next/README.md)保留输入模型的视觉契约。

在线由 Rust 处理图片预处理、位置生成、请求私有特征表及模型执行。视觉使用一个动态 patch 模板，支持图片和有序多图输入。默认最大 32768 patches，即 8192 个视觉 tokens；实际输入还受上下文与请求内存预算限制。使用方式见[服务说明](../../docs/serving.md)。

## 算子与参考

GPU 验证使用 CUDA uv 环境、物理机工具链及共享 GPU 锁，使用新输出目录：

```bash
bash tools/environments/cuda/run.sh tools/vision/validate_kernels.py artifacts/vision/kernels-check
bash tools/environments/cuda/run.sh tools/vision/validate_mrope.py artifacts/vision/mrope-check
```

`validate_mrope.py` 检查查询/KV/indexer 旋转、非对齐压缩块和特征注入的改输入 Graph replay。`official_reference.py` 按 checkpoint 配置调用 Transformers 的原生视觉模型，导出独立 BF16 或 FP16 特征；须注明参考 dtype。

`preprocess_reference.py` 为当前模型导出预处理夹具，再由 Rust ignored test 比对：

```bash
bash tools/environments/cuda/run.sh tools/vision/preprocess_reference.py artifacts/vision/preprocessor-check \
  --model /path/to/prepared-model
ORINFER_IMAGE_REFERENCE="$PWD/artifacts/vision/preprocessor-check" \
  cargo test --offline checkpoint_image_processor_reference -- --ignored --nocapture
```

完整状态、分块、prefix 恢复和图片后文本隔离通过 `make check-gpu` 的 Rust 模型验收检查。

## API 验收

在准备好的服务上运行：

```bash
python3 tools/vision/smoke.py --output artifacts/vision/smoke.json
python3 tools/vision/scenarios.py --output artifacts/vision/scenarios.json
python3 tools/vision/limits.py --manifest /path/to/prepared-model \
  --output artifacts/vision/limits.json
```

工具支持 `--base-url` 与 `--model`。Smoke 覆盖单图、多图顺序、SSE 及状态隔离；场景覆盖计数、图文历史、函数调用和排队。容量验收检查超限错误、错误恢复与不同图片并发排队，要求默认 32768-patch 容量。场景使用 seed 20261002，记录回答、usage、停止原因与时延；任务质量另外与原始 BF16/FP8 对照。
