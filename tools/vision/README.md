# 图片和多图

在线路径使用Rust解码PNG/JPEG/WebP、bicubic缩放、归一化与patch排列；GPU encoder与文本桥接均由TileLang导出AOT cubin。支持当前checkpoint的27层ViT、2D RoPE、learned position interpolation、pre-shuffle LayerNorm merger、5120／2560维输出与文本交错MRoPE。每张图片独立编码，无视频/DeepStack分支。

## Flash Next

Flash Next 使用共享 ViT 算子与 2560 维 merger。离线构建在已准备的 MTP 文本目录上附加视觉权重、QSA/indexer 三轴 RoPE 和移位 MTP embedding；在线仍为 Rust。完整命令见[Flash Next 构建](../model/flash_next/README.md#图片与多图)。原始视觉权重从同一 pinned revision 单独下载，约 898 MB；部署采用 FP16 视觉计算，累加和位置插值为 FP32。

默认单图最大 8192 patches、多图合计 16384 image tokens，可在构建时调整。图片和多图使用现有 OpenAI `image_url` API，可同时启用 thinking、MTP 和 prefix cache；不包含视频。

```bash
bash tools/operators/run.sh tools/vision/validate_mrope.py artifacts/flash-mrope-check
bash tools/operators/run.sh tools/vision/validate_encoder.py artifacts/flash-encoder-check \
  --model /path/to/flash-serving-mm/cache/build.json
```

`validate_mrope.py` 覆盖查询/KV/indexer 旋转、非对齐压缩块、文本位置回归及视觉/MTP 特征的改变输入 Graph replay。`official_reference.py` 根据 checkpoint 配置调用 Transformers 的 `Qwen4ExpVisionModel` 或 `Qwen3_5VisionModel`，可用其 `features.f16` 作为 `validate_encoder.py --official-reference` 的独立对照。FP16 与 BF16 参考须分别注明，视觉验证不替代完整文本主干的 BF16 量化评测。

## 27B 构建

先按[文本模型构建](../model/README.md)获得W4文本manifest，再附加原始checkpoint的视觉权重。checkpoint需包含`config.json`及未量化`model.visual.*`的`model.safetensors`。默认把原始BF16视觉权重转为FP16，视觉激活为FP16，累加为FP32；文本权重以hard link保留原字节和hash。`--vision-dtype bf16`是实验路径，使用NVCC编译TileLang导出的CUDA源以避开当前NVRTC的BF16向量转换问题；该路径的全编码器精度检查尚未通过，不作为默认配置。

```bash
bash tools/operators/run.sh tools/vision/build.py artifacts/vision/model01 \
  --text-model /path/to/text/model.json --checkpoint /path/to/checkpoint
./target/release/orinfer serve artifacts/vision/model01/model.json /path/to/tokenizer-dir
```

默认最大32768 patches/8192 image tokens，单张图片容量还受总上下文约束。`--max-patches 1024|2048|4096|8192|16384|32768`可降低workspace和输入分辨率上限。encoder使用256至最大容量的固定graph桶，实际grid/length动态上传并屏蔽padding；多图复用encoder workspace，输出按顺序拷入请求私有特征表。请求之间重置特征索引和全部MRoPE坐标。

27层视觉权重共460,730,096参数，占921,460,192字节；这些字节计入manifest全模型权重预算。merger最终输出为FP16以连接文本主干。当前工作区和视觉attention尚未针对大分辨率做性能调优，attention为双向全局attention。

构建时检查视觉激活、位置表、attention head布局、交错MRoPE、词表和group-128 W4 embedding布局；不兼容的checkpoint或文本计划须提供新的adapter，不能直接套用本桥接kernel。

## 验证

以下工具的output路径必须不存在。GPU验证遵循全仓库GPU实验锁，勿与常驻服务同时运行。

```bash
bash tools/operators/run.sh tools/vision/validate_kernels.py artifacts/vision/kernels01
bash tools/operators/run.sh tools/vision/validate_bridge.py artifacts/vision/bridge01 \
  --model artifacts/vision/model01/model.json
bash tools/operators/run.sh tools/vision/validate_encoder.py artifacts/vision/encoder01 \
  --model artifacts/vision/model01/model.json
python3 tools/vision/preprocess_reference.py --model artifacts/vision/model01/model.json \
  --output artifacts/vision/preprocessor01
ORINFER_IMAGE_REFERENCE="$PWD/artifacts/vision/preprocessor01" \
  cargo test --offline checkpoint_image_processor_reference -- --ignored --nocapture
```

kernel检查包含非对齐矩阵、真实length尾部以及改变输入/恢复输入的graph replay；encoder检查实际checkpoint权重的全27层图，与同方程FP16/BF16 PyTorch参考比较。预处理参考使用Transformers原始image processor，覆盖PNG/JPEG/WebP、非方图和放大/缩小。解码库差异可能造成JPEG像素的小误差。

独立视觉参考可直接调用Transformers的`Qwen3_5VisionModel`，避免参考方程和kernel共享实现错误。输入为image processor输出的FP32原始patch文件，转换到指定精度前不经过FP16中转：

GPU参考关闭TF32及FP16/BF16 GEMM的低精度中间归约，采用与生产kernel一致的FP32累加基准。`--trace`可保存patch、各block及merger归一化的输出，用于固定中间输入的逐算子诊断。

```bash
python3 tools/vision/official_reference.py --checkpoint /path/to/checkpoint \
  --pixels /path/to/pixels.f32 --grid 20 34 --dtype bf16 \
  --output artifacts/vision/official01
```

`tools/api/template_reference.py --vision-model ... --image ...`还生成官方图文token和MRoPE参考，Rust的`checkpoint_matches_reference_tokens`检查全部输入位置及后续16个生成位置。离线逐token诊断使用忽略的`multimodal_token_probe`测试；`ORINFER_VISION_PROBE`指向包含model、input_tokens、images、prefixes、target_tokens及output的JSON，model和output使用绝对路径，记录原始生成token、固定历史下top3及目标logprob。运行时须在外层持有GPU实验锁；该诊断不经过Chat API输出解析。

```bash
ORINFER_VISION_PROBE=/path/to/probe.json flock artifacts/gpu-experiment.lock \
  cargo test --release --offline -p orinfer-engine multimodal_token_probe -- --ignored --nocapture
```

运行真实服务后执行：

```bash
python3 tools/vision/smoke.py --output artifacts/vision/smoke01.json
python3 tools/vision/scenarios.py --output artifacts/vision/scenarios01.json
python3 tools/vision/limits.py --manifest /path/to/vision/model.json \
  --output artifacts/vision/limits01.json
```

Smoke覆盖单图、多图次序与反序、SSE和图片后文本状态隔离。场景验证包含跨512-token块的双图、非方图OCR、计数、图文历史、函数调用和两个排队的HTTP图片请求。工具固定seed 20261002；输出记录完整回答、usage、时延和是否符合场景预期。场景任务检查不能替代完整LLM benchmark或整文本主干BF16/FP8对照。

`limits.py`验证默认32768-patch配置的最大图片请求、图片/上下文超限拒绝、非法输入、不同图片同时排队，以及大图和错误请求之后的状态隔离。较小容量的manifest不适用这个最大容量fixture。OCR场景按完整文字判定质量；即使原生参考也答错，该项仍记录失败，不能将实现一致视作OCR识别正确。

OCR的thinking对照使用同一图片、提示词、seed及输出预算，分别记录`reasoning_content`、最终`content`、usage和耗时：

```bash
python3 tools/vision/ocr_thinking.py --image /path/to/image.png \
  --output artifacts/vision/ocr-thinking01.json --reasoning-effort xhigh
```

默认每次输出预算1024 tokens，包含thinking；检查`finish_reason`以判断是否因预算耗尽而没有最终答案。
