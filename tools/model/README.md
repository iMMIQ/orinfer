# 离线模型构建

本目录提供checkpoint转换、kernel验证与导出、模型组装和Rust运行入口。Python不参与在线推理。

## 本机依赖

checkpoint默认位于`/home/nvidia/model/vllm-comparison-20260930/awq-http/`。投影验证使用`artifacts/experimental-vllm/activations/`中的真实L0输入；运行前需准备这些外部数据，文件名见验证入口。LUT4验证支持`--activations-dir`。

GPU入口`bash tools/operators/run.sh RUNNER NEW_OUTPUT [ARGS]`使用本机NVIDIA Docker镜像和GPU锁；镜像名为`lada-orin-tilelang:0.11.0-exp7`，实际TileLang0.1.13/Torch2.9.1/CUDA12.6。镜像及checkpoint不随源码分发。CPU组装入口设`PYTHONPATH=.`；全部输出目录应为新目录，原产物不修改。构建和组装脚本产生的`model.json`及裸权重是离线中间产物，不能直接交给在线模型加载器；最后必须执行下述safetensors打包。

## 从checkpoint重建

先分别构建512/2048/8192基线计划，使用以下参数，并把每次的`--prefill-tokens`设为相应长度。各计划需要相同权重及量化身份；可用`--reuse-weights`复用第一个完成的构建目录，避免重复转换。

```bash
bash tools/operators/run.sh tools/model/build.py artifacts/model/rebuild512 \
  --dense-u4 --prefill-w4a8 --prefill-tokens 512 \
  --w8-expand-mode aligned --prefill-grid-order nfirst \
  --decode-register-mma --decode-register-scope all \
  --decode-state-mode inplace --decode-attention-mode staged \
  --prefill-norm-a8 --swiglu-a8-mode lut \
  --gdn-math factored --gdn-wy-mode compensated \
  --gdn-solve-mode columns-register --prefill-attention-mode staged64

PYTHONPATH=. python3 tools/model/assemble_plans.py \
  --output artifacts/model/rebuild-plans \
  artifacts/model/rebuild512/model.json \
  artifacts/model/rebuild2048/model.json \
  artifacts/model/rebuild8192/model.json
```

随后用基线三计划manifest生成并验证采用的AOT导出。下列路径`artifacts/model/rebuild-plans/model.json`必须已存在。

```bash
bash tools/operators/run.sh tools/model/validate_lut4_projection.py \
  artifacts/operators/rebuild-lut4 --model artifacts/model/rebuild-plans/model.json
bash tools/operators/run.sh tools/model/screen_expand_i8layout.py \
  artifacts/operators/rebuild-expand --model artifacts/model/rebuild-plans/model.json
bash tools/operators/run.sh tools/model/screen_decode_i8layout.py \
  artifacts/operators/rebuild-decode --model artifacts/model/rebuild-plans/model.json \
  --vector-load --byte-permute --vector-words 4 --production-split
PYTHONPATH=. python3 tools/model/assemble_lut4.py \
  --model artifacts/model/rebuild-plans/model.json \
  --short-exports artifacts/operators/rebuild-lut4 \
  --expand-exports artifacts/operators/rebuild-expand \
  --decode-exports artifacts/operators/rebuild-decode \
  --output artifacts/model/rebuild-lut4

bash tools/operators/run.sh tools/model/screen_gdn_precision.py \
  artifacts/operators/rebuild-gdn --policy high
bash tools/operators/run.sh tools/model/screen_norm_no_y.py \
  artifacts/operators/rebuild-norm
PYTHONPATH=. python3 tools/model/assemble_gdn_precision.py \
  --model artifacts/model/rebuild-lut4/model.json \
  --gdn-exports artifacts/operators/rebuild-gdn \
  --norm-exports artifacts/operators/rebuild-norm \
  --output artifacts/model/rebuild-high

bash tools/operators/run.sh tools/model/screen_gdn_wy_precision.py \
  artifacts/operators/rebuild-wy --policy high --value-tile 32
bash tools/operators/run.sh tools/model/screen_gdn_gated_norm_a8.py \
  artifacts/operators/rebuild-gatednorm --model artifacts/model/rebuild-high/model.json \
  --threads 512
PYTHONPATH=. python3 tools/model/assemble_gdn_fusions.py \
  --model artifacts/model/rebuild-high/model.json \
  --wy-exports artifacts/operators/rebuild-wy \
  --gatednorm-exports artifacts/operators/rebuild-gatednorm \
  --gatednorm-threads 512 --output artifacts/model/rebuild-final
```

组装器从实际导出host ABI绑定参数、grid和shared，校验数据hash和消费链；不假定PrimFunc声明顺序。新工具链编译会产生不同的cubin/manifest hash；须按新身份重新验证和计时，不能沿用旧速度数字。

## 运行与分析

### 原生MTP

Qwen3_5 adapter可在现有模型上添加单层MTP。准备包含checkpoint原生`mtp.*`参数的safetensors；支持BF16/FP16及带128×128块scale的FP8。社区AWQ文本checkpoint通常不包含MTP，需要另行提供同一原生模型的MTP参数。导入工具只转换草稿权重，主模型权重保持原来的单份常驻表示。

```bash
bash tools/operators/run.sh tools/model/mtp_weights.py \
  artifacts/model/mtp-weights --checkpoint /path/to/mtp.safetensors --format w4
bash tools/operators/run.sh tools/model/assemble_small_m.py \
  artifacts/model/verification --model /path/to/model.json --tokens 2 4 8
bash tools/operators/run.sh tools/model/assemble_mtp.py \
  artifacts/model/with-mtp --model artifacts/model/verification/model.json \
  --weights artifacts/model/mtp-weights --verification-tokens 4
python3 tools/model/prepare.py \
  --model artifacts/model/with-mtp/model.json \
  --checkpoint /path/to/checkpoint-dir --output artifacts/models/qwen3.8-27b
./target/release/orin-llm validate-model artifacts/models/qwen3.8-27b
./target/release/orin-llm serve artifacts/models/qwen3.8-27b \
  --listen 127.0.0.1:8088 --model qwen3.8-27b
```

`verification-tokens`包含一个已提交、尚未处理的输入token；4对应最多3个新草稿。验证图保存每个GDN/卷积前缀以恢复拒绝状态；主模型最终归一化hidden用于MTP预填充和验证后的KV更新。主模型embedding/head与MTP共享，不存第二份主模型权重。FP8 scale按原生语义相乘；norm保留zero-centered形式。`weights.json`和`mtp-build.json`记录权重身份、所有草稿参数字节数及合计平均bits。

算子验证使用`validate_mtp_kernels.py`，覆盖实际权重布局、因果注意力、GDN恢复、hidden捕获和改变输入后的graph replay。完整生成验证使用Rust ignored test `validate_mtp_generation`（环境变量`ORIN_MTP_FIXTURE`指向含model/output/cases/repetitions/eos的JSON）。先用CLI ignored test `export_mtp_chat_fixture`按原生Chat模板导出请求（`ORIN_MTP_FIXTURE_SPEC`），再在独占GPU锁下执行验证。测试对照关闭MTP的相同主模型，比较真实生成输出及全部有效主模型状态。HTTP计时工具`tools/bench/mtp_chat.py`通过主模型参考token IDs确认首个SSE片段的token数量，只统计最终交付输出。

`run.sh MODEL REQUESTS NEW_OUTPUT`运行Rust完整模型并记录机器状态、源代码、二进制身份和报告。`scenarios.py prepare DIR`生成固定场景请求，`scenarios.py score INPUT_DIR REPORT OUTPUT`做任务判分。`score.py`提供同历史概率诊断；这些社区权重对照不替代BF16/FP8质量评估。

`profile.sh MODEL REQUESTS NEW_OUTPUT`采集Nsight节点trace；`profile_summary.py`校验manifest映射。四输出profile不是正式TPS验收。

## Safetensors模型目录

离线编译、融合、视觉与MTP组装全部完成后，使用`prepare.py`发布新的模型目录。普通文本/视觉模型也使用同一入口；把`--model`替换为最终中间产物。无需GPU或PyTorch，只依赖Python的safetensors库。

```bash
python3 tools/model/prepare.py \
  --model artifacts/model/rebuild-final/model.json \
  --checkpoint /path/to/checkpoint-dir --output artifacts/models/qwen3.8-27b
./target/release/orin-llm validate-model artifacts/models/qwen3.8-27b
./target/release/orin-llm run-model artifacts/models/qwen3.8-27b examples/requests.json
```

工具检查checkpoint词表、所有源payload与构建资产的hash；不重新量化、不重编译kernel，包含只读权重以及RoPE/索引等可写buffer的初始值。默认每片约1 GiB，`--shard-mib`可调整；单个tensor不会拆分，转换内存由最大分片决定。写完并用标准safetensors reader校验后，才原子发布完整目录；已有输出不覆盖，失败时删除本次临时目录。

目录根保留checkpoint配置、generation config、tokenizer、chat template和图片预处理配置；不复制原始checkpoint的大权重。内部`cache/weights/`保存带dtype/shape和物理layout元数据的safetensors及HF分片索引；`cache/manifest.json`只引用tensor名字及payload SHA256，由索引定位文件；`cache/kernels/`保存去重、独立复制的构建资产。文件在加载期间必须保持不变。原始checkpoint与此物理布局缓存用途不同，不能用Transformers直接执行缓存tensor。

在线Rust不提供旧模型格式兼容分支，也不在首次请求中执行Python或量化。后续重新编译或改变布局时，完成离线组装后发布新的目录。GPU校验fixture的`model`字段可指向模型目录或schema-2执行计划；算子测试夹具仍采用独立的原格式。
