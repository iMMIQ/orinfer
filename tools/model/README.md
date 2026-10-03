# 离线模型构建

本目录提供checkpoint转换、kernel验证与导出、模型组装和Rust运行入口。Python不参与在线推理。

## 本机依赖

checkpoint默认位于`/home/nvidia/model/vllm-comparison-20260930/awq-http/`。投影验证使用`artifacts/experimental-vllm/activations/`中的真实L0输入；运行前需准备这些外部数据，文件名见验证入口。LUT4验证支持`--activations-dir`。

GPU入口`bash tools/operators/run.sh RUNNER NEW_OUTPUT [ARGS]`使用本机NVIDIA Docker镜像和GPU锁；镜像名为`lada-orin-tilelang:0.11.0-exp7`，实际TileLang0.1.13/Torch2.9.1/CUDA12.6。镜像及checkpoint不随源码分发。CPU组装入口设`PYTHONPATH=.`；全部输出目录应为新目录，原产物不修改。

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

`run.sh MODEL REQUESTS NEW_OUTPUT`运行Rust完整模型并记录机器状态、源代码、二进制身份和报告。`scenarios.py prepare DIR`生成固定场景请求，`scenarios.py score INPUT_DIR REPORT OUTPUT`做任务判分。`score.py`提供同历史概率诊断；这些社区权重对照不替代BF16/FP8质量评估。

`profile.sh MODEL REQUESTS NEW_OUTPUT`采集Nsight节点trace；`profile_summary.py`校验manifest映射。四输出profile不是正式TPS验收。
