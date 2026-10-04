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

算子验证使用`validate_mtp_kernels.py`，覆盖实际权重布局、因果注意力、GDN恢复、hidden捕获和改变输入后的graph replay。完整生成验证使用Rust ignored test `validate_mtp_generation`（环境变量`ORIN_MTP_FIXTURE`指向含model/output/cases/repetitions/eos及可选cuda_graph的JSON）。case可携带sampling、images、stop_after；未指定sampling时使用无惩罚greedy。先用API ignored test `export_mtp_chat_fixture`按原生Chat模板导出请求（`ORIN_MTP_FIXTURE_SPEC`），再在独占GPU锁下执行验证。greedy比较关闭MTP时的真实输出；随机采样验证固定seed复现，并把已提交的输出逐token重放到普通decode，对比全部有效KV、GDN和卷积状态。测试还覆盖取消后的请求隔离。计时只统计实际交付的token，拒绝草稿不计入TPS。

已发布的早期greedy MTP模型可离线更新绑定，不改权重payload：`python3 tools/model/upgrade_mtp.py /path/to/prepared-model /path/to/new-model`。工具复用原算子包的视觉embedding和MRoPE导出ABI，添加移位feature index及完整验证logits的绑定，并用Rust加载器验证新目录后原子发布。在线加载器仅接受当前数据契约。

`run.sh MODEL REQUESTS NEW_OUTPUT`运行Rust完整模型并记录机器状态、源代码、二进制身份和报告。`scenarios.py prepare DIR`生成固定场景请求，`scenarios.py score INPUT_DIR REPORT OUTPUT`做任务判分。`score.py`提供同历史概率诊断；这些社区权重对照不替代BF16/FP8质量评估。

`profile.sh MODEL REQUESTS NEW_OUTPUT`采集Nsight节点trace；`profile_summary.py`校验manifest映射。四输出profile不是正式TPS验收。

## Safetensors模型目录

离线编译、融合、视觉与MTP组装全部完成后，使用`prepare.py`发布新的模型目录。普通文本/视觉模型也使用同一入口；把`--model`替换为最终中间产物。无需GPU或PyTorch，依赖Python的safetensors库及已构建的Rust CLI（`make build`）；打包前用CPU `plan-model`逐项校验注册计划与离线原计划一致。

```bash
python3 tools/model/prepare.py \
  --model artifacts/model/rebuild-final/model.json \
  --checkpoint /path/to/checkpoint-dir --output artifacts/models/qwen3.8-27b
./target/release/orin-llm validate-model artifacts/models/qwen3.8-27b
./target/release/orin-llm run-model artifacts/models/qwen3.8-27b examples/requests.json
```

工具检查checkpoint词表、所有源payload与构建资产的hash；不重新量化、不重编译kernel，包含只读权重以及RoPE/索引等可写buffer的初始值。默认每片约1 GiB，`--shard-mib`可调整；单个tensor不会拆分，转换内存由最大分片决定。写完并用标准safetensors reader校验后，才原子发布完整目录；已有输出不覆盖，失败时删除本次临时目录。

目录根保留checkpoint配置、generation config、tokenizer、chat template和图片预处理配置；不复制原始checkpoint的大权重。内部`cache/weights/`保存带dtype/shape和物理layout元数据的safetensors及HF分片索引；`cache/model.json`引用tensor名字、payload SHA256及算子包digest，并声明weight/sequence/workspace作用域。`cache/operators/<digest>/`包含独立的`package.json`、cubin、源码和ABI；包不含权重payload或执行程序，执行顺序由Rust架构模块生成。文件在加载期间必须保持不变。原始checkpoint与此物理布局缓存用途不同，不能用Transformers直接执行缓存tensor。

在线Rust不提供旧模型格式兼容分支，也不在首次请求中执行Python或量化。后续重新编译或改变布局时，完成离线组装后发布新的目录。GPU校验fixture的`model`字段可指向模型目录或新数据描述文件；算子测试夹具仍采用独立的原格式。

## 独立算子包

### Decode INT8 FFN

已采用`u4_warp_n64_k128_mma_i8`布局的批处理模型，可离线添加质量优先的INT8 FFN包：

```bash
bash tools/operators/run.sh tools/model/optimize_int8_decode.py artifacts/operators/int8-decode \
  --model /path/to/prepared-batch-model --model-output /path/to/int8-model
```

GateUp与Down直接从原W4解包到寄存器，通过INT8 Tensor Core逐128通道group累积，再使用原scale进行FP32缩放；不增加常驻W8副本，也不重新量化权重。Post RMSNorm融合每token A8量化，SwiGLU保留FP16激活边界并融合每128通道A8量化，Down使用FP32 split-K及原归并。其它投影、norm、head和持续状态沿用原精度；这不是所有算子强制INT8。27B新增Down scale workspace为34 KiB。

1/2/4/8行使用固定行数kernel，较大batch使用动态行数kernel；32行及以上复用W4 tile计算两个MMA行块。短prefill、混合prefill和MTP主模型验证使用同一FFN策略，大块prefill沿用原包。执行计划仍由Rust注册，CLI和在线加载流程不变。工具验证布局、源包digest和生成后的计划，向新的目录原子发布；输出目录不可覆盖。

`validate_int8_decode.py`使用真实权重、独立FP32参考、尾部保护及改变输入后的Graph replay验证kernel。可用ignored test `capture_decode_projections`（`ORIN_BATCH_FIXTURE`含model/output/cases）导出原路径的真实输入，再传入`--activations`；随机输入只检查实现。Down验证需使用`--families Down --modes group --group-activation`。`--dynamic-rows --tile-m 32 --tile-n 128`覆盖动态分支。验证报告与当前W4路径比较，用于判断新增计算误差；BF16/FP8量化质量需单独验收。完整模型必须另测连续请求、MTP已提交历史状态与实际吞吐。

### 连续批处理

已有批处理算子包可加入GPU历史惩罚greedy；`--specialize-projections`同时加入27B的B4/B8 GateUp及B2/B4/B8 Down固定行数kernel。其他行数仍由动态kernel处理，权重payload保持不变，输出是独立的新模型目录：

```bash
make build
bash tools/operators/run.sh tools/model/optimize_decode.py artifacts/operators/decode-optimized \
  --model /path/to/prepared-batch-model --model-output /path/to/new-model \
  --specialize-projections
```

`validate_greedy_sampling.py`验证FP64历史惩罚、并列排序、非有限值和改输入的Graph replay；`validate_batch_projection.py --static-rows`用真实权重对照动态行数投影，检查逐元素输出、非对齐尾部和Graph。完整模型仍需执行`validate_continuous_requests`并测试实际API并发，不能以kernel微测代替吞吐验收。

含原生MTP、直接INT8 KV以及当前27B投影布局的prepared模型，可离线加入批处理算子与32/64/128-token混合prefill计划：

```bash
make build
bash tools/operators/run.sh tools/model/upgrade_batching.py artifacts/batch-build \
  --model /path/to/prepared-model --model-output /path/to/batch-model
target/release/orin-llm serve /path/to/batch-model \
  --max-active-requests 32 --max-batch-tokens 128 --prefill-budget-ms 200
```

工具编译动态行数TileLang投影，按实际host ABI生成2/4/8/16/32/64/128绑定；相同不可变权重与原kernel资产使用hardlink，模型描述和新包独立发布。源码目录中不包含这些二进制资产。Rust先验证注册计划，完成后原子发布新的模型目录；已有目录不覆盖，失败时清理本次临时目录。在线请求不触发编译或量化。

每请求状态驻留独立GPU地址，GDN验证前缀和统计临时量由执行线程共享；混合计划只合并无状态投影，因果attention和FP32 GDN逐段执行，padding不写入请求状态。Graph按有序槽位/段长缓存，地址保持稳定，最多16个batch捕获；新成员组合首次出现需捕获。执行计划由架构模块生成，算子包不提供用户程序。

`orin_engine::model::Model`提供`start_request`、`advance_requests`、`finish_request`，API worker负责队列与输出解析；退出或取消必须调用`finish_request`释放槽位。GPU ignored test `validate_continuous_requests`使用`ORIN_BATCH_FIXTURE`（model/output/cases/cuda_graph，cases包含原生token IDs及sampling），验证批处理输出、全部私有状态的请求隔离、取消/复用、相同历史下的概率/top-3及固定seed重排。这里的概率参考是同权重串行执行，用于检验重构；不会替代BF16/FP8量化质量评测。

已有schema-2 safetensors缓存可离线拆分为新的目录；这是一次性构建工具，在线加载器不读取旧manifest。相同文件系统上的不可变权重与kernel资产通过hardlink复用，避免额外复制整套权重；配置和描述文件独立复制。相关目录在使用期间必须保持不变。

```bash
make build
python3 tools/model/package.py split \
  --model /path/to/old-prepared-model --output /path/to/new-model
./target/release/orin-llm plan-model /path/to/new-model
```

算子包以内容hash命名，支持tar.gz归档和离线安装。安装器检查归档路径、package digest及全部kernel资产hash，校验完成后原子发布缓存；运行时再次检查ABI、配置、buffer布局与资产身份。包契约与权重payload身份分离，同配置和布局的不同checkpoint可以复用包。

```bash
python3 tools/model/package.py archive \
  /path/to/new-model/cache/operators/PACKAGE_DIGEST operators.tar.gz
python3 tools/model/package.py install operators.tar.gz ~/.cache/orin-llm/operators
./target/release/orin-llm serve /path/to/new-model
```

可用`ORIN_OPERATOR_CACHE`指定共享缓存位置。只有一种`int8_quality`策略，暂不提供compute-dtype切换。配置或构建变体不受当前包/架构recipe支持时，在准备或加载阶段报错；不在首次请求中编译或重新量化。

## 扩展已准备模型的上下文

已经包含视觉和MTP的本机Qwen3_5缓存可用以下命令扩容，无需重新量化权重：

```bash
bash tools/operators/run.sh tools/model/resize_context.py artifacts/context-build \
  --model artifacts/models/qwen3.8-27b-uncensored \
  --destination artifacts/models/qwen3.8-27b-uncensored-256k \
  --max-context 262144 --max-prefill-tokens 2048
target/release/orin-llm validate-model artifacts/models/qwen3.8-27b-uncensored-256k
bash tools/operators/run.sh tools/model/context_probe.py artifacts/context-probe --hidden-ring 2048
target/release/orin-llm serve artifacts/models/qwen3.8-27b-uncensored-256k
```

上下文必须按128 tokens对齐，且不超过checkpoint声明的原生容量。工具重新编译容量相关TileLang算子、按实际host ABI重新绑定参数并扩展位置表与KV；学习得到的权重分片使用硬链接。MTP随target prefill分块预热，以最大prefill块大小的hidden环形缓存代替整段hidden存储；主模型与MTP仍保留完整上下文的KV。视觉特征容量单独限制，不随文本扩容。`--max-prefill-tokens`可选择已有的较小profile，缩小文本临时workspace；提示总容量不变。

256k、2048-token最大prefill块配置显式CUDA分配约36.60 GiB，另需driver、graph及CPU内存。扩容影响常驻内存；长提示的attention计算量也随长度增加。`context_probe.py`检查最后一个位置的KV写入、因果边界、KV gather及hidden环形缓存的输入变化后graph replay；完整请求和长上下文质量仍需用实际模型另行验证。


## KV 存储优化

先用 `resize_context.py` 确定容量，再做 KV 优化；已优化目录不能直接用于上下文重编译，需要从原始 prepared 模型重建。已准备的 Qwen3_5 模型可以离线发布为直接读取 KV、按需映射物理内存的模型目录；权重 payload 不变，算子包由新目录独立固定：

```bash
python3 tools/model/optimize_kv.py \
  --model /path/to/prepared-model --destination /path/to/fp16-kv-model

bash tools/operators/run.sh tools/model/optimize_kv.py artifacts/kv-build \
  --model /path/to/prepared-model --destination /path/to/int8-kv-model \
  --storage int8
```

`fp16` 保留无损 KV。`int8` 使用每 token、每 KV head、每 64 个通道一组的对称 INT8，FP16 scale 计入存储；attention 以 packed half2 在共享内存 tile 中反量化后，继续使用 FP16 tensor-core 计算和 FP32 累加；反量化覆盖 INT8 全取值、FP16 次正规 scale 和最大有限值。当前单请求页表必须为 identity 顺序；转换工具校验 safetensors 中的页表及其 hash，省去 prefill 的 KV gather 和整份 FP16 workspace。文本与 MTP KV 都采用所选存储方式，GDN 状态保持 FP32。

CUDA VMM 只保留最大上下文的虚拟地址，按实际执行位置、写入块大小和驱动粒度映射物理内存；增长发生在图外，指针保持稳定。新请求开始时释放上一请求的 KV 映射，重新清零状态；已有 graph 可以继续重放。容量错误和物理内存不足返回错误，不缩短模型上下文。`run-model` 报告中的 `buffer_bytes` 为加载时固定常驻字节，`buffer_capacity_bytes` 为全部逻辑容量，`peak_kv_bytes` 为执行时 KV 映射峰值。

`tools/model/kv_probe.py` 验证 INT8 writer 的 RNE codes、Q/Gate 不变、非连续物理页、尾部和变化输入后的 graph replay。Rust ignored test `vmm_graph_growth_reset_and_last_token` 验证分配粒度、256k 最后一个 token、扩容和回收后的 graph replay；完整模型状态与图片/MTP回归继续使用 `validate_mtp_generation`。量化质量需要额外固定历史对照，不能由微测误差替代。


KV 质量对照可以用 `run-model` 的 `logits_steps` 导出完整分布。先运行 FP16 KV，再将它的 `output_tokens` 的前 `max_new_tokens - 1` 个作为 INT8 请求的 `forced_tokens`，保持输入、历史和采样 seed 一致。`compare_kv_quality.py --reference REF_REPORT --candidate INT8_REPORT --reference-requests REF_REQUESTS --candidate-requests INT8_REQUESTS --output NEW_RESULT` 校验历史，报告完整分布 KL、参考 token NLL 差、top-3 概率误差和重叠率；逐 token 不一致本身不判失败。这项对照隔离 KV 存储误差，权重质量仍应对照 BF16/FP8 并结合任务结果。

INT8 KV 的长文本 prefill 可以使用每层共享的 FP16 临时 workspace，避免每个 query tile 重复反量化历史 KV：

```bash
bash tools/operators/run.sh tools/model/stage_kv_prefill.py artifacts/kv-prefill-build \
  --model /path/to/int8-kv-model --destination /path/to/staged-int8-kv-model
bash tools/operators/run.sh tools/model/kv_prefill_probe.py artifacts/kv-prefill-probe
```

该工具保留 INT8 权重与 KV、decode 和 MTP 算子，只替换 512/2048-token 文本 prefill 的 KV 读取。所有主模型 attention 层顺序复用同一份 K/V workspace，按实际 prefill 位置映射物理内存，并随请求 reset 释放；不保存到模型权重或恢复状态中。每 token 临时容量为 4096 字节，8k 为 32 MiB，256k 上限为 1 GiB；加载时只预留虚拟地址。报告的 `peak_prefill_workspace_bytes` 单独记录临时映射峰值，`peak_kv_bytes` 仍只包含长期 KV。`kv_prefill_probe.py --context 262144 --query-tokens 64` 可验证最大容量、非对齐尾部和真实 graph replay；`ORIN_OPERATOR_SANITIZER=memcheck` 可检查越界。

## Dense prefill 流水线与 MTP warm

已有按需映射INT8 KV及FP16 prefill workspace的Qwen3_5模型，可发布新的SM87算子包：

```bash
bash tools/operators/run.sh tools/model/optimize_prefill.py artifacts/prefill-kernels \
  --model /path/to/prepared-model --destination /path/to/new-model
```

工具保持权重字节和完整dense attention，替换512/2048-token主模型attention为64×32查询/KV tile，使用单阶段CUDA异步复制，并跳过完全位于因果区域内的逐元素mask。KV反量化同时清零最后32-token块中的无效位置，避免异步读取旧值。主模型和MTP顺序复用同一份FP16 scratch。

默认添加64/128/512-token MTP warm计划，`--warm-sizes`可指定不超过主prefill块容量的32倍数；空列表只替换主模型attention。新增warm计划使用按16行分块的W4投影，增加有界临时buffer，不增加权重副本。运行时在主prefill分块间预热草稿状态，只在实际生成需要时计算草稿词表head。

源模型目录不修改；新目录通过完整加载器校验后发布。缓存预算是运行时配置，模型不包含prefix快照。`screen_prefill_attention.py`提供TileLang候选筛选、非对齐尾部及改变输入后的Graph replay验证；所有产物写入指定的新输出目录。

`validate_mtp_warm.py --model /path/to/new-model`通过相同GPU入口运行，使用实际W4权重逐位比较17/64/128/512行投影与16行执行，覆盖FP32 split-K、尾部及改变输入后的Graph replay。`kv_prefill_probe.py --context 262144 --query-tokens 64 --async-stages 1`验证反量化padding与异步attention完整链，可配合`ORIN_OPERATOR_SANITIZER=memcheck`检查越界。
