# Qwen3.5 Dense 原生 Q4A8 构建

当前基线是官方 27B 的 beam8 重新编码、两级 E8P Q4A8 量化目录。它使用 signed-block128 旋转、两级固定整数码表和输出行 scale；Embedding、mixer 输出投影和 MTP 草稿矩阵保留 W8，敏感小权重及视觉保留浮点。本机量化产物为 `artifacts/quantization/qwen35-e8p-beam8-20261016/converted`，完整模型为 `artifacts/models/qwen3.8-27b-e8p-q4a8-current`。

同一构建入口也支持 Index-Translate-2B 的 24 层和 Index-Translate-9B 的 32 层架构，使用 `python/orinfer_build/quantization/qwen_e8p.py` 从固定 BF16 来源编码。GDN/GQA、FFN、视觉、MTP、动态行及状态控制按配置复用同一执行流程。9B 使用独立的 embedding 和输出头 E8P Q4 bank。2B 的 embedding 和输出头共享一份 E8P Q4 bank：embedding 按需解码并逆旋转所选行，输出头复用相同压缩权重和旋转 A8 投影。视觉大矩阵也使用相同 Q4A8 投影，bias/GELU 保留原 FP16 边界；patch、位置表、norm 等小参数保留浮点，敏感输出/MTP 矩阵保留 W8。主干使用相同的两级 E8P Q4A8 契约，实际平均位数包含这些高精度例外。

两级 E8P-Q4 的数据契约为 `quant_method=orinfer_e8p_q4_int8`、`coefficients=[4,1]`、`basis_tensor=orinfer.e8p_q4.basis`、`layout=u32-low-coarse-high-residual`。输入压缩矩阵为 U32 `[N,K/8]`，构建时无损重排为 `[K/128,N,16]`，每个输出行一个 F16 scale。它复用原生执行计划、公共整数计算和离线 tile 调优。非零 basis 经验证后，直接 INT4 MMA 将完整 A8 拆成两个位平面，恢复相同的 INT32 点积；与 INT8 解码路径读取同一份压缩权重，由自动调优选择。

Qwen3.5-4B 同样复用此入口，按官方配置使用 2560 hidden、9216 FFN、32 层和共享 embedding/输出头。原始 checkpoint 缺少 `generation_config.json` 时，构建器从本模型配置和 tokenizer 固定 EOS 与 padding，覆盖模型结束和 chat 结束标记；显式提供的生成配置保留原始字节。

```bash
make build
bash tools/environments/cuda/run.sh python/orinfer_build/models/qwen35/build.py artifacts/qwen35-build \
  --source /path/to/converted --model-output artifacts/models/qwen35-q4a8 \
  --gpu-lock-held
```

构建只接受完整量化产物，按校验身份读取已有 safetensors 分片；两级 E8P 无损重排压缩码字，其余张量保持原始字节。构建复用已有量化值，展开 W8 限定用于临时计算。必要的浮点布局转换与常量放在一个额外分片中。默认上下文 262144，可通过 `--context` 降低。

发布目录结构与 Flash Next 相同：HF 配置/tokenizer、`cache/model.json`、标准权重分片，以及由 digest 固定的 schema-1 执行包。执行包包括 Rust 模型库和合并去重的 cubin/host ABI；层序、状态提交和 batching 逻辑由 Rust 模型适配器注册。

构建时将 HF `tokenizer_config.json` 的 added-token 配置合并进部署用 `tokenizer.json`，验证每个 token 的 ID 与原配置一致。chat template 优先读取独立文件，否则从 `tokenizer_config.json` 提取文本模板，保持字节内容，在编译前校验并冻结。独立 Rust tokenizer 因此与 Transformers 识别相同的 thinking、工具及视觉标记，在线由 Rust 完成分词。

直接 INT4 路径将每级 E8P 整数写成 `E=2n+1`，将 A8 写成 `A=lo+16hi`；分别累加两个位平面的整数乘积后，加回 `(a+b)·sum(A)`，再按原顺序乘权重 scale 和 token scale。它保留 A8 的全部信息。压缩权重以 K128 流水读取，CTA 内可用 1/2/4 组 warp 分摊 K 维，先在 shared memory 中合并 INT32 部分和，再执行一次原有 rescale，归约全部在 CTA 内完成。合法的 split-K 与 N tile 组合由同一个离线调优器按几何和实测成本选择，各请求场景共用结果。

Prefill、普通 decode 和 MTP target 验证共享行计算模板。常见行数 1/2/3/4/8/16/32 离线自动调优，其他长度使用动态参数 cubin。动态 prefill 按多个实际行数的总 GPU 时延选出一个配置，再以完整请求验收；decode 回退按各行数的相对代价平衡。两者均验证完整尾部与数值契约。投影读取原有权重并直接写入 QKV 与 gate/up 的输出区间。Gate 与 up 在同一临时 buffer 中分平面存放，各平面的行容量固定为 4096；SwiGLU/旋转通过显式平面偏移读取，短尾与单行均使用相同的固定偏移。请求分别保留 FP32 GDN、卷积历史、INT8 KV 和 MTP 状态；MTP 接受提交使用紧凑更新因子。

Prefill 在模型声明的层边界暂停，保留同一大块的真实行数；跨层存活的 Hidden 和 FP32 residual 由公共运行时保存，请求私有状态由各自的 arena 独立管理。相邻层按剩余调度时间和实测层成本连续执行，耗尽预算才暂停，层组大小随预算动态确定。分层注册、动态行调用和地址表绑定与 Flash Next 共用组件。普通 decode 的 1–128 路私有 GDN／attention 通过 `independent_rows` 包装同一单行计算体，运行时传入真实请求数。

多请求 MTP 使用现有 ABI 的 Verification、Draft、Commit、Warm 和 DraftHead；混合 prefill／verification 通过 MixedRows 显式携带角色边界。共享 target 前后置计算及草稿投影，只分别处理各请求的因果状态；普通请求收集末位置，验证请求收集全部验证位置。接受后恢复 GDN 因子、卷积历史和位置，再批量推进各请求自己的草稿条件。单请求验证复用相同 verification packet，草稿上限由 CLI 配置与运行时成本模型共同约束。

每种矩阵几何、输出 dtype/stride 和压缩布局只发布一个动态投影 fallback。离线调优覆盖几何长度、tile 边界及其相邻行数，以代表性请求的总 GPU 时延选择一个配置；构建阶段比较候选，发布时固定一个动态投影配置。常用 1/2/3/4/8/16/32 行保留同源模板的 AOT 特化。候选执行包仍须通过完整 Rust CLI 性能、概率、全部状态、并发及 MTP/prefix 验收。

E8P 的 INT8 Tensor Core 路径先将压缩 K128 slab 异步读入 CTA 共享内存，与激活预取共用流水；随后按原码表展开并累加。候选 tile 包含较大 M 以复用解码结果，选择同时测量短尾、512 和 4096 行，并以实际总时延决定配置。压缩缓冲限定在 block 内，常驻权重保持单份。GDN 顺序递推的循环展开次数也由离线行数调优选择，验证 FP32 状态和 FP16 输出逐元素一致，发布时保留选中的展开配置。

残差加法、zero-centered RMSNorm、block128 旋转和行 A8 量化融合为一个动态行 kernel，复用独立归一化与旋转组件，保留 FP32 residual 以及两次 FP16 舍入边界。GDN A/B 的 FP16 Tensor Core 投影按实际行数离线选择 tile，保持原 MMA 的 K 累加顺序。量化代码的公共 helper 使用 TileLang macro 保留局部绑定和原输出类型。

宽 FFN 的 SwiGLU、block128 旋转按列 tile 并行，再从局部最大值计算同一个输出行 A8 scale；旋转、舍入及 FP16 边界复用公共实现，保持每个输出行一个 scale。因果注意力使用公共行模板，离线选择 query-head 分组与 split 上限，支持动态拆分与编译期固定为单个 split 的特化。所有配置复用 INT8 KV 读取、FP32 softmax 和输出处理；真实行数与 KV 长度均在运行时传入。KV 读取使用紧凑的线程条带循环以控制指令量；split merge 按输出列保留短向量，使用 FP32 树形归约后再执行原有 FP16 边界和 gate。调优包含非整 tile 的 KV 长度。Prefill、decode 与 MTP 复用同一计算体，Rust 分别绑定每个请求的 KV 和位置。

1–8 行短段为每个 query 使用独立的 attention tile。普通 decode 与 MTP 验证共享一个离线选择的几何、固定 split 数和归约顺序；分段边界由该 query 的因果位置决定。请求数及同段其他 query 的位置保持这一算术契约。较大 prefill 使用原有动态行选择。每个 query 的临时部分结果具有相同的内部布局，普通地址表包装与共享验证模板读取相同的 FP32 中间表示。

对已发布、使用紧凑单行 decode tile 和共享验证工作区的 Qwen 包，可只重建短段 attention 与原生计划：

```bash
bash tools/environments/cuda/run.sh python/orinfer_build/models/qwen35/rebuild_attention.py \
  artifacts/qwen35-attention-rebuild --model /path/to/model \
  --target artifacts/models/qwen35-attention-fixed
```

该入口复用输入包已测量的 decode 几何、权重及大 prefill 实现。独立的卷积 history 更新绑定自动保留为原生构建契约；使用原地 history 更新的包保持其对应计划。包含 owner-parallel verification table 的包使用完整构建入口，以同时重建私有工作区布局。输入几何或 ABI 不匹配时重建会报错，并保留原模型目录。

整数投影保留一个大行数动态 fallback 与常用精确 AOT；128 行内非 AOT packet 的 tile 由离线测量选择，有界动态特化按真实行数执行。选择区间排除固定 AOT 点，复用同一投影计算体，部署包保留选中的实现。

上述流水线通过 `python/orinfer_build/tuning/row_tuning.py` 离线选择 tile，复用 Flash Next 的相邻选择区间合并规则；执行包只包含被选中的候选。各 batch 与上下文长度复用动态 cubin，运行时显式传入真实行数与 KV 长度。共享 workspace 保存临时旋转区和有界注意力部分结果；权重保持压缩表示。

图片和多图使用共享 ViT 算子，一个动态 patch 模板覆盖 4–32768 patches，merger 按实际 patch 数除以 4。量化、执行和视觉均按真实 checkpoint 语义校验。

两级 E8P 的整数、尾部和 Graph 检查可运行 `python/orinfer_build/quantization/check_e8p_q4.py` 与 `python/orinfer_build/quantization/check_e8p_direct.py`（通过同一个 GPU 包装器）。

卷积的 1/2/4/8/16 时间 tile 由离线测量选择，共用八个 K 部分和、十六 lane 归约及原有 FP16 边界；MTP、prefill 短尾和大块使用同一计算体，配置由离线测量选择。普通 decode 的私有卷积在缓存旧尾部后原地提交历史；批量请求按地址表分别完成计算与历史提交。E8P 的符号与 parity 合并为最终 XOR，保持全部码字和 INT32 点积一致。

算子检查：

```bash
bash tools/environments/cuda/run.sh python/orinfer_build/models/qwen35/check_kernels.py artifacts/qwen35-kernels \
  --gpu-lock-held
bash tools/environments/cuda/run.sh python/orinfer_build/models/qwen35/check_rows.py artifacts/qwen35-rows
bash tools/environments/cuda/run.sh python/orinfer_build/models/qwen35/check_decode.py artifacts/qwen35-decode
bash tools/environments/cuda/run.sh python/orinfer_build/models/qwen35/check_causal_attention.py artifacts/qwen35-causal-attention
```

构建缓存默认位于 `artifacts/build-cache/qwen35`。失败可用包装器 `--resume` 重跑，已校验编译对象会复用。算子检查验证局部数值与 ABI；新执行包还须完成真实模型加载、生成、prefix 恢复、并发和 MTP 验收。

Qwen3-ASR-1.7B 使用相同构建器并传入 `--context 65536`。它复用文本投影、动态 GQA、FFN 和请求状态，音频部分由 `qwen_asr/audio.py` 绑定共享 FP16 encoder 组件。其 RMSNorm、完整 RoPE 和 ungated attention 按配置传递，音频前端与转写用法见[语音识别](../../../docs/audio.md)。
