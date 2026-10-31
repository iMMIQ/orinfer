# 工程约定

本文件适用于整个仓库；用户明确指令优先。

## 范围

- 推理基线为 Jetson AGX Orin 64GB，aarch64/CUDA SM87；共享运行时同时支持 Altus A2 / Ascend 910B 的原生资源、异步传输、W4A8 grouped matmul、ACL Graph 和 HCCL 集合通信验收，DSv4.1 原始 W4A8 完整 40 层通过实验性原生 provider 接入公共请求接口、HTTP/SSE、连续调度及 prefix cache，保留原生 bench 入口。CUDA 数值、Graph 和性能验收仍在 Orin 上执行。
- 支持 Qwen3.8-27B、Qwen3.5-4B（Qwen3_5）、Qwen3.8-Flash-Next（qwen4_exp）和 Index-Translate-2B/9B/35B（Qwen3_5）的文本、图片和多图输入。
- Qwen3-ASR-1.7B 使用 Qwen3 全注意力解码器、标准 RMSNorm、128 维全 RoPE 和无输出 gate；复用 Qwen 动态行执行流程。文本采用两级 E8P Q4A8，前 4 层 Q/K/V 与 FFN 保留同一 signed-block128 输入基下的 W8，attention 输出投影保留 W8；音频编码器和 projector 保留 FP16，KV 使用 group-64 INT8。Rust 负责音频解码、16 kHz 重采样和 128-bin log-mel，TileLang 执行有界 CNN 与 104-token 窗口编码器。音频特征、位置和输入身份属于请求私有状态及 prefix 缓存身份。
- ASR 的 Q/K/V 与 gate/up 通过无损拼接权重输出行共享整数投影模板。音频 CNN 使用 NHWC；前处理按帧组并行，多个录音按 104-token 窗口边界打包编码，再写回各 owner 的特征。逐窗口有效长度保持录音边界；GPU 验收覆盖改变输入、反向准入、空闲 arena 复用、prefix 恢复及完整请求。
- ASR 文件前处理复用 Symphonia、Rubato 固定比例 FFT 重采样和启用 NEON 的 RealFFT；checkpoint 专用窗函数、mel 滤波与归一化保持参考语义。重采样改动检查时长、时间对齐、通带和抗混叠，并以真实录音及固定历史概率验收。
- ASR 按本轮选中的文本 packet 准备录音，音频编码单独计入启动成本；特征通过 owner 地址写回，保留统一 stream 顺序和范围校验。吞吐策略的冷请求准入以模型 prefill packet 容量为预算。空闲 arena 选择在连续内存估算与租用之间复用，租用、释放、重置和回收使选择缓存失效。
- Qwen 公共动态行执行器将普通 decode 的 token 输入、prepare、embedding、媒体覆盖、位置推进、MTP hidden 捕获、最终 norm 和输出分发作为地址表 packet 执行；混合普通 packet 中的 decode 前缀复用同一边界。离线通过 independent_rows 包装原始一行算术，层状态表后追加固定顺序的边界列，Rust 与 Python 共享列序契约并验证反向 owner、batch 变化、取消复用与 MTP 衔接。完整 logits 与选中 token 保留各请求的输出语义。
- 在线运行时用 Rust，生产 GPU kernel 以 TileLang 为源；Python 只用于离线转换、编译、参考和验证。A2 已有 ACLNN W4A8 grouped matmul 保留为接通模型的基线；新增 A2 算子用 TileLang-Ascend 离线编译，由 Rust 的指针表 C ABI 加载。AOT 库内描述必须与外部 dtype、shape、访问方式和编译身份一致，Graph 和完成事件持有全部原生资源。
- DSv4.1 A2 的 W4 bank、scale 和 assist 在离线阶段准备并落盘；在线直接读取物理布局，按计划校验形状和字节范围，不执行 W4 转置、全量参考对照或权重 payload 哈希扫描。
- DSv4.1 A2 多请求共享权重和聚合 token 行，KV、C2 环及解码 KV workspace 按 owner 隔离，私有状态复用公共对齐 arena 布局。Graph 准备关闭私有状态写入，缓存按实际显存预留回收；捕获前统一探测各 rank 缓存，任一 rank 未命中时全体失效同一形状并捕获，保持 HCCL 序列一致。普通推理使用默认高性能通信；`HCCL_DETERMINISTIC=strict` 作为显式复现选项。请求隔离和固定历史概率误差独立验收，不要求默认通信下跨 batch 逐位相同。TP8 为当前接通模型的方案，后续按实际显存、通信和完整请求性能选择并行策略。原生 batch 包默认 2048 行与每 owner 4096 上下文，准入同时校验 owner 数及显存预算。
- 模型实现、公共执行组件、TileLang kernel 和离线构建工具在统一 workspace 维护，共享架构与计算策略组件。orinfer-models 分别注册模型适配器与计算策略，部署包内 Rust cdylib 通过版本化 C ABI 生成执行程序、动态 batch 和状态绑定，匹配的 TileLang cubin 与库同包发布。公共 CUDA 执行器从模型契约读取模型族、层数、维度和层序；新增架构通过适配器注册接入。
- 第一阶段包含 prefix cache 与多请求 batching；缓存恢复和 batch 调度必须保持完整 KV/GDN/卷积/MTP 状态及请求隔离。
- 项目许可证为 LGPL-3.0-or-later，保留第三方版权和许可证声明。

## 实现

- 按真实 checkpoint 配置核对 GDN、GQA、RoPE、zero-centered RMSNorm、embedding 和 head 语义。
- GDN 持续状态为 FP32；KV、卷积、GDN 和位置状态均属于请求私有状态。
- Index-Translate-2B/9B 复用 27B 的动态 GDN/GQA、FFN、视觉、MTP 和请求状态组件，按配置传递头数与维度，共用模型执行流程。主干、embedding/共享输出头和视觉大矩阵使用两级 E8P Q4A8，敏感输出/MTP 投影保留 W8；2B 的 tied embedding 与输出头共享同一份 K128 压缩 bank；9B 按官方配置分别保留 embedding 与输出头 bank，embedding 按需解码并逆旋转选中行，输出头使用旋转 A8 输入，大权重保持单份压缩表示。
- Qwen3.5-4B 复用相同动态行与请求状态组件，按官方配置使用 2560 hidden、9216 FFN、32 层、32 个 GDN value heads 和 16/4 个 GQA heads；tied embedding/head 共享 E8P Q4 bank。缺少 generation_config.json 时，构建器从本模型配置和 tokenizer 固定 EOS 与 padding；显式提供的生成配置保持原始字节。
- 改变 prefill/decode、恢复或分支行为时校验 KV、GDN、卷积、位置和 MTP 等全部状态。
- KV 增长契约可以同时引用请求私有 KV 和按上下文增长的共享 workspace；私有映射按 owner 查找，共享映射从会话查找，逐项按声明的作用域解析增长列表。
- 单份常驻 W4；scale、zero、LUT、padding 和持久重复表示计入权重平均 bits。展开 W8 的存活期限定为临时计算。
- Flash Next 扩展采用 Q2A8 和质量优先的混合精度；其 QSA 长上下文 KV 使用 group-64 对称 INT8 及 FP16 scale，压缩索引与未完成块也纳入请求私有状态；Q2 专家直接索引固定地址的压缩权重库，默认全量常驻；`--resident-experts N` 使用模型包声明的专家 bank 与路由计数，按需从 SSD 加载到 CUDA VMM 页，并共享全局 SLRU。有限模式暂关闭 CUDA Graph，MTP 专家全量常驻；专家数约束常驻容量；实际字节占用另含临时路由集合和页粒度开销，W8 展开限定在临时计算中。HC 保留 BF16 权重；PLE 表使用受预算限制的 CPU 缓存，其 n-gram 历史与卷积历史必须纳入请求状态及 prefix 恢复。
- 27B 后续实验与部署使用已重新编码的两级 E8P Q4A8 权重（coefficients=[4,1]、beam8 搜索），本机当前模型目录为 `artifacts/models/qwen3.8-27b-e8p-q4a8-current`，离线量化产物为 `artifacts/quantization/qwen35-e8p-beam8-20261016/converted`。当前权重作为后续唯一部署与回退基线，旧量化和 TCQ 实验保留结果记录。更换权重值而配置和 buffer 契约相同时，可复用匹配执行包，仍需验证质量、状态和真实请求性能。
- Qwen3_5 整数投影每种矩阵几何、dtype/stride 及压缩布局只发布一个 prefill 动态 fallback；常用 1/2/3/4/8/16/32 行可保留同源 AOT，128 行内的非 AOT packet 可由离线测量选择同源有界动态特化，显式传递真实行数、排除固定 AOT 点，各 batch 复用同源计算体，仅发布被选择的配置。离线调优汇总多个长度的代价，为 prefill 选择一个动态配置；候选保留在离线调优阶段。收敛后检查实际发布包，验证尾部、完整请求性能和全部状态。
- 27B 的 prefill 以模型声明的层边界暂停，保留同一动态行块，仅将 Hidden 和 FP32 residual 列为跨层存活 workspace；状态增长与最终提交复用公共 Frame 流程。普通 decode 的 1–128 路私有 GDN/attention 复用单行 TileLang 计算体的地址表包装。普通、验证和混合 packet 复用共享 target 前后置计算，MixedRows 显式携带角色边界；验证输出全部位置，普通请求输出末位置。批量 draft/commit/warm 沿用公共 MTP 流程，草稿条件、紧凑 GDN 因子、卷积历史、KV 及隐藏环均按 owner 绑定。公共 packet 调用、地址表绑定和 prefill 分层注册与 Flash Next 共用，模型各自声明 HC/PLE/QSA 及层布局。
- Flash Next 采用已确定的混合量化方案：专家保留整数 E8P Q2A8；主干 GDN QKV/Z、QSA Q/gate 和 target/MTP 共享输出头采用 signed-block128 旋转、输出行 scale 的 Q4A8。GDN/QSA 输出投影、共享 FFN 和 MTP 草稿 Q/O 保留 W8；其余精度敏感部分维持原混合精度。原生适配器统一校验和执行该策略。
- Flash Next 使用我们自己的整数 E8P＋旋转量化表示，运行时从自有编码反量化到 INT8 计算。当前优先完成算子和真实端到端推理，允许免校准或小样本校准；原始 BF16 按矩阵或专家分块读取，分块释放。完整校准需单独验证专家覆盖和层间误差；BF16 基线必须直接来自原始浮点权重。
- E8P 可使用无损 K64/K128 压缩权重重排及完整 A8 位平面分解，以两个 INT4 Tensor Core 乘积恢复原 INT32 点积；此过程保持完整 A8 精度，必须保留原 scale 与 FP16 边界，并验证非零码本前提。MTP、并发及大 prefill 配置通过验证后启用该路径，其余配置保留 INT8 路径，读取同一份压缩 bank。
- 在线模型采用 HF 配置/tokenizer 目录和 `cache/model.json` 数据描述；标准分片 safetensors 保留当前 W4 表示，默认在线校验 dtype、shape、物理 layout 和容器边界；权重 payload hash 由 `--verify-weights` 显式开启，`validate-model` 始终完整校验。配置签名和原生 ABI 仍校验；正常加载不计算资产内容 SHA256，前端资源、CPU 资产元数据、原生执行库、kernel 与权重的内容完整性通过独立 `validate-model` 校验。模型数据通过 execution_package digest 固定独立模型执行包，第一阶段采用 INT8 为主、质量优先的混合精度，关键部分可 FP16/FP32；本阶段实现 int8_quality 计算策略和原生格式加载。离线 AOT manifest 只作构建中间产物，在线模型数据描述与执行包均保留 schema 1，统一使用当前原生模型包结构。算子夹具格式独立。
- 服务协议位于 orinfer-api；CLI 只处理命令。CUDA 执行器、加载器、模型适配器、计算策略与生成/视觉/MTP 控制分离；权重、序列状态和 workspace 分作用域管理。
- HTTP 与离线入口并行准备前端和模型；CUDA 模型在所属 worker 创建、使用及销毁，前端失败或 panic 也需等待并正确回收，模型销毁完成后再释放 GPU 锁。只读权重 arena 保持每张量对齐且仅释放单一 owner，padding 计入权重与显存预算；请求状态、KV 和按需专家单独管理。权重读取和 DMA 有界流水化，缓冲区复用及失败清理等待对应传输完成；启动验收同时测进程到 API 就绪、首输出及首请求完成。
- JSONL 离线入口与 HTTP 复用 Chat 前端、模型加载和连续 worker 及状态提交流程；输入有界预处理、完成即写结果，续跑固定输入和模型/参数身份，截断与错误分别统计。
- `serve` 默认 latency、`batch` 默认 throughput 调度。吞吐策略复用动态 packet 和原生 prefill，按吞吐目标连续准入和执行 prefill；物理内存、状态隔离和输出预算保持共同约束，联合路径按实测成本选择。排序使用一次性冻结的时间，比较函数在同一轮保持稳定顺序。
- JSONL 全局规划复用 Chat 前端提取实际 token 前缀，索引与工作区受预处理预算的四分之一约束，每条最多保留 4096 个前缀 token，预算紧张或规划范围外的内容保留流式处理。重排保持物理行号、custom_id、输入身份和续跑语义；缓存提示限定用于快照选择，输入和输出均保持完整。长度估计依据自然完成请求的实际长度，输出预算作为生成上限。异步结果队列保留内存计费，返回成功或中断结果前须排空并确认写入/同步；错误必须传播。
- 主机输入通过有界、只追加的 pinned slab 按 stream 顺序上传，大输入同样分块，超过单 slab 容量时继续沿用 stream 级异步传输。调用者的内存可在上传方法返回后释放；slab 复用和退出清理必须等待对应 DMA 完成，读回结果仍须确认 GPU 完成。
- 离线构建将 HF tokenizer_config 的 added_tokens_decoder 合并到部署 tokenizer.json，逐项验证 ID 一致。验收比较 Transformers 实际生成的 token ID；thinking/工具标记必须按原始特殊 token 映射切分。
- 执行包 catalog 仅保存共享编译定义/资产及动态 launch 模板；逐层 tensor 角色与参数映射以共享调用模板、逐层参数表和去重特化表离线编译进 Rust 模型库；生成模块共享特化绑定，经 orinfer_model_plans_v1 的有 owner C 数组导出，在线直接消费原生计划。目录身份与原生绑定注册表必须匹配，重建后校验完整程序/参数及全部请求状态。plan-model 保留构建契约以支持离线重建；在线导出范围限定为最终程序、局部特化和专家常驻契约引用的绑定，融合前的输入留在离线构建契约中。
- Rust 按实际导出 host ABI 绑定参数、grid/block/shared；检查 CUDA 返回值、unsafe 边界、资源生存期和 stream 顺序。
- 动态 launch 全行数验证放在离线构建使用的 plan-model 和 validate-model，同容量及表达式去重；在线加载保留逐绑定 ABI 校验，实际 launch 保留真实行数与算术检查，行数穷举限定在离线验证。
- Flash Next 普通 decode（1–128 请求）、混合 batch 和 MTP 验证共享 1–1024 token 行的 target 执行模板；单／多流草稿和 warm 共享 1–512 行 draft 模板，输出头复用相同实现。因果段、接受前缀因子、捕获、提交和请求隔离独立处理，投影／HC／FFN／专家复用同一计算流程。整数投影和专家按 1/2/3/4/8/16/32 离线特化，其余行数使用动态参数 cubin；TileLang 自动调优只选择保持 scale、dtype 与归约边界的合法 tile，路由与专家 tile 必须一致。运行时消费编译生成的行数选择表，单流也使用相同模板；辅助算子统一使用动态模板，真实请求直接组成混合 packet；离线中间绑定只用于构建最终模板，MTP 短尾也走一行 packet，单请求 target 程序仅保留状态准备／参考用途。Graph 分支按模板槽位注册，全部特化验证读写依赖后复用；batch 档位与配置由离线调优结果生成；缓存按源码、编译器和设备身份失效。
- CUDA graph 需要稳定地址和预分配 workspace；改输入后 replay 验证实际执行。
- 共享片段与完整 packet 复用同一套 Graph 分支捕获；片段只启用完整且无歧义的 fork/join 区间。动态槽位展开的连续辅助 kernel 属于同一边界，重复私有状态槽位的处理须保持其他共享分支可用；优先级按实际 launch grid 判断。模型声明支持 batch 验证时，单请求 MTP 也复用 Verification packet。
- Flash Next 因果段与 MTP 状态控制使用动态长度 kernel：target 段及 draft 最大 512 行，验证与接受前缀恢复最大 8 行；运行时显式传递真实段长。GDN head-major 因子按真实行数定 stride，卷积/PLE 历史与隐藏环使用发布 buffer 的物理几何。协议长度、调度分块和轻量程序别名共用动态 cubin。验证/恢复/捕获必须沿动态行 launch 传递段长，并保持每个请求的 owner 及状态视图。
- Flash Next 最终包通过原生 build.py 和共享 row_selection.py 离线调优，按测量结果生成局部选择表，只发布被选择的实现。混合因果段及 MTP warm 支持连续 1–512 行；剩余长度与延迟预算决定实际片段。MixedRows 在现有 ABI v1 flags 中显式携带前置 prefill 片段数，以该字段识别 prefill/验证角色；缓存身份包含该角色边界。warm 用一个 min_tokens=1 的容量模板，MtpLength 必须在映射动态 KV 前更新。最终执行包重打包资产，移除未引用的候选及旧 cubin。共享大行数计算通过 row_tuning.py 与 row_convergence.py 按测量结果发布；专家路由与上下投影共同选择。HC injection 的直接与矩阵 tile 统一 BF16 MMA K16 累加顺序，单行参考也保持相同契约；候选必须保持相同的 MMA K16 累加与舍入契约。动态绑定合并只改身份，保留原始 grid、实参表达式和 token 到专家行倍率。动态 MTP 在映射 KV 之前写入真实验证长度。视觉使用一个动态 patch 模板，merger launch 按实际 patch 数除以 4；提高视觉容量时须保持现有文本程序的行数契约。MTP 验证、恢复、捕获各用一个容量 8、min_tokens=2 的入口，运行时传递真实长度，固定形状参考包仅作离线构建输入。
- 跨 block 同步使用显式声明的 cooperative launch，并校验 occupancy。
- 性能验收使用真实模型的完整请求 TPS。合成探针、算子微测和四输出 profile 用于定位局部成本。
- DSv4.1 A2 的最终性能目标以 Mini-Weka agent 端到端回放衡量，先接通 orinfer 的 HTTP、长上下文和完整 prefix 状态。只运行用户授权的短测，调优阶段不执行一小时回放；保留固定输入／输出长度、同源主 agent 片段和每轮瓶颈证据，参考 vLLM-Ascend、SGLang 与 DeepInfer 后实现优化、验证、提交并推送。
- 质量对照官方 BF16 或同模型 FP8，固定 seed20261002 及 teacher-forced 历史；质量结合任务结果、target NLL、token 概率和长上下文行为判断，token 一致率作为诊断指标。

## 开发

- 文档直接说明功能、用法、条件和验证范围，优先使用清晰的肯定句；技术限制按准确的支持范围和错误行为表述。

- 先用 rg 检查调用者和依赖；保持修改小而完整。subagent 仅在用户或适用指令明确要求时启动。
- 每轮性能优化采用分阶段检查：候选阶段只做相关算子数值和 Graph 验证、2K/2049、一个代表性 batch 及固定历史 decode 短测；契约未变时复用已通过的验证。确认收益后运行一次 240 秒 Mini，与冻结基线比较；仅在波动或疑似超过 10% 回退时补充配对复测。提交前统一运行一次 `make check`；状态、prefix 和隔离检查限定于实际改动的契约，架构首次验收完成后不随每个 tile 候选重复。禁止过度检查拖慢迭代，不因外部 CPU 活动阻塞测试。
- 模型 packet 通过公共 Rows、视图及地址表绑定组件计算行布局；模型声明 owner 角色候选、计算顺序与状态边界。普通、混合、验证和 draft 复用这些组件。
- 离线编译复用 compile/private_rows，调优复用 tuning 的候选、缓存及 Graph 测量；文件身份、原子元数据和共享 GPU 锁位于 tools/build。共享组件源码计入消费者缓存身份。
- CPU Python 依赖由根目录 `pyproject.toml`/`uv.lock` 管理，CUDA 离线依赖由 `tools/environments/cuda` 中的独立 uv 项目管理；分别用 `make python-env` 和 `make cuda-env` 同步，工具通过 `uv run --locked` 执行。`make check` 执行 Rust 格式/编译/clippy/测试及 CPU 评测检查；CUDA GPU 验证用 `bash tools/operators/run.sh`，直接使用当前物理机的 CUDA 工具链和独立 uv 环境。
- GPU 工作通过 `artifacts/gpu-experiment.lock` 串行，保留其他服务，并保持全局频率、功耗和缓存设置。
- GPU 单测使用本轮构建输出的测试可执行文件，核对测试确实执行，并从本轮构建结果解析实际可执行文件路径。
- 当前原生 Flash Next 的整数算子重建使用 `tools/model/flash_next/build.py`：从权重几何和 host ABI 读取构造契约，真实权重调优，编译对象按内容校验并原子缓存，最终模型经 Rust 计划检查后原子发布。其输入为模型固定的执行契约；HC/QSA/状态及专家联合选择表保留输入已验证实现，专家真实几何另行验证。正式重建统一使用该入口。
- 可变的 HC 无损指数、QSA radix shift 使用 ABI 标量复用 cubin；私有状态包装器必须保留标量、列 stride 及字节偏移。QSA 条带数离线测量；动态整数投影调优覆盖多个实际行数及容量上限。选择区间仅在连续相邻且完整参数/launch 契约一致时合并，保持专家路由 tile。
- 新 kernel 验证数值、非对齐尾部和实际 graph replay；状态改动另测分块、恢复和请求隔离。
- 投影组件按权重读取、计算调度、输出处理组合：W8/Q4 及融合 SwiGLU 复用 integer_core，BF16 矩阵复用 floating_core，E8P 的位平面、双 INT4 乘积和输出复用 e8p_core。这些编译期组件供各模型的 prefill/decode 和 MTP 共享计算循环；访存调度、浮点归约和舍入契约显式保留。27B 两级 E8P Q4A8 同样复用 e8p_core，直接 INT4 与 INT8 回退读取同一份 K128 压缩权重。调优缓存必须包含内联组件源码身份。
- Flash Next 适配器统一使用当前动态 prefill、target/draft 行模板和因果状态契约。Q4 前置变换按 section、激活/scale 工作区及符号表关联消费者，QKV/Z 共享变换必须先于首个消费者，并对每个程序别名单独生成；按实际依赖关系生成执行顺序。
- 源码库保留实现、必要构建工具、测试、示例与使用文档。研究过程、agent 交付记录、历史实验报告和本机环境证据保存在忽略的 artifacts 目录。
- 权重、二进制、cubin、cache、日志及 trace 保留在忽略的 artifacts/target 目录；密钥和 token 通过仓库外的环境变量或凭据管理提供。
- 部署使用 Rust CLI 的 `package install` 安装已编译的执行包；新引擎、执行包和源码发行归档统一使用 `.tar.xz`，安装器仍接受已发布的 `.tar.gz`。Python 只用于离线构建、归档和发布校验。
