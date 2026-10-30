# 模型执行包

在线引擎通过版本化 C ABI 加载模型执行逻辑。模型适配器、计算策略、共享执行组件、kernel 和构建工具在同一源码 workspace 维护。部署包包含原生 `.so`、匹配的 cubin 和配置/布局契约；权重仍在模型目录中。部署包从统一 workspace 构建。A8/A4 表示主要计算策略，包同时声明各关键路径的实际混合精度。

核心负责 API、调度、CUDA context/stream、内存、graph 和 prefix 索引；包负责配置校验、模型执行顺序、融合、动态 batch、状态绑定和视觉位置。新增模型族通过包内适配器注册。当前原生库为 `orinfer-models`，注册 Qwen3.5 和 Flash Next 的 `int8_quality` 路径。

```text
MODEL_DIR/
  config.json, tokenizer.json, chat_template.jinja, ...
  cache/model.json
  cache/weights/*.safetensors
  cache/cpu/                         # 可选 CPU 查表数据
  cache/packages/<sha256>/
    package.json
    lib/model.so
    lib/bindings.rs, lib/BUILD.txt    # 生成模块的对应源码与重建方法
    assets.safetensors               # 或独立 cubin/source/host ABI 文件
    LICENSE
```

`model.json` schema 1 的 `execution_package` 固定整个包。共享缓存位于 `~/.cache/orinfer/packages`，可通过 `ORINFER_EXECUTION_CACHE` 覆盖。包 schema 1 / runtime ABI 1 描述目标 SM87、架构与计算策略、支持的配置、buffer 契约、kernel ABI 和 `.so` 身份。包自身通过 `describe` 返回身份及能力，运行时与 manifest 交叉检查。部署包以内容摘要固定；公共 ABI 的兼容性与源码组织分别管理。破坏协议兼容性的变化必须更新 ABI。

`.so` 的运行依赖限定为 `orinfer-model-sdk`。入口 `orinfer_model_v1` 返回 C 函数表，包含 describe/create/batch/visual 和对应资源释放函数。完整 C 声明位于 [`orinfer_model.h`](../crates/orinfer-model-sdk/include/orinfer_model.h)。跨动态库边界统一传递 C 兼容数据。

包内 `catalog` 保存共享资产、编译实现定义和动态行数 launch 模板。定义包含 cubin/source/host ABI 索引、symbol、block、shared memory、cooperative、实际参数类型顺序及只读契约。层号、权重名、逐层参数、grid 及完整执行程序由原生模型库提供。同一 cubin 可对应多个 ABI/launch 契约，定义去重须比较完整契约。

离线编译得到的参数映射编译为 `.so` 内的 Rust 张量角色表、共享调用模板、逐层参数表和去重特化选择表；该映射只记录实现与存储角色，执行顺序和融合仍由注册的 Rust 模型适配器生成。目录与映射分别固定身份，交换实现、参数类型、偏移或动态 launch 后必须重新构建匹配的库。运行时在 Rust 中校验目录身份、buffer dtype/shape/layout 和边界，并直接执行已编译的包。

同类层共享调用模板和特化表，每层仅提供层号、权重、状态、workspace 角色及必要标量。模板实例化严格保留实参顺序、slice 偏移、标量位模式、grid 和动态 launch 契约；计算与精度由显式契约确定。独立 continuation 参数查找直接查询模板。此因子分解在发布时逐条做精确往返校验，适用于两种模型适配器。

Flash Next 的 HC 无损缓存指数和 QSA radix shift 通过已有 ABI 的整数标量绑定，共享相同计算实现；独立请求的 QSA 复用同一私有状态包装器。QSA 条带数离线测量选择。相邻行数区间仅在全部实参、资产及动态 launch 契约相同时合并。整数投影的固定行数和动态行数使用相同模板与候选空间，动态回退在多个实际行数上测量，选择结果随编译器、设备和源码身份失效；发布包仅包含离线选中的实现。

在线创建只导出最终程序、全部局部特化候选和专家常驻契约引用的绑定；融合前的输入保留在离线构建契约中。`plan-model` 和 `validate-model` 通过内部创建参数 `build_bindings` 保留完整构建契约，供离线重建、完整资产校验和诊断。该参数保持 schema 1 与 C ABI 1；运行时的实际逐层执行节点仍需实例化，CUDA Graph replay 直接执行已捕获的程序。

动态 launch 的全行数验证只在离线构建使用的 `plan-model` 和 `validate-model` 执行，相同容量和表达式只验证一次。正常加载检查每个绑定的参数类型和容量 ABI；实际 launch 继续检查真实行数、算术溢出、除零和非零 grid，行数穷举检查限定在离线验证。该行为由引擎实现，现有执行包可直接复用。

Qwen MTP 执行包可声明 `mtp.packet_table`，将验证、提交、草稿、接受后补算和草稿输出中的请求私有操作数放入固定地址表。包装保持原 kernel 的计算体、grid、标量与共享 workspace 字节视图；Rust 每轮更新当前请求地址，完整 packet Graph 按执行阶段和实际执行形状跨请求复用。提交的缓存行数按请求数计，命中仍要求实际 kernel、grid 和标量一致；验证与补算保留各段实际长度、动态参数和混合角色边界。单请求沿用相同 packet；未声明该契约的包继续使用原有执行方式。启用该能力需重新构建匹配的模型数据与执行包，并验证改变输入、替换请求和 Graph replay 后的全部私有状态。

原生 Qwen 构建入口自动生成该地址表。已有 dense Qwen 执行包采用 sequential prefill、dense decode GDN 且尚无地址表时，可通过 `tools/model/qwen35/rebuild_mtp.py --model 原模型目录 --target 新模型目录` 增量构建，保留原权重与矩阵实现。该入口由 `bash tools/operators/run.sh tools/model/qwen35/rebuild_mtp.py 输出目录` 调用，并在末尾传入上述模型参数；发布前验证模型计划与实际 host ABI。

地址表的列序、kernel 映射及标量映射作为共享的不可变描述保存，逐轮复制 MTP 策略时只复制其引用。实际请求地址每轮独立刷新，JSON 格式和 C ABI 保持不变。

启动预加载选择实际 packet 使用的私有 kernel 和拷贝实现，并在权重读取的等待间隙由同一 CUDA worker 完成；已完成的读取优先上传，输入槽复用仍等待对应 DMA。

完整 Graph 缓存保留动态 packet 执行计划的不可变引用。同一计划引用可以快速命中；其他计划仍比较完整调用、grid、标量和字节视图。声明地址表的包按阶段和有序段长复用 owner 无关的计划，新请求只生成地址绑定；混合角色边界属于阶段身份。请求状态绑定分别缓存。prefill、prefix 桥接和生成中的 warm 复用同一 packet 实现。

完整 packet Graph 缓存最多保留 16 个条目，并与共享片段一起遵守 12288 个节点预算；满额时淘汰最久未使用的完整图。阶段切换后，连续执行的新形状逐步替换旧形状，历史累计命中次数不参与淘汰。

创建时用 JSON 传递配置、数据契约与小型编译目录。独立入口 `orinfer_model_plans_v1` 用 C 数组导出生成的 kernel 参数绑定和 kernel/copy/zero 程序；主 `orinfer_model_v1` 表保持 ABI 1。导出接口显式声明表及记录大小，数组、字符串和嵌套参数由同一个 owner 管理并释放。逐层绑定和完整程序通过原生 C 数组直接导出。动态 batch 热路径继续使用 C 数组，一次返回指令、buffer views、动态 launch 参数和请求状态绑定。原生内存由分配方释放，库保持加载直到模型及 CUDA 资源释放完成。模型实例与 GPU 执行线程绑定。

有限专家常驻模式通过创建请求的 `expert_contracts` 选择更完整的原生绑定契约。模型库返回 `expert_banks`：专家数、按专家连续存储的张量，以及消费 kernel 的 I32 路由计数参数位置。核心按实际绑定读取计数，按模型契约声明的几何管理全局 SLRU 和 CUDA VMM 页。默认全量常驻从原生模型库获取计划所需的验证契约。

模型可声明 hash 固定的 `input_assets`，通过独立可选入口 `orinfer_model_inputs_v1` 接收执行程序名、当前 token 和请求历史，返回命名 buffer 的上传数据；原有 ABI v1 函数表保持兼容。核心校验访问权限与上传范围，并在库释放数据前完成上传。Flash Next 在包内完成 CPU W8 embedding／E8P PLE 查表，有预算的行缓存共享，n-gram 历史从请求提供。CPU 资产随模型克隆和执行包更新一起保留。主模型、验证和 shifted draft 输入由模型包区分，架构专属输入分派由模型包注册。

MTP 的验证计划可声明始终提交紧凑 recurrent 更新，也可提供 draft 分支的保存/恢复程序；调度器在提出草稿前保存、刷新已提交前缀前恢复。Flash Next 使用这些通用接口保留 GDN、卷积、PLE、HC 和 QSA pending/index 的完整状态。

CUDA context、stream 和显存统一由引擎管理。包声明执行程序，核心负责执行；CUDA Graph 命中时直接重放已有程序。状态绑定缓存最多 16 个槽位/段长组合，仅缓存逻辑名称，GPU 地址仍从当前请求 arena 解析。prefix 身份包含包 digest；快照复用要求执行策略和状态布局一致。

## 构建与更新

### 设备后端边界

`orinfer-device` 定义后端、主机 ABI、设备架构、Graph 策略、逻辑内存区间及会话归属，不加载设备驱动。`orinfer-engine/src/backend/executor.rs` 是共享运行时的执行入口；CUDA 的 context、stream、VMM、Graph、kernel 和传输实现位于 `backend/cuda/`。生成、embedding 和 rerank 使用同一个入口，设备句柄留在后端内部。

包的 `target` 决定后端和主机 ABI：原有 `sm_87` 对应 aarch64/CUDA，`ascend910b` 对应 aarch64/Ascend，`ascend910b-x86_64` 对应 x86_64/Ascend。安装时静态检查 ELF 与声明的主机匹配；加载原生模型库前检查当前主机。模型架构及 `compute_policy` 保持独立，Ascend 目标本身不表示已有 DeepSeek 或 W4A8 模型实现。

现有 CUDA catalog 保留原字段和 JSON 形式。Ascend 定义使用 `launch: {"backend":"ascend","block_dim":24,"workspace_bytes":8192}`，tiling 数据沿显式参数 ABI 传递。Ascend 定义中 CUDA 的 grid/block/shared/cooperative 必须省略或为零/false；混用后端或几何字段会在校验阶段失败。动态行 launch 和 Graph 的具体实现仍由对应设备后端提供。

prefix 的逻辑区间、状态片段、预算及统计由共享层管理；物理映射和复制由后端执行。快照记录后端、设备和会话代次，恢复、继承和共享映射前拒绝其他会话的 backing，防止同卡重建 worker 后复用失效地址。缓存持久化或跨 worker 迁移需要单独的复制/导入协议。

```bash
orinfer devices --backend cuda
# 先按本机 CANN 安装配置运行库环境：
orinfer devices --backend ascend
# 可显式选择 ACL 库：
ORINFER_ASCEND_LIBRARY=/path/to/libascendcl.so orinfer devices --backend ascend
```

Ascend 探测共享进程内 ACL 初始化，读取卡数与 SoC，不分配设备内存或执行 kernel。最后一个会话、buffer、图和完成事件释放后才结束本进程持有的初始化。输出中的 `api_symbols` 仅表示当前库存在对应入口，芯片能力需要实际验证。`model_execution_available` 表示已注册实验性 DSv4.1 原生 provider；服务范围见下文。现有 CUDA 执行包 schema 1、C ABI v1、`--cuda-graph` 和统计字段保留；`--graph` 是相同参数的别名。

原生设备资源位于 `backend/ascend/runtime.rs`：每会话独占 context、compute/transfer stream，显存与 pinned 内存受独立预算约束；异步提交持有 buffer、主机副本和原生算子资源直到事件完成。跨流等待也持有源事件。完成确认失败时隔离仍可能被 DMA 使用的存储和 context，避免提前释放；释放资源只销毁所属 context，不重置整卡。Ascend 数字错误码进入统一错误分类，执行失败会使 worker 退出而非按普通请求错误继续服务。

Session 缓存最多 64 个 pinned 缓冲，按实际物理容量计入预算。完成事件经过硬等待、缓冲仅由缓存持有时才可复用；需要更大容量时回收空闲项，在途存储继续由完成事件持有。`pinned_bytes()` 包含空闲缓存，`trim_pinned_cache()` 释放空闲容量；原生验收在完成全部工作后清理缓存并检查资源归零。

`backend/ascend/operators.rs` 直接调用 CANN 的 `aclnnGroupedMatmulV5`，使用 ND packed signed INT4、UINT64 channel scales、FP32 assist/token scales 及 BF16 输出，支持空专家分组。准备阶段查询 workspace 并显式启用 repeatable executor，执行阶段复用准备结果；同一 compute stream 上的串行计划可共享 workspace，分支并行需要独立存储。Graph 持有稳定地址和原生计划直至最后一次重放完成。运行时通过动态库加载 CANN，无 PyTorch/ATen 依赖。

```bash
orinfer validate-ascend --device 0
```

此命令有界验证异步上传/读回、W4A8 数值、改变输入后的 executor 复用和 Graph 重放；包含单行、16 行及空专家分组。测试显存预算为 512 MiB，workspace 额外检查至少保留 256 MiB 空闲 HBM。输出的 wall time 包含提交和完成等待，只作诊断，不代表模型性能。已在 910B3 上验证；完整模型、CSA2、prefix 恢复及通信与模型的集成仍需后续接入及验收。

新增 A2 文本主干算子位于 `tools/operators/ascend`，由 TileLang-Ascend 离线编译。它们覆盖 RMSNorm、动态 A8、限幅 SwiGLU、W8/BF16 与 FP32 敏感投影、mHC、复数 RoPE、Engram 旋转与门控、MoE 路由和还原、KV 行搬运、稀疏索引与 attention、C2 池化及 32 行 FP32 ring 提交。索引 TopK 采用分块选择与分层归并，保留 FP32 分数；K 为 512 或 2048。FP32-sensitive 投影和稀疏 attention 当前采用 Vector 正确性基线。

local TopK 的空块使用交错的 `(-inf, -1)` 候选对。`tools.operators.ascend.validate_topk_empty --baseline BASE_PLAN --plan PLAN --output REPORT` 对照发布库的完整候选 buffer，覆盖空／短／完整前缀、并列分数、行尾及改变分数和长度后的 Graph replay；报告另含短、长上下文的算子耗时。

已有原生计划可使用 `tools.model.deepseek_v41.build --rebuild-plan PLAN --rebuild-op topk.local` 只重建相关算子，输出同目录的 `PLAN-selected.json`。`--rebuild-op` 可重复提供，未出现在计划中的算子名会报错；未选中的库和权重保持原引用。重建复用公共编译任务及内容缓存，最终部署仍需发布新的执行包。

当前离线编译镜像面向 aarch64，使用调用者提供的 CANN 9.1 / Python 3.12 运行时基础镜像，固定 TileLang-Ascend wheel 的版本与 SHA256。基础镜像需包含匹配的 PyTorch、torch_npu 和 CANN 编译器。`run.sh` 将所选基础镜像解析为本地不可变镜像 ID；板上运行时串行取得 GPU 实验锁，无卡编译使用 `--without-devices`。运行前先结束占用同一批 NPU 的开发容器；驱动的容器设备归属可能持续到容器退出。编译产物、参考数据和缓存保存在 `artifacts`：

```bash
ORINFER_ASCEND_BASE_IMAGE='your-cann-runtime:9.1-py312' \
  bash tools/operators/ascend/run.sh build
# 先在不映射 NPU 的容器编译，再在相同 cache 中执行板上验收：
bash tools/operators/ascend/run.sh --without-devices python3 -m tools.operators.ascend.validate \
  --compile-only --output artifacts/ascend-runtime/attention-cpu \
  --rows 16 --attention-selected 128 --only sparse_attention
bash tools/operators/ascend/run.sh python3 -m tools.operators.ascend.validate \
  --output artifacts/ascend-runtime/attention-cpu \
  --rows 16 --attention-selected 128 --only sparse_attention
bash tools/operators/ascend/run.sh python3 -m tools.operators.ascend.validate \
  --output artifacts/ascend-runtime/aot --rows 16 --width 5120
# BF16 尾数覆盖与动态 A8 舍入边界：
bash tools/operators/ascend/run.sh python3 -m tools.operators.ascend.validate \
  --output artifacts/ascend-runtime/a8-grid --rows 128 --width 512 --quant-grid --only quantize
# 专家行重排、空路由与异常索引的精确检查：
bash tools/operators/ascend/run.sh python3 -m tools.operators.ascend.validate \
  --output artifacts/ascend-runtime/route --rows 3 --only regroup
flock -w 5 artifacts/gpu-experiment.lock orinfer validate-ascend-kernels \
  --device 7 --fixtures artifacts/ascend-runtime/aot/fixtures.json
```

`backend/ascend/kernels.rs` 验证 AOT 库 hash、主机 ELF、设备型号及库内的版本化算子描述；外部描述与编译库的 operand、dtype、shape、访问方式和参数必须一致。指针表只绑定同一会话的精确 buffer 范围，输出与其他 operand 分离。`Program` 按同一 stream 的依赖顺序连续提交算子，并用一个完成事件持有整段原生资源。Graph 同样保留库、工作区和稳定地址。在线执行不加载 Python 或 Torch。

离线编译显式选择 NPU storage rewrite，编译器身份记录 `storage_rewrite=explicit-npu-v1`。固定版本编译器原本根据容器中的设备可见性选择存储复用，在无卡编译时可能让循环临时 KV 覆盖仍存活的 query/累积状态。显式目标配置使无卡编译和板上编译使用相同的 NPU 存储规则；无卡生成的产物需经过上述板上验收。

AOT 算子描述是独立的验收/编译格式，现有模型包与模型 C ABI 保持原版本。验收比较四组改变输入的结果，其中两组使用 Rust 原生 Graph replay；INT8 输出、行搬运和 ring 提交逐字节对照。FP32 敏感投影与 mHC 点积使用独立的 FP64 参考，再舍入到 FP32，避免 CPU BLAS 的累加顺序影响判断。C2 先读取上次残余再单独提交最后 32 个已接受位置，metadata 明确传递请求页、前驱行、位置和提交边界。完整模型绑定与分布式单流由下述实验入口执行；prefix 快照恢复和 AgentX 并发性能仍需接入。

`backend/ascend/hccl.rs` 提供原生 AllReduce SUM、AllGather、ReduceScatter SUM、变长 AlltoAllV 和异步错误检查。Bootstrap 在指定会话的 context 内生成，opaque root bytes 可传给其他 rank 进程。通信与计算使用同一 compute stream 顺序，完成事件持有 communicator、buffer 和 host count arrays；空 peer 计数和重复调用仍保持所有 rank 的共同 collective 顺序。调用者在 step 边界检查异步通信错误。

```bash
# 与现有服务共卡时，为本次验收选择独立的 NPU 通信端口：
HCCL_NPU_SOCKET_PORT_RANGE=auto orinfer validate-hccl --devices 0,1,2,3,4,5,6,7
```

验证器在同一进程的独立线程/context 上检查 2–8 卡的 AllReduce、AllGather、ReduceScatter 和含空 peer 的非均匀 AlltoAllV，并在全部 rank 完成显存预检查后统一准入。ReduceScatter 使用 FP32／BF16 可精确求和的数据验证连续行分片、重新汇总及改变输入后的 Graph replay；混合尺度浮点数据用于单独评估求和顺序的差异。HCCL 自身的通信缓冲不计入会话显存预算，每卡验收前要求至少 512 MiB 空闲 HBM。已在 2 卡和 8 卡 910B3 上验证。实验入口使用同一进程的独立 worker/context；AllReduce、AllGather 和 ReduceScatter 可与设备计算共同捕获为 Graph；跨进程编排和设备侧 MoE dispatch 仍需接入。A2 上加入 MC² 算子后需重新按对应端口约束验收；本命令只使用独立 collective。

DSv4.1 Engram 的 CPU 路径使用离线导出的 tokenizer 压缩映射、确定的整数 hash 参数和显式 token 历史；prefill 分块、恢复和 speculative 回滚无需读取可变 KV 页镜像。`host_rows` 以稳定分组和去重生成 owner 计划，保留原查询顺序的恢复索引；只读映射 INT8 表与 FP32 group-32 scales，按预算打包选中行，整表不注册到 ACL。实验入口已连接 NPU 投影；NUMA worker 绑定和预取重叠仍需接入。

在离线工具环境安装 NumPy 和模型 tokenizer 对应的 `tokenizers` 后导出元数据；导出器验证 compressed vocabulary 与桶数，元数据固定配置和 tokenizer 的 SHA256：

```bash
python -m tools.model.deepseek_v41.export_engram \
  --model /path/to/DeepSeek-V4.1-Flash-1017-w4a8-Ascend \
  --output artifacts/engram-metadata.json
orinfer validate-engram --metadata artifacts/engram-metadata.json \
  --model-dir /path/to/DeepSeek-V4.1-Flash-1017-w4a8-Ascend
```

CPU 验收读取真实表的小批量行，检查所有 token 分块边界的 hash 一致性、八路 owner 计划、去重和 staging 字节预算；不执行完整模型。

### A2 单流模型实验入口

`backend/ascend/model/` 提供 DSv4.1 Flash-1017 原始 W4A8 checkpoint 的实验性单流执行入口 `bench-ascend`。八个 Rust worker 各持有独立 ACL context，以 TP8 执行 attention、每卡 48 个专家执行 MoE，并通过 HCCL 合并结果。执行计划包含全部 40 层、Engram CPU 行查找、C2/C1 缓存、索引 TopK 与输出头。W4 grouped matmul 保留 ACLNN，其余设备计算从 TileLang AOT 库加载；连续 TileLang、W4 和 HCCL 调用共同捕获为 ACL Graph，按原执行顺序复用同一 W4 workspace。设置 `ORINFER_A2_W4_NO_CAPTURE=1` 可单独提交 W4，供诊断使用。

该入口使用独立的实验计划描述，公共模型包 schema 1 和 C ABI v1 保持原样；公共服务通过下述实验性原生 provider 接入。当前上下文容量限定为 128–16384，须为 32 的倍数；默认 8448，覆盖 8K 输入和 128 个输出。此范围内候选块覆盖完整历史。prefill 使用容量有界的动态行模板，`--prefill-capacity` 默认 16，支持 2–4096，且不超过上下文容量。单行 decode 复用独立 AOT；prefill 根据剩余长度和共享 `PrefillPlan` 契约选择真实块长，尾块保持真实长度。请求间清空历史和残余压缩状态，通过因果索引隔离旧 KV。prefix 复用、并发调度及 MTP 需要后续接入。

动态 AOT 的指针表 ABI v2 显式传递真实 token 行数；operand 描述保存容量几何，库内外同时固定容量和物理 grid 上限。专家算子保留 `6*m` 等行数倍率，各 kernel 使用同一个有界行标量。编译器将逻辑 grid 映射到有限物理块上的循环；显式 Cube/Vector 计算体拆成同一库内的两个 stream 有序阶段，阶段间通过原 workspace 传递结果。通信长度、W4 tensor 视图和输出头末行随真实块长绑定。新行数的 Graph 注册在请求 reset 之前完成；每个 Graph 保留不可变的行数、库与全部 operand。因果索引在设备上从位置生成。

普通 packet 用一次完成事件保留图、W4 executor、通信组及其设备资源。坐标、embedding 和 Engram 输入通过 compute stream 有序上传，复用的输入地址在原层边界更新。Engram 行包在对应层边界准备，使 CPU 查找与此前已提交的设备计算重叠。完成查询使用 `aclrtQueryEventStatus` 的 recorded 状态，pinned 内存统一通过硬完成栅栏回收。各 rank 共用只读 Engram 行包，缓存身份包含有序行 ID，掩码保持本次请求私有。ACLNN W4A8 会原地变换激活字节，每次调用前均由上游整数投影准备新的 A8 输入；离线复用 executor 的数值检查也需重新上传原始激活。

专家张量的行数表示容量，group counts 总和等于本卡的有效路由数，可以小于容量或为零。空闲 gather 槽保持 `-1`，W4 只计算有效行。路由错误标记和扫描计数器由同一个标量核心更新；算子验收覆盖异常索引、反向行序、空路由和改变输入后的 Graph 重放。

`ORINFER_ASCEND_IMAGE` 可选择编译镜像，`ORINFER_ASCEND_DRIVER_ROOT` 可选择主机驱动目录。编译 runner 将主机 `/data/models` 只读挂载到相同路径；其他模型根目录通过 `ORINFER_ASCEND_MODEL_ROOT` 指定绝对路径。在 CANN 和离线编译镜像准备完成后，先导出上述 Engram 元数据，再构建实验计划和固定 token 输入：

```bash
bash tools/operators/ascend/run.sh --without-devices python3 -m tools.model.deepseek_v41.build \
  --model-dir /data/models/DeepSeek-V4.1-Flash-1017-w4a8-Ascend \
  --engram-metadata artifacts/engram-metadata.json \
  --output artifacts/ascend-model --compile-workers 8
python -m tools.model.deepseek_v41.prepare \
  --plan artifacts/ascend-model/model.json \
  --model-dir /data/models/DeepSeek-V4.1-Flash-1017-w4a8-Ascend \
  --output artifacts/ascend-model/model-prepared.json --workers 16
bash tools/operators/ascend/run.sh python3 -m tools.bench.ascend_single \
  --prepare artifacts/ascend-inputs.json \
  --model-dir /data/models/DeepSeek-V4.1-Flash-1017-w4a8-Ascend
HCCL_DETERMINISTIC=true HCCL_OP_EXPANSION_MODE=AIV HCCL_NPU_SOCKET_PORT_RANGE=auto \
  flock -w 5 artifacts/gpu-experiment.lock \
  orinfer bench-ascend --plan artifacts/ascend-model/model-prepared.json \
  --model-dir /data/models/DeepSeek-V4.1-Flash-1017-w4a8-Ascend \
  --inputs artifacts/ascend-inputs.json --output-tokens 128 --repeats 3
```

大块 prefill 在构建时增加 `--strategy optimized --prefill-capacity 2048`。整数/BF16 投影可从同一模板离线测量多个真实长度，复用公共候选构建、Graph 测量及汇总选择，只替换每种几何的一个动态 fallback；decode AOT 保留原绑定：

```bash
bash tools/operators/ascend/run.sh python3 -m tools.operators.ascend.validate_rows \
  --output artifacts/ascend-row-validation
bash tools/operators/ascend/run.sh python3 -m tools.model.deepseek_v41.tune \
  --plan artifacts/ascend-model/model.json \
  --model-dir /data/models/DeepSeek-V4.1-Flash-1017-w4a8-Ascend \
  --output artifacts/ascend-model/tuning --rows 17 128 512 2048
```

调优读取真实 checkpoint 投影权重，检查候选数值、尾部和改变输入后的 Graph replay，发布同目录的 `model-tuned.json`；随后对选中的计划执行 `prepare` 和完整请求验收。调优只发生在离线工具中，完整请求 TPS 仍由 `bench-ascend` 测量。

`tools.model.deepseek_v41.tune_auxiliary` 接受同样的 `--plan`、`--model-dir` 和 `--output`，离线选择 mHC post/pre 的列块与 FP32 分解投影的行块。后者读取计划中已准备的真实 BF16 分量，候选保持原 K 累加顺序；所有候选检查逐元素一致、尾部及改变输入后的 Graph replay。输出计划名增加 `-aux-tuned`。两种调优缓存均固定当前源码、编译器、设备、测量长度和真实权重身份，并校验被选库的内容。

这两个调优入口的 `--pipeline-only` 搜索更大的行块与合法的两级／三级 L1 流水线；辅助投影同时选择 HC 方差计算的行块。候选校验 A2 的 L0A、L0B、L0C 与 L1 容量，并保留 K、FP32 归约及输出舍入边界。投影调优也可读取 `prepare` 后的物理 bank，避免在线权重变换。

整数投影的 `--wide-integer` 比较当前完整配置与 M128/N256 的流水线配置，覆盖输出列数能被 256 整除的 INT8 几何；其余投影沿用输入计划。当前行块、列块、K 块和流水级数共同构成基线，重复调优也保留已有的流水线选择。该选项与 `--pipeline-only` 互斥，选择结果仍需完整模型请求验收。

辅助投影的宽 N tile 连续写出完整 Cube 结果，并配套发布 `floating.reduce_partials`；归约按生产者的实际布局读取，保留每个 32 列组的八段 FP32 求和。调优检查 partial 的布局还原及配套归约的完整 Graph replay。`--combine-only` 为专家 combine 选择列块，覆盖无效路由、输出尾部及修改输入后的 replay。`--rows` 指定代价汇总长度，`--compile-only` 可提前预编译候选；最终选择仍由设备测量决定。

`tune_auxiliary --split-only` 为 FP32 激活的三分量 BF16 分解选择列块，复用相同的逐次舍入和残差计算。验证按真实行数检查紧凑的三个物理平面、未写尾部与修改输入后的 Graph replay，并逐位对照离线分解结果。`--convert-only` 为逐元素精度转换选择列块，保留 FP32 缩放与目标 dtype 的舍入。

`--hc-reuse-only` 比较 mHC post/pre 的原始临时缓冲和 residual 缓冲复用。原 residual 转为 FP32 后，BF16 缓冲存放各路 post 结果；四路计算完成后，FP32 缓冲承载下一次 pre 的加权结果。该实现保留 BF16 post 边界和四路 FP32 加法顺序，并允许调优 5120 列的完整隐藏行。

`tune_auxiliary --reuse-operand-only` 在 FP32 三分量投影的每个 K 块内复用不随分量变化的 L1 操作数，比较当前配置与零／两级／三级流水线。三个 BF16 乘积及 K 累加顺序保持一致；已有宽列块的基线和选中结果均按各自物理布局检查，配套归约也使用对应的列块。

`tune_auxiliary --quantize-only` 为不超过 256 列的动态量化比较当前行块与 8／16／32 行候选，保持原 FP32 除法残差修正、scale 位拆分及 INT8 舍入。query 的 32 个 head 按物理行数 `32*m` 参与编译和范围检查，ABI 仍接收实际 packet 行数 `m`。FP16 scale 下限从精确的 FP16 值转换，避免固定编译器截短 FP32 常量；静态 query 量化同步重建该下限，保留逐行调度。`tools.operators.ascend.validate_quantize` 独立以 FP64 除法检查 INT8 码和 scale，覆盖 FP16／FP32 scale、零值、小值、半整数、最小次正规 scale、非对齐尾部和改输入后的 Graph replay。可先用 `--compile-only` 离线预编译候选，再进行设备检查与完整请求验收。

`tune_auxiliary --mask-only` 比较当前 selected mask 与显式同步下的 1／8／16／32 项写入循环，保留有符号索引范围检查和 I32 的 0／1 输出。该算子为 DMA、Scalar、Vector 和 UB 复用声明配对事件；`orinfer.explicit_sync` 使编译器保留这些依赖并跳过自动同步插入，AOT 编译身份记录 `pipeline_sync=explicit-v1`。此属性限定为单个原生 stage，多 stage 编译返回错误。`tools.operators.ascend.validate_selected_mask` 验证该限制，并以独立整数参考覆盖重复索引、全无效索引、I32 极值、动态尾部及修改输入和行顺序后的 Graph replay。设备调优后还需完整模型请求验收。

`bench-ascend` 可通过 `ORINFER_A2_WORKER_CPUS` 按 rank 指定 worker 的 CPU 列表，八组以分号分隔，组内支持逗号和闭区间。按设备的 PCIe NUMA 拓扑选择列表；配置在创建原生 session 前应用，仅绑定本进程拥有的 worker 线程，测量报告记录所选 CPU。未设置时沿用进程的亲和性；无效列表、超出当前可用集合的 CPU 和设置失败均返回错误。此选项在 Linux 上支持。

`tune_indexer --vector-only` 保留输入计划的 query／key tile、scale 和 mask 策略，比较直接读取 UB head 切片累加及提前加载 scale、权重和索引的组合。32 个 head 保持顺序执行 FP32 加法；当前配置参与基线测量，多种行数汇总代价后选择发布候选。`validate_indexer --direct-sum --prefetch-inputs` 以独立参考检查该组合，并在同一 Graph 中改变 query、scale、权重和因果 mask。生产选择和完整请求验收沿用离线调优流程。

`tools.model.deepseek_v41.tune_softmax` 为 compact、live-bucket prefill 比较原 softmax 与共享 head 的实现。`--plan`、`--output`、`--rows` 和 `--compile-only` 沿用离线调优约定；默认 `--head-tiles 4 2 1` 分别用于结束位置不超过 2048、不超过 4096 和更长的前缀桶。各桶发布独立内核并在设备上检查位置，互斥写入同一概率缓冲，保留 FP32 倒数、归约和 BF16 概率边界。测量在前缀起点 0 汇总指定长度的代价；数值检查覆盖上下文末端，并在同一 Graph 中改变分数、位置、选择 mask 和前缀桶。候选通过逐位一致和尾部检查后，按测量结果发布原实现或整组分桶实现，输出计划名增加 `-softmax-tuned`。更改执行计划后仍需完整模型请求验收。

`--sinkhorn-only` 选择每组 1/2/4/8/16/32 行的 mHC Sinkhorn 实现。批量版本按 path-major 布局承载独立的 4×4 矩阵，沿用四列行归约、四路列归约和原迭代顺序；UB 别名的跨引擎传输与向量依赖显式同步。候选逐位对照单行实现，并检查动态尾部与改变输入后的 Graph replay。`tools.operators.ascend.validate_sinkhorn --tile-rows N` 另行验证 FP64 参考及重复 Graph。

`tools.model.deepseek_v41.tune_indexer --plan PLAN --output DIR --rows 2048` 对因果索引器联合选择 query 行块与 key 列块，输出计划名增加 `-indexer-tuned`。它复用公共候选、测量和汇总选择组件，以原逐头顺序为参考，保持 INT32 点积、FP16 权重缩放和 FP32 累加顺序；另检查短尾、上下文末端及改变输入后的 Graph replay。`--compile-only` 可通过 `run.sh --without-devices` 预编译候选，实际选择仍须设备测量。

`tune_indexer --mask-only` 在既定行列块下比较标量与向量因果掩码，以逐位输出一致和设备测量选择实现。向量掩码保留原逐头求和顺序，支持部分 key 块及全掩蔽行。

`tools.model.deepseek_v41.tune_prefill_qk --plan PLAN --output DIR` 为 `min_rows>=512` 的 prefill packet 选择 QK 遍历方式，输出同一资产目录的 `*-qk-tuned.json`。候选按当前 SWA 窗口和长 KV 前缀枚举有效块，对早晚 query 块折叠排列，并比较 key 分块数与物理 grid；共享同一 BF16 MMA 和 FP32 累加计算体。离线调优包含原库与新默认实现，逐位检查全部分数和未写区域，在同一 Graph 中同时改变 query 与位置，并抽查独立 FP32 点积。512 行和容量上限的测量经几何平均选择每种几何的一个实现；短片段复用原计划的行数选择范围。`--compile-only --compile-workers N` 通过现有 `build.compile_task` 预编译候选，后续测量按源码、编译器、设备和库身份缓存；输出计划只引用选定实现。完整请求另验收 2K、分块、4K／8K 和全部请求状态。

`tools.model.deepseek_v41.shard_indexer --plan PLAN --output PLAN_SHARDED --min-rows 512` 为 TP8 动态 prefill 计划生成交错分行的 indexer 和 TopK 路径，输出与输入位于同一资产目录。真实行数为 `M` 时，每卡打包 `rank + 8*i` 的 `ceil(M/8)` 行，复用原 INT32 点积和逐头 FP32 累加，再以相同真实行数做 INT32 AllGather 和交错还原；权重和请求私有状态保持原布局。局部行容量为 packet 容量的八分之一，尾部填充行的选择长度为零。较短片段复用原 packet 的物理容量和算子，以独立的 `min_rows`／`max_rows` 范围选择；加载时拒绝间断或重叠的范围。`--fixed-capacity` 保留固定局部容量参考。AOT 容量和逐操作数字节数继续严格匹配。

`tools.model.deepseek_v41.sequence_parallel --plan PLAN --output PLAN_PARTITIONED --min-rows 512` 在已有 TP8 大行数 packet 上复用本地行模板，输出与输入位于同一资产目录。每卡持续保存连续八分之一行的残差，mHC、归一化、FP32 路由和 A8 量化复用原计算体与归约顺序。默认保留原 AllReduce，将结果的本卡行视图交给局部后处理；全局注意力和专家投影读取 AllGather 汇总的归一化输入、A8/scale 以及路由结果。A8 通过四字节 INT32 视图无损传输。两处 Engram 和最终输出头沿用汇总后的原计算；层快照启用时才汇总完整残差。较短 packet 保留原实现，范围由 `--min-rows` 分开。

分片的容量只决定分配大小。真实 packet 行数为 `M` 时，每卡执行 `ceil(M/8)` 行，动态视图、AllGather 和 ReduceScatter 使用相同的行数表达式，填充量最多为七行。`--fixed-capacity` 保留原固定容量参考。`--reduce-scatter` 生成原生 ReduceScatter 的实验候选，未使用的尾行在计算流上清零。原生接口检查源／目标范围、跨 rank 输入容量和资源生存期，支持改变输入后的 ACL Graph replay。此集合通信与 AllReduce 的浮点求和顺序可能不同；该候选的模型质量需要通过相同 teacher 历史的官方对照验收，默认使用 AllReduce 路径。

`tools.model.deepseek_v41.converge` 从一个保留 `min_rows=2` 未分片动态主体的原生计划生成收敛包，统一调用现有 indexer 分片、序列分片、融合 attention、后缀重放和投影调优工具。大行数 indexer 的分片在规范化主体之后重建，保留已有优化；打包、局部计算、集合通信和还原共用真实局部行数。输出计划与输入共用资产目录；`--package-output` 原子发布只含被引用资产的独立目录，准备好的权重 bank 通过硬链接复用，发布目录与资产目录需位于同一文件系统。默认保留完整 decoder；融合归约和近似后缀重放由显式参数启用。基础构建命令为：

```bash
bash tools/operators/ascend/run.sh --without-devices \
  python3 -m tools.model.deepseek_v41.converge \
  --plan ASSET_DIR/model-kv-int8.json \
  --output ASSET_DIR/model-converged.json \
  --package-output artifacts/ascend-runtime/converged
```

`--shared-attention` 将跨 rank 相同的 KV/Qa 投影和归一化复用到局部行模板，Qb 保留原 TP 分片；构建时检查权重身份及局部输出在下一次覆盖前的全部消费者。INT8 KV 的写入与原有 dtype 边界保持不变。局部几何直接读取实际算子工厂，包含按 Cube tile 填充的 scratch，分配大小不再直接按八分之一推导。

设备调优时去掉 `--without-devices`，增加 `--shared-attention --tune-projections local --model-dir MODEL_DIR`。调优使用真实权重，默认覆盖局部容量内的所有 1–128 行和更大行数采样，检查输出尾部、scratch 边界及改变输入后的 Graph replay。共享选择组件按逐测量点至多 5% 回退的限制选择一个通用配置；只有完整测量全部整数的短行区间才生成局部 `row_selections`。相邻区间仅在实现相同时合并，发布前再按 packet 的实际调度范围和截短边界裁掉不可达的选择与库。执行器复用 SDK 的 `KernelVariant` 和行数算术，在准备 Graph 时消费选择表；在线执行保持编译好的同源实现。替换实现必须保持源码、编译器、权重、dtype、scale、数学参数和非 scratch 操作数一致，scratch 可以使用已分配范围内的较小视图。

`--packet-cache-capacity N` 限制每个 rank 的 Graph 形状缓存，默认 64、范围 4–4096；普通单行 decode 常驻，其余按最近使用顺序淘汰。相同几何和 GPU 指针的 ACLNN W4 计划由仍存活的 Graph 共享；Graph 和完成事件持有原生资源。请求重置按模板中声明的状态去重，与缓存过多少种形状无关。离线构建穷举行数表达式和物理范围，在线保留实际行数、操作数和 ABI 校验。

`bench-ascend` 的 `ttft_s` 保留准备与重置之后的计时范围；`request_ttft_s` 和 `request_e2e_s` 包含两者。`preparations` 记录每个输入第一次执行时的形状准备、重置和首输出，预热使用同一完整输入。`python -m tools.bench.ascend_regression --reference OLD.json --candidate NEW.json --output GATE.json` 按相同输入身份、分块、重复次数和执行环境逐场景比较完整请求中位数，任一场景超过默认 5% 回退即失败；新增请求计时字段必须两侧同时存在。输出 token 差异单独报告，质量验收仍需固定历史概率和状态检查。

上述收敛入口用于单请求的动态 token 行；`M` 表示一个请求的片段长度。原生多 owner 计划通过下面的入口构建，公共 A2 服务复用同一执行团队和行模板。

### A2 原生多请求 batch

`tools.model.deepseek_v41.batch` 从保留 `min_rows=2` 完整未分片主体的 INT8-KV 计划构建多请求执行包。投影、HC、路由和专家合并执行 `M=sum(q_i)` 行；attention、KV、索引缓存和 C2 环通过请求地址表调用同源算子。执行包提供一个动态 prefill 家族和一个动态 decode 家族，支持容量内的任意整数 batch；默认最多 128 个 resident owner、每轮 2048 token 行、每请求 4096 上下文位置。实际准入同时受设备内存预算限制。

默认构建复用现有 `sequence_parallel` 与 `fuse_attention` 离线变换，保留动态行表达式及共享 buffer 的视图偏移。Prefill 按总行数使用连续的短行／序列分片区间，大行数同时复用共享 KV/Qa 投影与 ReduceScatter；当前序列变换要求行容量与输入计划一致。各 owner 的 prefill attention 复用一个融合计算体，保持全局归一化后的 BF16 概率与连续 FP32 PV 累积；独立的 `prefill_quantum` 控制序列分片调度。Decode 投影与 PV 保留同源融合 Cube/Vector 模板及显式完成握手。各 batch 通过真实行数和离线配置选择复用这些计算体。原完整参考可用 `--no-sequence-parallel --no-shared-attention --no-reduce-scatter --no-fused-attention` 构建。

```bash
bash tools/operators/ascend/run.sh --without-devices python3 \
  -m tools.model.deepseek_v41.batch \
  --plan artifacts/ascend-model/model-kv-int8.json \
  --output artifacts/ascend-model/model-batch.json \
  --package-output artifacts/ascend-model/batch-package \
  --owners 128 --rows 2048 --capacity 4096 --compile-workers 4

orinfer bench-ascend-batch \
  --plan artifacts/ascend-model/batch-package/model.json \
  --model-dir /data/models/DeepSeek-V4.1-Flash-1017-w4a8-Ascend \
  --inputs inputs-batch.json
```

可选的小行数调优使用现有真实权重、Graph 测量和共享选择表组件，比较同源 64/128/256 CTA 边界。默认完整覆盖 1–128 行及各几何的大行数采样；发布阶段保留间接地址表声明的 owner 状态、视图及工作区，只包含最终选中的实现：

```bash
bash tools/operators/ascend/run.sh python3 \
  -m tools.model.deepseek_v41.tune \
  --plan artifacts/ascend-model/model-batch.json \
  --model-dir /data/models/DeepSeek-V4.1-Flash-1017-w4a8-Ascend \
  --output artifacts/ascend-model/batch-tuning \
  --decode-only --grid-only \
  --package-output artifacts/ascend-model/batch-selected
```

随后将 `bench-ascend-batch --plan` 指向 `batch-selected/model.json`。稳定布局的地址表、工作表和输出索引按执行家族、owner 顺序、片段长度和输出位置缓存，准入及释放使缓存失效。主机输入通过最多 4 MiB 的 pinned 分块在 compute stream 上传；预算不足时等待已提交 DMA 的完成事件，再复用缓冲。调用者可在方法返回后释放源数据。

输入可直接声明不同长度的请求；Rust 按 `chunk_tokens` 预算自动分块，每个请求在最后一个 prefill 片段执行输出头，再合并后续普通 decode：

```json
{"cases":[{"name":"mixed","requests":[{"owner":7,"tokens":[0,27620,3494]},{"owner":91,"tokens":[0,27620]}],"chunk_tokens":2048,"decode_steps":16,"repeats":3,"compare_serial":true}]}
```

自动分块从执行包的 owner 选择区间读取优选片段长度，轮转选择请求并填满剩余预算，保留每个请求的 token 顺序与最终输出位置。默认优先保留长请求的 512 行片段，短请求使用其真实剩余长度；单请求继续使用全部可用预算。相邻轮次推进轮转位置，避免将多路长输入平均切成短片段或产生可合并的极短尾包。

也可用 `frames` 代替 `requests`，逐轮声明 `requests` 与 `decode`；每条请求的 `output` 控制该轮是否执行输出头。`decode:true` 要求各请求正好一行；默认家族接受不同片段长度及一行续算。自动 decode 要求每个 owner 的最后一个输入片段请求输出。owner 身份与行顺序独立，未出现在某轮的请求保留完整状态。

`compare_serial` 使用 batch 实际生成的固定历史，按相同计算家族逐请求重放，比较完整词表概率和 greedy 输出。它验证请求隔离及批量算术，不替代官方 BF16/FP8 质量对照。报告区分 forward 时间与执行计划准备时间；完整请求吞吐包含 CPU 坐标、Engram 查找、传输、计算、HCCL、输出头和同步，预热测量排除加载、准入、重置与 Graph 捕获。

离线短测的 case 可设置 `record_routing:true`，在首次预热的每个 frame 完成后读取各 rank 的 W4 专家行计数。报告的 `routing` 保存 case、frame、角色、实际 token 行数及按计划 rank 顺序排列的计数；每项保留原始 groups buffer 名。读取由资源所属 worker 执行，耗时不计入 forward 测量。默认关闭，在线服务不采集该统计。

可选的 decode router 分片复用现有 `dense.fp32_partials`、`dense.fp32_reduce` 和 `memory.gather_columns`。各 rank 计算连续的输出列，AllGather 后恢复完整 FP32 logits，再执行原来的 TopK。TP8 的 384 个专家按每 rank 48 个逻辑列分配，投影使用 64 列对齐视图，末端重叠列在恢复时裁剪；权重值、K 归约顺序和专家布局保持原契约。该转换作用于 `batch_decode` packet，prefill 继续使用完整投影。

```bash
bash tools/operators/ascend/run.sh python3 \
  -m tools.model.deepseek_v41.router_sharding \
  --plan artifacts/ascend-model/batch-selected/model.json \
  --output artifacts/ascend-model/router-sharded
```

输出计划位于 `router-sharded/package/model.json`。单算子验收可运行 `tools.operators.ascend.validate_router_sharding --baseline BASE_PLAN --plan SHARDED_PLAN --output REPORT`，比较同一输入的完整／分片 FP32 投影、非对齐行尾和改变输入后的 Graph replay。完整请求另测固定历史概率及实际吞吐；默认高性能通信下，各 rank 输入的数值差异仍需纳入跨包概率对照。

`compare_whole_prompt:true` 配合 `compare_serial:true`，对 `requests` 使用单请求最大容量分块作为参考，检查批量自动分块及后续 teacher-forced decode 的状态一致性。显式 `frames` 沿用原片段边界。设置 `ORINFER_A2_BATCH_LOGITS_DIR` 时，首次预热各输出位置的完整 FP32 logits 按 case、frame 和 owner 写入指定目录，供跨包概率／NLL 对照；独立导出不要求启用逐请求串行重放。

该入口默认使用高性能通信，不要求设置 `HCCL_DETERMINISTIC`。需要固定跨 batch 的 BF16 归约顺序时，在进程启动前显式设置 `HCCL_DETERMINISTIC=strict`；此模式单独测量性能。`compare_serial` 以固定历史的最大概率绝对差 0.02 为门限，greedy 差异另报；默认通信在 B128 的跨 batch 对照尚未通过，详见 [性能与验证记录](performance.md#融合动态-decode-与地址表复用2026-10-29)。请求私有状态和 owner 隔离保持独立检查。当前原生验收使用 CANN 9.2，TP8 为当前模型接通方案，后续并行策略按实际显存和完整请求性能选择。

Graph 按计算家族、片段长度集合与输出数缓存，地址表和位置在 replay 前更新。捕获期间关闭私有状态访问；完成事件持有实际请求 buffer。请求状态复用公共对齐 arena 布局，额外请求的私有状态各使用一块原生分配，避免小张量独立分配的页池开销。Graph 缓存同时受实际空闲显存约束；各 rank 在捕获前统一探测缓存，任一 rank 未命中时全体失效并重建该形状。首次准入、重置、捕获和首轮 forward 的成本保留在 `preparations`，`request_s` 另包含本轮重置与准备。私有 scratch 采用声明的物理行补齐规则；Engram AllGather 复用 `gather_columns` 还原行布局。上下文使用包内固定容量，跨包扩容和公共服务接入另行实现。该入口使用完整 decoder；近似后缀重放仍通过单请求实验入口选择。

query/head 交换的宽 BF16 行重排可通过现有 `memory.partition_rows` 和 `memory.interleave_rows` 的 `tile_columns` 参数调优。离线工具比较 256/1024/4096 列连续搬运，共用原来的动态行、rank 排列及补齐语义；每种几何只发布一个配置。选择器汇总 1、32、511、512 和容量边界的实际测量，要求各采样点相对原配置回退不超过 10%。测量前后检查逐元素相等、未使用行和改变输入后的 Graph replay；一个 Graph 内重复调用以摊薄主机 replay 开销，结果按每次搬运报告。编译及测量缓存包含算子、编译器、设备、共享测量组件和选择器身份。

```bash
bash tools/operators/ascend/run.sh python3 \
  -m tools.model.deepseek_v41.tune_movement \
  --plan artifacts/ascend-model/router-sharded/package/model.json \
  --output artifacts/ascend-model/movement-tuned
```

最终计划位于 `movement-tuned/package/model.json`，只引用被选择的算子；候选及逐行测量留在离线输出目录。权重、KV 状态和 decode 执行程序继续复用输入包。


`tools.model.deepseek_v41.quantize_kv --plan BF16_PLAN --output INT8_PLAN` 将尚未量化 KV 的 DSv4.1 TP8 原生计划转换为 INT8 缓存，输出与输入共用资产目录。40 层 SWA 和四个共享压缩 KV bank 均按每 64 个值使用一个 FP16 scale，512 维每行共 528 字节。写入按实际存储的 FP16 scale 做对称舍入，整数范围为 −127..127；scale 下限为 FP16 最小正数，上限为 65504。零组写零，`-1` 位置跳过写入，其他越界位置设置错误标志。压缩器的 FP32 ring 状态和索引器现有 INT8 格式保持各自的契约。

prefill 使用一个 SWA 和一个压缩 KV 的共享 BF16 工作区，仅解码当前窗口／前缀；同一压缩 source 的后续层复用已经解码的值。decode 将选中行读取与 group64 解量化融合，后续 QK／PV 继续使用 BF16 输入与原 FP32 累加边界。工作区属于计算暂存，不保留逐层 BF16 缓存副本。`--reference-bf16` 生成保存相同 INT8 舍入值的 BF16 对照计划，用于核对实际存储布局和状态，不用于部署。

`tools.operators.ascend.validate_kv_int8 --plan INT8_PLAN --reference REFERENCE_PLAN --output REPORT` 通过固定编译镜像运行数值验收，覆盖独立 FP64 编码参考、FP16 scale、舍入边界、跳过／越界写入、动态尾部、选择读取、前缀及改变输入和位置后的 Graph replay。完整模型另用固定历史比较概率／NLL、分块与长上下文状态，并以完整请求测量性能。

`tools.model.deepseek_v41.fuse_attention --plan PLAN --output FUSED_PLAN` 生成融合 prefill attention 实验计划，使用分块在线 softmax 与同一次 launch 内的 Cube／Vector 交换，代替完整 scores／probabilities 矩阵。`--tile-m` 选择 16／32／64 个 query-head 行；`--tile-k` 选择 128／256 个 key 行，压缩 bank 不满足 256 对齐时使用 128。`--reuse-kv` 在 L1 保留 query，并让 QK／PV 共用同一份 KV tile。`--min-rows 512` 只转换声明最小行数至少为 512 的 packet，较短 packet 继续复用原实现。A2 的存储与计算边界仍为 INT8 缓存解码后的 BF16；在线 softmax 改变浮点归约和概率舍入位置，需要完整模型质量对照。`tools.operators.ascend.validate_attention_fused --manifest KERNEL_JSON --output REPORT` 检查独立 FP64 参考、尾部、长上下文以及改变数据和位置后的 Graph replay。

`tools.model.deepseek_v41.tune --joint --geometry N K` 在真实权重上联合测量 M／N／K tile 和流水候选；`--geometry` 可重复，只发布跨所选行数的胜出配置。`tools.model.deepseek_v41.reduce_scatter --plan SP_PLAN --output RS_PLAN` 将已有 TP8 行分片计划中的全量 AllReduce 替换为本地结果 ReduceScatter，检查消费者视图并定义未使用的尾行。该替换保留各自的 BF16／FP32 dtype，浮点求和顺序需要独立质量验收。

`tools.model.deepseek_v41.decoder_replay --plan PLAN --tail-template INT8_PLAN --keep 140 --output REPLAY_PLAN` 生成近似 decoder SWA 后缀重放实验。第 20 层（从零编号）的完整共享 compressed KV 和 index K 先写入，再收集跨边界状态、保留绝对位置并压缩查询行；后续层复用未做 SP 的动态行模板。原生 `limit_rows` 边界同时限制后续算子、集合通信、W4 以及输出头的容量与真实行数，需使用匹配版本的 Rust CLI。重放区间不包含被截掉的 SWA 前缀；未截短的小片段继续读取已有历史。该计划改变 decoder 状态，质量验收必须覆盖固定 teacher 历史、长上下文、不同分块、请求重置和后续 decode。`tools.operators.ascend.validate_decoder_replay --plan REPLAY_PLAN --output REPORT` 检查边界两侧行数、完整数据搬运、位置、窗口掩码与改变参数后的 Graph replay。上述实验均保留原计划，按真实完整请求质量和 TPS 选择部署实现。

`tools.model.deepseek_v41.sequence_parallel --decoder-tail --plan REPLAY_PLAN --output TAIL_SP_PLAN` 为截短后的 decoder 复用相同的 TP8 私有行模板；构建 replay 时的 `--keep` 必须能被 8 整除，例如 128。跨边界的 residual 与 collapsed 输入先映射到各 rank 的有效行分片，再沿原有集合通信边界执行；`--reduce-scatter` 同时为后缀启用 ReduceScatter。随后可用 `tools.model.deepseek_v41.reduce_scatter` 转换完整前缀中的集合通信，保留各部分独立验收的计划。

上述工具可通过 `run.sh --without-devices` 离线编译。`tools.operators.ascend.validate_sharded_indexer --baseline PLAN --plan PLAN_SHARDED --output DIR` 在设备上逐卡比较原始分数、TopK、填充行和索引重排，以独立参考抽查分数，并验证修改输入后的 Graph replay。完整请求另用 `bench-ascend` 验证分块、缓存状态、输出与 TPS。

`--fast-engram` 为动态 prefill 启用批量 32 阶蝶形旋转。构建器要求 checkpoint 的旋转块恰为标准顺序、统一幅值、带行符号的 Walsh 矩阵；不匹配时构建失败。计算读取原始 FP32 系数，保留输入乘系数与输出 FP32 边界。`validate_rotation` 检查原 bank、FP64 参考误差、动态尾部和改变输入后的 Graph replay；单行 decode 保留通用旋转。

`--parallel-regroup` 为 prefill 生成稳定的分块专家分组：每个 32-token 块按路由位置排序，再计算专家与块的前缀和，写回 gather 和 restore。专家、token、路由 lane 的顺序与串行扫描一致；六个有效路由之外的 padding 及最后一个块中超出真实 token 行数的位置均不参与分组。`tools.operators.ascend.validate_regroup --base 0` 覆盖 expert 0 的短尾，默认的 `--base 144` 覆盖中间 rank；验证包含重复路由、空专家、异常索引和改变输入后的 Graph replay。

`--vector-prefix` 在 `--parallel-regroup` 上为动态 prefill 启用有界向量前缀和。扫描容量由 packet 的行数上限固定，计数及其中间和不超过 6×4096，FP32 表示保持整数精确性；单行 decode 保留原扫描。`validate_regroup --vector-prefix` 验证完整分组，`tools.operators.ascend.validate_prefix` 覆盖 4096 行上限、分组 padding、错误标志、未写尾部和改变输入后的 Graph replay。

实验参数 `--fp32-cube` 将 prefill 的敏感投影改为三个 BF16 分量的 Cube 乘积，保留 FP32 partial 和原有归约边界。权重分量离线准备，路由输入分量由设备生成；原始 W4 专家 bank 保持单份。该路径改变浮点累加顺序，启用前需单独比较任务质量、teacher-forced 概率和状态。默认继续使用原 FP32 投影。

`--packed-attention` 为 prefill 使用 token/head 合并的 Cube tile，直接读取共享 SWA 和压缩 KV bank，以因果窗口和唯一 TopK 索引掩码选择有效项。它支持 256 对齐、至多 8448 的上下文容量，保留 FP32 softmax 和 BF16 概率。`--batched-hc` 在多行之间复用 mHC 的同源八段归约与 FP32 归一化。两者均为独立实验开关，decode 继续使用单行实现；算子验证入口分别为 `validate_prefill_attention` 和 `validate_hc`，完整模型需另外验收。

`--compact-attention` 配合 `--packed-attention`，按 query 块裁剪 SWA 的 QK/PV 计算，使用至多三个 128-key tile 的概率窗口；压缩 KV 的 softmax 按块内最后位置选择有界前缀。它保留因果掩码、FP32 softmax、BF16 概率和输出 inverse RoPE。`validate_prefill_attention --compact --live-buckets` 检查该布局，包括非对齐起点和上下文边界。

`--batched-rope` 在 prefill 中合并同一 token 的 head，保留原旋转算术及 BF16 边界，使用 `validate_rope` 验证。`--active-expert-rows` 要求 `--parallel-regroup`，让专家 SwiGLU／量化读取设备前缀扫描给出的实际行数，跳过未被 restore 引用的容量尾部；Graph 地址和行数 ABI 保持固定，`validate_active_quant` 检查零路由、尾部及 replay 时改变有效行数。以上开关保持实验性质，完整请求、质量及请求状态需另外验收。

完整模型的重复性及固定历史概率对照使用 `HCCL_DETERMINISTIC=true`。BF16 集合归约的顺序会影响后续路由及输出，确定性配置与实际吞吐一并记录。单独同步 Graph 提交不能替代确定性归约。

`ORINFER_A2_LOGITS_DIR=目录` 在普通 Graph 完成后保存各 rank 的 FP32 输出头分片，文件名携带 rank、前缀位置和本次行数，重复执行覆盖同名文件。输入计划须声明 `logits` buffer；每个固定历史使用独立目录。该诊断保持原 Graph 分段，报告标记 `logits_dump: true`，耗时包含额外读回和文件写入，供概率对照使用。

`tools.model.deepseek_v41.compare_logits` 用 `--reference` 指定官方 `reference` 输出目录，`--inputs` 和按位置排序的 `--cases` 指定原 prompt 加 teacher token 的完整历史，`--logits` 指定原生分片目录，`--output` 保存 NLL、argmax 和 KL。`--baseline-logits` 可同时比较同权重基线。每个 case 使用单个 prefill packet；工具核对 prompt 和 teacher 身份、历史、分片长度及有限值。少量位置的概率对照用于诊断，任务质量与长上下文仍需单独验收。

构建器引用原 checkpoint。离线 `prepare` 使用 NumPy、系统 C 编译器与 ARM NEON 8×8 转置，将 W4 bank、ND per-channel scale 和辅助 bias 写入物理布局文件。数值与尾部检查放在独立 CPU 测试，批量转换只执行 NEON 转置和顺序写盘。通过顺序 `pread` 读取单个专家，以有界 bank 缓冲写盘，避免对 ZFS 映射做逐字节转置。此转换保持原始 INT4 nibble，准备产物保留 W4，每卡在线仍只加载一份 W4；原 checkpoint 保留用于参考和其余张量。断点续转核对原文件的大小/mtime/inode、转换器身份、产物长度与完成标记；权重准备和在线加载均不扫描 payload 计算哈希。全部 bank 完成后原子发布准备计划，现有 AOT 与资产共享同一目录。加载前核对配置、tokenizer、Engram、AOT 库和绑定身份；设备分配使用大页优先策略及独立会话预算。

八卡 910B3 的单流验收使用进程环境变量 `HCCL_OP_EXPANSION_MODE=AIV` 选择通信展开方式；先运行 `HCCL_OP_EXPANSION_MODE=AIV orinfer validate-hccl --devices 0,1,2,3,4,5,6,7` 验证目标设备，再以相同环境启动模型测量。

构建参数 `--strategy optimized` 启用单流优化计划：decode 使用 Cube QK/PV 与 FP32 masked softmax；共享专家按 TP8 离线分片，down 投影保留实际 K288；Engram WKV 投影按输出列分片并 AllGather；HC 参数、Q/KV 与共享／路由专家通过嵌套 Graph 分支重叠；mHC post／下一次 pre 和量化行／scale gather 分别融合。`prepare` 同时生成共享专家 bank，在线直接读取物理文件。`--strategy baseline` 保留串行参考计划。优化计划的 attention 概率使用 BF16，共享专家使用分片激活 scale 和跨卡归约，需要独立质量验收。

Graph 分支在捕获前校验实际 buffer 字节范围的读写依赖，禁止并行分支写入重叠视图，HCCL 保持在主分支。每个 fork／join 使用独立的捕获事件，Graph 持有辅助 stream、事件、通信组和所有算子资源；改变输入后的重放属于原生验收。

官方混合精度参考使用原始 FP4 专家和 FP8 主干 checkpoint。`tools.model.deepseek_v41.reference` 复用该目录的 `inference/model.py`，有界解码原始权重，保留官方激活和 KV 量化、RoPE、mHC 及 attention 的概率舍入边界。该离线工具使用软件解码后的矩阵计算；输出身份明确记录其数值路径，性能验收仍使用原生 Rust 执行器。

```bash
bash tools/operators/ascend/run.sh python3 -m tools.model.deepseek_v41.reference \
  --model-dir /data/models/DeepSeek-V4.1-Flash-1017 \
  --inputs artifacts/ascend-inputs.json --teacher artifacts/ascend-result.json \
  --case p1024 --positions 129 --device npu:0 \
  --output artifacts/ascend-official-reference
```

`--teacher` 接受 `bench-ascend` 的结果 JSON，以同一 token 历史逐位置保存官方 logits、target NLL 和 argmax；`--inspect-only` 只检查官方参数映射。大 Engram 表按选中行读取，矩阵缓存由 `--cache-gib` 控制。

固定版本的 TileLang-Ascend 编译器会为纯 Vector 程序生成混合任务。离线构建器识别仅使用 UB、没有跨核同步且网格固定的独立程序，移除未使用的 Cube 分配，生成 AIV-only 任务；物理网格扩大一倍并保留原有逻辑块和双 Vector 索引。含 Cube 计算、跨核同步、未知网格或仍使用 Cube 工作区的程序保留原调度。转换规则及源码计入编译缓存身份，改变输入的数值和 Graph replay 在离线验收中检查。

测量排除权重加载和预热，输出固定输入的 token hash、全部生成 token ID、TTFT、decode 速度及端到端时延。TTFT 包含末位置输出头；由输入长度除以 TTFT 得到的值是 prefill 吞吐代理。Rust 计时包含 CPU 查找、提交、等待与 HCCL；`tools/bench/ascend_single.py` 对 vLLM 的测量还包含本机 HTTP 和调度开销。两端使用同一组原始 completion token，固定输出长度，关闭跨请求 prefix 复用和推测解码。输入 `cases` 可分别指定 `output_tokens`、`repeats` 和 `trace`，省略时沿用命令行设置。`--trace DIR` 下载各 rank 的快照用于排错，rank 0 写在根目录，其余写入 `rankN/`；正式测量使用 `trace: false` 或省略 `--trace`。

### 源码入口

统一源码入口：

- `crates/orinfer-engine`：通用运行设施。
- `crates/orinfer-device`：无驱动依赖的设备及执行契约。
- `crates/orinfer-model-sdk`：计划、状态和原生 C ABI 契约。
- `crates/orinfer-models`：模型适配器、独立的精度策略选择和注册表。
- `kernels/`：可复用的 TileLang 算子与融合实现。
- `tools/`：离线编译、权重准备、组装与验证。

适配器描述架构语义与执行组合，策略决定可用精度路径；同一策略可被多个适配器支持，融合可以跨执行组件边界。架构/策略组合完成验证后注册。部署时只绑定模型配置实际需要的 kernel 资产。

```bash
make build
# 或仅构建原生模型库：
cargo build --release --locked --offline -p orinfer-models
```

该产物提供注册的模型逻辑；可部署库还需要离线构建工具生成匹配当前 kernel ABI 和物理 tensor 角色的绑定模块。发布器通过 `ORINFER_BINDINGS_SOURCE` 编译该模块，原生库的链接依赖限定为 SDK；结果位于包的 `lib/model.so`。包同时保存生成的 `lib/bindings.rs` 和 `lib/BUILD.txt`，与发行版内的 workspace 源码及 Cargo.lock 一起用于重建该库；在线执行使用编译后的原生库。

在保留权重和 cubin 的情况下，重建原生库和目录：

```bash
uv run --locked python -m tools.model.catalog \
  --model /path/to/prepared-model --output /path/to/rebuilt-model
```

工具需要项目固定 Rust 工具链及已缓存的 Cargo 依赖，使用 `--locked --offline` 构建；同文件系统的不可变权重通过 hardlink 复用，模型描述和 `.so` 使用独立文件。构建输出经加载器验证后原子发布到新目录，源目录保持原样。普通在线启动直接使用已发布的二进制包。

`tools.model.attach_execution` 要求替换库与原包具有**相同绑定注册表和目录身份**，并在替换前校验这些条件。更新注册模型逻辑后优先使用上述重建工具。

在线数据与包统一使用 schema 1、`catalog` 及原生绑定导出接口。离线构建输入仍可包含展开的 bindings，用于精确对照与生成 Rust 模块。

## 安装执行包

下载引擎、匹配的执行包和 `SHA256SUMS` 后，先检查下载内容。Rust CLI 在 CPU 上完成安装：

```bash
sha256sum --ignore-missing -c SHA256SUMS
orinfer package install /path/to/package.tar.xz
orinfer plan-model /path/to/new-model
orinfer validate-model /path/to/new-model
```

安装器按文件内容识别 `.tar.xz` 和旧 `.tar.gz`，校验 package digest、schema/ABI、设备目标、目标主机的 ELF 库和全部独立／safetensors 内嵌 kernel 资产的 SHA256；安装期间对库文件做静态校验。路径越界、链接和重复归档成员被拒绝，全部校验成功才原子写入缓存，失败清理临时目录，已安装的同 digest 包保持原样。解压与校验只发生在安装时，后续启动直接读取已安装的资产。

默认安装位置与运行时一致：优先 `ORINFER_EXECUTION_CACHE`，其次 `$XDG_CACHE_HOME/orinfer/packages`，最后 `~/.cache/orinfer/packages`。`--cache /path/to/packages` 只改变本次安装位置；自定义位置需要同时为后续运行设置 `ORINFER_EXECUTION_CACHE`。

`plan-model` 会在 CPU 加载原生执行库并生成计划；`validate-model` 还完整校验权重、CPU 表和全部 kernel 文件，均在 CPU 上执行。在线启动默认只检查大体积权重和 CPU 表的结构，`--verify-weights` 开启完整内容校验；配置签名仍校验，前端资源、CPU 资产元数据、执行库和 kernel 身份仍校验 hash。JSON 创建请求的可选 `verify_weights` 字段由模型库处理 CPU 表，C ABI v1 和 schema 1 保持兼容；更新执行库后即可使用新的 CPU 表加载行为。

### 安装已发布的 v0.1.3

v0.1.3 使用系统 tar 安装执行包：下载并通过 `SHA256SUMS` 校验后，将执行包的 digest 目录解压到共享缓存，再运行 `validate-model`：

```bash
export ORINFER_EXECUTION_CACHE="$HOME/.cache/orinfer/packages"
mkdir -p "$ORINFER_EXECUTION_CACHE"
tar -xf /path/to/package.tar.gz -C "$ORINFER_EXECUTION_CACHE"
orinfer validate-model /path/to/prepared-model
```

### 离线归档

模型构建者使用 Python 工具生成 `.tar.xz` 执行包，供部署时安装：

```bash
uv run --locked python -m tools.model.package archive /path/to/package /path/to/package.tar.xz
```

## 合并发布布局

离线调优结束后，将 GPU 权重合并成约 2 GiB 的标准 safetensors 分片，CPU 查表按连续编码段合并。张量名称、物理布局、量化位模式及 CPU 行序保持不变；更新索引、CPU 文件偏移与身份 hash。模型目录只包含模型数据，匹配的执行包独立写入指定缓存：

```bash
uv run --locked python -m tools.model.compact \
  --model /path/to/prepared-model --output /path/to/compact-model \
  --package-cache ~/.cache/orinfer/packages
orinfer validate-model /path/to/compact-model
```

Flash Next 的正式权重发布采用稳定的组件和层组名称：主模型专家每三层一片，其余原权重每八层一片，Q4 投影、输出头、HC 无损压缩表示、MTP 和视觉各自独立；CPU 表复用已合并的连续行分片。单片上限默认为 2 GiB，超限会返回发布错误。重打包直接复用每个张量的量化字节、布局、身份和已编译 kernel：

```bash
uv run --locked python -m tools.model.flash_next.repack \
  --model /path/to/optimized-model --output /path/to/published-model
orinfer validate-model /path/to/published-model
```

输出目录及 Hugging Face、ModelScope 模型仓库保存配置、tokenizer、模型描述、GPU 权重与 CPU 表；匹配执行包通过 GitHub Release 独立发行，先安装到公共执行缓存再加载模型。模型描述中的 `execution_package` 固定对应包 digest，重打包保持该 digest。

包内 `assets.safetensors` 用具名 U8 tensor 收纳 cubin、生成源码与 host ABI，去重字节相同的资产。schema 1 的文件身份可带 `tensor` 字段，SHA256 校验对应 tensor 的原始字节；原生 `.so` 保持独立文件。引擎复用 bundle 的 mmap 和头部索引，直接从 bundle 读取具名资产；CUDA 模块仍分别加载。此布局需要支持具名资产的引擎版本。

执行包单独归档和安装，模型通过 `execution_package` digest 在共享缓存中找到它。合并用于最终发布。离线调优工具通过 `plan-model` 从 Rust 库读取当前绑定，在编译阶段展开和修改，再重新生成原生模块及目录；在线格式保持一致。

## 验证

`make check` 统一执行引擎、SDK 及模型 crate 的格式、clippy、单元测试，同时覆盖 C ABI 跨语言加载、错误拒绝和更新原子性。更新还应比较完整程序、launch 参数、buffer 与状态契约；GPU 验证需要匹配模型权重，覆盖请求状态、graph 输入变化、prefix 恢复及 MTP 分支。

通用加载链的 GPU 夹具可独立运行：

```bash
bash tools/operators/run.sh tools/model/validate_model_package_gpu.py artifacts/model-package/check01
```

夹具验证三种 Graph 模式、输入变化和完整 logits；真实模型质量与性能另行验收。

Flash Next 的当前原生包可通过 `tools/model/flash_next/build.py` 重新生成整数投影与执行包。该入口读取模型固定的架构绑定契约，按真实矩阵调优，支持内容校验与断点恢复；状态、浮点归约和联合专家组合保持输入中的已验证实现。输入须为当前原生模型并包含匹配的执行包；量化中间权重需先完成模型组装。用法见 [Flash Next 构建](../tools/model/flash_next/README.md)。

### A2 文本服务与长上下文

实验性 DSv4.1 A2 provider 复用公共 `Model`／`RequestState`、连续 worker、Chat/SSE 前端和 radix prefix cache。原生八个线程持有各自的 ACL context、HCCL communicator、请求私有状态及 Graph；`bench-ascend-batch` 使用同一团队。普通 decode 和混合 prefill packet 复用已发布的动态行模板。

离线生成 262K 计划并发布服务目录：

```bash
bash tools/operators/ascend/run.sh --without-devices python3 -m tools.model.deepseek_v41.long_context \
  --plan /path/to/prepared-int8/model.json --output artifacts/a2-long \
  --capacity 262144 --owners 32 --length-input --gather-slots 16
python3 -m tools.model.deepseek_v41.service \
  --plan artifacts/a2-long/package/model.json --model-dir /path/to/original-checkpoint \
  --output artifacts/models/dsv41-a2-service
source /usr/local/Ascend/cann/set_env.sh
orinfer serve artifacts/models/dsv41-a2-service --model deepseek-v41-flash-1017-w4a8 \
  --listen 127.0.0.1:18818 --mtp-drafts 0 --max-active-requests 32 \
  --prefix-cache-mib 8192 --gpu-lock artifacts/gpu-experiment.lock
```

输入是已准备物理 W4 bank 的原始单请求 INT8 KV 计划。构建器扩展 RoPE、完整因果索引及递归 TopK，复用已有 gather、QK、softmax、PV；索引 Cube 中间结果使用有界工作区。SWA 保留 4096 行环，覆盖 2048 行 packet 和 128 行因果窗口。压缩 KV 与 key 按各层 1／2 压缩率保留完整上下文，KV 为 group-64 INT8 加 FP16 scale，计算边界保持 BF16。checkpoint 复制有效压缩历史、SWA 尾部和完整 FP32 压缩环；恢复后重新绑定 owner 地址及位置。

A2 服务在当前没有活跃 decoder 时按执行包的 token 行容量填充 prefill packet；单个请求仍在每 4096 token 的缓存检查点暂停并提交完整状态。有活跃 decoder 时，decode 行优先，混合 packet 继续按 `--prefill-budget-ms` 和原生吞吐估计限制总行数。不同分块长度复用同一个动态 prefill 模板；新的请求与取消在 packet 边界处理。

`--length-input` 让同一索引 Cube 工厂读取每行的有效前缀长度，按 owner 内单调递增的位置决定实际 key tile 数，写入有效 score 前缀。TopK 使用相同长度屏蔽其余容量，完整有效候选与原有归约顺序保持一致。此模式以长度向量替代稠密因果索引矩阵，同时适用于 decode 和 prefill。`--gather-slots` 选择同一 INT8 gather 工厂每次转换的槽位数。可重复提供 `--reuse-cache /path/to/compiled/cache`，复用完整编译对象；编译器仍校验对象身份和库哈希。

当前范围为纯文本 system/user/developer/assistant 消息、连续 batch、流式与非流式生成、采样及 prefix 恢复。工具定义／工具调用、媒体、MTP、prompt lookup、有限专家常驻及 `--verify-weights` 返回不支持错误。服务目录采用 schema 1 的实验性原生计划封装，由上述 Python 发布器生成；通用模型 cdylib C ABI 打包、`package install` 和 `validate-model` 尚未接通此封装。默认 HCCL 用于性能；`HCCL_DETERMINISTIC=strict` 用于逐值相等的独立状态诊断。状态恢复检查与官方 BF16／FP8 质量验收分别记录。

`GET /metrics` 导出请求排队、首 token 前、解码、结束处理的 wall time histogram，以及运行／等待请求数和原生计算／计划准备累计时间。统计从前处理完成、请求进入 worker 队列开始，准入／恢复计入排队阶段；前处理与网络需结合客户端 TTFT 分析。阶段包含调度等待，不能相加解释成独占 GPU 时间。Chat 请求的 `ignore_eos:true` 保持指定生成上限，适用于固定输出长度回放。

A2 的 Engram CPU 行表使用共 2 GiB 的跨 packet 行缓存，由各 rank 共享，按行 ID 保留原始 INT8 权重及 FP32 scale 字节。预算包含缓存数组、标签及替换元数据；最近 packet 和在途上传沿用原有的有界 staging。`ORINFER_A2_HOST_ROW_CACHE_MIB` 设置总预算，`0` 关闭跨 packet 行缓存；`ORINFER_HOST_ROW_CACHE_REPORT=1` 在模型释放时输出累计 packet 命中、唯一行命中／读取数及保留字节，便于定位主机取表成本。

主机内存充足时，可用 `ORINFER_A2_HOST_TABLE_MIB` 设置 Engram 完整常驻预算，例如 DSv4.1 当前 INT8 表使用 `212000` MiB。非零预算须容纳全部表及分配元数据，不足时在分配前报错；启动通过现有 Rayon 工作池按 32 MiB 块读取各表，rank 共享唯一常驻副本，后续行读取直接复制内存字节，并关闭重复的行缓存。该模式额外使用约 206 GiB 主机内存，读取成本计入模型启动时间，原生统计的 `resident_bytes` 报告实际保留量。默认 `0` 保持文件读取与有界行缓存；表常驻和行缓存均保留原始 INT8／FP32 字节，设备 staging 沿用已有边界。

`python3 -m tools.bench.agentx_mini --help` 提供固定 Mini-Weka 短测入口：保留原始 16 个窗口、64 个 profiling 请求、并发 16、seed 42、180 秒 profiling 和每 lane 一次 warmup，记录输入哈希、服务二进制、A2 主机预算与 HCCL 参数、健康状态及 metrics。计时区间内通过 `/proc/status` 采样 RSS，PSS 在启动回放前单独记录，并保存采样耗时及进程 I/O、CPU 计数。默认总时限 900 秒，到时终止本轮客户端并保留诊断；中断结果不能作为通过的短测分数。该入口连接已运行服务，不部署后端或修改设备设置。

私有算子的离线调度重建可直接复用已编译计划：

```bash
bash tools/operators/ascend/run.sh --without-devices python3 -m tools.model.deepseek_v41.private_schedule \
  --plan /path/to/native-batch/model.json --output artifacts/a2-kv-schedule \
  --operator kv_int8.gather_attention --parameters '{"tile_slots":16,"bulk_store":true}'
```

该入口复用 `batch.compile_task` 与 `private_rows`，检查输入／输出 dtype、owner 指针列序和物理字节范围，生成新的 work 表及最终资产。`--min-owner-rows N` 沿现有 owner range 契约保留更短 owner 的原绑定；分片计划按 owner 的局部选中行数应用该范围，并保留指针表／work 表的分片角色。decode 程序及权重保持已验证的编译资产。`gather_attention` 的 1／4／8／16 slot 配置共享整数读取后的 FP16／FP32 转换、scale 广播及 BF16 输出边界，按离线测量选定配置后发布。`validate_private_rows --op kv_int8.gather_attention --rows 1 17 128` 对照独立参考，并检查无效索引、环形 SWA、反向 owner、非对齐尾部及修改 KV／scale 后的 Graph replay。

`bulk_store:true` 使用与下游 attention 一致的连续 512 维视图，一次写回整个 slot tile，合并逐 slot 的输出等待与拷贝。`validate_private_rows` 加 `--bulk-store` 验证此调度，并覆盖 FP16 scale 的零值、次正规值及极值。完整模型的 prefix 恢复诊断覆盖 17、129、2049 和 8193 token，包含环形 SWA 跨界后的 owner 迁移。

上述 `private_schedule` 命令使用 `--parameters '{}' --reuse-long-kv` 时，在同一 prefill packet 内复用已经转换的长 KV；每层继续更新私有的 128 个 SWA slot。离线检查共享 bank、scale、选中索引、输出及 mask 的指针列和行布局，遇到相关写入或别名写入时重新完整转换。每个 packet 的首次读取、owner range 未覆盖部分和未知指令后的读取均执行完整转换。所有配置复用 `kv_int8.gather_attention`，保持原始 INT8／FP16 scale 到 BF16 的边界，普通 decode 保留原绑定。`validate_private_rows --op kv_int8.gather_attention --kv-compressed --bulk-store --reuse-long --rows 1 17 129` 对照完整 gather，覆盖初始化后 SWA 改变、反向 owner、尾部和 Graph replay。

Cube PV 的输出列 tile 通过同一算子工厂与离线测量选择。下面的验证对照原 32 维 tile，检查完整输出、FP32 workspace、反向 owner、原地址数据改变和 Graph replay；测量覆盖连续 1–128 行、容量上限及常见大分块两侧：

```bash
bash tools/operators/ascend/run.sh python3 -m tools.operators.ascend.validate_private_rows \
  --output PV_TUNING --op attention_cube.pv --owners 3 --rows 1 17 129 \
  --pv-tile-d 32 64 128 256 \
  --benchmark-rows {1..129} 255 256 257 511 512 513 799 800 801 1023 1024 1025 2047 2048
bash tools/operators/ascend/run.sh --without-devices python3 -m tools.model.deepseek_v41.private_schedule \
  --plan PLAN --output OUTPUT --operator attention_cube.pv \
  --tuning-record PV_TUNING/result.json --tuning-parameter tile_d --maximum-regression 0.1
```

该入口复用共享调优缓存、Graph 测量及逐样本回退约束，选择一个动态配置和至多一个小 owner 边界；回退约束覆盖记录中的实测点。测量、算子及编译辅助源码、编译器和目标身份均参与检查，缺少数值验证、连续小行数或容量上限时拒绝发布。新包只保留被选中的资产，普通 decode 保留原绑定。PV 的 BF16 概率、每 128 个 KV slot 的 FP32 累加与 BF16 输出边界保持一致，发布前继续验收完整模型请求及 2048／2049 等尾部。

`bench-ascend-batch` 的显式 `frames` 可携带 `decode:true` 和固定 token 历史；阶段时间与 decode 输出计数跟随 frame 角色。比较 decode 调度时保持相同输入历史，并单独保留自由生成结果。

owner-private 计划可复用查询行分片构建器：

```bash
bash tools/operators/ascend/run.sh --without-devices python3 -m tools.model.deepseek_v41.shard_indexer \
  --plan PLAN --output artifacts/a2-query-partition --compile-workers 4
```

该构建器对完整动态 prefill 行域使用同一套 INT8 indexer 和 TopK 工厂，各卡计算间隔分配的查询行，回收每条查询的 512 个 INT32 索引并恢复全局行序。共享 `Rows` 派生每个原 owner 的紧凑行数和偏移；`partitioned_tables` 声明指针表／work 表的分片步长，行列的 `row_partition` 声明物理容量。没有选中行的 owner 保留原槽位并使用空 work guard；私有 key、scale、KV 和压缩状态继续按原 owner 绑定。Graph 身份包含有序的 owner 行数布局，原地址表仍随 owner、真实行数和输出角色更新。构建器验证物理范围并裁掉失效的指针表、列和工作区；decode 使用原绑定。全局 TopK value 消费者需要显式回收契约，当前入口拒绝这类输入计划。 `validate_sharded_indexer --baseline BASELINE --plan PARTITIONED --output OUTPUT` 对照已发布的原算子链，覆盖不同私有 key、完整有效分数、TopK 值／索引、精确 tie、空 owner 分片、尾部和改变输入后的 Graph replay；`--rows` 可指定检查的全局查询行数。实际集合通信与请求状态由完整模型及 prefix 恢复验收。

私有查询分片构建在输入未声明 `direct_sum` 时默认启用直接累加：沿原有 32 头 FP32 顺序读取 UB 分数行，减少逐头临时复制。输入中显式声明的配置继续保留。已有包可用 `private_schedule --operator indexer_cube.causal_logits --parameters '{"direct_sum":true}'` 复用同一计算体；默认只替换 prefill，decode 沿用原资产。索引权重使用合并后的头／维度缩放因子，在 FP16 舍入后参与逐头累加；独立参考保持这一边界，包括 FP16 次正规数的中点舍入。

完整 attention 可继续沿用查询分片，并复用现有 Cube QK、softmax、PV 和私有 KV gather 工厂：

```bash
bash tools/operators/ascend/run.sh --without-devices python3 -m tools.model.deepseek_v41.query_attention \
  --plan QUERY_PARTITIONED_PLAN --output artifacts/a2-query-attention \
  --min-rows 512 --pv-tile-d 256 --compile-workers 4
```

该变换要求 TP8 和已经启用查询分片的私有计划。大行数 packet 先通过 AlltoAllV 将每卡的全部查询／8 个头换为间隔分配的查询／64 个头，复用本地 TopK 和按 owner 绑定的 KV；PV 后进行反向交换，恢复原始行序与头分片，再执行原有位置变换、gate 和输出投影。短 packet 与 decode 沿用输入计划。各 owner 的 KV 更新仍覆盖全部输入行，私有列通过显式物理视图表达分片容量，尾部补零仅覆盖无效查询。共享计算体保持 K128 累加和 BF16 概率／输出边界。

原生 `exchange` 指令以每个 peer 的元素数声明均匀 AlltoAllV，支持动态行表达式。捕获的操作持有通信器、源／目标 buffer 和稳定地址的 host counts；集合通信沿主 stream 顺序执行，资源保留到最后一个完成事件。`validate_attention_heads` 验证共享多头计算体；`validate_query_layout --help` 提供已发布移动算子的尾部、头序及改变输入后的 Graph 验证入口。架构验收完成后的配置调优按分阶段检查约定复用这些结果。

QK 和 PV 在同一计算体中支持 `head_tile=16/32/64`，默认 16。tile 须整除按 16 对齐的头维度；不足一个 tile 时使用实际对齐维度，保持原物理行布局。K128 累加顺序、FP32 workspace 和 BF16 概率／输出边界保持一致。`validate_attention_heads --heads 64 --head-tiles 16 32 64 --rows 1 17 65 --benchmark-rows 64 65 256` 对照独立的 8 头执行，检查改变输入后的 Graph replay，并复用共享 Graph 测量记录各候选耗时；离线编译默认使用四个 worker，复用已编译缓存。

根据实测选择配置后，可在 `query_attention` 首次构建时通过 `--qk-head-tile` 和 `--pv-head-tile` 传入；已有包沿用 `private_schedule` 重建。例如选中 64 时，分别对 `attention_cube.qk` 和 `attention_cube.pv` 使用 `--parameters '{"head_tile":64}' --match-parameters '{"heads":64}'`，后一阶段读取前一阶段的输出包。`--match-parameters` 匹配已发布的 primitive 参数，仅替换对应几何；其他 prefill 绑定和 decode 资产继续复用。完整请求短测覆盖实际 owner 聚合、2K/2049 和固定历史 decode，确认收益后再运行一次 Mini。
