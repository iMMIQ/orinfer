# 离线模型构建

Python 用于权重准备、TileLang 编译、调优和验证；在线加载、执行与服务由 Rust 完成。

实现位于 `python/orinfer_build`，构建命令通过 `python -m orinfer_build...` 运行。共享契约、编译、调优
和发布组件分别归入该包，目录与工作区约定见[开发说明](../../docs/development.md)。

## 支持的构建入口

| 模型 | 输入 | 构建入口 |
| --- | --- | --- |
| Qwen3.8-27B | 完成量化的自有码表 Q4A8 checkpoint | [qwen35/build.py](qwen35/README.md) |
| Qwen3-ASR-1.7B | 文本 E8P Q4A8、音频 FP16 checkpoint | [qwen35/build.py](../../docs/audio.md)，`--context 65536` |
| Nemotron-3.5-Embed-1B | 文本和视觉 E8P Q4A8 checkpoint | [nemotron_embed/build.py](../../docs/embeddings.md) |
| Qwen3-Reranker-0.6B | 因果文本 E8P Q4A8、双行原始分类头 checkpoint | [qwen_reranker/build.py](../../docs/reranking.md) |
| Qwen3.8-Flash-Next | 当前 schema-1 模型及其匹配执行包 | [flash_next/build.py](flash_next/README.md) |
| DSv4.1 Flash-1017 / A2 | 当前计划及已预处理 W4A8 权重 | `python -m orinfer_build.models.deepseek_v41.build --rebuild-plan PLAN`，使用 CANN 9.2 前端 |

27B 的当前部署方案为两级整数 E8P Q4A8。Flash Next 使用 E8P Q2A8 专家及质量优先的混合精度。量化组件与存储约定见[量化说明](../quantization/README.md)。

```bash
make python-env
make build
bash tools/environments/cuda/run.sh python/orinfer_build/models/qwen35/build.py artifacts/qwen35-build \
  --source /path/to/converted --model-output artifacts/models/qwen35-ready \
  --gpu-lock-held
bash tools/environments/cuda/run.sh python/orinfer_build/models/flash_next/build.py artifacts/flash-build \
  --model /path/to/native-flash-model --model-output artifacts/models/flash-ready
```

CUDA 编译使用物理机 CUDA 12.6 和独立 uv 环境，A2 使用固定的 CANN 9.2 前端；环境说明见[编译说明](../build/README.md)。外部输入通过 `ORINFER_CHECKPOINT_DIR` 只读挂载。所有 GPU 工作共用 `artifacts/gpu-experiment.lock`；构建输出使用新目录，已发布产物保持不可变。

## 产物与公共组件

模型目录保留 HF 配置、tokenizer、chat template、标准 safetensors 分片和 `cache/model.json`。数据描述通过 digest 绑定独立执行包，包内包含 Rust 模型库、去重的 cubin、源码及 host ABI；在线使用版本 1 的原生模型数据和 C ABI；AOT manifest 用于离线构建。

- `compile.py`、`row_contract.py`、`row_selection.py`：动态行工厂、host ABI 合同与离线选择规则。
- `host_compiler.py`、`build_cache.py`：公共 host ABI 生成与内容缓存；`projection_policy.py`、`projection_tuning.py`：共享整数投影候选与离线测量；`row_convergence.py`：动态 launch 表达式及相邻选择合并。
- `catalog.py`、`binding_templates.py`：共享编译定义和原生参数模板。
- `private_rows.py`：普通 decode 与 MTP 共用的请求私有地址表、scratch 布局和动态行包装；各入口保留自己的状态生命周期。
- `packet_graph.py`、`native_packets.py`：从原生计划提取私有操作，将普通、混合和 MTP packet 的请求地址统一放入固定地址表；保留模型选择的计算体、标量 ABI 和 launch 几何。Qwen 与 Flash Next 构建器在发布前生成该契约，ASR 复用同一绑定。Embedding 和 reranker 按完整序列边界复用 Graph。
- `attention_reference.py`：独立 FP32 attention 参考，供离线候选和质量验证使用。
- `publication.py`、`compact.py`：不可变资产复用、safetensors 合并及原子发布。
- `package.py`：离线执行包校验、XZ 归档、发布工具内部安装及前端资源固定；部署使用 Rust 的 `orinfer package install`。
- `attach_execution.py`：为匹配的数据目录安装执行包。

更新执行包须核对完整程序、参数、动态 launch、状态及 buffer 契约。权重布局改变时重新准备权重；仅重新编译相同计算时可复用匹配的权重分片。

已有 Qwen、Flash Next 或 ASR 原生包可单独迁移公共 packet 地址表，复用原有权重和算术：

```bash
bash tools/environments/cuda/run.sh python/orinfer_build/compiler/packet_graph.py artifacts/packet-build \
  --model /path/to/prepared-model --target artifacts/models/packet-ready
```

Graph 缓存按操作、行数和角色边界区分形状；请求 owner 只改变表中地址。同一形状更换请求后复用已捕获的 Graph。模型适配器继续声明 GDN、KV、卷积、HC、PLE、QSA 和 MTP 的状态顺序。

## 验证与测量

```bash
./target/release/orinfer validate-model /path/to/prepared-model
./target/release/orinfer plan-model /path/to/prepared-model
./target/release/orinfer run-model /path/to/prepared-model examples/requests.json
make check
make check-offline
make check-gpu MODEL=/path/to/prepared-model OUTPUT=artifacts/check-gpu/new-run
```

`validate-model` 完整校验文件与 payload，`plan-model` 只生成 CPU 执行计划。GPU 验收覆盖动态 batch、请求隔离、prefix 恢复、MTP 和图片输入；量化质量另外与同模型 BF16/FP8 固定历史对照。详细口径见[评测说明](../eval/README.md)。

`run.sh` 记录完整 Rust 请求和机器状态；`profile.sh` 与 `profile_summary.py` 用于 Nsight 分析。算子微测、短输出 profile 和完整请求吞吐分别记录，性能参考见[性能文档](../../docs/performance.md)。

共享 GPU 采样由 Qwen3_5 与 Flash Next 正式构建器发布，也可只重建现有原生包的采样部分，保持权重与前端资源：

```bash
bash tools/environments/cuda/run.sh python/orinfer_build/models/sampling.py artifacts/sampling-build \
  --model artifacts/models/source-model \
  --model-output artifacts/models/model-with-gpu-sampling
```

该入口重建 Rust 绑定库并原子发布当前 schema-1 模型，沿用完整程序与 ABI 检查。新包仍需完整请求、状态隔离及 prefix/MTP 验收。greedy 沿用原有路径；需要 top_logprobs 或 MTP p/q 分布的调用保留 CPU 路径。

采样包同时发布同源单请求程序和 1–128 行地址表程序。普通 decode 中至少两个符合条件的随机请求按 owner 批量执行，分别绑定 logits、seed、惩罚、bias 和 grammar mask，集中回读选中 token；单个随机请求复用单请求程序，greedy、logprobs 和 MTP 保持各自的输出语义。临时采样 scratch 按实际行数映射，行数和 owner 改变时复用相同动态 kernel。

Rust 执行器为大块共享工作区保留 CUDA 虚拟地址，在首次上传或执行依赖程序前映射并清零物理内存，保留映射供 Graph replay 使用。文本启动可延后视觉、音频和采样 scratch 的物理分配；请求准入预留尚未映射的计算工作区与采样行。`LOAD STAGES.deferred_workspace_bytes` 记录启动时延后的逻辑字节，scheduler 的 `batch_execution` 记录映射次数、物理字节、耗时及实际采样 packet／行数。该运行时优化可加载已有匹配执行包；批量采样需重建采样包并配合当前引擎使用。

Qwen3_5 的 `gdn_execution` 明确声明 prefill 的 sequential/chunked 策略，以及 decode 的 dense/deferred 策略和日志容量；构建、验证、恢复与提交消费同一契约。原生适配器依据显式契约校验策略及其全部状态和工作区。Attention 的 INT8 读取与临时 FP16 KV 暂存候选按完整流水线实测选择，以小行数、非对齐尾部和大 prefill 的实测结果决定选择区间。

`auxiliary_tuning.py` 可为已有原生包测量并发布 MoE top-k、QSA 选择和 FP16 PLE norm/gate 融合，参数为 `--model`、`--model-output` 和 `--output`；`--families router` 只处理共享 MoE 路由。Flash Next 正式 `build.py` 包含该步骤，Qwen3_5 构建器复用同一 MoE 调优器。几何来自 checkpoint 签名及物理 buffer 视图，支持展开或扁平 workspace；数值、尾部和 Graph 检查通过后才参与选择。QSA 与 PLE 汇总多长度测量选择一个配置，Rust 适配器验证完整生产者链后应用融合并生成执行顺序。
