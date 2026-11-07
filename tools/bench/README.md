# 性能工具

六模型统一基准、2B/4B/9B 完整网格及 Rust CLI / Chat API 实测见[性能参考](../../docs/performance.md)。该文档分别列出执行层 prefill、输入 tokens / API TTFT、排除 TTFT 的 decode、冷进程启动及完整 prefix 命中时延；本目录工具分别按对应口径报告结果，比较时应使用相同指标。

`reference_bench.py` 测量兼容 OpenAI SSE 接口的 prefill/TTFT 与 decode 时间；`test_reference_bench.py` 检查 token chunk 和计时间隔。

参考服务的配置及身份须记录在输出目录的 `reference-lock.json` 中，输入 token 由 `--seed-tokens /path/to/input-tokens.json` 显式提供。请求要求服务支持 token-ID 输入/输出、固定长度生成和可选的独立计时诊断。普通 Chat 请求使用下方 HTTP 基准测量。

`run_aot_artifact.sh MANIFEST NEW_OUTPUT` 构建 Rust CLI 并执行 AOT fixture，保留实际运行身份和机器状态。模型完整推理使用 `tools/model/run.sh`。

`sample_machine.py` 和 `sample_continuous.py` 以只读方式记录机器温度、频率、内存和功耗，保持全局设置。GPU 运行入口共享 `artifacts/gpu-experiment.lock`。

`bash tools/environments/cuda/run.sh tests/cuda/operators/sampling.py NEW_OUTPUT --batch-only` 对照同源单请求采样，覆盖不同 seed/参数、非对齐词表、1–128 行、owner 换序、空行、无效行隔离和改变输入后的 Graph replay。`--sanitizer` 缩小诊断集合，配合 `ORINFER_OPERATOR_SANITIZER=memcheck` 检查设备访存。

完整请求验收 `validate_batched_gpu_sampling_requests` 使用下方 `ORINFER_BATCH_FIXTURE`，需要包含批量采样程序的新模型包；`batches` 默认 1/3/8。它比较逐请求／批量 GPU 采样的完整 token 和因果状态，覆盖混合 greedy、不同参数、grammar 解除、反向 owner 以及取消后的 arena 复用。`validate_lazy_workspace_requests` 需要同一夹具内的纯文本和图像／音频 case，检查先文本、首次媒体、再文本时的结果、状态与虚拟地址。按本轮构建返回的实际测试可执行文件运行并核对测试数；吞吐比较使用 release 完整请求。

`runtime::requests::profile::profile_continuous_decode` 的夹具可设置 `sampling_batch=false`，在同一执行包上诊断逐请求 GPU 采样；默认使用包声明的批量路径。`hot_only=true` 记录固定 batch 的主干 GPU 时间及 CPU 分段，批量采样另记录控制回读、地址表、输入上传、提交和输出回读边界。分段用于定位，完整请求吞吐需另行验收。

`validate_plan.py` 校验 benchmark 与快速评测配置，输出 CPU 配置检查结果。

`output_stress.py` 对已启动的 Chat 服务执行定时混合负载，复用 `concurrency.py` 的 SSE 客户端，按 usage 统计真实输出 token。默认依次运行 1/8/16/32 并发，共发放请求 600 秒，每段结束等待在途请求完成；实际总时长含各段排空。输入为 `{"cases":[{"id":"math","request":{"model":"example","messages":[{"role":"user","content":"1+1，只输出JSON对象，字段value为整数。"}]},"checks":{"json_equals":{"value":2}}}]}`；`checks.contains` 可指定答案必须包含的字符串。请求原样保留采样参数和 EOS 设置，省略 seed 时使用服务默认行为。

```bash
uv run --locked python -m tools.bench.output_stress \
  --cases artifacts/output-cases.json --tokenizer /path/to/model/tokenizer.json \
  --base-url http://127.0.0.1:8088/v1 --output artifacts/output-pressure \
  --phases 1:30 8:120 16:210 32:240
```

新输出目录保留完整请求、逐请求正文及 thinking、usage、首输出及流间隔、循环片段诊断和每 5 秒的 health 采样。请求错误、答案检查失败或 health 错误使工具返回非零退出码。384-token 以上的三次连续循环作为异常线索；长度截断单独统计。自然语言、代码正确性、循环误报及性能长尾需结合保留的回答和相同配置基线复核，工具退出成功只表示这些自动检查通过。

`code_generation.py` 用相同 Chat 请求测量 thinking、MTP 和 prefix cache 的组合，输入采用下方 `concurrency.py` 的 cases 格式：

```bash
python3 -m tools.bench.code_generation --requests code-requests.json \
  --repetitions 3 --output artifacts/code-generation.json
```

每个 case 先执行一次预热，再重复请求；输出保留全部正文、思考文本、usage、首 token、首正文时间、实际缓存 token 命中率和 scheduler 差值。吞吐按已提交的 completion tokens 除以完整 HTTP 耗时计算，包含 thinking、prefill 和传输。token 数以提交的 usage 为准，MTP 草稿与 SSE 片段分别统计。报告单独列出空正文及预算截断，完整代码的性能与质量使用自然完成请求验收。测冷请求需先重启自己的服务或使用新前缀；服务配置和输入保持一致，测量期间应独占接口。

该工具也适用于翻译任务：以 `--requests fixtures/translation-benchmark.json` 测量完整译文，核对三个段落、时间和标识符。固定 batch 的 `benchmark_flash_decode_template` 诊断通过引擎输出预算固定长度，并保持原始 EOS 分数和普通 greedy 路径；计时范围为就绪后的固定 batch decode。

`concurrency.py` 以真实 SSE 请求测量 1..128 并发，输入为 `{ "cases": [{ "id": "case", "request": { ...Chat API body... } }] }`：

```bash
python3 tools/bench/concurrency.py --requests requests.json \
  --concurrency 1 2 4 8 16 32 128 --repetitions 3 --output artifacts/concurrency.json
```

报告包括真实 usage 输出 token 数、整轮吞吐、TTFT 分位数、缓存 tokens 及每请求结果。吞吐按包含排队、prefill、HTTP 和输出的整轮时间计算；流式 chunk 间隔按传输片段计时，token ITL 需单独测量。重复请求会命中 prefix cache；测冷请求时需构造不同输入或关闭 cache。`/health.scheduler_statistics` 提供 prefill 实际计算 tokens、引擎提交的 decode tokens、batch 直方图、混合迭代及峰值活跃数。

健康接口还提供准入、request startup、prefix 恢复、prefill 完成、batch 输入准备/计划/提交及 Graph 统计。`batch_execution` 记录 batch Graph 命中、未命中、捕获的 kernel/copy/memset 操作数、淘汰、地址变化失效和分项耗时；单请求/MTP 程序运行期间的新捕获另记 `sequence_captures/sequence_capture_s`。这些指标是包含同步等待的累计主机计时；GPU 时间使用 CUDA event 单独测量。`request_start_s` 包含 `prefix_restore_s`，batch Graph 分项包含在 `compute_s` 中；单程序捕获可能发生在 startup 或计算阶段，总耗时按外层计时读取，分项用于归因。

工具在发送请求前、整轮完成后取得健康快照，并输出 `scheduler_delta` 和 `admission_delta`；累计峰值保留原值。测量期间应独占服务，重启导致计数器回退时会报错。`admission_statistics` 记录各轮处理的请求数、完整缓存命中的文本请求数、准入直方图及墙钟时间；它包含策略、显存准入检查和 request startup。

`benchmark_prefill_scheduling` 是默认跳过的真实 GPU 基准，比较同一模型的联合 prefill 与逐请求大块 prefill。夹具包含 `model`、新的 `output` 路径、`cuda_graph`、`batches`（2..8）、`repetitions`、`output_tokens` 及 `cases`；case 使用 `GenerationInput` 字段 `input_tokens`、`max_new_tokens`、`sampling`，只接受文本输入。采样应设置 `temperature: 0`、`seed: 20261002`。关闭本项目 GPU 服务后，在独占实验锁下运行：

```bash
ORINFER_BATCH_FIXTURE=artifacts/prefill-scheduling-fixture.json \
  flock artifacts/gpu-experiment.lock \
  cargo test --release --offline -p orinfer-engine benchmark_prefill_scheduling -- --ignored --nocapture
```

基准关闭 MTP 和 prefix 复用，交换两条路径的测试顺序，分别记录准入、所有请求完成 prefill 的吞吐、每请求 TTFT 和后续 batch decode。Prefill 计时包含 head 和首 token 选择，TTFT 另含准入；整个 cohort 完成 prefill 后才开始 decode，在线混合调度的 ITL 由独立混合请求基准测量。联合路径使用算子包声明的分块，逐请求路径使用其最大可用单请求块；不同块可能采用不同量化码本。输出 token 差异用于诊断，量化质量以独立 BF16/FP8 对照验收。每个请求另校验最终位置及 KV 回收。

`mixed.py` 先预热一个文本 `anchor`，启动指定数量的缓存命中请求，在每条流收到若干有效输出片段后注入一个冷请求。输入为 `{ "anchor": { ...Chat API body... }, "cold_cases": [{ "id": "cold", "anchor_count": 2, "request": { ... }, "expected_prompt_tokens": 8192, "target_text": "CODE-641" }] }`。所有请求使用 `temperature: 0`、`seed: 20261002`；anchor 须足够长以保持生成。每个并发档位的冷请求应在输入开头使用独立标识，以测量冷请求；`anchor_count` 限定 case 适用的档位，省略时适用于所有档位。提示 token 数和任务答案字段可省略。

```bash
python3 -m tools.bench.mixed --requests mixed-requests.json \
  --anchor-counts 1 2 4 8 --trigger-chunks 4 --output artifacts/mixed.json
```

报告包含无注入对照、长请求 TTFT、anchor 输出吞吐、长请求 prefill 期间重叠的 SSE 片段间隔及引擎计数器差值。跨越注入边界的停顿也计入；片段间隔按 SSE 传输单元计时。要求 anchor 完整命中 prefix、冷请求命中率至多 5%，并通过准入计数确认流量全部来自本轮测试。测试采用服务当前配置，包括自适应 MTP；纯 decode 吞吐与关闭 MTP 的基准应另测。

`acceptance.py` 执行带延迟到达的异构任务验收，覆盖长短文本、图片、多图和多轮分支。JSON 含 `cases: [{id, request, expected_words}]`、可选 `warmup: [id]` 与 `rounds: [{id, jobs: [{case, delay_s}]}]`；每轮最多 128 个提交请求。答案检查仅用于明确的标签/简短答案任务，忽略大小写和分隔符；并发吞吐从真实 usage 计算。输出保存每轮 scheduler/admission 差值、活跃/排队峰值及每请求结果；SSE 完整且任务答案正确时通过。

```bash
python3 -m tools.bench.acceptance --requests acceptance-requests.json \
  --output artifacts/acceptance.json
```

`validate_flash_decode_batches` 使用同一 `ORINFER_BATCH_FIXTURE` 格式，检查 2/3/4/8/16/32 请求的同历史 logits/top3/NLL、图片与多图、取消/槽位复用，以及 target batch 后继续 MTP。需 native Flash batch 包和独占 GPU 锁：

```bash
ORINFER_BATCH_FIXTURE=/absolute/path/to/flash-fixture.json \
  flock artifacts/gpu-experiment.lock \
  cargo test --release --offline -p orinfer-engine validate_flash_decode_batches -- --ignored --nocapture
```

该检查关闭 prefix cache，对照同权重、相同冷 prefill 和 teacher-forced 历史的单序列执行，验证 batch 迁移的算术与隔离；原始 BF16/FP8 量化质量另行验收。性能与缓存命中的功能验收另用真实 Chat/SSE 的 `concurrency.py`、`acceptance.py` 测量。

`validate_flash_heterogeneous_scheduling` 使用相同夹具格式，第一条 case 的输入须至少 4096 tokens，第二条为短请求。它分别检查两条 cold prefill 和已开始 decode 的短请求与长 prefill 混跑，确认长请求保留原生大块、活跃 decoder 期间块长受限、连续 decode 有推进，以及相同分块和 teacher-forced 历史下的 KL/NLL/top3；同样需要独占 GPU 锁。独立最大块与中断分块的首 token 分布差异另行记录，用于诊断分块数值差异；状态隔离和 BF16 量化质量分别验收。`validate_flash_bounded_prefill` 按完整成本选择联合或独立路径，两者均为合法执行方式。

`validate_flash_long_generation_reuse` 使用相同夹具格式，前两条 case 为压力提示和后续验证提示；后者须包含原生 4096-token prefill。它先记录后续提示的冷请求 logits 与完整已提交状态，再强制执行 16384 个 decode tokens（含 MTP）、保存生成缓存并复用槽位，检查冷请求和完整 prefix 命中与参考的分块、logits、KV/GDN/卷积/PLE/MTP 状态一致。引擎测试刻意继续处理 EOS 来保证压力长度，任务输出正确性另用真实 Chat API 验证；同样需要独占 GPU 锁。

固定到达时间的混合请求验收使用 `python -m tools.bench.scheduling --requests TRACE.json --output RESULT.json`。输入为 `{"requests":[{"id":"short-1","at_s":0.1,"request":{"model":"MODEL_ID","messages":[{"role":"user","content":"Hello"}],"max_tokens":16}}]}`，最多 128 条；同一时刻可提交多条请求。请求按固定时间独立到达，报告客户端到达误差、TTFT、整请求耗时和最大流式 chunk 间隔的 P50/P95/P99，以及失败数、有效输出吞吐、成功请求的 prefix 请求/token 命中率和调度统计。chunk 间隔按传输片段统计，token ITL 需单独测量。比较调度版本时使用相同 trace、模型包、缓存/MTP/Graph 配置，分别测试冷缓存与预热后的缓存。

`profile_vision_stages` 是默认跳过的真实模型诊断，使用 `ORINFER_BATCH_FIXTURE` 中的 `model`、新 `output` 路径和带 `images` 的 `GenerationInput` cases。它按实际图片网格绑定视觉程序，记录整段热 Graph 与逐算子 GPU 时延；`ncu: true` 对实际 launch 添加 NVTX 区间。该计时覆盖 GPU 视觉程序；图片解码、resize、特征复制和文本生成的完整耗时通过 Chat API 另测。通过同一个 GPU 锁运行 `cargo test --release -p orinfer-engine profile_vision_stages -- --ignored --nocapture --test-threads=1`。

冷启动端到端测量使用 `python -m tools.bench.startup --model MODEL_DIR --request REQUEST.json --output artifacts/startup --server-args --mtp-drafts 0 --cuda-graph decode_only`。每轮启动独立 Rust CLI，通过 HTTP 等待就绪并提交首条流式 Chat 请求；记录进程启动到 ready、首个非空文本／thinking／工具片段及请求完成的耗时、usage 和原始输出，默认三轮并报告中位数。计时从进程启动开始，到请求完成结束；GPU 锁等待和退出清理另计，文件缓存保持原状。该结果衡量保留系统缓存的新进程启动，冷盘读取需在对应条件下另测。工具在子进程退出前持有共享 GPU 锁；`--gpu-lock-timeout 60` 可等待当前 GPU 工作结束。

## 固定工作量与并发 prefill

`runtime::requests::tests::benchmark::benchmark_fixed_requests` 通过真实请求状态机测量固定工作量。`ORINFER_BATCH_FIXTURE` 指定 JSON 夹具，包含 `model`、新 `output` 路径、`repetitions`、`cuda_graph` 和 `cases`。每个 case 包含 `name`、`prompts`（原生模板展开后的 token-ID 数组）、`output_tokens` 及可选的 `concurrent_prefill`。

单流 prefill 设置 `output_tokens=1`；decode 设置 65，首输出完成后再计时后续 64 个提交 tokens。并发 prefill 设置 `concurrent_prefill=true`、`output_tokens=1`，同时推进全部等待请求，按最后一个首输出完成时刻计算吞吐。并发 decode 采用首输出屏障，随后计量固定活跃请求集合。每组先预热一次，再测 `repetitions` 轮；汇总采用热态中位数。该诊断固定忽略 EOS、关闭 MTP/prefix cache，greedy 与 seed `20261002` 保持一致。

```bash
ORINFER_BATCH_FIXTURE=/absolute/path/to/fixed-requests.json \
  flock artifacts/gpu-experiment.lock \
  cargo test --release --locked --offline -p orinfer-engine \
    runtime::requests::tests::benchmark::benchmark_fixed_requests \
    -- --ignored --exact --nocapture --test-threads=1
```

输出保存完整 token 序列与各轮首输出、decode 和墙钟计时。自然完成的翻译、代码质量及 HTTP 排队性能分别使用上方 Chat API 工具验收。最新测量表及执行包身份见[性能参考](../../docs/performance.md)。


`encoders.py` 测量完整嵌入或重排 HTTP 请求。冻结输入文件格式为 `{"cases":[{"id":"case","request":{...}}]}`；嵌入使用完整 `input` 数组和 `input_type`，重排使用 `query` 与完整 `documents` 数组。两个后端使用同一份文件，工具为 vLLM 的嵌入接口转换为 Cohere `/v2/embed`，保留图片顺序和 query/document 角色并禁止截断。测量期间由调用者持有全局 GPU 锁，独立服务使用私有 worker 锁，避免重复获取全局锁。身份文件记录引擎版本、模型来源及 revision、精度、执行包或容器 digest、启动参数和设备；输入 hash 固定在报告中。

```sh
python -m tools.bench.encoders --task rerank --backend orinfer \
  --url http://127.0.0.1:8089 --model bench --requests inputs.json \
  --identity native-identity.json --output native-results.json
python -m tools.bench.encoders --compare vllm-results.json native-results.json \
  --output comparison.json
```

默认每个 case 预热一次、测量五次，记录中位数时延、每秒完成项数、全部测量时延和完整响应。比较同时列出各 case 加速比、几何均值、整套工作量加速比及是否每项达到 2 倍；输出比较包含嵌入 cosine 或重排 top1 与概率误差。算子微测与某个 case 的结果各自说明，完整性能结论以相同配置和全部 workload 的真实结果为准。

重排基线启动 vLLM 时使用 `--chat-template tools/bench/templates/qwen3_reranker.jinja`，保持 Qwen 官方 prefix/body/suffix 及末尾的两个换行。`check_kernels.py` 核对模板输出与原生前端参考的 token ID。比较工具要求两边报告实际输入 token 用量，并拒绝用量不一致的结果；模板、分词和截断策略在测量前保持一致。

原生构建逐项比对最终 `plan-model` 的 `row_execution.variants` 与离线选中的行数变体，确认执行计划实际引用这些绑定。构建基准身份文件时，记录所用引擎文件的 SHA256、执行包 digest 和实际选择表数量。
