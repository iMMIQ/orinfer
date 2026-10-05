# 性能工具

`reference_bench.py`测量兼容OpenAI SSE接口的prefill/TTFT与decode时间；`test_reference_bench.py`检查token chunk和计时间隔。

参考服务的配置及身份须记录在输出目录的`reference-lock.json`中，输入token由`--seed-tokens /path/to/input-tokens.json`显式提供。请求要求服务支持token-ID输入/输出、固定长度生成和可选的独立计时诊断；不能直接用普通Chat请求代替。

`run_aot_artifact.sh MANIFEST NEW_OUTPUT`构建Rust CLI并执行AOT fixture，保留实际运行身份和机器状态。模型完整推理使用`tools/model/run.sh`。

`sample_machine.py`和`sample_continuous.py`只读记录机器温度、频率、内存和功耗；不会改动全局设置。GPU运行入口共享`artifacts/gpu-experiment.lock`。

`validate_plan.py`校验benchmark与快速评测配置，属于CPU检查，不产生性能测量。

`concurrency.py`以真实SSE请求测量1..128并发，输入为`{ "cases": [{ "id": "case", "request": { ...Chat API body... } }] }`：

```bash
python3 tools/bench/concurrency.py --requests requests.json \
  --concurrency 1 2 4 8 16 32 128 --repetitions 3 --output artifacts/concurrency.json
```

报告包括真实usage输出token数、整轮吞吐、TTFT分位数、缓存tokens及每请求结果。吞吐包含排队、prefill、HTTP和输出，不能直接当作GPU纯decode TPS；流式chunk间隔也不能当作token ITL。重复请求会命中prefix cache；测冷请求时需构造不同输入或关闭cache。`/health.scheduler_statistics`提供prefill实际计算tokens、引擎提交的decode tokens、batch直方图、混合迭代及峰值活跃数。

健康接口还提供准入、request startup、prefix恢复、prefill完成、batch输入准备/计划/提交及Graph统计。`batch_execution`记录batch Graph命中、未命中、捕获的kernel/copy/memset操作数、淘汰、地址变化失效和分项耗时；单请求/MTP程序运行期间的新捕获另记`sequence_captures/sequence_capture_s`。这些是累计主机计时，replay包含同步等待，不能视为CUDA event的GPU时间。`request_start_s`包含`prefix_restore_s`，batch Graph分项包含在`compute_s`中；单程序捕获可能发生在startup或计算阶段，不能把这些重叠计时重复相加。

工具在发送请求前、整轮完成后取得健康快照，并输出`scheduler_delta`和`admission_delta`；累计峰值不作差值。测量期间应独占服务，重启导致计数器回退时会报错。`admission_statistics`记录各轮处理的请求数、完整缓存命中的文本请求数、准入直方图及墙钟时间；它包含策略、显存准入检查和request startup。

`benchmark_prefill_scheduling`是默认跳过的真实GPU基准，比较同一模型的联合prefill与逐请求大块prefill。夹具包含`model`、新的`output`路径、`cuda_graph`、`batches`（2..8）、`repetitions`、`output_tokens`及`cases`；case使用`GenerationInput`字段`input_tokens`、`max_new_tokens`、`sampling`，只接受文本输入。采样应设置`temperature: 0`、`seed: 20261002`。关闭本项目GPU服务后，在独占实验锁下运行：

```bash
ORINFER_BATCH_FIXTURE=artifacts/prefill-scheduling-fixture.json \
  flock artifacts/gpu-experiment.lock \
  cargo test --release --offline -p orinfer-engine benchmark_prefill_scheduling -- --ignored --nocapture
```

基准关闭MTP和prefix复用，交换两条路径的测试顺序，分别记录准入、所有请求完成prefill的吞吐、每请求TTFT和后续batch decode。Prefill计时包含head和首token选择，TTFT另含准入；整个cohort完成prefill后才开始decode，未测量在线混合调度的ITL。联合路径使用算子包声明的分块，逐请求路径使用其最大可用单请求块；不同块可能采用不同量化码本。输出token差异用于诊断，不能替代独立BF16/FP8质量验收。每个请求另校验最终位置及KV回收。

`mixed.py`先预热一个文本`anchor`，启动指定数量的缓存命中请求，在每条流收到若干有效输出片段后注入一个冷请求。输入为`{ "anchor": { ...Chat API body... }, "cold_cases": [{ "id": "cold", "anchor_count": 2, "request": { ... }, "expected_prompt_tokens": 8192, "target_text": "CODE-641" }] }`。所有请求使用`temperature: 0`、`seed: 20261002`；anchor须足够长以保持生成。每个并发档位的冷请求应在输入开头使用不同标识，避免跨轮缓存；`anchor_count`限定case适用的档位，省略时适用于所有档位。提示token数和任务答案字段可省略。

```bash
python3 -m tools.bench.mixed --requests mixed-requests.json \
  --anchor-counts 1 2 4 8 --trigger-chunks 4 --output artifacts/mixed.json
```

报告包含无注入对照、长请求TTFT、anchor输出吞吐、长请求prefill期间重叠的SSE片段间隔及引擎计数器差值。跨越注入边界的停顿也计入；片段间隔不是token ITL。要求anchor完整命中prefix，冷请求命中不超过5%，通过准入计数检查无其他流量。测试采用服务当前配置，包括自适应MTP；纯decode吞吐与关闭MTP的基准应另测。

`acceptance.py`执行带延迟到达的异构任务验收，覆盖长短文本、图片、多图和多轮分支。JSON含`cases: [{id, request, expected_words}]`、可选`warmup: [id]`与`rounds: [{id, jobs: [{case, delay_s}]}]`；每轮最多128个提交请求。答案检查仅用于明确的标签/简短答案任务，忽略大小写和分隔符；并发吞吐从真实usage计算。输出保存每轮scheduler/admission差值、活跃/排队峰值及每请求结果，拒绝不完整SSE或任务答案错误。

```bash
python3 -m tools.bench.acceptance --requests acceptance-requests.json \
  --output artifacts/acceptance.json
```
