# JSONL 离线推理

`orinfer batch` 通过 Rust CLI 直接加载模型、处理本地文件并写入结果。它与 `serve` 共用 Chat 模板、tokenizer、采样、停止判断、连续批处理、prefix cache 和模型执行包。

```bash
orinfer batch /path/to/model \
  --model index-translate-2b \
  --input requests.jsonl --output results.jsonl \
  --max-active-requests 64 --mtp-drafts 0 --prefix-cache-mib 512 \
  --default-request-params '{"temperature":0,"seed":20261002,"enable_thinking":false,"max_completion_tokens":512}'
```

并发、Graph、MTP、缓存和预处理预算参数与 `serve` 共用，默认最多 32 个活跃请求。完成一条即补位；请求状态仍受内存准入限制。64 与 128 路的吞吐取决于输入和输出长度，应按实际工作负载测量。

`batch` 默认 `--scheduler-policy throughput`，以整份文件的完成时间为目标：有容量时继续准入长短请求；复用同一套动态执行模板，按完整迭代的实测成本选择联合或独立 prefill。联合形状可先探测再选择，prefill 与 decode 按吞吐成本连续安排。该策略的迭代时长由实际工作量决定；`--prefill-budget-ms` 和 `--target-tpot-ms` 用于 latency 策略，单条请求的首 token 或输出间隔可能增加。内存、活跃数、执行包容量、输出预算及停止规则仍然生效；MTP 继续按实测收益选择。

使用 `--scheduler-policy latency` 可恢复在线策略，`serve` 的默认值仍为 `latency`。续跑可切换调度策略。

`--batch-order auto`（默认）在吞吐模式下先扫描文本请求，用相同 Chat 前端提取实际 token 前缀。关闭 prefix cache 时先安排较长请求；开启时按压缩前缀树安排组内次序、优先处理较重子树，为共享分支提供 checkpoint 提示。前缀索引与规划工作区合计不超过预处理预算的四分之一，每条最多保留 4096 个前缀 token；预算不足时回退到输入顺序。图片及非字符串 content 直接走有界预处理。`--batch-order input` 可关闭全文件规划；GPU 连续调度仍可重排已准备好的请求。

吞吐模式从自然完成的请求学习输入/输出长度比例，按预计剩余输出配合已有 cohort 准入；请求的 `max_completion_tokens` 只作为上限。完成一条即补位，等待过久的请求仍优先推进。

全局文本规划还会减少一次性 prompt 和生成续写的缓存快照；已知共享的完整 prompt、部分 checkpoint 和请求状态仍按公共缓存规则处理。对于短且互不重复的翻译，可用 `--prefix-cache-mib 0`；长公共前缀场景应另测缓存收益。

## 输入

UTF-8 JSONL，每行一个 OpenAI Batch 格式的 Chat 请求：

```json
{"custom_id":"paragraph-1","method":"POST","url":"/v1/chat/completions","body":{"messages":[{"role":"user","content":"请翻译为中文：The service is ready."}]}}
```

也接受直接 Chat 请求，可附带 `custom_id`：

```json
{"custom_id":"paragraph-2","messages":[{"role":"user","content":"请翻译为英语：服务已就绪。"}],"max_tokens":128}
```

缺省 `model` 使用 CLI 的 `--model`，显式指定时必须匹配。其他 Chat 字段遵循[服务指南](serving.md)，包括模型支持的图片、多图和工具调用；本地图片可使用 data URL，远程图片沿用服务端下载限制。每条请求单独应用模型模板并保持独立 prompt。

- `custom_id` 必须唯一且为 1–256 字节的字符串；省略时使用物理行号 `line-N`。空白行跳过，重复 ID 在加载 GPU 模型前报错。
- 单行最多 32 MiB。语法和请求错误写入该行的错误结果，其他行继续；文件读取失败、超大行等作业级错误终止任务。
- 离线返回完整响应，显式 `stream=true` 或 `stream_options` 报错。部署默认值中的流式选项被关闭，其余默认值仍按 Chat 的显式覆盖规则合并。
- 输入、图片资源及模型文件在任务和续跑期间应保持一致。程序检查输入文件变化；远程图片按 URL 读取，内容稳定性由资源提供方保证。

[翻译示例](../fixtures/translation-batch.jsonl) 可直接作为输入。设置足够的输出预算，并按 finish_reason 识别完整与截断译文。

## 输出与续跑

结果按完成顺序写入，可用 `custom_id` 对应原始记录。成功结果遵循 OpenAI Batch 的响应结构：

```json
{"id":"batch-paragraph-1","custom_id":"paragraph-1","response":{"status_code":200,"request_id":"chatcmpl-...","body":{"choices":[{"index":0,"message":{"role":"assistant","content":"服务已就绪。"},"finish_reason":"stop"}],"usage":{"prompt_tokens":24,"completion_tokens":8,"total_tokens":32}}},"error":null}
```

错误结果包含 `custom_id`、`response:null` 和 `error.code/message`。完整响应仍包含正常的模型、usage、thinking 或工具结果等字段。

新任务使用新输出文件。输出旁的 `results.jsonl.meta.json` 保存输入 SHA256、模型描述身份和请求默认值；请连同结果一起保留。续跑使用相同命令加 `--resume`：

```bash
orinfer batch /path/to/model \
  --model index-translate-2b \
  --input requests.jsonl --output results.jsonl \
  --max-active-requests 64 --mtp-drafts 0 --prefix-cache-mib 512 \
  --default-request-params '{"temperature":0,"seed":20261002,"enable_thinking":false,"max_completion_tokens":512}' \
  --resume
```

已提交的正常和截断响应都会跳过，失败及未完成请求重试。同一 ID 的重试结果追加在末尾，以最后一条为准。已提交的截断结果单独统计并在续跑时跳过；增加预算后应使用新的输出文件。输入、模型描述、模型 ID、模型默认策略、请求默认值和 MTP 配置均相同时才可续跑到原结果文件；调度、读取次序、并发和内存预算可调整。模型默认参数策略变化会使续跑身份不匹配；旧策略生成的结果文件保留，使用新输出文件启动任务。

每条完成结果进入有界写入队列，由独立线程顺序落盘，约每秒及退出时同步到存储；排队结果仍计入预处理内存预算。SIGINT/SIGTERM 停止新工作，排空已进入写入队列的结果后退出；写入或同步失败会使任务失败。续跑只移除末尾未以换行提交的半条记录，已完成行损坏时报错。输出由进程锁保证单进程追加。同步完成的写入具有落盘保证，断电可能丢失待同步结果；中断续跑可能重复执行对应输入，外部副作用需由调用方实现幂等。

标准输出为 JSON 汇总：调度策略（`scheduler_policy`）、读取次序（`batch_order`）、规划记录数/前缀 tokens/耗时（`planned_requests`、`planning_prefix_tokens`、`planning_s`），以及任务总量、成功、截断、失败、未完成、跳过、本次处理量、实际生成 tokens、峰值活跃请求及耗时。`output_tps` 按本次生成 tokens / 完整调用耗时计算，包含扫描、加载、规划、预处理、排队和落盘，是完整作业吞吐。正常全部完成退出码为 0；存在错误、截断或中断时非零。

MTP 自动调度在同一验证行预算内，按各请求此前实际测试位置的条件接受率分配不同验证长度，并遵守包声明的合法长度、剩余输出及上下文容量；本轮预算使用此前已完成轮次的统计。`mtp_request_depth_histogram` 记录实际逐请求深度，`mtp_depth_histogram` 保留总预算对应的名义深度。连续 worker 在 packet 提交后、读回完成前执行有界队列维护和输出发送；`maintenance_calls/maintenance_s` 记录该 CPU 工作，其中无 packet 的迭代在返回前执行。下一轮仍按既有请求状态提交及缓存完成条件启动。

`maintenance_s` 可与 GPU 等待及 `compute_s` 重叠，总耗时使用独立墙钟计时。

MTP 冷启动先试探最短合法深度；其他预计收益较低的深度需等待累计时间预算（普通迭代耗时的 2%），每次探索均计入短批次的时间成本。已有实测收益的深度仍按收益选择，长期运行可继续探索待测量的深度。
