# DSv4.1 A2 部署

v0.3.2 支持 glibc 2.34 及以上、CANN 9.2、Ascend 910B TP8 的文本、HTTP/SSE、连续 batch、DSpark、设备及 CPU prefix cache，以及 DSML 工具与 thinking 历史。匹配视觉执行包及视觉权重后支持单图、多图和 OpenCode 读图工具交互。引擎与执行包分别安装，模型目录提供对应的前端和权重。支持范围见[模型包说明](model-packages.md#a2-文本服务与长上下文)。性能按实际工作负载与基线比较。DSH 接入方式见 [DeepSeek Harness](deepseek-harness.md)。

## 安装与升级

解压 `orinfer-v0.3.2-linux-aarch64.tar.xz`，使用其中的二进制安装匹配的执行包：

```bash
bin/orinfer --version
bin/orinfer package install \
  orinfer-exec-deepseek-v41-flash-1017-w4a8-ascend910b-v0.3.2.tar.xz \
  --cache /srv/orinfer/execution-cache
```

本版 A2 执行包 digest 为 `8931eb00b3221fee90ca2469322bd65e7ffab414bff8369f719818eaefc5d1f0`，包含已验收的 mHC 融合与 192 行 decoder 验证容量。自动 DSpark 默认最多 5 个草稿，32 路请求可按成本选择同一动态验证实现。正常部署使用 `--max-speculative-tokens 0`，由执行包提供验证预算。

已有 v0.3.1/fix1 模型的权重保持不变。通过 Hugging Face 模型仓库的 `v0.3.2` revision 下载配套元数据，确保 `cache/model.json` 声明上述 digest；旧 revision 与执行包仍可用于回滚。算子与执行计划继续单独安装。

工具前端需要模型 `cache/model.json` 声明 `metadata.chat_format=deepseek_v41`。已有模型可通过包内离线工具生成新目录，保留旧目录用于回滚；同一文件系统上的权重使用硬链接：

```bash
uv run --locked python -m tools.model.deepseek_v41.service --frontend-only \
  --model-dir /data/model/current --output /data/model/dsv41-v0.3.2

hf download immiq/DeepSeek-V4.1-Flash-1017-Orinfer-A2-W4A8 \
  cache/model.json README.md publication.json SHA256SUMS \
  --revision v0.3.2 --local-dir /data/model/dsv41-v0.3.2
```

该命令只更新工具前端；启用图片还需包含 `vision.safetensors` 的模型目录和声明原生视觉程序的执行包。视觉分片使用与语言权重相同的 Ascend 预处理基，算子和执行计划保留在独立执行包中。离线工具的 Python 环境在源码目录执行 `make python-env` 准备。在线服务使用 Rust 二进制。正常加载校验结构和范围，内容完整性由独立 `validate-model` 命令校验。权重、执行包和 DSpark 成本配置的既有完整性验收在身份相同时复用。

## 部门入口

服务绑定本机地址，通过部门网关提供访问、用户配额及请求期限。将 `ORINFER_API_KEY` 放在仓库外的受保护环境文件中，客户端使用 `Authorization: Bearer ...`。网关应保留 SSE 流式转发并关闭响应缓冲。

以下配置适用于至少 96 核 CPU、1 TB 主机内存和完整 8 卡独享的 A2 部署。DSpark 使用 `auto`，预处理线程及 batch 行数保留默认值。视觉特征缓存设为 1024 MiB；前处理缓存默认 256 MiB，按实际图片历史与内存压力调整。已有与执行包、CANN 9.2、TP8 和成本 schema 2 匹配的离线成本文件时，可显式设置 `ORINFER_A2_DSPARK_COST_MODEL`；未设置时由运行时估计成本。

```bash
source /usr/local/Ascend/cann/set_env.sh
export ORINFER_EXECUTION_CACHE=/srv/orinfer/execution-cache
export ORINFER_A2_HOST_TABLE_MIB=212000
export ORINFER_VISION_CACHE_MIB=1024
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_NPU_SOCKET_PORT_RANGE=auto

bin/orinfer serve /data/model/dsv41-v0.3.2 \
  --model deepseek-v4.1 --listen 127.0.0.1:8088 \
  --mtp-drafts auto --max-speculative-tokens 0 --graph full \
  --prefix-cache-mib 8192 --kv-offload-mib 262144 \
  --request-body-mib 64 \
  --queue-timeout-ms 60000 --output-timeout-ms 180000 --drain-timeout-ms 120000
```

默认最多 32 个活跃请求，普通 decode 行预算为 128；投机验证行数由执行包自动约束。设备 prefix 预算按每卡计，CPU offload 预算为 TP ranks 合计。CPU 的 256 GiB 是 prefix 状态缓存容量上限，实际占用按需增长，短测没有形成 CPU 层命中时不能据此声称吞吐收益。它不扩展活跃请求的 NPU KV 容量。常驻 CPU 表另计约 207 GiB，主机还需为缓存、前端和系统保留空间。交互客户端可从 16 并发起步，32 为服务准入上限，不要求每轮凑满。队列容量固定为 128；`--queue-timeout-ms` 只限制排队，`--output-timeout-ms` 限制输出通道等待进展，二者均不等于请求总期限。

长上下文通道单独设置并发、排队和客户端总期限；冷请求的准入等待可能超过交互入口的排队预算，达到该期限会结束请求。性能对照固定相同的排队预算与缓存准备条件，并同时报告成功请求、排队超时和运行时故障。

队列满返回 HTTP 429 和 `Retry-After: 1`。排队期限到达时，非流式请求返回 HTTP 408，流式请求在已建立的 SSE 中返回终止错误 `request_queue_timeout`，携带 `retryable=true` 和 `retry_after_ms=1000`。客户端从该等待时间开始指数退避，限制尝试次数和总期限。已经收到文本、thinking 或工具调用片段后停止自动重试，避免重复执行 agent 工具。慢客户端输出超时使用 `client_output_timeout`，携带 `retryable=false`；客户端关闭连接后，请求进入回收流程。

需要精确字段或机器读取输出时，显式使用严格的 `response_format: {type: "json_schema", ...}` 和本任务字段 schema。语法约束保证合法格式，字段语义通过任务答案另行验收。自由文本保留模型实际输出，调用方按业务要求检查格式。

模型容量为 262144 tokens，输入与输出合计受此上限约束。历史压力测试中 128K/256K 冷请求曾触及 210 秒客户端期限，成功完成的压力样本最长约 196K；容量不构成延迟承诺。交互入口先使用 32K 上下文预算，较长请求在网关独立准入；客户端及网关可从 300 秒总期限起配置，长上下文通道可从 600 秒起配置，并根据部署实测调整。示例服务保留 180 秒输出等待期限，长上下文通道可提高到 300 秒；仅服务短上下文时可缩短为 60 秒。总期限与 SSE 空闲期限分别设置，取消时关闭上游连接，限制自动重试，避免高负载下重复放大请求。

`/health` 提供就绪、失败、活跃及排队信息。发布后观察非预期 5xx、进程退出、请求卡住、队列恢复和资源预算；将限流、排队期限、主动断连与运行时错误分别统计。服务处理 SIGTERM 并释放资源；外部进程管理器的停止期限应大于 120 秒。

## 工具与质量边界

OpenCode 使用[配置示例](../examples/opencode.json)中的 `orinfer/deepseek-v4.1`，模型 ID 与服务一致。需要参数类型和格式保证时采用 `strict` 工具或 `response_format` JSON Schema，检查 `finish_reason` 后再执行工具。`length` 表示输出可能截断；工具应校验参数，并将执行错误作为对应 `tool_call_id` 的结果回传。

图片通过 Chat 消息的 `image_url` 内容传入，同一消息可按顺序包含多张图片。OpenCode 配置声明 `attachment` 和 `modalities.input=["text","image"]`，工具读取图片后沿同一接口提交。图片内容与预处理网格参与 prefix 身份；替换图片或调整图片顺序会重新匹配缓存。每图至多 1024 个视觉 token，原生 A2 图片历史按模型上下文和内存预算管理；可用 `image_processing.max_images`、`max_pixels`、`max_tokens_per_image` 和 `max_tokens` 收紧入口预算。`--request-body-mib` 调整 HTTP 请求体上限，`--image-cache-mib` 调整前处理缓存；UUID 引用与淘汰重传见服务文档。排队、图片编码、语言 prefill 和 decode 均计入客户端请求期限。

固定样例、代码执行、长上下文检索及短压测用于验收部署行为；它们不代表完整编码任务基准。自由文本的严格格式遵循和工具任务正确性分别记录。原有 32K 检索样例曾正确找到值，但多输出 `RELEASE_CODE=` 前缀；程序消费结果时使用受约束的结构化输出。

## 回滚

部署记录固定引擎版本及 revision、模型目录、执行包 digest、成本配置和环境。升级前保留上一套已在相同硬件及 CANN 环境验收的完整部署单元。停止新服务并确认资源释放，再使用旧单元启动，完成真实请求和 prefix 恢复检查。执行包可在同一个 cache 中共存，模型前端保留独立目录。

公开 v0.3.0 的执行包与本次 CANN 9.2/DSpark 包不同；回滚时同时恢复对应模型描述与环境。仅替换二进制不能证明兼容。发布包校验、安装和回滚使用隔离目录完成后，再切换部门入口。
