# DSv4.1 A2 部署

v0.3.1 候选支持 CANN 9.2、Ascend 910B TP8 的文本、HTTP/SSE、连续 batch、DSpark、设备及 CPU prefix cache，以及 DSML 工具与 thinking 历史。匹配视觉执行包及视觉权重后支持单图、多图和 OpenCode 读图工具交互。引擎与执行包分别安装，模型目录提供对应的前端和权重。支持范围见[模型包说明](model-packages.md#a2-文本服务与长上下文)。正式发布仍以图片正确性、并发性能及部署验收完成为条件。

## 安装与升级

解压 `orinfer-v0.3.1-linux-aarch64.tar.xz`，使用其中的二进制安装匹配的执行包：

```bash
bin/orinfer --version
bin/orinfer package install \
  orinfer-exec-deepseek-v41-flash-1017-w4a8-ascend910b-v0.3.1.tar.xz \
  --cache /srv/orinfer/execution-cache
```

工具前端需要模型 `cache/model.json` 声明 `metadata.chat_format=deepseek_v41`。已有模型可通过包内离线工具生成新目录，保留旧目录用于回滚；同一文件系统上的权重使用硬链接：

```bash
uv run --locked python -m tools.model.deepseek_v41.service --frontend-only \
  --model-dir /data/model/current --output /data/model/dsv41-v0.3.1
```

该命令只更新工具前端；启用图片还需包含 `vision.safetensors` 的模型目录和声明原生视觉程序的执行包。视觉分片使用与语言权重相同的 Ascend 预处理基，算子和执行计划保留在独立执行包中。离线工具的 Python 环境在源码目录执行 `make python-env` 准备。在线服务使用 Rust 二进制。正常加载校验结构和范围，内容完整性由独立 `validate-model` 命令校验。权重、执行包和 DSpark 成本配置的既有完整性验收在身份相同时复用。

## 部门入口

服务绑定本机地址，通过部门网关提供访问、用户配额及请求期限。将 `ORINFER_API_KEY` 放在仓库外的受保护环境文件中，客户端使用 `Authorization: Bearer ...`。网关应保留 SSE 流式转发并关闭响应缓冲。

以下为本机已使用的资源配置；CPU 绑核按部署机器拓扑调整。本次发布的 `dspark-costs-deepseek-v41-flash-1017-ascend910b-v0.3.1.json` 可复制为 `/srv/orinfer/dspark-costs.json`；它匹配当前执行包、CANN 9.2、TP8 及成本 schema 2。其他离线成本文件可由 `tools/bench/dspark_costs.py` 生成；已有匹配文件可直接复用。

```bash
source /usr/local/Ascend/cann/set_env.sh
export ORINFER_EXECUTION_CACHE=/srv/orinfer/execution-cache
export ORINFER_A2_HOST_TABLE_MIB=212000
export ORINFER_A2_HOST_ROW_CACHE_MIB=2048
export ORINFER_A2_DSPARK_COST_MODEL=/srv/orinfer/dspark-costs.json
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_NPU_SOCKET_PORT_RANGE=auto

bin/orinfer serve /data/model/dsv41-v0.3.1 \
  --model deepseek-v4.1 --listen 127.0.0.1:8088 \
  --mtp-drafts auto --max-active-requests 16 --max-batch-tokens 128 \
  --max-speculative-tokens 192 --prefix-cache-mib 8192 --kv-offload-mib 16384 \
  --queue-timeout-ms 30000 --output-timeout-ms 300000 --drain-timeout-ms 120000
```

设备 prefix 预算按每卡计，CPU offload 预算为 TP ranks 合计。常驻 CPU 表另计约 212000 MiB，主机还需为行缓存、offload、前端和系统保留空间。灰度从 16 活跃请求开始，依据实际请求长度、排队和首 token 时延再放宽到已验收的 32。队列容量固定为 128；`--queue-timeout-ms` 只限制排队，`--output-timeout-ms` 限制输出通道等待进展，二者均不等于请求总期限。

模型容量为 262144 tokens，输入与输出合计受此上限约束。历史压力测试中 128K/256K 冷请求曾触及 210 秒客户端期限，成功完成的压力样本最长约 196K；容量不构成延迟承诺。交互入口先使用 32K 上下文预算，较长请求在网关独立准入；客户端及网关可从 300 秒总期限起配置，长上下文通道可从 600 秒起配置，并根据部署实测调整。示例服务为长请求保留 300 秒输出等待期限；仅服务短上下文时可缩短为 60 秒。总期限与 SSE 空闲期限分别设置，取消时关闭上游连接，限制自动重试，避免高负载下重复放大请求。

`/health` 提供就绪、失败、活跃及排队信息。发布后观察非预期 5xx、进程退出、请求卡住、队列恢复和资源预算；将限流、排队期限、主动断连与运行时错误分别统计。服务处理 SIGTERM 并释放资源；外部进程管理器的停止期限应大于 120 秒。

## 工具与质量边界

OpenCode 使用[配置示例](../examples/opencode.json)中的 `orinfer/deepseek-v4.1`，模型 ID 与服务一致。需要参数类型和格式保证时采用 `strict` 工具或 `response_format` JSON Schema，检查 `finish_reason` 后再执行工具。`length` 表示输出可能截断；工具应校验参数，并将执行错误作为对应 `tool_call_id` 的结果回传。

图片通过 Chat 消息的 `image_url` 内容传入，同一消息可按顺序包含多张图片。OpenCode 配置声明 `attachment` 和 `modalities.input=["text","image"]`，工具读取图片后沿同一接口提交。图片内容与预处理网格参与 prefix 身份；替换图片或调整图片顺序会重新匹配缓存。每图至多 1024 个视觉 token，单请求总量由执行包限制；可用 `image_processing.max_images`、`max_pixels` 和 `max_tokens` 收紧入口预算。排队、图片编码、语言 prefill 和 decode 均计入客户端请求期限。

固定样例、代码执行、长上下文检索及短压测用于验收部署行为；它们不代表完整编码任务基准。自由文本的严格格式遵循和工具任务正确性分别记录。原有 32K 检索样例曾正确找到值，但多输出 `RELEASE_CODE=` 前缀；程序消费结果时使用受约束的结构化输出。

## 回滚

部署记录固定引擎版本及 revision、模型目录、执行包 digest、成本配置和环境。升级前保留上一套已在相同硬件及 CANN 环境验收的完整部署单元。停止新服务并确认资源释放，再使用旧单元启动，完成真实请求和 prefix 恢复检查。执行包可在同一个 cache 中共存，模型前端保留独立目录。

公开 v0.3.0 的执行包与本次 CANN 9.2/DSpark 包不同；回滚时同时恢复对应模型描述与环境。仅替换二进制不能证明兼容。发布包校验、安装和回滚使用隔离目录完成后，再切换部门入口。
