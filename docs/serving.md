# 服务使用指南

## 启动与模型加载

```bash
./target/release/orinfer validate-model /path/to/prepared-model
./target/release/orinfer serve /path/to/prepared-model \
  --model qwen3.8-27b --listen 0.0.0.0:8088 --cuda-graph decode_only
```

服务加载一次模型，由单个 GPU worker 执行请求。CPU 使用模型目录中的 `tokenizer.json`、`chat_template.jinja` 和 `generation_config.json` 渲染模板及分词；线上不运行 Python、量化或 kernel JIT。

权重加载采用并行读取和有界的 pinned 缓冲区，并与异步 GPU 上传重叠。`--load-workers N` 可设置 1–32 个读取线程；默认按可用 CPU 核心选择，最多 4 个。暂存内存最多 256 MiB，加载结束后释放。开启 `--verify-weights` 时使用完整校验路径，启动较慢。

执行包中的全部 kernel 资产仍校验 hash；普通文本、配置的 MTP、视觉及小 batch 计划预加载 CUDA 模块，其余计划在首次使用时加载，并在 CUDA Graph 捕获前完成绑定。未预加载的形状可能有额外的首次执行开销。

模型准备时封存上述前端文件的 SHA256。需要有意修改时，先确认它们与 checkpoint 的 token 映射和语义一致，再执行 `python3 tools/model/package.py pin-assets MODEL_DIR` 重新封存；此命令不改权重。模型文件在加载期间必须保持不变。

先查共享执行包缓存中的 `<digest>`，缺失时再查模型内的 `cache/packages/<digest>`。`ORINFER_EXECUTION_CACHE` 可覆盖共享缓存位置；默认目录是 `$XDG_CACHE_HOME/orinfer/packages`，未设置 XDG 时为 `~/.cache/orinfer/packages`。存在但损坏的包会报错。专用 safetensors 布局沿用 `orin.layout.<tensor>` 标识，执行包 digest 固定原生执行逻辑与 kernel。

## Chat API

提供 `GET /health`、`GET /v1/models` 和 `POST /v1/chat/completions`。设置 `ORINFER_API_KEY` 后，`/v1` 请求需携带 `Authorization: Bearer <key>`。

```bash
curl http://127.0.0.1:8088/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"37×24−19×13等于多少？"}],"temperature":0,"max_tokens":512,"enable_thinking":true,"stream":true,"stream_options":{"include_usage":true}}'
```

支持 `temperature`、`top_p`、`top_k`、presence/frequency/repetition penalties、有符号 64 位 `seed`、`logit_bias`、EOS 和 `stop`。temperature、top_p、top_k 和 repetition_penalty 默认采用模型的 generation config；presence/frequency penalties 默认 0。默认 seed 为 20261002。`logit_bias` 的 key 是 tokenizer 的 token ID，值范围为 −100 到 100。

默认关闭 thinking；`enable_thinking=true` 或非 `none` 的 `reasoning_effort` 开启后，思考文本位于扩展字段 `reasoning_content`。`none` 关闭，`minimal/low` 映射 checkpoint 的 `low`，`medium` 保留，`high/xhigh/max` 映射 `xhigh`；显式 `enable_thinking=false` 可关闭。模型原生模板要求存在用户消息，system/developer 指令应放在开头。

`thinking_token_budget` 限制本轮初始思考段的 token 数，支持 0；到达上限时解码强制选择 `</think>`，随后继续回答或调用工具。结束标记计入 completion，不计入 reasoning。预算自动缩小到总输出上限减 2，为结束标记和至少一个回答 token 留空间；开启 thinking 且指定此参数时，总输出预算至少为 2。关闭 thinking 时不应用此预算。限制是请求私有状态，MTP 的草稿和 target 共同遵守；无正文约束时结束思考后恢复普通 GPU 采样。思考期间使用 CPU 约束采样，会增加开销。effort 控制模型倾向，不能替代这个硬上限。

默认输出预算为 8192 tokens，并缩小到剩余上下文容量；可用 `max_tokens` 或 `max_completion_tokens` 指定，二者不能同时传入。提示、历史、图片、thinking 和输出合计计入上下文容量；显式预算超出模型包容量时在准入前返回 400。HTTP body 上限为 32 MiB。

上下文溢出返回 `error.code=context_length_exceeded`，客户端可据此压缩历史并重试；其他无效参数仍为 `invalid_request`。

`stream` 默认 false；显式 true 使用 SSE deltas 和 `[DONE]`。`stream=null` 等同默认值，`tools=null` 等同无工具。请求 `stream_options.include_usage=true` 时，普通 chunk 包含 `usage:null`，结束前另发 `choices:[]` 的完整 usage。usage 包含 reasoning 和 prefix cache token 计数。客户端断连后取消生成。异常流返回 error，不将失败后的部分输出伪装为成功完成。

`logprobs=true` 返回正文 token 的 `token/logprob/bytes/top_logprobs`，`top_logprobs` 支持 0–20；SSE 返回对应正文 delta 的概率。概率来自 target 模型，应用惩罚、bias 和约束后、top-k/top-p 截断前；temperature=0 的 greedy 请求以 T=1 softmax 报告概率。`bytes` 可用于还原跨 token 的 UTF-8 字符；thinking、工具协议标记和工具参数不计入正文概率。开启此选项需要将 logits 下载到 CPU，会增加开销。

非流式概率结果按输出预算预留 CPU 内存，计入 `--preprocess-memory-mib`，预算保持到 HTTP body 结束；过大的单请求在准备前返回 400。SSE 逐步发送概率，不累计完整概率响应，慢客户端的增量缓冲仍有上限。

参数错误返回 HTTP 400 及 `error.message/type/param/code`，响应带 `x-request-id`。`/health.chat_capabilities` 声明当前能力。`store=false`、`metadata`、`safety_identifier` 和 `service_tier=auto/default` 可传入，但本地服务不持久化 completion，也不提供云端审核或服务等级调度。`store=true`、`n>1`、音频、视频和未知参数会明确报错。

### JSON 输出

支持 `response_format.type=text/json_object/json_schema`。后两种使用 Rust llguidance 做逐 token 约束；JSON Schema 支持嵌套对象、数组、类型、required、enum/const 及本地 `$ref` 等可编译约束。外部引用、超大或过深 schema，以及编译器无法保证的约束在请求准备阶段返回 400。

```json
{
  "model": "qwen3.8-27b",
  "messages": [{"role": "user", "content": "用 JSON 返回 2+3 的结果"}],
  "response_format": {"type": "json_schema", "json_schema": {
    "name": "answer", "strict": true,
    "schema": {"type": "object", "properties": {"answer": {"type": "integer"}},
      "required": ["answer"], "additionalProperties": false}
  }},
  "max_completion_tokens": 128
}
```

thinking 可与 JSON 输出组合，正文仍受约束。MTP 的 target 和 draft 使用相同约束，拒绝的草稿不会改变解析器状态。达到输出预算会返回 `finish_reason=length`，正文可能尚未完成 JSON；调用方需检查 finish reason。约束输出不接受 `stop`，避免截断 JSON 或工具参数。

### 图片与多图

27B 和附加视觉编码器的 Flash Next 均支持图片。用户消息可交错文本和多张图片，支持 PNG/JPEG/WebP、HTTP(S) URL 和 base64 data URI；历史消息中的图片也会重新编码。

```json
{
  "model": "qwen3.8-27b",
  "messages": [{"role": "user", "content": [
    {"type": "text", "text": "比较两张图片"},
    {"type": "image_url", "image_url": {"url": "https://example.com/first.png"}},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
  ]}],
  "max_tokens": 512
}
```

`detail=low` 使用约 256×256 像素预算；`auto/high` 使用 checkpoint 预算与模型包视觉容量中的较小者。预处理保持宽高比并按 32 像素对齐，每 32×32 像素占一个提示 token。单张压缩图片上限 24 MiB，URL 读取超时 20 秒；patch、特征及总上下文超出模型包容量时返回 400。纯文本模型收到图片也返回 400。

每张图片独立执行双向视觉 attention，merger 输出注入对应 image tokens；文本使用交错 MRoPE。视觉 encoder 每次仍执行，相同图片的预处理结果可命中文本主干 prefix cache。当前不支持视频或音频。构建与验证见[图文工具](../tools/vision/README.md)。

### 工具调用

支持 `tools`、`tool_choice`、`parallel_tool_calls`。模型的 XML 工具调用转换为标准 `tool_calls`，arguments 为 JSON 字符串。客户端执行工具，再把带 `tool_call_id` 的 `role=tool` 消息连同历史发送回来。

流式调用先发 index/id/type/name，再增量发送 `function.arguments`；客户端按 index 拼接参数。`strict=true`、`tool_choice=required`、指定函数，或 `parallel_tool_calls=false` 时启用约束解码，限制函数名、参数 schema 和调用数；`required` 至少调用一次，指定函数仅能调用该函数，关闭 parallel 最多一次。内部约束调用使用 JSON envelope，API 和历史仍为标准 `tool_calls`。

未开启约束的普通 auto 调用沿用原生 XML 协议。未知函数、缺失参数或错误参数类型仍作为调用交给客户端，客户端应校验并返回 `role=tool` 错误结果，让模型重试；服务端不执行工具。XML 参数按声明类型转换，无法转换的值保留为字符串。语法损坏到无法辨认函数的输出仍报错。输出预算用尽时返回 `finish_reason=length` 并保留已生成的调用前缀，流式与非流式参数一致；不要直接执行截断调用。回传历史的参数不是有效 JSON 对象时，原始字符串在原生模板中以 `__orinfer_raw_arguments` 参数保留，不猜测或补齐内容。

### OpenCode 与 Pi

[OpenCode 配置](../examples/opencode.json)使用 `@ai-sdk/openai-compatible` 和默认编码 agent，启用自动压缩，声明图片、工具、thinking 及 `reasoning_content` 历史回传；不限制 agent 步数。将其复制为项目中的 `opencode.json`，按服务的 `--model` 选择对应模型：

```bash
opencode run --model orinfer/qwen-flash-next '检查并修复当前项目的测试失败'
opencode run --model orinfer/qwen-flash-next --variant low '检查并修复当前项目的测试失败'
```

默认关闭 thinking；low/medium/high 分别使用 512/1024/2048 思考 token 上限，总输出上限为 8192。工具权限沿用 OpenCode 默认设置。

[Pi 配置](../examples/pi-models.json)对应 `earendil-works/pi`。将 provider 合并到 Pi agent 目录中的 `models.json`，默认目录为 `~/.pi/agent`；配置使用 `openai-completions`，可通过 `/model` 和 `/thinking` 切换模型与思考级别：

```bash
pi --provider orinfer --model qwen-flash-next --thinking low
```

配置中的 `orinfer-local` 是无鉴权本地服务的占位 key，使模型在 Pi 中可选；启用服务鉴权后用实际凭据或环境引用替换。Pi 的 thinking budgets 通过 `thinking_token_budget` 发送；可在 Pi 设置中调整 `thinkingBudgets`。示例不声明固定缓存寿命，也不启用 long cache retention。图片和多图由用户附件或客户端 read 工具送入 messages。两端的模型 ID 必须与正在服务的模型一致，contextWindow/context 与输出上限应按实际模型包填写。

## 执行与缓存

### CUDA Graph

`serve`、`run-model` 和 `score-model` 支持 `--cuda-graph decode_only|full|off`。

- `decode_only`：默认模式。普通 decode、MTP 草稿、验证、恢复和短步刷新使用 Graph；文本 prefill、视觉编码和 MTP 首次预热直接提交。
- `full`：捕获并使用全部执行计划。
- `off`：按相同计划直接提交 kernel、copy 和 memset。

prefill 尾部即使复用 decode 计划，在 `decode_only` 模式下也不使用 Graph。地址及 workspace 变化会使相关 Graph 失效或重新捕获。

单请求 Graph 在程序首次执行时按需捕获，加载时不预先捕获未使用的程序。batch Graph 在成员组合重复出现后捕获，缓存常见的 2 的幂次批次，同时限制条数和总操作节点数；过渡批次及超出节点预算的计划直接提交；内存准入紧张时先淘汰空闲 Graph，再回收空闲请求状态。首次执行的捕获耗时计入请求预热。

短请求结束后，每个 KV 缓冲区最多保留一个 CUDA 分配粒度的页，清零后供同一槽位复用；较大映射释放，内存紧张时空闲页也释放。准入只计入需要新增的 KV 页，空闲槽位的页不会抵扣其他活跃请求的未来预算。

新请求优先选择新增分配最少的空闲槽位。回收后重新计算准入预算；没有活跃请求时，临时内存不足会排队重试最多两秒，持续不足返回 503 `service_unavailable`，服务继续处理能够准入的请求。

Flash Next 的主模型支持 2048／4096 token 大分块，MTP 预热保持 512 token 分块；使用 `full` 可同时对这两条 prefill 路径启用 Graph。索引计算和临时评分缓冲区按当前有效上下文推进，缓存容量仍为 256K，Graph 使用的虚拟地址保持稳定。

### MTP

包含原生 `mtp` 计划的模型自动启用 MTP，适用于文本、图片、多图、thinking、工具调用和支持的采样参数。无惩罚 greedy 使用 GPU top-1；随机采样对 target/draft 应用相同的采样处理，接受概率为 `min(1,p/q)`，拒绝后从归一化的 `(p-q)+` 分布采样，以保持主模型分布。

`--mtp-drafts 0` 移除草稿和验证执行计划，加载时跳过它们独占的权重；主模型、视觉及 batch 所需的目标状态捕获仍保留。模型文件不变，重新启用 MTP 时正常加载草稿权重。

拒绝时恢复私有 GDN、卷积、位置和有效 KV 状态。固定 seed 可复现同模式结果；随机 MTP 与普通 decode 不要求同 seed 下逐 token 相同。输出尾部不足一个验证块时执行普通 decode。

单请求使用 MTP；2–4 个 decoder 按实测每提交 token 成本选择 MTP 或 target batch，更大并发和混合 prefill 使用 target batch。收益取决于接受率及采样开销。构建方法见[模型工具](../tools/model/README.md)。

### Prefix cache

`--prefix-cache-mib` 默认 12288，单位 MiB，0 关闭复用；按需分配，不在加载时预占。缓存保存完整 KV、FP32 GDN、卷积、位置及 MTP 检查点。Rust radix tree 按 token 和图片身份匹配，并结合实际 prefill/恢复成本选择可用端点。

KV 区间不可变并按引用共享，持续状态单独保存；恢复仍执行 GPU 复制到 Graph 绑定地址。短前缀妨碍大块执行时可能跳过。保存最终提示、周期及按成本准入的分叉检查点；生成结束也可保存已计算的输出前缀。模型重载后缓存清空。

命中免除对应文本主干计算，视觉编码、剩余提示和生成继续执行。`usage.prompt_tokens_details.cached_tokens` 报告实际恢复长度，SSE 需请求 usage。缓存不是 paged attention 或零复制映射，不保证接近满上下文时命中完整提示；内存不足会驱逐或跳过保存。

接受 1–64 字符的 `prompt_cache_key` 和 `prompt_cache_retention=in-memory|24h` 作为路由、保留偏好。当前单 worker 不需要路由，缓存仍按精确 token 和图片身份匹配，key 不划分缓存或改变模型输出。缓存是可驱逐的内存缓存，`24h` 不预留容量、不写 SSD，也不保证 24 小时命中；`/health.chat_capabilities.prompt_cache` 声明这些限制。未传这些字段也自动复用前缀。

## 并发、上下文与资源

支持批处理的算子包启用连续 batching；未包含相应算子时串行执行，`/health.continuous_batching` 报告实际模式。每个请求拥有独立 KV、GDN、卷积、位置、视觉和采样状态，权重及执行 workspace 共享。

单请求冷 prefill 使用大块计划；兼容的多个短请求可联合执行。与 decode 混合时，按预测耗时选择小块 prefill。常见 batch 可使用固定形状 kernel，其余 2–128 行可由支持动态行数的 AOT 包执行，线上不编译。请求按迭代加入和结束；不为凑 batch 额外等待。

Flash Next 的 batch 包共享投影、MoE 路由、专家和输出头计算，GDN、卷积、PLE、QSA/KV/index/pending 与 MTP 状态按请求隔离。提供 2/4/8/16/32/64/128 专用档；附加动态回退包后，其他数量复用邻近容量的 AOT 算子，按真实行数执行。未附加动态回退的包仍补零到下一档。冷 prefill 保留单请求大块路径；有 decoder 等待时，根据 `--target-tpot-ms` 的剩余时间，预留下一轮 decode 的预计耗时，再在 `--prefill-budget-ms` 上限内选择真实 prefill 小块。该包不合并不同请求的 prompt chunk；两轮 decode 之间的独立 prefill 耗时跨迭代累计；每次 decode 后允许一个受预算限制的块，额外 prefill 还受预计 decode 耗时约束。预算是预测软目标，冷启动、长上下文和 checkpoint 可能超出；预算不足时允许最小有效块（最多 16 tokens），避免退化为反复读取权重的逐 token prefill，尾部仍按真实长度执行。预算控制文本主干分块，视觉编码器仍独立执行。调度统计提供 `prefill_chunk_histogram`、`bounded_prefill_iterations` 和 `max_bounded_prefill_s`；`batch_execution` 中的 `dynamic_graph_captures`、`dynamic_direct_iterations` 与 `graph_hits` 可用于核对动态回退和重放。构建入口见[Flash 模型工具](../tools/model/flash_next/README.md)。

包含私有地址表的 Flash 包在每层用一套 GDN/QSA kernel 并行处理全部请求，包含临时缓冲区的隔离和批量卷积历史提交。请求地址在 Graph 重放前更新，补零行的地址为空；单请求仍走 M1 程序。

请求的 KV 和 MTP hidden ring 按总 token 预算分配，长请求仍可使用完整容量。槽位复用时收缩过大的私有分配；显存准入受阻时保留一个空闲槽位并回收其余空闲槽位的普通缓冲区，下次准入前重新分配。实际活跃数取决于请求预算和可用显存，超过准入容量的请求排队。

默认最多 32 个活跃请求、128 个待处理请求，队列满时返回 429。活跃数还受每请求上下文/输出预算、workspace 和可用内存约束。准入检查 CUDA 可用内存和 Linux `MemAvailable`，保留系统余量；不足时先收缩 prefix cache，再排队。等待队列按初始化、图片编码、有效缓存恢复和剩余 prefill 成本排序，等待时间平滑提升优先级，30 秒后优先处理最老请求。图片身份在队列描述中只计算一次，缓存匹配随 checkpoint 更新。大于每轮 token 预算的冷 prefill 通常最多提前准入两个，短请求可成批准入；可有效复用的共同 prefix 由一个请求先计算，其余暂留 CPU 队列。等待 30 秒仍因内存不足无法准入的请求会阻止新增工作，直到已有请求结束、腾出空间；取消或超时会解除此保护。统计见 `/health`。

prefill 按实际耗时扣减执行额度，较慢的大块不会仅因轮次少而获得更多服务。有 decoder 时根据距上次提交 token 的时间决定下一轮工作；至少定期推进一个 prefill token，目标不可满足时计入 `prefill_progress_overrides`。`--target-tpot-ms` 是软目标，不限制客户端输出预算，不保证视觉编码、缓存保存或冷启动期间的最大流式间隔。`max_decode_gap_s` 和 `decode_budget_overruns` 记录引擎提交 token 的间隔，HTTP chunk 间隔需另行测量。

实际活跃数还结合已测量的相邻 batch 耗时与吞吐决定，未知档位允许探索。Graph 首次遇到新 batch 成员组合时直接执行，重复出现且仍有足够输出预算时再捕获；已有图可用于最后几步。MTP 在算子包和 CLI 草稿上限内比较真实轮次耗时与有效提交量，选择验证长度；这是请求级在线探索，不能保证短请求已经完成调优。

模型包决定实际上下文容量，上限可配置为 262144 tokens。长提示分块计算，首次 dense attention 的复杂度不变。INT8 group-64 KV 包含 FP16 scale，CUDA VMM 按已执行位置映射物理页，同时保持 Graph 地址稳定；prefill 的单层 FP16 临时 KV workspace 可共享并按需增长。持续 GDN 状态保持 FP32。上下文扩展、KV 优化和验证见[模型构建](../tools/model/README.md)。

| 参数 | 默认值 | 行为 |
| --- | --- | --- |
| `--listen` | `0.0.0.0:8088` | HTTP 监听地址 |
| `--model` | `qwen3.8-27b` | API 模型 ID |
| `--cuda-graph` | `decode_only` | Graph 模式 |
| `--verify-weights` | 关闭 | 启动时完整校验已加载权重和 CPU 表内容 |
| `--prefix-cache-mib` | `12288` | GPU 缓存字节预算，0 关闭 |
| `--max-active-requests` | `32` | 活跃请求上限，1–128 |
| `--max-batch-tokens` | `128` | 每轮 target 计算 token 预算，1–128 |
| `--prefill-budget-ms` | `200` | 混合块的预测耗时上限，非严格延迟上限 |
| `--target-tpot-ms` | `400` | 引擎提交 token 的软间隔目标 |
| `--memory-reserve-mib` | `1024` | 系统余量以外的准入保留内存 |
| `--preprocess-workers` | `2` | 同时执行图片/模板/tokenizer CPU 任务数 |
| `--preprocess-memory-mib` | `2048` | 预处理和排队图片张量预算 |
| `--queue-timeout-ms` | `0` | 排队期限，0 不限制等待；不限制已开始计算 |
| `--output-timeout-ms` | `60000` | 输出通道无进展时终止，prefill 不计入 |
| `--drain-timeout-ms` | `30000` | SIGINT/SIGTERM 后 HTTP 排空期限 |
| `--gpu-lock` | `artifacts/gpu-experiment.lock` | GPU worker 使用的共享锁 |

`/health` 显示 starting/ready/draining/failed，并提供调度、准入、缓存恢复和 Graph 统计。GPU worker 故障时返回 503；CUDA 或内部状态故障后需要重启服务。统计和真实 API 性能测量见[性能工具](../tools/bench/README.md)。

启动默认校验配置、执行库、kernel 身份，以及权重和 CPU 表的 safetensors 结构、dtype、shape、布局和偏移边界，跳过大体积权重的 SHA256 扫描。`serve`、`run-model` 和 `score-model` 添加 `--verify-weights` 可校验加载的权重内容；`validate-model` 始终完整校验模型权重。默认模式无法发现仅修改数值、但不改变结构的损坏。

## 诊断 CLI

`plan-model` 在 CPU 加载模型包原生库并输出其生成的计划，`validate-model` 校验模型、执行库、kernel 和张量 hash；两者不初始化 GPU。执行包接口与旧数据迁移见[模型执行包](model-packages.md)。

`run-model MODEL_DIR REQUESTS.json` 接收 token-ID 请求，用于固定块、普通 decode 的诊断基准，逐请求重置状态。输入长度须能由声明的 prefill 计划整除；它不代表 API 的任意长度提示、MTP 和缓存路径。任意长度和多请求生成通过 Chat API 或 Rust `Model::generate` 执行。

`score-model` 对冻结 token 历史评分，服务的固定场景检查见[API 验证](../tools/api/README.md)，BF16/FP8 质量对照见[质量评测](../tools/eval/README.md)。
