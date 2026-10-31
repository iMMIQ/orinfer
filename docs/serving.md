# 服务使用指南

## 启动与模型加载

```bash
./target/release/orinfer validate-model /path/to/prepared-model
./target/release/orinfer serve /path/to/prepared-model \
  --model qwen3.8-27b --listen 0.0.0.0:8088 --cuda-graph decode_only
```

服务加载一次模型，由单个 GPU worker 执行请求。CPU 使用模型目录中的 `tokenizer.json`、`chat_template.jinja` 和 `generation_config.json` 渲染模板及分词；在线推理直接使用 Rust 运行时和预编译 kernel。

权重加载采用并行读取和有界的 pinned 缓冲区，并与异步 GPU 上传重叠。`--load-workers N` 可设置 1–32 个读取线程；默认按可用 CPU 核心选择，最多 4 个。暂存内存最多 256 MiB，加载结束后释放。开启 `--verify-weights` 时使用完整校验路径，启动较慢。

模型就绪后、API 对外前，worker 会预热执行一次单 token 请求并输出 `WARMUP READY`。预热把序列租用映射、packet 计划和模块加载移入启动阶段（Flash Next 实测约 0.9s），避免首个用户请求多付同等的首 token 延迟。

正常加载消费已声明的资产身份，不计算内容 SHA256；`validate-model` 完整校验执行包及 kernel 资产。普通文本、配置的 MTP、视觉及小 batch 计划预加载 CUDA 模块，其余计划在首次使用时加载，并在 CUDA Graph 捕获前完成绑定。延迟加载的形状会在首次执行时承担模块加载开销。

模型准备时封存上述前端文件的 SHA256。需要有意修改时，先确认它们与 checkpoint 的 token 映射和语义一致，再执行 `python3 tools/model/package.py pin-assets MODEL_DIR` 重新封存前端身份；权重保持原样。加载期间应保持模型文件内容稳定。

先查共享执行包缓存中的 `<digest>`，缺失时再查模型内的 `cache/packages/<digest>`。`ORINFER_EXECUTION_CACHE` 可覆盖共享缓存位置；默认目录是 `$XDG_CACHE_HOME/orinfer/packages`，XDG 采用系统默认值时为 `~/.cache/orinfer/packages`。存在但损坏的包会报错。专用 safetensors 布局沿用 `orin.layout.<tensor>` 标识，执行包 digest 固定原生执行逻辑与 kernel。

### 专家常驻容量

`serve`、`run-model` 和 `score-model` 提供 `--resident-experts full|N`，默认 `full`，保持专家权重全量常驻。Flash Next 的有限模式要求执行包包含专家分页声明；旧执行库通过模型工具更新为匹配的原生执行库，保留原权重及 cubin。

`N` 是每层等效容量：48 层主干共享最多 `48 × N` 个专家的全局 SLRU 缓存，容量由各层动态共享；可指定 1–512，512 等效全量常驻。MTP 专家使用独立的全量常驻权重。缺失专家的压缩权重和 scale 从模型文件按需读取，以压缩形式映射到稳定的 CUDA 虚拟地址，计算时临时解码为 INT8。

```bash
./target/release/orinfer serve /path/to/prepared-flash-model \
  --model qwen-flash-next --resident-experts 256
```

有限模式在路由后执行 CPU 调度和 SSD 加载，自动使用 `cuda_graph=off`；`full` 保留所选 Graph 模式。容量限制按专家数量计算，实际物理内存还包含 CUDA 分配粒度和共享页开销。一轮 prefill 或 batch 路由的专家集合超过容量时，临时映射额外专家，随后释放，实际字节峰值需另计临时映射和页开销。较小容量会增加 I/O、降低吞吐；模型文件在整个服务期间必须保持可读且内容稳定。

### 请求默认参数

`--default-request-params` 接受一个 JSON 对象，为本次部署填充客户端省略的 Chat 请求字段。例如，默认开启 thinking，同时设置采样温度：

```bash
./target/release/orinfer serve /path/to/prepared-model \
  --default-request-params '{"enable_thinking":true,"temperature":0.7}'
```

省略此选项时使用 API 默认值。显式请求字段优先，包括 `false`、空数组和 `null`；`null` 取消对应部署默认值，再按该字段原有空值语义处理。对象字段采用整体替换。`model` 和 `messages` 由每条客户端请求提供，其余字段可配置部署默认值。配置的字段类型、取值和组合在加载 GPU 模型前校验；默认值必须能组成有效请求。

`enable_thinking` 与 `reasoning_effort`、`max_tokens` 与 `max_completion_tokens` 分别按组覆盖：请求传入其中一项时，该组完全使用请求参数。因此，即使服务默认开启 thinking，请求仍可用 `enable_thinking=false` 或 `reasoning_effort=none` 关闭。请求自身的冲突仍返回 400。显式 `stream=false/null` 会清除默认 `stream_options`，显式 `logprobs=false/null` 会清除默认 `top_logprobs`。实际请求默认参数可在 `/health.default_request_params` 查看。

## Ollama API

无模型路径的 `orinfer serve` 同时提供 Ollama 原生接口，客户端地址为 `http://127.0.0.1:8088`：

- `GET /api/version`、`GET /api/tags`、`POST /api/show`、`GET /api/ps`：版本、全部已注册模型、详情和已加载模型；发现接口只使用 CPU。
- `POST /api/chat`、`POST /api/generate`：聊天与应用原生 chat template 的生成；默认 `stream=true`，输出 `application/x-ndjson`；`stream=false` 返回单个 JSON。
- `POST /api/copy`、`DELETE /api/delete`：复制别名、移除注册。移除前等待活动请求结束；导入的权重文件由用户管理，保留原路径。

```bash
curl http://127.0.0.1:8088/api/chat -d '{"model":"index-translate-2b","messages":[{"role":"user","content":"Translate to English: 你好"}],"think":false,"options":{"num_predict":256},"keep_alive":"60s"}'
```

支持 `think` 布尔值/推理级别、消息中的 base64 PNG/JPEG/WebP 图片、工具及工具结果历史、`format="json"` 或 JSON Schema、`logprobs/top_logprobs`。`options` 支持 `temperature/top_p/top_k/seed/stop/presence_penalty/frequency_penalty/repeat_penalty` 与正数 `num_predict`。省略的参数继承部署默认值；显式 `think=false` 可以覆盖默认 thinking。工具参数返回 JSON 对象，流式时完整解析后发布；截断的无效参数返回错误。

`keep_alive` 支持数字秒数、`"1h30m"` 等时长、负数永久保留和 `0` 请求结束后回收；空 messages/prompt 请求用于预加载，空请求加 `keep_alive=0` 用于卸载。未带 tag 的注册名也接受 `:latest`。两种协议共用准入、连续调度、prefix cache 和取消逻辑。

模型内存预算充足时直接并发处理，各模型可并行冷启动，同一模型共享一个 worker。加载或请求状态预算不足时，优先立即驱逐最近最少使用的空闲模型，内存压力回收同样适用于 `keep_alive=-1`；空间仍被活动请求或在途加载占用时才等待。正在生成的请求持续执行至结束或取消。加载中的模型单独预留内存，以准确计入在途内存占用，预留在就绪或失败后释放。

返回实际测量的 token/cache 计数和请求总时长（纳秒）。响应按实际测量提供字段；`/api/ps.estimated_load_bytes` 报告内存准入估算值，实际 VRAM 占用需独立测量。`ORINFER_API_KEY` 同时保护 `/api` 与 `/v1`。

当前模型由 CLI 导入原生 orinfer 包；Ollama registry 的 pull/push/create、embedding 接口、raw prompt、suffix、Go template 覆盖、旧 token context 和自动截断/滑窗明确返回错误。上下文由模型执行包确定，`options.num_ctx` 可省略或使用该值。

## Chat API

提供 `GET /health`、`GET /v1/models` 和 `POST /v1/chat/completions`。设置 `ORINFER_API_KEY` 后，`/v1` 请求需携带 `Authorization: Bearer <key>`。

```bash
curl http://127.0.0.1:8088/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"37×24−19×13等于多少？"}],"temperature":0,"max_tokens":512,"enable_thinking":true,"stream":true,"stream_options":{"include_usage":true}}'
```

支持 `temperature`、`top_p`、`top_k`、presence/frequency/repetition penalties、有符号 64 位 `seed`、`logit_bias`、EOS 和 `stop`。temperature、top_p、top_k 和 repetition_penalty 默认采用模型的 generation config；presence/frequency penalties 默认 0。默认 seed 为 20261002。`logit_bias` 的 key 是 tokenizer 的 token ID，值范围为 −100 到 100。

默认关闭 thinking；`enable_thinking=true` 或非 `none` 的 `reasoning_effort` 开启后，思考文本位于扩展字段 `reasoning_content`。`none` 关闭，`minimal/low` 映射 checkpoint 的 `low`，`medium` 保留，`high/xhigh/max` 映射 `xhigh`；显式 `enable_thinking=false` 可关闭。模型原生模板要求存在用户消息，system/developer 指令应放在开头。

`thinking_token_budget` 限制本轮初始思考段的 token 数，支持 0；到达上限时解码强制选择 `</think>`，随后继续回答或调用工具。结束标记仅计入 completion。预算自动缩小到总输出上限减 2，为结束标记和至少一个回答 token 留空间；开启 thinking 且指定此参数时，总输出预算至少为 2。该预算在 thinking 开启时生效。限制是请求私有状态，MTP 的草稿和 target 共同遵守。无惩罚、无概率报告的 greedy 请求在 GPU argmax 满足约束时直接复用结果；预算耗尽时可直接选择唯一的结束标记。其他约束选择保留 CPU 回退，随机采样保持相同目标分布。effort 控制思考倾向，`thinking_token_budget` 提供硬上限。

默认输出预算为 8192 tokens，并缩小到剩余上下文容量；可用 `max_tokens` 或 `max_completion_tokens` 之一指定。提示、历史、图片、thinking 和输出合计计入上下文容量；显式预算超出模型包容量时在准入前返回 400。HTTP body 上限为 32 MiB。

上下文溢出返回 `error.code=context_length_exceeded`，客户端可据此压缩历史并重试；其他无效参数仍为 `invalid_request`。

`stream` 默认 false；显式 true 使用 SSE deltas 和 `[DONE]`。`stream=null` 等同默认值，`tools=null` 等同无工具。请求 `stream_options.include_usage=true` 时，普通 chunk 包含 `usage:null`，结束前另发 `choices:[]` 的完整 usage。usage 包含 reasoning 和 prefix cache token 计数。客户端断连后取消生成。执行异常时，流以 error 明确报告失败。

`logprobs=true` 返回正文 token 的 `token/logprob/bytes/top_logprobs`，`top_logprobs` 支持 0–20；SSE 返回对应正文 delta 的概率。概率来自 target 模型，应用惩罚、bias 和约束后、top-k/top-p 截断前；temperature=0 的 greedy 请求以 T=1 softmax 报告概率。`bytes` 可用于还原跨 token 的 UTF-8 字符；正文概率的统计范围为最终回答文本。开启此选项需要将 logits 下载到 CPU，会增加开销。

非流式概率结果按输出预算预留 CPU 内存，计入 `--preprocess-memory-mib`，预算保持到 HTTP body 结束；过大的单请求在准备前返回 400。SSE 通过有界增量缓冲逐步发送概率，缓冲上限同样适用于慢客户端。

参数错误返回 HTTP 400 及 `error.message/type/param/code`，响应带 `x-request-id`。`/health.chat_capabilities` 声明当前能力。`store=false`、`metadata`、`safety_identifier` 和 `service_tier=auto/default` 可传入，本地服务按兼容字段接收；completion 随响应返回，持久化与云端审核由客户端负责，服务等级按统一的本地调度处理。`store=true`、`n>1`、音频、视频和未知参数会明确报错。

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

thinking 可与 JSON 输出组合，正文仍受约束。MTP 的 target 和 draft 使用相同约束，解析器只提交已接受 token 的状态。达到输出预算会返回 `finish_reason=length`，正文可能为 JSON 前缀；调用方需检查 finish reason 并按完整性处理。约束输出要求省略 `stop`，由 grammar 和输出预算控制结束。

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

每张图片独立执行双向视觉 attention，merger 输出注入对应 image tokens；文本使用交错 MRoPE。视觉 encoder 每次仍执行，相同图片的预处理结果可命中文本主干 prefix cache。视觉输入范围为静态图片和有序多图。构建与验证见[图文工具](../tools/vision/README.md)。

### 工具调用

支持 `tools`、`tool_choice`、`parallel_tool_calls`。模型的 XML 工具调用转换为标准 `tool_calls`，arguments 为 JSON 字符串。客户端执行工具，再把带 `tool_call_id` 的 `role=tool` 消息连同历史发送回来。

流式调用先发 index/id/type/name，再增量发送 `function.arguments`；客户端按 index 拼接参数。`strict=true`、`tool_choice=required`、指定函数，或 `parallel_tool_calls=false` 时启用约束解码，限制函数名、参数 schema 和调用数；`required` 至少调用一次，指定函数仅能调用该函数，关闭 parallel 最多一次。内部约束调用使用 JSON envelope，API 和历史仍为标准 `tool_calls`。

auto 调用在宽松模式下沿用原生 XML 协议。未知函数、缺失参数或错误参数类型仍作为调用交给客户端，客户端应校验并返回 `role=tool` 错误结果，让模型重试；工具由客户端执行。XML 参数按声明类型转换，无法转换的值保留为字符串。语法损坏到无法辨认函数的输出仍报错。输出预算用尽时返回 `finish_reason=length` 并保留已生成的调用前缀，流式与非流式参数一致；客户端仅执行完整且通过参数校验的调用。回传历史中存在无效 JSON 对象参数时，原始字符串在原生模板中以 `__orinfer_raw_arguments` 参数完整保留。

### OpenCode 与 Pi

[OpenCode 配置](../examples/opencode.json)使用 `@ai-sdk/openai-compatible` 和默认编码 agent，启用自动压缩，声明图片、工具、thinking 及 `reasoning_content` 历史回传；agent 按客户端自身的运行策略推进。将其复制为项目中的 `opencode.json`，按服务的 `--model` 选择对应模型：

```bash
opencode run --model orinfer/qwen-flash-next '检查并修复当前项目的测试失败'
opencode run --model orinfer/qwen-flash-next --variant low '检查并修复当前项目的测试失败'
```

默认关闭 thinking；low/medium/high 分别使用 512/1024/2048 思考 token 上限，总输出上限为 8192。工具权限沿用 OpenCode 默认设置。

[Pi 配置](../examples/pi-models.json)对应 `earendil-works/pi`。将 provider 合并到 Pi agent 目录中的 `models.json`，默认目录为 `~/.pi/agent`；配置使用 `openai-completions`，可通过 `/model` 和 `/thinking` 切换模型与思考级别：

```bash
pi --provider orinfer --model qwen-flash-next --thinking low
```

配置中的 `orinfer-local` 是无鉴权本地服务的占位 key，使模型在 Pi 中可选；启用服务鉴权后用实际凭据或环境引用替换。Pi 的 thinking budgets 通过 `thinking_token_budget` 发送；可在 Pi 设置中调整 `thinkingBudgets`。示例使用服务默认的可驱逐内存缓存。图片和多图由用户附件或客户端 read 工具送入 messages。两端的模型 ID 必须与正在服务的模型一致，contextWindow/context 与输出上限应按实际模型包填写。

## 执行与缓存

### CUDA Graph

`serve`、`run-model` 和 `score-model` 支持 `--cuda-graph decode_only|full|off`。

- `decode_only`：默认模式。普通 decode、MTP 草稿、验证、恢复和短步刷新使用 Graph；文本 prefill、视觉编码和 MTP 首次预热直接提交。
- `full`：捕获并使用全部执行计划。
- `off`：按相同计划直接提交 kernel、copy 和 memset。

prefill 尾部即使复用 decode 计划，在 `decode_only` 模式下也直接提交。地址及 workspace 变化会使相关 Graph 失效或重新捕获。

单请求 Graph 在程序首次执行时按需捕获。batch Graph 在成员组合重复出现后捕获，缓存常见的 2 的幂次批次，同时限制条数和总操作节点数；过渡批次及超出节点预算的计划直接提交；内存准入紧张时先淘汰空闲 Graph，再回收空闲请求状态。首次执行的捕获耗时计入请求预热。

短请求结束后，每个 KV 缓冲区最多保留一个 CUDA 分配粒度的页，清零后供同一槽位复用；较大映射释放，内存紧张时空闲页也释放。准入只计入需要新增的 KV 页，各活跃请求的未来预算独立计费。

新请求优先选择新增分配最少的空闲槽位。回收后重新计算准入预算；没有活跃请求时，临时内存不足会排队重试最多两秒，持续不足返回 503 `service_unavailable`，服务继续处理能够准入的请求。

Flash Next 的主模型支持 2048／4096 token 大分块，MTP 预热保持 512 token 分块；使用 `full` 可同时对这两条 prefill 路径启用 Graph。索引计算和临时评分缓冲区按当前有效上下文推进，缓存容量仍为 256K，Graph 使用的虚拟地址保持稳定。

### MTP

包含原生 `mtp` 计划的模型自动启用 MTP，适用于文本、图片、多图、thinking、工具调用和支持的采样参数。无惩罚 greedy 使用 GPU top-1；随机采样对 target/draft 应用相同的采样处理，接受概率为 `min(1,p/q)`，拒绝后从归一化的 `(p-q)+` 分布采样，以保持主模型分布。

`--prompt-lookup` 可选开启本请求 token 历史的续写草稿，默认关闭。它适合保留大量输入原文的代码修改或重复模板；普通生成可能因候选拒绝或验证批次选择而变慢。仅在无惩罚 greedy 且未请求 logprobs 时尝试，所有候选仍由主模型验证。查找窗口有界，较长的匹配可在 `--mtp-drafts` 上限和其他请求的时限内扩大验证批次，连续低接受率后暂时退回模型草稿；随机采样继续使用原生 MTP。

`--mtp-drafts 0` 移除草稿和验证执行计划，加载时跳过它们独占的权重；主模型、视觉及 batch 所需的目标状态捕获仍保留。模型文件保持原样，重新启用 MTP 时正常加载草稿权重。

拒绝时恢复私有 GDN、卷积、位置和有效 KV 状态。固定 seed 控制采样随机流；自适应验证长度受实测耗时影响，同一 seed 下的随机输出仍会随调度和验证长度变化。随机 MTP 与普通 decode 保持相同目标分布，逐 token 复现应在固定执行条件下单独验证。输出尾部不足一个验证块时执行普通 decode。

单请求使用 MTP；包含批量草稿/验证能力的 Flash 包支持 2–8 个 decoder 共享草稿投影、主模型验证与接受状态提交，接受后的移位 token 按真实因果段成组预热，各请求只在自己的端点产生下一轮草稿。卷积、GDN、QSA、位置、采样与约束解析器保持私有。其他并发、混合 prefill 或输出尾部使用普通 target batch；缺少批量 MTP 能力的包保留逐请求选择。

草稿长度在 CLI 上限内按整轮耗时和实际提交量选择，包含草稿、验证、提交和预热。多请求先测普通 target batch 的成本，再判断 MTP 是否更快；跨上下文档位保留低收益结论，通过同一个有界探测周期复测，以复用已有成本结论。各档先采集预热与稳定样本，稳定估计使用完成 CUDA 模块加载及 Graph 捕获后的样本；低收益时退回普通 batch，按已提交 token 数定期复测。探索仍受其他请求的 decode 时间预算约束。收益取决于接受率与并发规模，短请求的调优样本可能较少。构建方法见[模型工具](../tools/model/README.md)。

#### 单请求深度探索（可选）

`ORINFER_SINGLE_MTP_POLICY=explore_local` 可选启用单请求 MTP 深度探索：在时间预算内为合法草稿深度采集稳定样本，第二次观测时丢弃首次冷执行耗时，并使用当前请求的接受率估计收益，避免其他提示词的历史拒绝率压低本请求的预期提交量。当前请求的稀疏样本收缩估计仍保留。

```bash
ORINFER_SINGLE_MTP_POLICY=explore_local orinfer serve /path/to/prepared-model --mtp-drafts 3
```

该策略仅作用于当前执行轮次选中一个 decoder、且没有混合 prefill 的情况，不要求整个服务只有一个活跃请求。CLI 草稿上限、合法执行计划、剩余输出长度和其他请求的时间预算仍约束候选选择；多 decoder 和混合 prefill 保留原有策略。接受率低或请求短时，探索成本可能抵消收益，应使用实际工作负载测量。

未设置该变量或设为 `auto` 时保留默认行为。`explore` 仅启用深度探索和冷样本排除，保留跨请求收益上限；`fixed0`–`fixed3` 用于诊断，分别限制为最多 0–3 个草稿，并选择预算内最宽的合法候选。这些诊断模式同样只作用于上述单 decoder 轮次。其他值会报错。

### Prefix cache

`--prefix-cache-mib` 默认 12288，单位 MiB，0 关闭复用；随实际缓存内容按需分配。缓存保存完整 KV、FP32 GDN、卷积、位置及 MTP 检查点。Rust radix tree 按 token 和图片身份匹配，并结合实际 prefill/恢复成本选择可用端点。

KV 区间不可变并按引用共享，持续状态单独保存。完整对齐的 KV 区间通过 CUDA VMM 共享物理内存，恢复后保留各请求的 Graph 虚拟地址；边界片段和 GDN、卷积、PLE、未完成压缩块等状态仍逐请求复制。共享区间只读，写入前复制为请求私有内存，复用 arena 前解除共享映射。创建新检查点仍需保存一份前缀数据。短前缀妨碍大块执行时可能跳过。保存最终提示、周期及按成本准入的分叉检查点；生成结束也可保存已计算的输出前缀。保存输出前缀时保留它所属的最长可复用提示；预算不足则跳过这次输出快照。正常内存回收仍可驱逐提示。模型重载后缓存清空。

命中免除对应文本主干计算，视觉编码、剩余提示和生成继续执行。`usage.prompt_tokens_details.cached_tokens` 报告实际恢复长度，SSE 需请求 usage。VMM 通过映射共享完整对齐区间，并单独恢复边界与持续状态。完整提示的保存和命中取决于内存余量；接近满上下文时可能驱逐或跳过快照。

接受 1–64 字符的 `prompt_cache_key` 和 `prompt_cache_retention=in-memory|24h` 作为路由、保留偏好。当前单 worker 统一处理请求；缓存按精确 token 和图片身份匹配，`prompt_cache_key` 作为兼容的路由提示接收。缓存采用可驱逐的内存存储，`24h` 作为保留偏好接收，实际寿命取决于内存压力和模型驻留时间。`/health.chat_capabilities.prompt_cache` 声明该语义。省略这些字段时同样自动复用前缀。

## 并发、上下文与资源

支持批处理的算子包启用连续 batching；未包含相应算子时串行执行，`/health.continuous_batching` 报告实际模式。每个请求拥有独立 KV、GDN、卷积、位置、视觉和采样状态，权重及执行 workspace 共享。

单请求冷 prefill 使用大块计划；兼容的多个短请求可联合执行。与 decode 混合时，按预测耗时选择小块 prefill。常见 batch 可使用固定形状 kernel，其余 2–128 行可由支持动态行数的 AOT 包直接执行。请求在每轮迭代按就绪状态加入和结束。

Flash Next 的 batch 包共享投影、MoE 路由、专家和输出头计算，GDN、卷积、PLE、QSA/KV/index/pending 与 MTP 状态按请求隔离。提供 2/4/8/16/32/64/128 档；支持动态行数的包让其他数量复用邻近容量的 AOT 算子，按真实行数执行。包含混合执行能力的包会把 decode 行和真实 prefill 片段拼接，共享 HC、投影与专家计算；单 token decoder 子组继续使用批量 GDN/QSA，较长因果段各自推进。独立长 prefill 保留大块路径。

有 decoder 时，选择能容纳完整混合迭代的最大已编译 prefill 块；预测包含输入准备、MTP 预热、缓存检查点和首 token 收尾。耗时估计按上下文建立单调上界，并采用滚动 P95 正向误差和 EWMA 误差余量。冷启动、长上下文和检查点仍可能超出软预算；最小有效块保证 prefill 进展，尾部只处理真实 token。视觉编码器独立执行。`prefill_chunk_histogram`、`max_bounded_prefill_s`、`decode_budget_overruns` 可用于核对实际行为。构建入口见[Flash 模型工具](../tools/model/flash_next/README.md)。

包含私有地址表的 Flash 包在每层用一套 GDN/QSA kernel 并行处理全部请求，包含临时缓冲区的隔离和批量卷积历史提交。请求地址在 Graph 重放前更新，补零行的地址为空；单请求仍走 M1 程序。

请求的 KV 和 MTP hidden ring 按总 token 预算分配，长请求仍可使用完整容量。槽位复用时收缩过大的私有分配；显存准入受阻时保留一个空闲槽位并回收其余空闲槽位的普通缓冲区，下次准入前重新分配。实际活跃数取决于请求预算和可用显存，超过准入容量的请求排队。

默认最多 32 个活跃请求、128 个待处理请求，队列满时返回 429。活跃数还受每请求上下文/输出预算、workspace 和可用内存约束。准入检查 CUDA 可用内存和 Linux `MemAvailable`，保留系统余量；不足时先收缩 prefix cache，再排队。等待队列按初始化、图片编码、有效缓存恢复和剩余 prefill 成本排序，等待时间平滑提升优先级，30 秒后优先处理最老请求。图片身份在队列描述中只计算一次，缓存匹配随 checkpoint 更新。大于每轮 token 预算的冷 prefill 通常最多提前准入两个，短请求可成批准入；可有效复用的共同 prefix 由一个请求先计算，其余暂留 CPU 队列。等待 30 秒仍因内存不足无法准入的请求会阻止新增工作，直到已有请求结束、腾出空间；取消或超时会解除此保护。统计见 `/health`。

prefill 按实际耗时扣减执行额度，各块按实际占用时间公平计费。有 decoder 时根据距上次提交 token 的时间决定下一轮工作；至少定期推进一个 prefill token，目标不可满足时计入 `prefill_progress_overrides`。`--target-tpot-ms` 是引擎提交 token 间隔的软目标；输出预算独立设置，视觉编码、缓存保存或冷启动可能使流式间隔超出该目标。`max_decode_gap_s` 和 `decode_budget_overruns` 记录引擎提交 token 的间隔，HTTP chunk 间隔需另行测量。

实际活跃数还结合已测量的相邻 batch 耗时与吞吐决定，未知档位允许探索。Graph 首次遇到新成员组合时直接执行，重复出现且仍有输出预算时捕获；已有图可用于最后几步。共享计算段还可按算子、行数、视图和实际地址复用 Graph，私有状态段按请求执行，换成员前更新地址表。整批与共享段共用 12288 个操作节点的缓存预算；CPU 执行计划有独立的有界缓存，并只保留各算子实际使用的缓冲视图。`batch_execution.shared_graph_hits` 与 `shared_graph_captures` 报告共享段复用。`decode_only` 仍只在 decode/MTP 阶段使用 Graph，混合 prefill 的图需要 `full`。

模型包决定实际上下文容量，上限可配置为 262144 tokens。长提示分块计算，首次 dense attention 仍完整计算所需上下文。INT8 group-64 KV 包含 FP16 scale，CUDA VMM 按已执行位置映射物理页，同时保持 Graph 地址稳定；prefill 的单层 FP16 临时 KV workspace 可共享并按需增长。持续 GDN 状态保持 FP32。上下文扩展、KV 优化和验证见[模型构建](../tools/model/README.md)。

| 参数 | 默认值 | 行为 |
| --- | --- | --- |
| `--listen` | `0.0.0.0:8088` | HTTP 监听地址 |
| `--model` | `qwen3.8-27b` | API 模型 ID |
| `--default-request-params` | `{}` | JSON 请求默认值，显式请求优先 |
| `--scheduler-policy` | `latency` | 在线延迟优先；`throughput` 以整批吞吐为目标 |
| `--cuda-graph` | `decode_only` | Graph 模式 |
| `--resident-experts` | `full` | 主干专家每层等效常驻容量；有限模式全局缓存，MTP 全量常驻 |
| `--verify-weights` | 关闭 | 启动时完整校验已加载权重和 CPU 表内容 |
| `--prefix-cache-mib` | `12288` | GPU 缓存字节预算，0 关闭 |
| `--max-active-requests` | `32` | 活跃请求上限，1–128 |
| `--max-batch-tokens` | `128` | decode/验证行预算，1–128；混合 prefill 可使用执行包的行容量 |
| `--prefill-budget-ms` | `200` | `latency` 混合块的预测耗时上限，非严格延迟上限 |
| `--target-tpot-ms` | `400` | `latency` 引擎提交 token 的软间隔目标 |
| `--memory-reserve-mib` | `1024` | 系统余量以外的准入保留内存 |
| `--preprocess-workers` | `2` | 同时执行图片/模板/tokenizer CPU 任务数 |
| `--preprocess-memory-mib` | `2048` | 预处理和排队图片张量预算 |
| `--queue-timeout-ms` | `0` | 排队期限，0 持续等待；仅作用于排队阶段 |
| `--output-timeout-ms` | `60000` | 输出通道等待进展的期限，从输出阶段开始计时 |
| `--drain-timeout-ms` | `30000` | SIGINT/SIGTERM 后 HTTP 排空期限 |
| `--gpu-lock` | `artifacts/gpu-experiment.lock` | GPU worker 使用的共享锁 |

默认 `latency` 策略按完整迭代成本选择联合或独立 prefill。多个冷请求到达时保留长请求的原生大块；联合路径经过测量后才参与成本选择。混合 batch 与独立 prefill 共用 decode 时间预算，执行 prompt 工作后会留出受 `--prefill-budget-ms` 和 `--target-tpot-ms` 限制的连续 decode 时间，期间允许 MTP；等待过久的 prefill 保留推进机会。`throughput` 策略放开时间配额和冷 prefill 数限制，允许探测未知联合形状，仍保留内存准入与请求隔离，适合[离线任务](batch.md)。`/health` 的 `independent_prefill_choices`、`joint_prefill_choices` 和 `decode_burst_iterations` 可核对实际选择。请求日志分别记录排队、从进入模型队列起的首 token 时延、生成总时长和 thinking token 数；首 token 指引擎提交，可能早于客户端可见的完整文本。

`/health` 显示 starting/ready/draining/failed，并提供调度、准入、缓存恢复和 Graph 统计。GPU worker 故障时返回 503；CUDA 或内部状态故障后需要重启服务。统计和真实 API 性能测量见[性能工具](../tools/bench/README.md)。

HTTP 在加载模型前绑定监听地址，端口被占用时立即退出；模型和预热完成后开始处理 HTTP。就绪探针使用 `/health` 的 HTTP 200，TCP 连通只表示监听地址已绑定。A2 全量常驻 CPU 表的启动可能需要数分钟，部署时为启动探针预留足够时间，例如 600 秒。正常退出及启动失败会清理本服务创建的 Unix socket；已有 socket 绑定失败时保留原文件。

A2 rank 的命令使用一个共同的等待期限，`ORINFER_A2_COMMAND_TIMEOUT_MS` 默认 `30000`，支持 `1..1800000` 毫秒，同时设置 ACL 同步等待期限；初始模型加载单独允许 1800 秒。rank 驱动错误、线程退出或命令超时会使当前 TP 团队失效，保留首个故障的类型和驱动错误码，后续请求无法继续使用该团队。服务故障后由进程管理器重启；调整命令期限时也应相应配置 HTTP 和停机期限。

A2 DSv4.1 将请求私有状态按存活范围分配：有界 SWA/C2 状态和固定 workspace 放入 owner arena；声明为 append-only 的 KV、FP16 scale 和索引保留最大上下文的虚拟地址，物理页按实际执行位置增长。补页在 Graph 准备和执行前完成，已有地址与算子 ABI 保持稳定；prefix 恢复先映射对应的历史范围，再复制状态。取消和完成请求后，等待已有访问结束再释放映射；owner 0 保留启动时的固定缓冲区，回收可增长状态的物理页。实际字节数包含设备页粒度和各 buffer 的对齐开销。

A2 准入按请求的实际输入长度与声明的输出上限保留后续 KV 增长额度，物理页仍随执行进度映射。尚未映射的额度同时从物理显存和 session 余额中扣除，供新请求、prefix 快照和 Graph workspace 统一检查，避免已接收请求后续增长时与缓存争抢同一份空间。`--prefix-cache-mib` 按同一快照 bank 在各 rank 的最大物理占用计费，共享 bank 去重，CPU 输出也计入缓存预算。

A2 准入同时检查每张卡的物理空闲显存与原生 session 分配余额，并保留执行 workspace；不足时依次回收 prefix 快照和缓存的 Graph 程序，再决定是否排队。快照先规划新增 bank 并检查缓存预算和分配余量，再分配和复制；回收父快照后重新规划，避免把完整复制误算成增量复制。正常容量不足走准入等待；其他进程的显存变化或原生驱动分配失败仍可能使 worker 失效，需要重启服务。

A2 HTTP/JSONL 使用 `ASCEND_COREDUMP_SIGNAL=none`，由服务处理 SIGINT/SIGTERM 并回收资源。`serve`、`worker` 和 `batch` CLI 在创建线程前默认设置该值；直接调用 Rust API 时在进程启动环境中设置。A2 检测到缺失或冲突配置时在创建 GPU worker 前报错。[CANN 环境变量说明](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/800alpha002/apiref/envref/CANN%E7%A4%BE%E5%8C%BA%E7%89%88%208.0.0.alpha002%20%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E5%8F%82%E8%80%83%2001.pdf)说明了 trace 与业务信号捕获冲突时的配置；此设置关闭 CANN trace 的信号堆栈文件，Rust 错误报告及系统 core dump 配置继续生效。SIGINT/SIGTERM 会中断未完成请求，按 `--drain-timeout-ms` 等待 HTTP 响应及 worker 回收。

A2 停机时，常驻 CPU 表等待所有读取者释放后回收，与 rank 和 bootstrap 原生资源的销毁重叠；两者完成后释放 GPU 锁。全量常驻大 CPU 表的部署使用 `--drain-timeout-ms 120000`，为内存释放和驱动清理保留余量，外部进程管理器的强制退出期限应大于此值。

若启动或停机错误返回时仍有模型 worker 存活，CLI 记录错误后以退出码 1 直接终止进程，跳过会与原生线程竞争的第三方退出钩子，由操作系统回收进程资源。正常退出仍等待模型和 CPU 表完成销毁。Rust API 的 `has_live_model_workers()` 可供嵌入调用方判断后台资源是否仍在回收。

启动默认检查配置签名、执行库和 kernel 的 ABI，以及权重和 CPU 表的 safetensors 结构、dtype、shape、布局和偏移边界，不计算资产内容 SHA256。`serve`、`run-model` 和 `score-model` 添加 `--verify-weights` 可校验加载的权重内容；`validate-model` 始终完整校验模型权重。检测结构合法但数值受损的权重需开启完整内容校验。

HTTP 与离线入口并行准备 Chat 前端和模型，CUDA 资源始终在 GPU worker 上创建和释放；两者成功并核对声明的前端身份与词表后才就绪。`WORKER READY` 分别记录前端、模型与并行准备总耗时，`API READY` 的 `startup` 包含前端和 HTTP 初始化。`LOAD STAGES` 提供 CUDA 初始化、buffer 分配、权重流水线等待和算子资产准备计时；`kernel_assets_s` 包含在 `module_load_bind_s` 中，权重等待计时包含在 `weight_upload_s` 中；总耗时按外层计时读取。完整进程到首输出的测量见[性能工具](../tools/bench/README.md)。

已加载的只读权重共享一次 CUDA arena 分配，保留各张量的对齐和布局，padding 计入权重与显存统计；请求私有状态、KV 和按需加载的专家保持独立管理。读取线程复用分片文件描述符，按文件偏移调度读取，并与多块异步上传重叠；缓冲区在对应 DMA 完成后复用。启动失败时先完成 CUDA 清理，再释放 GPU 锁。

## 诊断 CLI

`plan-model` 在 CPU 加载模型包原生库并输出其生成的计划，`validate-model` 校验模型、执行库、kernel 和张量 hash；两者均在 CPU 上执行。执行包接口与更新方式见[模型执行包](model-packages.md)。

`run-model MODEL_DIR REQUESTS.json` 接收 token-ID 请求，用于固定块、普通 decode 的诊断基准，逐请求重置状态。输入长度须能由声明的 prefill 计划整除；该入口用于固定长度执行诊断；任意长度、多请求、MTP 和缓存生成通过 Chat API 或 Rust `Model::generate` 验收。

`score-model` 对冻结 token 历史评分，服务的固定场景检查见[API 验证](../tools/api/README.md)，BF16/FP8 质量对照见[质量评测](../tools/eval/README.md)。


图片预处理可通过 Chat 请求的 `image_processing` 配置：

```json
{
  "image_processing": {
    "max_pixels": 1048576,
    "max_tokens": 16384,
    "max_images": 20,
    "auto_resize": false
  }
}
```

这些字段均可省略。`max_pixels` 限制每张图缩放后的像素数（至少 65536，尺寸按 patch/merge 对齐）；`max_tokens` 限制所有图片合计的视觉 token；`max_images` 限制图片数量。总容量同时受执行包的 `max_features` 和完整文本上下文限制，主机预处理仍受服务的共享内存预算限制。默认按每张图的实际尺寸处理，超出总容量会返回所需和可用视觉 token 数。`auto_resize: true` 在超限时缩小 `detail: auto` 的图片；小图保留实际需求，将空余份额分配给大图，`high`/`low` 保留各自明确的分辨率语义。

排队期间仅保留缩放后的 RGB8。GPU worker 逐图生成 checkpoint 要求的 FP16/BF16 patch 输入，完成上传后释放该图的 CPU 数据。单图上传 scratch 与队列数据一起计费；请求的图片顺序、MRoPE、prefix 身份和私有特征空间保持绑定。原有离线 FP32 图片夹具继续支持。

新构建的 Qwen3.5 执行包按上下文声明图片总特征容量，单图 patch 容量和文本 packet 行数保持原有契约。特征 bank 启动时仅分配一行，每个请求按实际视觉 token 数租用；提高上限不会预先分配完整 bank。已有 Qwen3.5／Flash Next 执行包可离线升级总容量：

```bash
bash tools/operators/run.sh tools/model/vision_capacity.py artifacts/vision-capacity-build \
  --model /path/to/model --model-output /path/to/upgraded-model
```

该命令默认把总容量提高到模型上下文上限，也接受 `--features N`。它保留权重字节，重编译普通、MTP 和地址表 packet 的视觉覆盖入口，验证原有行数与 ABI 契约后原子发布新执行包。通过 `package install` 安装新包后使用升级模型目录。未升级的执行包继续遵守原容量上限。
