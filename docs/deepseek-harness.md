# DeepSeek Harness 接入

Orinfer 0.3.1-fix1 支持 DeepSeek Harness（dsh）的 OpenAI Chat Completions 路径，包括流式工具调用、多轮工具结果、thinking 和图片输入。适配测试使用 npm 包 `@deepseek-ai/dsh@0.2.1-rc.5`；协议由 [DSH 的 pi-ai 适配器](https://github.com/deepseek-ai/deepseek-harness/tree/master/packages/llm/llm-pi-ai)提供。

安装 Node.js 后，可通过 `npx @deepseek-ai/dsh@0.2.1-rc.5 --profile headless --help` 初始化 headless profile。将[配置示例](../examples/dsh-cordis.patch.yml)中的条目合并到 `$DSH_HOME/profiles/headless/cordis.patch.yml`，调整 `baseURL` 和模型 ID，通过环境变量 `ORINFER_API_KEY` 提供鉴权。配置覆盖同 ID 条目的完整 `config`，已有其他 provider 时须一同保留。

```bash
export ORINFER_API_KEY=your-local-service-key
npx @deepseek-ai/dsh@0.2.1-rc.5 --profile headless '读取当前目录的 README.md，简要总结内容。'
```

图片需匹配视觉权重与执行包，模型配置声明 `input: [text, image]`。工具读取的 PNG/JPEG/WebP 经 dsh 转为消息图片，沿同一 Chat API 处理。Orinfer 按上下文和主机预算管理图片历史；缓存复用、请求体和单图预算见[服务文档](serving.md)。

示例使用 `thinkingFormat: chat-template`，将 dsh 的开关和 effort 传入 `chat_template_kwargs`。`thinking_token_budget: 1024` 限制思考阶段，全部思考与答案仍受 8192 输出 token 上限约束。`qwen-chat-template` 格式同样受支持：该格式发送 `enable_thinking` 和 `preserve_thinking`，具体 effort 使用服务端默认值。修改 `agent-default-model.reasoningEffort` 为 `off` 可关闭 thinking；`low/high/max` 通过示例中的动态变量选择推理强度。

服务端会同步处理嵌套 thinking 参数、模板输入和 `reasoning_content`，并遵守部署默认值的覆盖规则。`/health.chat_capabilities.chat_template_kwargs` 为 `true` 表示引擎已包含此兼容能力。部署时更新引擎，继续使用原有权重和匹配的执行包。
