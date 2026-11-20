# DeepSeek Harness Integration

Orinfer supports [DeepSeek Harness (`dsh`)](https://github.com/deepseek-ai/deepseek-harness) through its OpenAI Chat Completions adapter (`@deepseek-ai/dsh@0.2.1-rc.5` via `llm-pi-ai`), including streaming tool calls, multi-turn tool results, thinking controls, and image inputs.

---

## Setup

1. Ensure Node.js is installed and initialize the headless profile:
   ```bash
   npx @deepseek-ai/dsh@0.2.1-rc.5 --profile headless --help
   ```
2. Merge the provider entry from [examples/dsh-cordis.patch.yml](../examples/dsh-cordis.patch.yml) into `$DSH_HOME/profiles/headless/cordis.patch.yml`, updating `baseURL` and the served model ID. Preserve any existing providers in the file.
3. Export your API key and run a test prompt:
   ```bash
   export ORINFER_API_KEY=your-local-service-key
   npx @deepseek-ai/dsh@0.2.1-rc.5 --profile headless 'Read README.md in the current directory and summarize it briefly.'
   ```

---

## Thinking & Vision Behavior

- **Thinking**: The example uses `thinkingFormat: chat-template`, passing `dsh`'s toggle and effort (`off`, `low`, `high`, `max`) through `chat_template_kwargs`, with `thinking_token_budget: 1024` inside an 8,192-token output cap. `qwen-chat-template` (`enable_thinking`, `preserve_thinking`) is also supported.
- **Vision**: Requires a vision-enabled model directory and execution package with `input: [text, image]`. PNG/JPEG/WebP files read by tools are forwarded as `image_url` parts and follow the budgets described in [Serving Guide](serving.md#3-images--multi-image-input).
- **Capability check**: Confirm `/health.chat_capabilities.chat_template_kwargs` is `true` on the running server.
