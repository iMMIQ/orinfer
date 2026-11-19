# orinfer-models

统一 workspace 内的模型适配器以 Rust rlib 静态链接到 `orinfer-engine`。CUDA 执行包使用 schema 2，包含 `package.json`、`bindings.json` 和 `assets.safetensors`。

模型适配器注册 Qwen3.5、Qwen3-ASR、Flash Next、Nemotron Embed 和 Qwen Reranker 的 `int8_quality` 路径，并按执行包的数据契约生成执行计划、动态 batch 和视觉位置。

Flash Next 只接受当前动态执行契约：prefill 1–4096 行、共享 target 最多 1024 行、draft/warm 最多 512 行、MTP 验证 2–8 行。普通 decode、混合请求和 MTP 复用计算模板，状态所有权、捕获与提交分别处理。Q4 使用固定混合精度策略，输入旋转按消费者的工作区与符号表绑定，在共享 QKV/Z 的首个消费者之前执行一次。

```bash
cargo build --release --locked --offline -p orinfer-cli
cargo test --locked --offline -p orinfer-models
```

构建、包契约和验证见[模型执行包](../../docs/model-packages.md)。
