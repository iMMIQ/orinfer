# orinfer-models

统一 workspace 内的原生模型实现。模型适配器与精度策略分别注册，共享执行组件、TileLang源码、构建工具和测试；部署时生成 `.so + cubin` 包。

当前支持 Qwen3.5 `int8_quality`，保持原有混合精度、融合、batch、MTP和视觉位置语义。模型库只链接 `orinfer-model-sdk`，不依赖 `orinfer-engine`；跨库仅使用 C ABI v1。

```bash
cargo build --release --locked --offline -p orinfer-models
cargo test --locked --offline -p orinfer-models
```

构建、包契约和验证见[模型执行包](../../docs/model-packages.md)。
