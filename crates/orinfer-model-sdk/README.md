# `orinfer-model-sdk`

Shared Rust data contracts for execution plans, dynamic row launches, packet bindings, request-private state descriptors, and Schema 2 execution packages.

## Role in the Workspace

- Consumed by both **`orinfer-models`** (statically linked model adapters) and **`orinfer-engine`** (runtime loader, scheduler, and graph executor).
- Defines backend-independent descriptions for model architectures, operator catalogs, dynamic row expressions, MTP/speculation contracts, vision/audio/embedding/reranking metadata, and conditional memory.
- Keeps no CUDA or CANN driver dependencies so `plan-model` and `validate-model` run purely on CPU.

See [Model Execution Packages](../../docs/model-packages.md) for the Schema 2 package specification.
