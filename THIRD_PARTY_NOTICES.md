Selected test vectors and adapted test scenarios originate from
[Ollama](https://github.com/ollama/ollama), Copyright (c) Ollama, under the MIT
License. The pinned revision, source test names and adaptations are recorded in
`fixtures/ollama/cases.json`; the full license is in `fixtures/ollama/LICENSE`.

The offline compiler image uses PyTorch and TorchVision built for Jetson by
[our PyTorch fork](https://github.com/iMMIQ/pytorch). The build profile records the
upstream revisions, NVIDIA toolchain image, and pinned dependencies.
`tools/build/jetson_wheels.json` pins the release artifacts and SHA256 values.
The wheels retain their upstream BSD licenses and bundled third-party notices.

The Engram tokenizer-normalization and addressing rules in
`tools/model/deepseek_v41/export_engram.py`, the DSv4.1 chat template and
`fixtures/chat/deepseek-v41.json`, the DeepSeek image geometry and
`fixtures/vision/deepseek-v41.json` are adapted from DeepSeek's V4.1
reference inference implementation, Copyright (c) 2023 DeepSeek, under the MIT
License. The full license is in `tools/model/deepseek_v41/LICENSE`.

The A2 offline compiler uses [TileLang-Ascend](https://github.com/tile-ai/tilelang-ascend),
Copyright (c) Tile-AI, under the MIT License. The A2 projection and vector
implementations use its public programming model and examples as references;
generated AOT libraries include its templates. The license is retained in
`tools/operators/ascend/LICENSE.tilelang`. The compiler image pins the wheel
and its SHA256; CANN and Catlass retain their upstream notices in that image.

The CANN 9.2 HCOMM Graph registry ABI declaration in
`tools/operators/ascend/hccl_graph_registry_abi.h` is adapted from
[CANN HCOMM v9.2.0](https://gitcode.com/cann/hcomm/tree/v9.2.0),
Copyright (c) 2025 Huawei Technologies Co., Ltd., under the CANN Open Software
License Agreement Version 2.0. Its full license is retained in
`tools/operators/ascend/LICENSE.cann`. The optional Ascend adapter uses this
declaration solely with the installed Huawei CANN runtime. Native HCOMM
methods and resources remain provided by that runtime.
