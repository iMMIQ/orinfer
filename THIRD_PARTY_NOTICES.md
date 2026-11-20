# Third-Party Notices

This repository is licensed under **LGPL-3.0-or-later** (see [LICENSE](LICENSE) and [COPYING](COPYING)). The following third-party components and adapted materials retain their respective copyrights and licenses:

1. **Ollama** (MIT License)
   - Selected test vectors and adapted test scenarios originate from [Ollama](https://github.com/ollama/ollama), Copyright (c) Ollama.
   - Pinned revision, source test names, and adaptations are recorded in `fixtures/ollama/cases.json`; full license in `fixtures/ollama/LICENSE`.

2. **PyTorch & TorchVision Jetson Wheels** (BSD Licenses)
   - The offline CUDA environment uses PyTorch and TorchVision built for Jetson from [our PyTorch fork](https://github.com/iMMIQ/pytorch).
   - Release artifacts and SHA256 hashes are pinned in `tools/build/jetson_wheels.json`. Upstream BSD licenses and bundled third-party notices are retained in the wheels.

3. **DeepSeek V4.1 Reference Implementation** (MIT License)
   - Engram tokenizer-normalization and addressing rules in `python/orinfer_build/models/deepseek_v41/export_engram.py`, the DSv4.1 chat template and `fixtures/chat/deepseek-v41.json`, and DeepSeek image geometry and `fixtures/vision/deepseek-v41.json` are adapted from DeepSeek's reference implementation, Copyright (c) 2023 DeepSeek.
   - Full license in `python/orinfer_build/models/deepseek_v41/LICENSE`.

4. **TileLang-Ascend** (MIT License)
   - The Ascend 910B / A2 offline compiler uses [TileLang-Ascend](https://github.com/tile-ai/tilelang-ascend), Copyright (c) Tile-AI.
   - A2 projection and vector implementations reference its public programming model and examples; generated AOT libraries include its templates.
   - Full license in `kernels/ascend/LICENSE.tilelang`. CANN and Catlass retain their upstream notices in the compiler image.

5. **CANN HCOMM v9.2.0** (CANN Open Software License Agreement Version 2.0)
   - The CANN 9.2 HCOMM Graph registry ABI declaration in `crates/orinfer-engine/native/ascend/hccl_graph_registry_abi.h` is adapted from [CANN HCOMM v9.2.0](https://gitcode.com/cann/hcomm/tree/v9.2.0), Copyright (c) 2025 Huawei Technologies Co., Ltd.
   - Full license in `kernels/ascend/LICENSE.cann`. The optional Ascend adapter uses this declaration solely with the installed Huawei CANN runtime.
