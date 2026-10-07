# Offline quantization

Use the project uv environment for CPU tools and the fixed compiler image for GPU fitting. These codecs are experimental. Validate each checkpoint against an original BF16/FP8 reference; execution completion and quality acceptance are separate. Short scene tests establish initial quality evidence, not general benchmark performance.

`e8p.py` generates our integer basis from lattice geometry without importing an external quantizer. `e8p_gpu.py` fits original floating matrices using a fused TileLang nearest-neighbor encoder, input rotation, and two weight-only scale updates. The stored table is part of the format: indices must never be interpreted with a different basis or basis ordering.

Flash Next 的固定源转换、safetensors 发布、完整请求和 MTP 验证、原始 BF16 teacher 用法见[模型准备](../model/flash_next/README.md)。执行路径使用自有 E8P 编码；下面的其他 codec 保留为独立量化研究工具，不属于 Flash Next 部署路径。

`vq.py` defines independent experimental integer-vector fixtures in standard safetensors. VQ4 has U8 `[N,K/4]` indices and I8 `[256,4]` tables. E8P has U16 `[N,K/8]` indices and an even-integer I8 `[256,8]` basis; sign/parity and the unit shift reconstruct the final integer coordinates exactly. Both use F16 `[N]` scales and either empty or I8 `[K]` rotation signs. `gpu_layout()` uses group-major indices and little-endian packed U32 tables. Optional U16 `[N,K/128]` patches contain one valid bit, eight signed-value bits and seven position bits, adding exactly 0.125 bit/weight; replacement values use the same row scale.

Block128 rotations are normalized input-side Hadamard transforms. The corresponding floating weight transform is mathematically equivalent before quantization; A8/FP16 boundaries add numerical error. Store/charge signs, tables, patches, scales and padding in the footprint, and include activation transformation in complete-FFN performance. These codecs do not establish full-model quality, and the E8P decoder is not a reproduction of the full QuIP# quantization/training pipeline.

`block_ldlq.py` provides regularized covariance factors and a CPU oracle for reverse block error feedback following the BlockLDLQ principle. `rotation.py` provides offline forward/inverse transforms for the Flash Next dimensions using an order20 Paley Hadamard and a power-of-two factor. A two-sided transform must restore the output coordinates before nonlinearities; its reference representation is not an online operator package. These helpers do not include QuIP# fine-tuning or full-model calibration.
