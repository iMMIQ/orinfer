# TileLang kernels

生产算子按后端组织：`cuda/` 面向 CUDA SM87，`ascend/` 面向 Ascend 910B / CANN 9.2，
`shared/` 保存两个编译前端复用的动态行契约。CUDA 的基础算子、模型计算、音频和视觉
分别位于 `cuda/operators`、`cuda/model`、`cuda/audio`、`cuda/vision`。
设备验证程序位于 `tests/cuda` 与 `tests/ascend`，编译和调优实现位于 `python/orinfer_build`。

kernel 经离线编译生成 cubin 和真实 host ABI，在线由 Rust CUDA Driver 执行。导出格式与模型组装入口见[模型工具](../tools/model/README.md)。新增实现应检查完整数值、非对齐尾部、状态语义和改变输入后的 graph replay。

`audio/encoder.py` 提供 Qwen3-ASR 的动态 chunk 卷积、特征重排与位置注入。音频 Transformer 复用 `vision/encoder.py` 的浮点投影、LayerNorm、GELU、残差与窗口 attention；文本解码器复用 Qwen 动态行模板。`python/orinfer_build/models/qwen_asr/check_kernels.py` 验证卷积、104-token 窗口尾部和改变输入后的 Graph replay。部署与精度策略见[语音识别](../docs/audio.md)。

## 共享计算组件

- `cuda/model/integer_core.py` 提供 W8、标量码表 Q4 及双 W8 融合 SwiGLU 共用的 DP4A/Tensor Core 循环、尾部处理和输出缩放。权重 loader 按编码选择；gate/up 各自先舍入 FP16 再执行 SwiGLU。外层工厂只保留几何校验、ABI 和 launch 映射。两个调度由离线调优选择，各模型与请求场景共享计算体。
- `cuda/model/floating_core.py` 提供通用浮点投影、分组 BF16、HC 投影与视觉 Linear 共用的 MMA 循环。输入布局、操作数方向和顺序 group 归约在编译时固定；K 尾部在 tile 内补零，HC 严格保留 K16 累加顺序与中间舍入边界。视觉的 Conv3D bias 舍入与 Linear/GELU 输出仍由各自 epilogue 保持。
- `cuda/model/e8p_core.py` 提供完整 A8 位平面拆分、双 INT4 MMA 和输出恢复/缩放。普通分组、K-major 流水和唯一路由缓存输入共享算术组件，保留各自的数据移动调度。唯一路由路径要求每专家至多一行，由调用方的路由契约保证，E/N/K 由调用方显式提供。

组件通过 TileLang 宏内联，codec 在编译期固定，在线直接加载编译后的实现。编译后的 host ABI 仍是权重/状态 buffer 与必要的动态行参数。Qwen 27B 的两级 E8P Q4A8 复用相同 E8P 算术、旋转/A8 和动态行调度；其编码与 K128 布局由构建器校验。

BF16 视觉 epilogue 使用显式 PTX RNE 舍入/写入，保留 Conv3D 的独立 dot 舍入及 Linear/GELU 的精度边界，直接完成 BF16 向量类型的舍入与写入。FP16 保留自动向量化。`tools/vision/validate_kernels.py` 验证两种精度、K/N 尾部和修改输入后的 Graph replay。

`python/orinfer_build/models/flash_next/checks/shared_compute.py` 用独立 CPU 整数点积验证 W8/Q4、FP16/FP32 输出与融合 SwiGLU，包括非对齐 N、动态尾部及改输入后的 Graph replay。浮点和专家另外使用独立参考验证既有归约/路由契约。

`cuda/model/moe.py` 提供 FP32 top-k 路由、确定性激活分发和输出合并；A8 量化及融合 SwiGLU 复用 `operators/op30_activation_quantization.py`。整数 E8P 专家通过 `cuda/model/e8p_exact.py`、`cuda/model/e8p_prefill.py` 等调度在 Tile 内临时解码，权重保持单份压缩表示。

`cuda/model/hyperconnection.py` 提供 Qwen4 gated residual 的分支归一化、低秩投影、混合及残差更新，保留 BF16 权重和 FP32 累加。小行数路径将 down/SiLU 和 up/分支混合分别融合，保持原始 FP16 中间舍入。`cuda/model/ple.py` 提供 PLE 门控、膨胀卷积及历史更新；CPU 的 n-gram 哈希与有容量限制的 E8P 行缓存入口为 `python/orinfer_build/models/flash_next/ple.py`。验证入口为 `tests/cuda/operators/hyperconnection.py` 和 `ple.py`。

`cuda/model/integer_vq.py` 保留离线构建和验证所需的 E8P INT8 参考实现，以及输入侧 block128 旋转。`cuda/model/rotation_a8.py` 将旋转、可选 SwiGLU 和行级 A8 量化融合，保留原始 FP16 舍入边界。`rotate_expert_a8` 复用相同旋转计算，显式接收专家数、top-k 及 shared/scatter 布局，融合完整 A8 位平面打包和路由。公共验证入口 `python/orinfer_build/validation/check_rotation.py` 覆盖不同几何、动态尾部及改输入后的 Graph replay。旋转必须在对应投影前执行，down 的旋转位于 SwiGLU 之后。数值验证入口为 `tests/cuda/operators/e8p_selector.py`，整模型状态和请求验证入口见[Flash Next](../tools/model/flash_next/README.md)。

`cuda/model/int8_projection.py` 与 `cuda/model/q4_projection.py` 提供行 scale 的 W8 和码表 Q4 投影。`python/orinfer_build/models/flash_next/native.py` 组装离线参考模型；正式整数算子重建入口为 `python/orinfer_build/models/flash_next/build.py`，部署包由 Rust 在线执行。QSA 使用 `cuda/model/qsa.py` 的 group-4 压缩索引、精确 radix top-k 及 group-64 INT8 KV，`cuda/model/qsa_attention.py` 用共享 KV tile 的 FP16 Tensor Core 计算 attention，softmax 和累加保持 FP32；上下文上限为 262144 token。压缩索引、pending ring 和 KV scale 均纳入请求状态。算子验证入口为 `tests/cuda/operators/qsa*.py`，真实长上下文执行入口为 `python/orinfer_build/models/flash_next/long_context.py`。转换和独立 BF16 对照用法见[量化工具](../tools/quantization/README.md)。

`cuda/model/flash_mtp.py` 提供 MTP 的完整 HC 输入归一化、embedding/分支融合和逐位置历史保存；`cuda/model/greedy.py` 支持多行验证输出。目标模型验证时保存卷积、PLE 及 QSA pending 历史；GDN 保存键、衰减和 FP32 更新量，以 `cuda/model/gdn_sequence.py` 重放接受前缀到 FP32 状态，以紧凑因子保存分支更新。按接受长度提交，KV 和压缩索引的拒绝后缀通过 live cursor 隔离。`python/orinfer_build/models/flash_next/mtp.py` 负责离线 greedy 接受、草稿刷新及完整 session 恢复；MTP 共用主模型的 embedding 和输出头，另有独立 INT8 KV。算子验证入口为 `tests/cuda/operators/flash_mtp.py`，真实请求对照入口为 `python/orinfer_build/models/flash_next/checks/mtp.py`，融合、GDN 提交和专家直接索引验证入口为 `tests/cuda/operators/flash_mtp_optimization.py`。

`cuda/model/sampling.py` 提供共享 GPU 随机采样：FP64 惩罚与 bias 处理，先 top-k 再 nucleus 截断，按分数及 token ID 稳定排序。概率质量使用有界 U64 定点数，最大未归一化截断误差为 `vocab / scale`，scale 由词表大小确定并保证总和不溢出。整数 radix 累计在不同 batch 组成下保持一致；随机数仍由 Rust 按 seed 和已提交位置产生。计数、mask 及直方图均属于可重建 workspace；prefix 快照保存请求的持久状态。普通采样和无需概率报告的 grammar 采样走 GPU；top_logprobs 与 MTP 的 p/q 概率及拒绝校正保留 CPU 参考路径。算子数值、尾部、异常输入及改变输入后的 Graph 检查使用 `tests/cuda/operators/sampling.py`。

MoE 归一化 top-k 可使用 `cuda/model/moe.py` 的单 warp 行计算，以 SM87 整数归约保持 FP32 分数、同分低 ID 优先和原概率求和树。`cuda/model/qsa_selection.py` 通过局部 top-512 和有界候选合并实现精确 QSA 选择，复用原直方图 workspace；输出保留大于阈值、等于阈值两组内的升序 ID 及未完成块尾部。`cuda/model/ple.py` 的 FP16 norm/gate 融合复用 HC 归一化计算体，保留逐步舍入、零门控和卷积历史输入。验证使用 `python/orinfer_build/validation/check_auxiliary.py`，覆盖尾部及改变输入后的 Graph replay。
