# TileLang kernels

目标为CUDA SM87。`operators/`包含基础算子，`model/`包含模型使用的投影、融合、GDN和attention实现，`projections/`包含基础投影的共用实现。

kernel经离线编译生成cubin和真实host ABI，在线由Rust CUDA Driver执行。导出格式与模型组装入口见[模型工具](../tools/model/README.md)。新增实现应检查完整数值、非对齐尾部、状态语义和改变输入后的graph replay。

`model/q2a8.py`提供Q2_0×A8投影，保留原始18字节/64权重的表示，支持直接索引专家权重库、分开的gate/up权重和GPU生成的紧凑专家分组。分组投影支持按K分组连续存放的布局；加载时的`q2a8_repack`只转置原始块，目标替换源表示，不增加常驻权重大小。`model/moe.py`提供FP32 top-k路由、确定性激活分发和输出合并；A8量化及融合SwiGLU复用`operators/op30_activation_quantization.py`。验证入口为`tools/operators/q2a8.py`、`q2a8_indexed.py`、`moe.py`和`moe_grouped.py`。

`model/hyperconnection.py`提供Qwen4 gated residual的分支归一化、低秩投影、混合及残差更新，保留BF16权重和FP32累加。`model/ple.py`提供PLE门控、膨胀卷积及历史更新；CPU的n-gram哈希与有容量限制的IQ4_NL实验行缓存位于`orinfer-engine::ple`，自有E8P权重的离线查表入口为`tools/model/flash_ple.py`。验证入口为`tools/operators/hyperconnection.py`和`ple.py`。

`model/q2i8.py`提供实验性2bit索引和局部INT8码表的投影，支持预分发及GPU紧凑专家分组。整个K维用INT32累加，最后按权重行scale及激活行scale统一转换；不使用组内浮点scale或全局展开W8。表示和离线校准契约见[量化工具](../tools/quantization/README.md)，验证入口为`tools/operators/q2i8.py`和`q2i8_grouped.py`。该格式尚未通过整模型质量验收。

`model/integer_vq.py`提供实验性VQ4和整数E8P的shared解码、INT8投影，以及输入侧block128旋转；可选每128个权重一个整数替换槽。旋转必须在对应投影前执行，down的旋转位于SwiGLU之后。数值验证入口为`tools/operators/integer_vq.py`，完整MoE性能探针复用`q2i8_grouped.py --vq-dir ... --vq-variant ...`。

`model/int8_projection.py`提供行scale的常驻INT8权重投影。`tools/model/flash_native.py`组装整数E8P专家、INT8普通投影及保留精度的HC/router，用于自有量化权重的离线整模型执行及质量对照；QSA使用`model/qsa.py`的group-4压缩索引、精确radix top-k及group-64 INT8 KV，`model/qsa_attention.py`用共享KV tile的FP16 Tensor Core计算attention，softmax和累加保持FP32；上下文上限为262144 token。压缩索引、pending ring和KV scale均纳入请求状态。算子验证入口为`tools/operators/qsa*.py`，真实长上下文执行入口为`tools/model/flash_long.py`。执行完成与质量验收分别记录，尚未提供Rust在线架构适配。转换和独立BF16对照用法见[量化工具](../tools/quantization/README.md)。
