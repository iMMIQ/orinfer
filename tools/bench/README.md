# 性能工具

`reference_bench.py`测量兼容OpenAI SSE接口的prefill/TTFT与decode时间；`test_reference_bench.py`检查token chunk和计时间隔。

`run_aot_artifact.sh MANIFEST NEW_OUTPUT`构建Rust CLI并执行AOT fixture，保留实际运行身份和机器状态。模型完整推理使用`tools/model/run.sh`。

`sample_machine.py`和`sample_continuous.py`只读记录机器温度、频率、内存和功耗；不会改动全局设置。GPU运行入口共享`artifacts/gpu-experiment.lock`。

`validate_plan.py`校验benchmark与快速评测配置，属于CPU检查，不产生性能测量。
