# 性能工具

`reference_bench.py`测量兼容OpenAI SSE接口的prefill/TTFT与decode时间；`test_reference_bench.py`检查token chunk和计时间隔。

`run_aot_artifact.sh MANIFEST NEW_OUTPUT`构建Rust CLI并执行AOT fixture，保留实际运行身份和机器状态。模型完整推理使用`tools/model/run.sh`。

`sample_machine.py`和`sample_continuous.py`只读记录机器温度、频率、内存和功耗；不会改动全局设置。GPU运行入口共享`artifacts/gpu-experiment.lock`。

`validate_plan.py`校验benchmark与快速评测配置，属于CPU检查，不产生性能测量。

`concurrency.py`以真实SSE请求测量1..128并发，输入为`{ "cases": [{ "id": "case", "request": { ...Chat API body... } }] }`：

```bash
python3 tools/bench/concurrency.py --requests requests.json \
  --concurrency 1 2 4 8 16 32 128 --repetitions 3 --output artifacts/concurrency.json
```

报告包括真实usage输出token数、整轮吞吐、TTFT分位数、缓存tokens及每请求结果。吞吐包含排队、prefill、HTTP和输出，不能直接当作GPU纯decode TPS；流式chunk间隔也不能当作token ITL。重复请求会命中prefix cache；测冷请求时需构造不同输入或关闭cache。`/health.scheduler_statistics`提供prefill实际计算tokens、引擎提交的decode tokens、batch直方图、混合迭代及峰值活跃数。
