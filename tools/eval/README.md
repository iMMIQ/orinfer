# 质量评测工具

`quick_quality.py`分析同一teacher-forced历史下的baseline/candidate token概率；概率按完整词表归一化，并显式查询baseline top-3 IDs。token身份不同用于诊断，不单独作为质量失败条件。

`scoring_common.py`校验固定seed、历史身份、概率字段及任务结果。`run_quality.py`连接配置的评分接口，使用`fixtures/quick-quality-scenarios.json`生成评测请求。参考服务需自行提供。

```bash
python3 -m unittest discover -s tools/eval -p 'test_*.py' -v
python3 tools/eval/run_quality.py --help
```

官方BF16或同模型FP8用作量化质量参考；同AWQ权重对照只用于实现诊断。场景检查、target NLL与长上下文任务需要分别评估。
