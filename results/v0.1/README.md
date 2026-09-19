# v0.1 结果

`summary.json` 由以下命令从原始实验日志生成：

```bash
python summarize_v0_1_results.py --output results/v0.1/summary.json
```

评估设置：seed 42--46，每个方法每个 seed 10 个 episode，Actor 冻结。

| 方法 | 成功率 | 平均步数 | 平均最小 Transfer MSE |
| --- | ---: | ---: | ---: |
| 原始最终目标 | 20/50 (40%) | 139.78 | 0.1466 |
| 无门控轨迹检索 | 26/50 (52%) | 123.52 | 0.0905 |
| 置信度门控轨迹检索 | 30/50 (60%) | 110.18 | 0.0864 |
| 学习型 MLP + 双门控 | 24/50 (48%) | 128.80 | 0.0940 |

完整解释、命令与局限见 [`../../docs/V0_1_RELEASE.md`](../../docs/V0_1_RELEASE.md)。
