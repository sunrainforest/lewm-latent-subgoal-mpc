# LeWM Latent Subgoal MPC

这是一个基于 LeWorldModel (LeWM) 的单任务 Cube 操作实验。当前冻结的第一版方法是**无置信度门控的成功轨迹检索**：只在 `TRANSFER` 阶段，从离线成功轨迹中寻找与当前 latent 最近的状态，并以同一轨迹最多前方 25 步的 latent 作为 MPC 的中间目标。Actor 在评估时冻结；LeWM、TransitionNet、三阶段 FSM、Actor、MPC 和 Value Ensemble 均保留。

代码中的 `oracle_subgoal` 是历史命名。这里使用的是**已有成功轨迹库的未来 latent**，不是测试回合尚未发生的未来真值，也不是已训练成功的 `LatentSubgoalPlanner`。库中含有 64 条固定任务成功轨迹；当前结果不证明跨任务泛化。

## 项目来源与许可证

本仓库不是 LeWM 官方仓库，而是从 **LeWM_SAIMO** 的 Cube / Proprio V2
实验代码中抽取、整理并继续改造的下游研究项目。LeWM_SAIMO 本身基于
LeWorldModel（LeWM）上游工作：

- [论文：LeWorldModel](https://arxiv.org/abs/2603.19312)
- [项目主页](https://le-wm.github.io/)
- [预训练模型与数据](https://huggingface.co/collections/quentinll/lewm)

作为本项目直接代码来源的 LeWM_SAIMO 快照采用 MIT License，并包含
`Copyright (c) 2026 Lucas Maes`。本仓库保留该版权与许可正文，见
[`LICENSE`](LICENSE)；更完整的来源、修改范围和第三方资产边界见
[`NOTICE.md`](NOTICE.md)。LeWM 权重、OGBench 数据及其他第三方二进制资产
未随本仓库分发，仍分别受其原始许可证或使用条款约束。

## 主要结果：`proprio_v2` latent 完成率

完成判据为 FSM 在 `TRANSFER` 阶段检测到最终目标 latent MSE ≤ 0.017，连续 2 帧后设置 `machine.complete`。不使用方块物理位置替代该判据。两组均使用相同 checkpoint、冻结 Actor、每回合独立重设随机种子，最多 300 环境步。

| 批次 | 回合 seed | Baseline | 无门控检索 H=25 | 差值 |
| --- | --- | ---: | ---: | ---: |
| 探索性对照 | 11100–11199 | 53/100 | 60/100 | +7 个百分点 |
| 新 seed 复核 | 11200–11299 | 51/100 | 59/100 | +8 个百分点 |

新 seed 复核的配对 bootstrap 95% 区间为 **−3 至 +19 个百分点**，双侧精确 McNemar `p=0.2005`。两批都观察到正向差异，但目前**不能声称稳定提升已获统计证实**。同 seed 的两个完整运行也不保证到达完全相同的 `TRANSFER` 状态。逐批配对计数和分析方法见 [结果说明](results/frozen_ungated_v1/README.md)。

## 复现

本仓库的 `source/` 是从 LeWM_SAIMO Cube / Proprio V2 实验复制并改造的
代码副本，不是完整 LeWM 权重或 OGBench 数据发行包。需要 CUDA 环境，
以及 [数据清单](docs/DATA.md) 所列的 LeWM 权重、`proprio_v2` checkpoint
和 64 条成功轨迹文件。二进制资产尚未获再分发确认，未放入 Git 历史；
**仅克隆此仓库不能直接重跑数值结果**。

资产齐备后，按 [复现指南](docs/REPRODUCE.md)检查版本和文件校验值，并运行：

```bash
PYTHON_BIN=/path/to/python bash experiments/run_frozen_ungated_v1.sh 100 100
PYTHON_BIN=/path/to/python bash experiments/run_frozen_ungated_v1.sh 200 100
```

脚本顺序运行 Baseline 和无门控 H=25，拒绝覆盖已有结果；原始日志、视频和模型输出写入 Git 忽略的 `runtime/`。也可用 [汇总脚本](experiments/summarize_frozen_ungated_v1.py)对已完成的 summary JSON 重新计算配对结果。复跑受仿真和 GPU 数值非确定性影响，未必逐回合 bitwise 相同。

## 范围与历史

- 当前第一版只将**轨迹检索辅助 MPC**作为主方法。仓库中保留的学习型 MLP、门控方案及早期 50 回合评估属于探索性历史，不应与此表混为同一结论。
- 仅 MuJoCo 仿真、单一方块起点/目标；没有真实机械臂实验。
- 需要离线成功轨迹库；这不是无需专家数据的通用规划器。
- 旧版 50 回合结果保存在 [`results/v0.1/`](results/v0.1/)，用于溯源，不是当前首页的主结果。

源码采用仓库中的 [MIT 许可证](LICENSE)，并保留 LeWM_SAIMO 代码来源的
版权声明；上游与第三方边界见 [`NOTICE.md`](NOTICE.md)。LeWM、OGBench、
模型权重和数据仍以各自许可或使用条款为准，未随本仓库再分发。
