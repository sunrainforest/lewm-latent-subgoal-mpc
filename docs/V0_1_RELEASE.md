# 历史探索记录：早期 50 回合 v0.1

> 本文档保留早期门控/学习型 MLP 实验的原始说明，不是当前冻结的无门控 H=25 主版本。当前主结果、latent 完成判据和复现方法分别见仓库根目录 `README.md`、`results/frozen_ungated_v1/README.md`、`docs/REPRODUCE.md`。本文下方的 50 回合结论不可覆盖后续两批各 100 回合的独立种子结果；原有复现命令也使用旧目录布局。

# LeWM_Saimo v0.1：Confidence-Gated Latent Subgoal MPC

> 来源与许可：本实验是基于 `LeWM_SAIMO` Cube / Proprio V2 代码的下游改造；
> LeWM_SAIMO 随附的 MIT License 及原版权声明已保留在仓库根目录
> [`LICENSE`](../LICENSE)，详细边界见 [`NOTICE.md`](../NOTICE.md)。

这是一个基于冻结 LeWorldModel（LeWM）的 Cube 机械臂长程控制实验版本。
它保留原有 `ALIGN -> GRASP -> TRANSFER` 状态机、Actor、MPC、Value
Ensemble 和 LeWM dynamics，只在 `TRANSFER` 阶段研究中间 latent 子目标。

v0.1 的目标不是宣称已经解决层次规划，而是提供一套可复现的诊断与实验链路：

1. 用成功轨迹中的未来 latent 检查中间子目标是否有价值；
2. 用当前状态到成功轨迹的匹配误差进行置信度门控；
3. 训练单步 MLP `LatentSubgoalPlanner`；
4. 先做 shadow 评估，再通过输入/输出双门控接入控制；
5. 用相同的 5 个随机种子评估所有方法。

## 核心结果

每种方法使用 seed 42--46，每个 seed 运行 10 个 episode。Actor 在评估期间冻结，
因此不同方法的主要变量是 `TRANSFER` 阶段的目标来源。

| 方法 | Seed 42 | 43 | 44 | 45 | 46 | 合计 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原始最终目标 | 1 | 4 | 4 | 6 | 5 | **20/50 (40%)** |
| 无门控轨迹检索 | 6 | 4 | 4 | 7 | 5 | **26/50 (52%)** |
| 置信度门控轨迹检索 | 8 | 5 | 7 | 6 | 4 | **30/50 (60%)** |
| 学习型 MLP + 双门控 | 4 | 5 | 4 | 7 | 4 | **24/50 (48%)** |

门控轨迹检索相对原版的配对精确检验为 `p = 0.0213`。学习型 MLP 相对原版
为 `p = 0.4240`，当前不能认为它带来了稳定提升。

这里的“轨迹检索”在代码中保留了历史名称 `oracle_subgoal`。它不是部署时可用的
未来真值，而是从离线成功轨迹库中找到当前 latent 的最近邻，再取同一条轨迹前方
最多 25 步的 latent。发布文档使用“trajectory retrieval”以避免误解。

## 方法

### 原版

```text
TRANSFER -> final transfer_goal_latent -> Actor + MPC
```

### 置信度门控轨迹检索

```text
current latent
    -> nearest successful TRANSFER latent
    -> match_mse <= 0.0031 ?
         yes: use its future latent (up to 25 steps)
         no:  fall back to final transfer_goal_latent
```

`0.0031` 是 64 条离线成功轨迹之间“跨 episode 最近邻 MSE”的 99% 分位数，
没有使用 50 个测试 episode 的成败结果调参。

### 学习型 LatentSubgoalPlanner

输入：

```text
current latent (192)
+ final goal latent (192)
+ phase embedding
+ proprioception (6)
```

网络：

```text
Linear -> ReLU -> Linear -> predicted future latent (192)
```

训练使用 64 条固定任务成功轨迹。原始 `TRANSFER` 帧为 1712 个，删除 64 个
终点预测终点的零步样本后得到 1648 个样本；按完整 episode 拆分为 1315 个训练
样本和 333 个验证样本。验证 latent MSE 为 `0.00609`。

控制时采用双门控：当前状态必须匹配成功轨迹分布，同时预测出的子目标也必须接近
成功轨迹流形，否则回退到最终目标。虽然离线 MSE 很低，最终成功率仍只有 48%，
说明 latent MSE 不是子目标可执行性的充分条件。

## 关键代码

| 文件 | 作用 |
| --- | --- |
| `run_proprio_v2_experiment.py` | 原版、轨迹检索、门控、shadow 和学习控制入口 |
| `latent_subgoal_planner.py` | 单步 MLP LatentSubgoalPlanner |
| `train_latent_subgoal_planner.py` | 按 episode 拆分数据并训练 Planner |
| `proprio_v2_models.py` | SemanticNet、TransitionNet、TargetNet、Actor、Value Ensemble |
| `privileged_teacher.py` | 固定任务专家轨迹与离线标签 |
| `proprio_v2_config.yaml` | 模型、MPC、门控和训练配置 |
| `summarize_v0_1_results.py` | 汇总 5-seed 结果并计算配对检验 |

## 复现命令

以下命令从本目录执行。需要先准备冻结 LeWM 权重、OGBench 环境、Proprio V2
checkpoint 和固定任务轨迹数据；具体文件见 [DATA.md](DATA.md)。

训练 LatentSubgoalPlanner：

```bash
python train_latent_subgoal_planner.py
```

原版冻结 Actor 对照：

```bash
python run_proprio_v2_experiment.py \
  --stage adapt --freeze-actor --seed 42 \
  --run-name oracle_control_frozen_seed42
```

置信度门控轨迹检索：

```bash
python run_proprio_v2_experiment.py \
  --stage adapt --freeze-actor \
  --oracle-subgoal --oracle-confidence-gate \
  --oracle-lookahead-steps 25 --seed 42 \
  --run-name oracle_gated_h25_frozen_seed42
```

学习模型 shadow 评估：

```bash
python run_proprio_v2_experiment.py \
  --stage adapt --freeze-actor \
  --oracle-subgoal --oracle-confidence-gate \
  --latent-subgoal-shadow --seed 42 \
  --run-name subgoal_shadow_gated_seed42
```

学习模型双门控控制：

```bash
python run_proprio_v2_experiment.py \
  --stage adapt --freeze-actor \
  --latent-subgoal-control --oracle-lookahead-steps 25 \
  --seed 42 --run-name learned_subgoal_h25_gated_seed42
```

汇总 seed 42--46：

```bash
python summarize_v0_1_results.py \
  --output results/v0.1/summary.json
```

## 测试环境

- Python 3.10.20
- PyTorch 2.12.1 + CUDA 13.0
- NumPy 2.2.6
- MuJoCo 3.10.0
- ImageIO 2.37.3
- PyYAML 6.0.3

## 数据范围与局限

- 控制实验来自 MuJoCo 仿真，不代表真实机械臂结果。
- 任务使用固定物块起点与目标点；没有证明跨目标或跨环境泛化。
- 每种方法为 50 个 episode，足以作为工程基线，不足以支持广泛结论。
- MLP 只用 64 条固定任务成功轨迹。服务器上另有 1000 条 OGBench play
  episode，但没有用于 v0.1 Planner 训练；它们计划用于后续预训练。
- 检索方法依赖离线成功轨迹库，适合作为固定任务基线和学习方法上界，不是最终的
  通用层次规划方案。

## 路线图

- v0.2：使用 1000 条 OGBench episode 中已编码的 H=25 样本预训练；
- v0.3：加入动作条件和可达性约束，减少 MSE 回归造成的 latent 平均化；
- v1.0：多目标任务、更大规模评估以及真实机器人验证。

## 致谢与许可

本项目建立在 LeWorldModel、OGBench、MuJoCo 和相关开源工作之上。发布代码前应保留
上游版权声明，并分别检查预训练权重、原始数据和衍生数据的再分发条款。仓库现有
代码许可证为 MIT；数据与权重不会随源码仓库直接提交。
