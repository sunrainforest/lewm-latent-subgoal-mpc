# 第二阶段：play 预训练初步实验

## 目的

检验“先用 OGBench play trajectory 学习 H=25 未来 latent，再用 64 条固定任务成功轨迹微调”能否改善单步 `LatentSubgoalPlanner`。本实验只评估离线 latent 预测，尚未接入控制成功率实验。

## 数据审计

`cube_semantic_latents.npz` 包含 64,000 个从 1000 条 play episode 抽取的样本。按当前 `proprio_v2` 规则重建 6 维 proprio 和阶段后：

- `ALIGN`：21,799；
- `GRASP`：10,289；
- `TRANSFER`：31,912；
- 严格满足同 episode、目标位于未来、目标/H=25 latent 已编码：5,962 个样本，覆盖 998 个 episode；
- 按 episode 拆分为 4,765/599/598 个训练/验证/测试样本，不按帧随机拆分。

固定任务数据仍为 64 条成功轨迹、1,648 个 `TRANSFER` 样本，按与旧模型完全相同的 episode split 拆成 1,315/333 个训练/验证样本。历史控制评估 seed 11100–11299 没有用于训练或选择 checkpoint。

一个明显的数据差异是：play 样本中 98.8% 能取满 25 步，平均前视 24.74 步；固定任务样本只有 10.6% 能取满 25 步，平均前视 13.67 步。两者的条件分布并不一致。

## 训练

- 网络保持 `Linear → ReLU → Linear`，输入仍为当前 latent、goal latent、phase embedding 和 6 维 proprio；
- play 预训练最多 150 epoch，以 play 验证集 early stopping；最佳 epoch 103；
- 固定任务微调最多 200 epoch，以任务验证集 early stopping；最佳 epoch 199；
- normalization 只由 play 训练 episode 计算，避免验证/测试泄漏；
- checkpoint 与第一版分开保存，没有覆盖冻结的检索版本或旧任务专用 Planner。

## 结果

play 测试集上：

| 方法 | latent MSE |
| --- | ---: |
| 直接使用当前 latent | 0.52514 |
| 直接使用 goal latent | 0.39080 |
| play 预训练模型 | **0.15731** |

说明模型确实学到了 play 数据本身的时序规律。但它对固定任务验证集的 zero-shot MSE 为 1.30924，存在明显分布差异。

在完全相同的 13 条固定任务验证轨迹、333 个样本上：

| 模型 | latent MSE | 样本 MSE 中位数 |
| --- | ---: | ---: |
| 只用 64 条成功轨迹训练的旧模型 | **0.00609** | **0.00240** |
| play 预训练后再用成功轨迹微调 | 0.01002 | 0.00519 |

预训练模型只在 13.2% 的验证样本上优于旧模型。因此这一版直接预训练**没有改善任务内 latent 预测，反而退化**；不进入控制成功率评估。

## 结论与下一步

1000 条 play trajectory 不是无用，而是不适合直接当成当前结构的“成功子目标监督”。主要风险包括：成功/失败行为混合、前视长度分布不同，以及没有动作输入时同一视觉状态的未来具有多模态性，MSE 回归容易学习平均 latent。

下一步更合理的方向是二选一：

1. 把 play 数据用于**动作条件的 latent reachability/dynamics 预训练**，而不是直接训练无动作的子目标输出；
2. 在新的固定任务训练 seed 上运行冻结的 H=25 检索器作为 teacher，保存当前状态与检索子目标，训练 student 模仿 teacher，再用未见 seed 做控制评估。

第二项与当前已验证有正向信号的轨迹检索器更一致，优先级更高。
