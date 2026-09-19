# v0.1 数据与模型清单

源码仓库不直接提交原始数据、生成日志、全部视频或模型 checkpoint。发布时建议将
允许再分发的小型衍生数据和权重放入独立 GitHub Release，并在此记录 SHA-256。

## 运行所需文件

相对 `latent_three_phase_mpc/`：

| 文件 | 大小 | 用途 | 源码仓库 |
| --- | ---: | --- | --- |
| `outputs/proprio_v2/data/proprio_v2_task_continuous.npz` | 6.7 MB | 64 条固定任务成功轨迹 | 不提交；待 Release |
| `outputs/proprio_v2/checkpoints/proprio_v2_offline.pt` | 12 MB | Semantic/Transition/Target/Actor/Critic | 不提交；待 Release |
| `outputs/proprio_v2/checkpoints/latent_subgoal_planner_h25.pt` | 1.2 MB | 学习型 Planner | 不提交；待 Release |
| `outputs/data/cube_semantic_latents.npz` | 123 MB | 1000 条 OGBench episode 的 latent 缓存 | 可选；单独下载 |
| `../../model/lewm-cube-mapped/` | 较大 | 冻结 LeWM 权重 | 使用上游来源 |
| `../../data/ogbench_state/cube-single-play-v0.npz` | 245 MB | OGBench 状态/动作数据 | 使用 OGBench 来源 |

## v0.1 Release 候选文件校验值

```text
29d9091599600cb13988f0cb6003260365cbd42752c14e2e9b1ed5adfad68305  proprio_v2_task_continuous.npz
f49db5c3b789db7221e9da9805ff5b73d3b275c90ae9f25ae9d6f9dc607fdfc5  proprio_v2_offline.pt
5fca1eaa9e781164b9c4937ac1f8369859a4c9b89f6afff181b49102e549d1dd  latent_subgoal_planner_h25.pt
```

在确认 LeWM、OGBench 与衍生数据的再分发条款前，不上传这些二进制文件。

## 固定任务轨迹内容

`proprio_v2_task_continuous.npz` 包含 64 条成功 episode、3007 帧：

- 前一、当前、下一帧 192 维 latent；
- 最终 goal latent 和阶段 keyframe latent；
- 6 维 proprioception；
- 5 x 5 action block；
- ALIGN/GRASP/TRANSFER 状态；
- predicate、reward、done、episode id 和 success 标记。

LatentSubgoalPlanner 使用其中 1648 个非零步 `TRANSFER` 样本。标签为同一 episode
中最多前方 25 步的 latent；由于轨迹接近终点时会截断，实际平均前视距离约13.7步，
只有约10.6%的样本能取满25步。

## 服务器上的扩展数据

OGBench训练数据包含1000条episode和1,001,000步；验证集另有100条episode。
现有 `cube_semantic_latents.npz` 覆盖1000条episode：

- 64,000个训练样本；
- 177,331个已经编码的latent；
- 31,912个TRANSFER样本；
- 5,983个TRANSFER样本已经具有可直接取得的准确 `t+25` latent。

这些数据没有用于v0.1学习型Planner，留作后续版本的预训练数据。
