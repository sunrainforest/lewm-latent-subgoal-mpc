# 数据与权重清单（冻结无门控 H=25 版）

Git 历史**不包含**以下二进制资产。下表给出评估用文件的仓库相对路径与 SHA-256，供有合法访问权的复现实验者核对。未经确认各自许可之前，不把权重、原始数据或衍生轨迹上传到 GitHub。

| 文件 | 用途 | SHA-256 |
| --- | --- | --- |
| `runtime/model/lewm-cube-mapped/config.json` | 冻结 LeWM 配置 | `4d446944fe28922cc2c5763f43d4ef9132a457bd89e9a0ce5dbceac183994999` |
| `runtime/model/lewm-cube-mapped/weights.pt` | 冻结 LeWM 权重 | `d17e5bba8c53d019492f303282f9355821a21635de05f8efcfcfb1239a98452f` |
| `runtime/outputs/proprio_v2/checkpoints/proprio_v2_offline.pt` | SemanticNet、TransitionNet、TargetNet、Actor、Value Ensemble 与动作统计 | `f49db5c3b789db7221e9da9805ff5b73d3b275c90ae9f25ae9d6f9dc607fdfc5` |
| `runtime/outputs/proprio_v2/data/proprio_v2_task_continuous.npz` | 64 条固定任务成功轨迹；检索库 | `29d9091599600cb13988f0cb6003260365cbd42752c14e2e9b1ed5adfad68305` |

检索库约 6.7 MB，含 64 条成功 episode、3007 帧及每帧 latent/状态/episode id；在线检索只保留成功样本。`source/run_proprio_v2_experiment.py` 在当前 latent 上按 MSE 寻找同阶段最近邻，再取同一 episode 中最多 25 步后的 latent；只在 `TRANSFER` 启用。没有置信度阈值回退。

原始环境训练数据 `runtime/data/ogbench_state/cube-single-play-v0.npz`（SHA-256 `80f3b6fd27f4f9d9e9eb6f0d07d6951559012f45b1e15ea4046ef8ecd8d3684e`）可用于重建离线训练产物，但已冻结的 `--stage adapt` 对照直接加载上表 checkpoint 与检索库；它不是这两批评估启动时的必需输入。`release_assets/` 中有本地候选副本，但该目录被 `.gitignore` 排除，不能视为已发布下载地址。

如需从头训练，可在满足上游依赖及原始数据授权后研究 `--stage train`，该流程会重新采集成功轨迹并生成 checkpoint；由于采集和训练可能存在数值差异，不保证得到上表完全相同的 SHA-256。先前学习型 `latent_subgoal_planner_h25.pt` 不用于此冻结方法。
