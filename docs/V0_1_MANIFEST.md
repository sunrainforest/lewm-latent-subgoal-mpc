# 历史清单：早期 v0.1

> 以下是旧版候选附件列表，不是当前冻结的无门控 H=25 GitHub 第一版发布清单。二进制附件未经再分发许可确认，不上传；当前范围见根目录 `README.md` 与 `docs/DATA.md`。

# v0.1 待发布清单

## 提交到源码仓库

- `README.md` 中的 v0.1 入口；
- `V0_1_RELEASE.md`、`DATA.md`；
- `latent_subgoal_planner.py`；
- `train_latent_subgoal_planner.py`；
- `run_proprio_v2_experiment.py`；
- `proprio_v2_config.yaml`；
- `summarize_v0_1_results.py`；
- `results/v0.1/README.md`；
- `results/v0.1/summary.json`；
- 根目录 `.gitignore` 更新。

## GitHub Release 候选附件

- `proprio_v2_task_continuous.npz`；
- `proprio_v2_offline.pt`；
- `latent_subgoal_planner_h25.pt`；
- 同一初始条件的原版失败视频：
  `outputs/proprio_v2/videos/oracle_control_frozen_seed42/proprio_v2_episode_0_failed.mp4`；
- 对应的门控轨迹检索成功视频：
  `outputs/proprio_v2/videos/oracle_gated_h25_frozen_seed42/proprio_v2_episode_0_success.mp4`；
- 可选诊断对照：门控检索在 seed 42 episode 5 成功，而学习型 Planner 失败。

## 不上传

- 原始 OGBench 数据；
- 原始 LeWM 权重；
- 123 MB latent cache（除非单独确认再分发与托管方式）；
- 全部逐步 JSONL 日志；
- 全部实验视频；
- `__pycache__`、临时文件和本地环境；
- 服务器绝对路径或账号信息。

## 发布前待确认

- GitHub账号和目标仓库名称；
- 新建仓库还是推送到现有仓库；
- 仓库公开或私有；
- 数据与checkpoint的再分发许可；
- Release附件是否包含示例视频。
