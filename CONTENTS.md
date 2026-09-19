# 项目内容

当前冻结的第一版为**仅在 TRANSFER 启用、无置信度门控、H=25 的成功轨迹检索 + MPC**。主结果和限制见 [`README.md`](README.md)，复现步骤见 [`docs/REPRODUCE.md`](docs/REPRODUCE.md)，权重/数据边界见 [`docs/DATA.md`](docs/DATA.md)。

- `source/`：从 Cube 实验复制出的实现副本；`run_proprio_v2_experiment.py` 与 `proprio_v2_config.yaml` 为评估入口和配置。
- `experiments/run_frozen_ungated_v1.sh`：顺序运行 Baseline 与无门控 H=25；不覆盖已有结果。
- `experiments/summarize_frozen_ungated_v1.py`：校验配置并汇总 `proprio_v2` latent 完成率。
- `results/frozen_ungated_v1/`：两批各 100 回合的可公开统计摘要，无个人路径或二进制资产。
- `results/v0.1/`、`docs/V0_1_RELEASE.md`：早期 50 回合与门控/学习型 MLP 探索记录，**不是当前冻结版的主结果**。
- `runtime/`、`release_assets/`、`vendor_reference/`：仅服务器本地，Git 忽略；不得误认为仓库自带数据和权重。

所有整理与实验均在独立副本进行，不改原版 LeWM_SAIMO。
