# 冻结版复现指南

## 1. 环境与代码

已验证环境：Linux、CUDA GPU、Python 3.10.20、PyTorch 2.12.1+cu130、TorchVision 0.27.1+cu130、NumPy 2.2.6、MuJoCo 3.10.0、OGBench 1.2.1、stable-pretraining 0.1.7、stable-worldmodel 0.1.1、Gymnasium 1.3.0、ImageIO 2.37.3、imageio-ffmpeg 0.6.0、PyYAML 6.0.3。版本清单另见 [`requirements-frozen-v1.txt`](../requirements-frozen-v1.txt)；CUDA 对应的 PyTorch wheel 需按平台选择。上游 LeWM 和 OGBench 的安装、权重获取仍需遵守其各自说明与许可。

评估所用 `source/run_proprio_v2_experiment.py` SHA-256：
`4646298736a24a425a79bb92a8b04523ed05a85028b4b2a0577424cbb7ea9ed1`。
评估所用 `source/proprio_v2_config.yaml` SHA-256：
`3344a21e6a772571086a5347705c0585816918d86d0834c725226c2733a8e958`。
两批评估运行时的仓库 commit 为 `51c84155a9b4e19e20385b9a0237f216cff37ced`；此后新增的文档与汇总脚本不改动这两个评估代码文件。

## 2. 放置资产

从仓库根目录看，需要：

```text
runtime/model/lewm-cube-mapped/{config.json,weights.pt}
runtime/outputs/proprio_v2/checkpoints/proprio_v2_offline.pt
runtime/outputs/proprio_v2/data/proprio_v2_task_continuous.npz
```

文件来源、校验值和再分发限制见 [DATA.md](DATA.md)。`--stage adapt` 使用 checkpoint 内的动作归一化统计量，不重新训练模型。当前服务器已具备这些文件；公开仓库暂不提供下载链接，所以外部读者需要先自行取得或按上游代码生成合法副本。缺少二进制资产时，只能复核脚本和已归档的统计摘要，不能声称端到端数值复现。

## 3. 顺序运行两组

从仓库根目录执行，`PYTHON_BIN` 指向已配置 CUDA、OGBench、LeWM 依赖的 Python：

```bash
PYTHON_BIN=/path/to/python bash experiments/run_frozen_ungated_v1.sh 100 100
PYTHON_BIN=/path/to/python bash experiments/run_frozen_ungated_v1.sh 200 100
```

第一条评估回合 seed 11100–11199；第二条为 11200–11299。每批先运行 Baseline，再运行仅 `TRANSFER` 检索、无门控、H=25 的方法。两组均 `--stage adapt --freeze-actor --episodes 100 --max-steps 300 --independent-episode-seeds`。脚本检查必要资产与依赖，并在已有输出时拒绝覆盖。所有运行生成物写入 `runtime/outputs/proprio_v2/`，不进入 Git 历史。

若只想复核已有运行的统计结果，可直接使用汇总器，例如新 seed 批次：

```bash
python experiments/summarize_frozen_ungated_v1.py \
  --baseline runtime/outputs/proprio_v2/heldout_v0_5_baseline_seed200_299_summary.json \
  --ungated runtime/outputs/proprio_v2/heldout_v0_5_ungated_h25_seed200_299_summary.json \
  --start-seed 200 --episodes 100 \
  --output-json runtime/outputs/proprio_v2/rechecked_v0_5.json \
  --output-markdown runtime/outputs/proprio_v2/rechecked_v0_5.md
```

汇总器先检查 seed 序列、100 回合、零 Actor 更新、相同 checkpoint 路径、方法开关和 H=25，再计算完成率、配对计数、20,000 次配对百分位 bootstrap 区间（随机种子 20260919）及双侧精确 McNemar 检验。

## 4. 结果定义与限制

主指标为 `system_success`：在 `TRANSFER` 中，执行后 latent 相对最终 transfer-goal latent 的 MSE ≤ 0.017 且连续 2 帧满足，FSM 才设置 `machine.complete`。不是 OGBench 的物理方块位置成功率。成功轨迹库由 64 条离线固定任务成功轨迹构成；在线推理只检索该库，不读取本次测试回合的未来观测。

种子配对确保初始随机数设定一致，但这是**两次完整仿真**，不是从同一个保存的 `TRANSFER` 状态分叉。渲染/模型数值差异可使轨迹偏离，因此配对检验应连同这一限制阅读。当前结果证明“观察到两批正向差异”，不证明在任意新任务或硬件上有稳定收益。
