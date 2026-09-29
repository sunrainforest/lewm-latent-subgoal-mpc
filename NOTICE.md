# 来源与第三方许可说明

## 项目关系

本仓库 `lewm-latent-subgoal-mpc` 不是 LeWorldModel（LeWM）官方仓库。
它是从 **LeWM_SAIMO** 的 Cube / Proprio V2 实验代码中抽取、整理并继续
改造的下游研究项目，重点研究 `TRANSFER` 阶段的 latent 子目标与 MPC。

LeWM_SAIMO 本身基于 LeWorldModel 上游工作：

- 论文：[LeWorldModel: Stable End-to-End Joint-Embedding Predictive
  Architecture from Pixels](https://arxiv.org/abs/2603.19312)
- 项目主页：<https://le-wm.github.io/>
- 预训练模型与数据：<https://huggingface.co/collections/quentinll/lewm>

本仓库保留 LeWM encoder、dynamics 以及 Cube 实验的若干接口，并在其上
整理或增加了三阶段状态机、轨迹检索、latent 子目标、评估与复现脚本。
具体实验范围及限制以 [`README.md`](README.md) 为准。

## 代码许可证

作为本仓库直接代码来源的 LeWM_SAIMO 快照随附 MIT License：

```text
Copyright (c) 2026 Lucas Maes
```

该版权声明和完整许可正文保留在 [`LICENSE`](LICENSE)。本仓库中从
LeWM_SAIMO 复制并修改的代码，以及本仓库新增且未另行标注的代码与文档，
均按该 MIT License 分发。保留许可证不代表 LeWM 或 LeWM_SAIMO 的原作者
认可、维护或背书本下游项目。

## 未随仓库分发的内容

本仓库不包含 LeWM 预训练权重、OGBench 原始数据、64 条成功轨迹、训练
checkpoint、MuJoCo 资产或其他第三方二进制文件。它们分别受各自许可证或
使用条款约束；获得和使用这些资产时，应遵循其原始发布方的规定。
