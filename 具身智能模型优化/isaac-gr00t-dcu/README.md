# 项目 11: Isaac-GR00T-DCU 适配优化(具身智能 · 人形机器人基础模型)

## 概述

英伟达 **Isaac GR00T**(人形机器人基础模型,GR00T-N1)在海光 DCU 上的微调适配与优化,移植自 GitLab 仓 `ts-models-opt/training/embodied-ai/isaac-gr00t-dcu`。

## 技术要点

- 模型:GR00T-N1(如 GR00T-N1.7-3B-local-cosmos)
- 数据集:LIBERO 系列(libero_10_no_noops_1.0.0_lerobot_h264),embodiment-tag `LIBERO_PANDA`
- DCU 适配:依赖替换为 DCU 生态版本;模型与训练代码修改集中在 `patches/` 文件夹
- 基础镜像:`harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.9.0-ubuntu22.04-dtk26.04-py3.10`
- 微调命令:`bash examples/finetune.sh --base-model-path <ckpt> --dataset-path <data> --embodiment-tag LIBERO_PANDA ...`

## 代码

[code](./code/) 为完整适配仓:

```
code/
├── gr00t/           # GR00T 源码
├── patches/         # DCU 适配与优化 patch
├── uv.lock
└── README.md        # 海光适配版部署说明(镜像/模型/数据集)
```

## 文档

暂无独立报告文档,部署与适配细节见 [code/README.md](./code/README.md)。
