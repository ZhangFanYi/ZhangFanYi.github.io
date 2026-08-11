# 项目 12: LeRobot-ACT-DCU 适配优化(具身智能 · 动作策略 ACT)

## 概述

LeRobot **ACT**(Action Chunking with Transformers)动作策略在海光 DCU 上的训练适配与优化,移植自 GitLab 仓 `ts-models-opt/training/embodied-ai/lerobot-act-dcu`。

## 技术要点

- 框架:LeRobot 上游源码(基线提交 `e40b58a8`),ACT 动作策略训练
- DCU 适配:`patches/` 内含 ACT on Hygon DCU 的优化 patch,镜像构建时通过 `ENABLE_OPTIMIZATION=1` 开关启用
- 目录结构参照 lingbot-vla-dcu 的交付方式

## 代码

[code](./code/) 为完整适配仓:

```
code/
├── lerobot/         # 上游 LeRobot 源码(基线 e40b58a8)
├── patches/         # ACT on Hygon DCU 优化 patch
├── dockerfile       # 镜像构建(可开关优化)
├── datadown.sh      # 数据下载
├── start_act.sh     # 训练启动
└── README.md        # 海光适配版部署说明
```

## 文档

暂无独立报告文档,部署与适配细节见 [code/README.md](./code/README.md)。
