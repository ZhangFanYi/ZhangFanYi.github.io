# 项目 10: DreamZero-DCU 适配优化(具身智能 · 视频生成 RL)

## 概述

面向具身智能的视频生成强化学习模型 **DreamZero**(Wan2.1-I2V-14B 基座)在海光 DCU 上的部署适配与优化,移植自 GitLab 仓 `ts-models-opt/training/embodied-ai/dreamzero-dcu`。

## 技术要点

- 模型:[DreamZero](https://github.com/dreamzero0/dreamzero)(Wan2.1-I2V-14B-480P 视频生成)+ google/umt5-xxl 文本编码器
- 数据集:DreamZero-DROID-Data(具身操作轨迹数据)
- DCU 适配:删除 NVIDIA 相关依赖,替换为 DCU 生态包(pypi.sourcefind.cn 对应版本)
- 视频解码组件 torchcodec 按海光文档单独适配
- 基础镜像:`harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.9.0-ubuntu22.04-dtk26.04-py3.10`

## 代码

[code](./code/) 为完整适配仓:

```
code/
├── groot/            # 模型与训练代码
├── eval_utils/       # 推理/评估
├── patches/          # DCU 优化 patch
├── docs/             # 文档
├── env.sh            # 环境配置
├── run.sh            # 启动脚本
└── README.md         # 海光适配版部署说明(镜像/模型/数据集下载)
```

## 文档

暂无独立报告文档,部署与适配细节见 [code/README.md](./code/README.md)。
