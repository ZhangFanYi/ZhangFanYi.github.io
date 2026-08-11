# 项目 13: LingBot-VA-DCU 适配优化(具身智能 · 因果世界模型)

## 概述

Robbyant 团队 **LingBot-VA**(Causal World Modeling for Robot Control,因果世界模型)在海光 DCU 上的部署适配与优化,移植自 GitLab 仓 `ts-models-opt/training/embodied-ai/lingbot-va-dcu`。

## 技术要点

- 模型:[LingBot-VA](https://technology.robbyant.com/lingbot-va)(arXiv:2601.21998),视频动作(VA)世界模型
- DCU 适配:`patches/` 内含 DCU 适配 patch;依赖(flash-attn 等)按海光生态替换
- 提供 `Makefile` / `script` / `evaluation` / `example` 等完整工程结构

## 代码

[code](./code/) 为完整适配仓:

```
code/
├── wan_va/          # 模型实现
├── patches/         # DCU 适配 patch
├── evaluation/  example/  script/
├── pyproject.toml   # 依赖定义(含 flash-attn 安装说明)
└── officical-README.md / INSTALL.md
```

## 文档

暂无独立报告文档。
