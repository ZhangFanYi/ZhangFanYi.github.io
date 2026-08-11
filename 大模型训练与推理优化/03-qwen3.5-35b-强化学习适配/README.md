# 项目 03:Qwen3.5-35B 强化学习训练适配(verl-das / verl-Megatron-GRPO)

## 概述

基于 **verl-das**(HCU 生态的 verl + Megatron-LM 强化学习训练套件)完成 Qwen3.5-35B-A3B 的 GRPO 强化学习训练适配,支持 FSDP2 与 Megatron 两种训练后端。

## 技术要点

- 套件基于 verl `release/v0.8.0` 与 Megatron-LM 的 HCU 适配版本,HCU 生态定制化修改均由海光信息完成
- 支持特性矩阵:
  - 训练算法:GRPO / fully_async_policy / one_step_off_policy / on_policy_distillation / multi_on_policy_distillation
  - 模型家族:Qwen2.5 系列、Qwen3 系列、Qwen3-VL、Qwen3.5-9B/27B/35B-A3B、DeepSeek-7B、Moonlight-16B-A3B、Seed-OSS-36B-Base 等
  - 推理引擎:vLLM / sglang
  - 训练后端:FSDP / FSDP2 / Megatron / VeOmni

## 代码

[code](./code/) 为 verl-das 的自定义部分(HCU 适配层、示例与文档):

```
code/
├── hcu_verl/        # HCU 生态适配层(adaptor / core / trainer / workers ...)
├── examples/        # grpo / sft / fully_async_policy / on_policy_distillation / one_step_off_policy 等示例
├── docs/            # 用户指南
├── README.md        # 上游说明(原仓 HYGON-AI/verl-das,Apache-2.0)
└── requirements.txt
```

> 注:`third_party/`(上游 verl / Megatron-LM / VeOmni 子模块)为纯上游开源代码,未包含在本仓库中,clone 后按 `.gitmodules` 拉取即可。
