# DeepSeek-V3 671B 千卡集群训练 & 线性度验证

## 概述

在合肥训练场 192 节点集群上,基于 Megatron 框架完成 **DeepSeek-V3 1.1T(万亿参数量)模型** 的千卡规模训练,并验证集群扩展线性度。

## 背景

- 客户要求:万亿参数量模型训练,128 节点与 256 节点两次训练,计算集群线性度,要求 **> 0.8**
- 线性度公式:`(T_128 / T_256) / 2`

## 方案

| 配置项 | 128 节点 | 256 节点 |
| --- | --- | --- |
| Global Batch Size | 2048 | 4096 |
| 单卡 micro-batch | 相同 | 相同 |
| 模型规模 | DeepSeek-V3 扩展至 1.1T 参数 | 同左 |

- 模型扩展方式:`num-layers 61 → 64`,`num-experts 256 → 384`,训练日志中参数量显示为 1049B
- 环境:conda 独立环境 + DTK(驱动版本限制 ≤ 25.04)+ 多机任务前先跑 rccltest 验证网络
- 训练框架:`dcu_megatron / examples/deepseek_v3`

## 结果

- 128 节点 / 256 节点各训练 3 小时,观察 loss 收敛并记录单次迭代耗时
- 256 集群 GBS 为 128 集群的一倍,单卡 mbs 一致(算力翻倍)
- 按前 40 步总耗时计算,**线性度 97.1%**,远超客户 >80% 的验收线

<img width="912" height="491" alt="image" src="https://github.com/user-attachments/assets/c6b4e1ca-8bfd-4327-a1c7-8eec8b078e4e" />

## 代码

[code](./code/) 为 DCU 版 Megatron 训练框架(`dcu_megatron`):

```
code/
├── Megatron-LM/    # Megatron-LM(DCU 适配版)
├── Megatron-Energon/  # Megatron-Energon(子模块,DCU 适配版)
├── examples/       # 训练示例(deepseek_v3 / GLM / llama / qwen 等)
└── README.md
```

DeepSeek-V3 相关脚本见 `code/examples/deepseek_v3/`。
