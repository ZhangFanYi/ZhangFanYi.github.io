# Qwen3.5-0.8B 训练优化(DeepSpeed ZeRO-0 + Flash Linear Attention)

## 概述

针对腾讯云客户的多模态小模型训练需求(Qwen3.5-0.8B VLM,seq_len=1423,图片 1000×720),在 **accelerate + DeepSpeed** 框架上通过三项优化大幅提升训练吞吐,远超客户预期 10000+ Tokens/sec。

## 三项优化

### 1. ZeRO-1 → ZeRO-0

- 客户原配置 ZeRO-1(切分优化器状态) 每张卡需在优化器更新时做额外 All-Gather 带来不必要的通信开销
- 模型 0.8B 单卡可完全放下,切分没有收益,**反推 ZeRO-0(什么都不切,所有卡保留完整参数)** 反而最优
- ZeRO-0 价值:自动计算全局 batch size、梯度裁剪、优化器融合、学习率调度集成,适合小模型高效训练
- 进一步优化点:torch 原生 AdamW 在 CPU 上更新,可换 `fused_AdamW`(DCU 上完成),但 trace 显示优化器占比不大,收益有限

### 2. 增大 Global Batch Size

`gbs = mbs × gradient_accumulation_steps × num_gpus`,增大后 FusedAdam、梯度裁剪、参数 copy 等操作从每个 micro-step 一次降为多个 micro-step 一次,显著降低参数更新频率开销。

### 3. 引入 Flash Linear Attention(FLA)

- 模型结构:标准 Gated Attention 与 **Gated DeltaNet 线性注意力** 层比例 3:1
- Gated Attention 用 FlashAttention 加速(Tiling 分块 + Online Softmax,IO 感知,最小化 HBM 读写)
- DeltaNet 是线性注意力的"升级版":传统线性注意力静态累积记忆(`状态 = 状态 + 新信息`),DeltaNet 根据预测误差动态修正记忆(`状态 = 状态 - 旧预测 + 新目标`)
- FLA 是线性注意力领域的 FA:分块并行 + 融合算子;TFLA(Tiled FLA)在块内再平铺一层,实现任意大分块、更高算术强度
- 实现:直接 `pip install` 社区版 fla,transformers 的 modeling_qwen3_5.py 中自动启用 `chunk_gated_delta_rule`

## 结果

- 三项优化叠加后吞吐大幅提升,满足并超出客户 10000+ Tokens/sec 预期

## 代码

- [code/demo_code](./code/demo_code/):1 条数据验证 Qwen3.5-0.8B VLM SFT 训练流程的最小 demo(含 ds_config ZeRO-0 配置)
- [code/train.py](./code/train.py):训练主脚本(accelerate + DeepSpeed 封装)

