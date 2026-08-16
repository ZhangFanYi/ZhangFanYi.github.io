# Qwen3-30B 训练优化(Megatron-SFT 框架)

## 概述

针对某客户 Qwen3-30B-A3B(Megatron-SFT 框架,TP4×EP4×PP2)训练性能不足的问题,给出 Megatron 框架下的 prof 抓取代码,分析 trace 定位到**大量小 GEMM 算子**,通过开启 grouped GEMM 融合使**训练性能翻倍**。

## 背景

- 客户环境:DTK 2504.2 + Python 3.10 + torch 2.5
- 性能表现:BW1000 5 次迭代约 7 分钟,友商(华为 910B)约 3.5 分钟,差距一倍

## 方法论:Megatron 框架 Prof 抓取

由于团队此前未在该框架下抓取过 prof,参考官方文档对 NPU 的抓取方式:

- 在 `base.py` 的 `train()` 函数中插入 `torch.profiler.profile` 代码块
- 第 10 步时触发,仅抓 rank 0;`activities=[CPU, CUDA]`(DCU 使用 CUDA 枚举)
- `record_shapes=True` + `with_stack=True`,输出按 `self_cuda_time_total` 排序的 Top 30 算子表,并导出 chrome trace(`megatron_prof.json`)
- 其余步骤正常训练,不干扰性能

## Prof 分析与定位

- 算子表 Top 耗时全部是通信算子(AllReduce / AlltoAll / AllGather / ReduceScatter) 但通信专项测试无问题
- 进一步分析 trace 发现:**海光 trace 中出现大量小 GEMM 算子**(对应矩阵乘栈)

## 优化

- 开启环境变量 `export NVTE_USE_HIPBLASLT_GROUPEDGEMM=1`,将小 GEMM 批量分组融合执行
- **结果:训练性能翻倍**,与友商持平/超越

## 文档

- [prof分析-基于megatron-sft框架.pdf](./prof分析-基于megatron-sft框架.pdf)(原始报告,含 prof 插桩代码)
