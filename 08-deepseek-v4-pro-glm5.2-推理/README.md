# 项目 08:DeepSeek-V4-Pro / GLM5.2 推理部署(多机 IFB / PD 分离 + 缓存命中测试)

## 概述

完成 DeepSeek-V4-Pro 与 GLM5.2 的多机推理部署与性能测试,方案要点:**多机 IFB(Incremental Batching)/ PD 分离部署,指定缓存命中率测试**。

## 技术要点

- **多机 IFB**:多机部署下使用连续批处理,提升吞吐与 GPU 利用率
- **PD 分离**:Prefill / Decode 分阶段独立部署,避免长 prefill 阻塞 decode,提升整体时延稳定性
- **指定缓存命中率测试**:控制 KV 缓存命中率进行基准测试,评估不同命中率下的性能表现

## 数据

- [glm5.2.xlsx](./glm5.2.xlsx):GLM5.2 测试数据
- [sglang_benchmark_results_20260805_104615.xlsx](./sglang_benchmark_results_20260805_104615.xlsx):SGLang 基准测试结果(2026-08-05)

## 代码

暂无独立代码仓。
