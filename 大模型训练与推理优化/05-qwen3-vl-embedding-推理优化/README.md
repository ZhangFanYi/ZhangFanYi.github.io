# 项目 05:Qwen3-VL-Embedding 推理优化(高并发 GEMM 定向优化)

## 概述

针对 Qwen3-VL-Embedding 在高并发场景下的性能劣化问题,通过定向优化 GEMM 算子解决。代码基于 Qwen 官方开源仓 [Qwen3-VL-Embedding](https://github.com/QwenLM/Qwen3-VL-Embedding)(Apache-2.0)进行适配与优化。

## 技术背景

- Qwen3-VL-Embedding 支持文本 / 图像多模态 embedding 与 rerank 场景
- 高并发下性能劣化通常源于:小 GEMM 算子过多、kernel 调度开销、显存带宽瓶颈

## 优化内容

- 定向优化 GEMM 算子,提升高并发吞吐(细节见代码 diff)
- 兼容 transformers 5.x 的 API 变更(`check_model_inputs` 适配)

## 代码

[code](./code/) 为完整源码(含个人优化改动):

```
code/
├── src/                  # 模型实现(models / evaluation)
├── examples/             # embedding / rerank / 多模态 RAG notebook
├── scripts/              # 训练与推理脚本
├── test/                 # 测试
├── all/                  # 压测与运行脚本(batch_test / bench.sh / run_test.sh)
└── README.md             # 上游官方说明
```

## 文档

暂无独立报告文档。
