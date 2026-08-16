# Qwen3-VL-Embedding 推理优化

## 概述

针对 Qwen3-VL-Embedding 在高并发场景下的性能劣化问题,通过定向优化 GEMM 算子解决。代码基于 Qwen 官方开源仓 [Qwen3-VL-Embedding](https://github.com/QwenLM/Qwen3-VL-Embedding)(Apache-2.0)进行适配与优化。

## 技术背景

- Qwen3-VL-Embedding 支持文本 / 图像多模态 embedding 与 rerank 场景
- 高并发下性能劣化通常源于gemm的性能劣化

## 优化内容
- 设置channel last修改算子布局
- 定向优化 GEMM 算子 提升高并发吞吐

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

对于bs>32时 性能劣化的情况 是典型的gemm的问题 所有gemm全部调优很费时间 可以先用dtk自带的rocblas-bench测试一下耗时和算力 找到性能最差的几个进行定向调优
qwen-vl出现的该问题在具身智能GR00T中也有出现 解决方法一样
