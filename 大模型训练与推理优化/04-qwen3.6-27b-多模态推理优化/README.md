# Qwen3.6-27B 多模态推理优化(VLLM 缓存命中 + 量化模型选型)

## 概述

针对客户 Qwen3.6-27B 多模态(视频)推理场景,通过**多模态缓存命中**与**更换正确的量化模型**两项优化,把推理吞吐从客户实测 0.078 samples/s 提升至 0.31 samples/s,达 A800 性能的 133%~335%。

## 关键问题与解法

### 1. 镜像与启动

- 基于vllm0.18和0.21不同镜像进行推理 
- 启动关键环境变量:`VLLM_HCU_USE_FLASH_ATTN=1`、`VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER=1`
- 量化方式 `slimquant_marlin`,开启 MTP 投机解码(`--speculative-config.method mtp`,3 个投机 token)


### 2. 性能瓶颈:量化版本选错

- 客户使用 NVIDIA 卡导出的量化模型, 原始模型是 FP8
- **换用沐曦官方 W8A8 量化版本后性能正常**(modelscope: metax-tech/Qwen3.6-27B-W8A8)

## 结果

| 配置 | 16 样本性能 | 全量性能 |
| --- | --- | --- |
| vLLM 0.18.1 | 0.6837 samples/s(148.5%) | 0.6141 samples/s(133.4%) |
| vLLM 0.21.0 | 1.549 samples/s(335.3%) | 1.3605 samples/s(295.6%) |
| A800 参考 | 0.4602 | — |

> 括号内为对 A800 的性能比

## 代码

[code](./code/):压测脚本与启动脚本

```
code/
├── vlm_speed_benchmark.py   # 多模态速度基准(视频/图片压测,worker 并发)
├── 1server.sh               # 服务端启动
└── client.sh                # 客户端压测
```

## 文档

- [Qwen3.6-27B多模态推理.pdf](./Qwen3.6-27B多模态推理.pdf)(原始报告)
