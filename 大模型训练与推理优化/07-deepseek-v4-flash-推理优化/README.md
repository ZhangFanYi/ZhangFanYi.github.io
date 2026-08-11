# 项目 07:DeepSeek-V4-Flash 推理优化(Flash-FP8 性能瓶颈排查)

## 概述

针对 DeepSeek-V4(Flash-FP8)推理性能短板,在自有芯片(NMZ1101)与 H20 上分别抓取 prof trace,通过 SQL 聚合生成算子级性能对比表,定位出**量化 GEMM、MLA/FlashAttention、MoE 融合**三大瓶颈。

## 方法:算子级 Prof 对比

对 trace 中的 slice 表执行聚合查询,生成算子级耗时表:

```sql
SELECT name, COUNT(*) AS record_count, SUM(dur)/1000000.0 AS total_duration_ms,
       SUM(dur)/COUNT(*)/1000000.0 AS avg_duration_ms
FROM slice
WHERE category IN ('kernel','gpu_memcpy','gpu_memset')
GROUP BY name ORDER BY total_duration_ms DESC;
```

## 三大瓶颈分析

### 1. 量化 GEMM 算子

| 平台 | 算子 | 平均耗时 | 调用次数 | 总耗时 |
| --- | --- | --- | --- | --- |
| H20 | per_token_group_quant_8bit_kernel | 0.0014ms | 9009 | ≈12.5ms |
| NMZ1101 | _per_token_quant_int8 | 0.0118ms | 3481 | ≈41.1ms |

NMZ1101 单次量化算子耗时高,H20 虽调用次数更多但总耗时反而更短。

### 2. MLA / Flash Attention

| 算子 | H20 平均耗时 | NMZ1101 平均耗时 | 差距 |
| --- | --- | --- | --- |
| sparse_fp8 flash decode kernel | 0.033ms | 0.098ms | 慢 3 倍 |
| flash_fwd_mla_combine_kernel | 0.0027ms | 0.0796ms | 慢 29 倍 |
| paged_mqa_logits_fp8 | 无耗时 | 0.483ms(调用 1701 次) | 显著 |

### 3. MoE

- `moe_align_block_size`:H20 总耗时 13.3ms vs NMZ1101 29.3ms(平均 0.004ms vs 0.008ms)
- MoE 实现策略不同:H20 用大融合算子 `fused_moe_kernel`(平均 0.102ms,总耗时 708ms,H20 最大算子);DCU 侧大量带 marlin 名的**小算子**
- Elementwise kernel 也慢 2 倍以上

## 结论

性能瓶颈集中在:量化 GEMM、MLA/FlashAttention 算子、MoE 融合粒度。优化方向:算子融合(减少小 kernel 数量)、MLA 相关 kernel 优化、MoE 大融合 kernel 替换。

## 代码

[code](./code/):prof 抓取与压测脚本

```
code/
├── begin.sh          # 通过 HTTP API 触发服务端 profile(指定输出目录/起止步数)
├── client_prof.sh    # prof 场景压测(基于 sglang bench_serving)
├── pro5000_client.sh # 5000 并发压测
├── tencent_client.sh # 腾讯场景压测
└── dockerstart.sh    # 容器启动
```

## 文档

- [Deepseek V4 性能瓶颈排查.pdf](./Deepseek V4 性能瓶颈排查.pdf)(原始报告)
- [dpsk-V4 nmz1101 & H20 内部测试.xlsx](./dpsk-V4 nmz1101 & H20 内部测试.xlsx)(详细数据)
