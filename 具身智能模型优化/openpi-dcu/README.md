# 项目 16: OpenPI-DCU 训练优化(π0 动作策略)

## 概述

客户基于 **DeepSpeed-Z2** 训练 **OpenPI(π0 动作策略)**。一开始在单卡上跑,显存占用 98% 难以叠加优化手段,因此先以单机 8 卡作为 baseline;客户最终要求必须单卡训练,据此调整优化思路,最终落地注意力 FA2 改造 + 算子融合。

## 优化记录

### 1. 单机 8 卡 baseline 与融合优化

| 配置 | 稳态迭代耗时 |
| --- | --- |
| baseline | 2.01 s/it |
| + torch.compile(max-autotune-no-cudagraphs) | 1.87 s/it |
| + 关闭 grad_ckpt | 1.36 s/it |

- loss 收敛正常,训练时显存 88%-89%
- **开启 triton_fusion 会 loss NaN**
- 当前脚本 `scripts/tc_cloud/train_tc.sh` 已合入最优配置,原始脚本保留为 `train_tc.sh-bak`
- offload 选项(`config/zero2_offload.json`)显存占用 90%,未启用

### 2. 单卡训练:注意力 eager → FA2

单卡关闭 grad_ckpt 会 OOM,offload 也无济于事,于是修改注意力实现路径。

**FA 底层逻辑要点**(详见 Qwen3.5-0.8B 训练优化 1.3.1 部分):

- FA 加速基于分块 + 融合,降低的是显存读写量而非计算量
- 原生支持 GQA(`8:1` 的 Q/KV 比例,不复制 KV)
- 支持 causal 与 bidirectional
- FA2 base 版本基本不能吃复杂 mask:`padding mask + varlen 打包` 处理有效/无效位

主要改动:

1. **接口设置**:启动脚本传参指定 `pi0_config.attention_implementation=fa2`,在 `paligemma_with_expert_pi05.py` 中把 `_attn_implementation` 设为 `flash_attention_2` 并强制 `is_causal=False`(OpenPI 前缀/后缀块内均为双向)
2. **4D additive mask → 2D bool padding mask**(`pi05/paligemma_with_expert_pi05.py`):

   ```python
   attention_mask = attention_mask[:, 0].eq(0).any(dim=-2)
   ```

   拆掉冗余 head 维、加性 → bool、在 Sq 维折叠成 `[B, Sk]`。因为 `split_forward` 模式下一次只跑 prefix 或 suffix,块内双向,掩码只需表达"哪些列是 padding",2D 掩码足够
3. **prefix KV cache dtype 对齐**:FA2 对 dtype 一致性要求严格,在梯度检查点重算前把 KV cache 转成当前输入 dtype

**收益**(prefix/suffix checkpoint attention 耗时):

| 场景 | Math | FA2 |
| --- | --- | --- |
| prefix attention | 20.962 ms | 7.330 ms |
| suffix attention | 4.098 ms | 2.101 ms |

收益来源:拆 5 个算子(QK^T / +mask / scale / softmax / PV)→ 1 个融合算子;GQA 从复制 KV → 原生 8:1 计算。

注:同事(patch_hf_gemma_sdpa)的 `fused_sdpa.py` 方案与上述方案**不能叠加**——其替换的是 `sdpa`/`eager` 后端,而 2 的方案已指定 `flash_attention_2` 后端;且本质上 varlen 打包计算逻辑一致。

### 3. 算子融合(hcu)

```bash
export HA_USE_RMSNORM=1
export HA_USE_ROPE=1
export HA_USE_FUSED_GELU_MUL=1
```

- **RMSNorm 融合**:原版 square→mean→add ε→rsqrt→mul 一串算子,每步读写 HBM;融合为单 kernel,26 层 × 两个 Gemma 大量省 HBM 读写
- **RoPE 融合**:Q/K 旋转的 cos/sin 构造 + rotate_half + 两次 mul + cat 融合为单 kernel
- **Fused GELU×Mul**:`gelu_tanh(gate) * up` 融合为单 kernel(B×Sq×intermediate 的中间激活读写是大头)

| 指标 | BW1000 | BW1000 优化后 | A800 |
| --- | --- | --- | --- |
| 单次迭代耗时 | 2.07 s/it | 1.53 s/it | 1.57 s/it |

## 文档

- [openpi.pdf](./openpi.pdf)(原始报告)