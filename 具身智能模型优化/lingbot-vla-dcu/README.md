# LingBot-VLA-DCU 适配优化

## 概述

**LingBot-VLA 4B**(pi0.5 架构,Vision-Language-Action 模型)在海光 BW1000(DCU)上的完整部署、适配与性能优化专项



## 一、环境与部署

- 节点:8× Hygon BW151(gfx936)64GB;容器 `lingbot-vla-dcu:dtk26.04-torch2.7.1`
- 数据集:pusht(LeRobot v3.0,50 episodes,`lerobot-manual-download` 下载)
- 模型:LingBot-VLA 4B 预训练权重 + Qwen2.5-VL-3B-Instruct(tokenizer)
- 关键适配(与已有 VA 适配一致):
  - config.json 删除 `dtype` 字段(Qwen2.5-VL 代码兼容)
  - 新建 `configs/robot_configs/pusht.yaml` & `assets/norm_stats/pusht.json`(bounds_99 归一化)
  - `base_dataset.py`:video_backend `torchcodec` → `pyav`(DCU 容器无 torchcodec;H20 用 torchcodec 可再省 ~10%)
  - lerobot 升级 0.3.3 → 0.4.2;补装 pyserial / deepdiff
  - torch inductor 补丁:fix flex_attention `block_m` AttributeError(torch 2.9.0 特有)

## 二、基准与优化路径(开箱 → 125%)

| 阶段 | 配置 | s/it | samples/s/gpu | BW1000/H20 |
| --- | --- | --- | --- | --- |
| 开箱 | 无优化 | 2.0 | 4 | 70% |
| +Fused 优化器 | optimizer fused | 1.951 | 4.39 | 71.9% |
| +全局 compile | torch.compile(flex_attention, max-autotune) | 1.281 | 6.23 | 110.3% |
| +triton 更换 | 更换 triton 包后单次迭代耗时缩短一半 | 1.198 | 6.67 | 117.2% |
| **+FSDP 通信优化 + flex attention + vision channels_last** | 全部优化叠加 | **0.7192** | **11.14** | **125%**(相对 H20 全局 compile) |

H20 参考(同配置):开箱 1.404s/it、全局 compile 0.899s/it、freeze_vision_encoder 下 0.73s/it。

## 三、关键优化(5 个 patch)

### 1. FSDP 通信优化(0.98s → 0.74s,最大单项收益)

双塔结构(36 层 VLM 塔 + 36 层 action 专家塔,每层一个 FSDP unit,共 74 个)导致反向传播中同一参数被 use 两次:

- 原实现:`reshard_after_backward=True`,第一次 backward 后就 reshard,第二次 use 需重新 all-gather → 重复通信
- 优化(`02_fsdp2_parallelize.patch`):**reuse-aware 延后 reshard** —— 若还有更早的 forward use 未消费,本次不 reshard,让参数在连续多次 backward 间共享同一份 unsharded 数据,只在最后一次 use 后真正 reshard,剔除中间重复的 all-gather + reduce-scatter
- 附带:`embed_tokens` 独立 shard(`enable_fsdp2_embed_tokens_shard: true`),避免每次 forward/backward 全量 all-gather 巨大的 embed_tokens,再省 0.06s

### 2. Flex Attention mask 复用(`03_flex_attention.patch`)

- 原实现:每次调用都按 128 对齐 padding,并重新构建 block mask(36 层 Q/K/V shape 完全一致却反复重建)
- 优化:解耦 `build_flex_block_mask()` + `flex_attention_forward()`;不 padding,直接传原始长度(create_mask 内部处理非对齐);**block_mask 按 shape 首次构建后跨层复用,零开销**

### 3. LLM Pipeline 优化(`04_llm_pipeline.patch`)

- `image_embeds` 去同步:用 `reshape(batch,-1,dim)` 替代 `split_sizes` 的 GPU→CPU 同步 + split/stack(graph break)
- `block_mask` 跨层复用(36 层 shape 一致则只构建一次)

### 4. Vision Conv3d 内存布局(`05_vision_channels_last.patch`)

- `Qwen2_5_VisionPatchEmbed.forward`:patch 的 view 结果用 `memory_format=torch.channels_last_3d` 摆放后再进 Conv3d,显著降低 DCU 上卷积访存

### 5. 环境变量与 torchcodec

- 新增 `TRITON_FLEX_TUNED=1` / `TRITON_FLEX_ASYMMETRIC_BWD=1`(flex asymmetric bwd)、`TORCHINDUCTOR_FORCE_POINTER_RANGE=1`、`HSA_FORCE_FINE_GRAIN_PCIE=1`
- 若 DCU 能支持 torchcodec 视频后端,预计可再提升 ~10%



## 四、代码与文档

```
lingbot-vla-dcu/
├── README.md               # 本项目说明
├── Lingbot-vla部署及其性能分析.pdf   # 原始部署+性能分析报告
├── LingBot-VLA_DCU_适配与优化记录.txt # 完整适配记录(环境/配置/patch/基准/命令)
└── code/
    ├── lingbot-vla/        # 上游 GitHub 工程(含 DCU 打补丁后代码)
    ├── patches/            # 兼容 patch + 可开关性能 patch
    ├── dockerfile / datadown.sh / start_pi.sh
    └── README.md           # 海光适配版部署说明
```



  **最终成果:BW1000 达 H20(开启全局 compile)的 125%(samples/s/gpu 11.14 vs 8.9),全流程收益 100% → 125% 以上。**
