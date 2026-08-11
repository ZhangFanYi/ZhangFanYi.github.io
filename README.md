# 张一凡

我是张一凡,海光信息客户应用优化部成员, 欢迎来到我的github主页，本人专注于 **国产海光芯片(DCU)生态下的大模型与具身智能性能优化**。这个站点是我的工作成果归档:每个项目包含优化思路、性能数据与代码。

---

## 一、大模型训练与推理优化

| # | 项目 | 类型 | 文档 | 代码 |
| --- | --- | --- | --- | --- |
| 01 | [DeepSeek-V3 671B 千卡集群训练及线性度验证](./大模型训练与推理优化/01-deepseek-v3-671b-线性度验证/) | 千卡集群训练 | ✅ 线性度 97.1% | ✅DCU-Megatron |
| 02 | [Qwen3.5-0.8B 训练优化(ZeRO-0 + Flash Linear Attention)](./大模型训练与推理优化/02-qwen3.5-0.8b-训练优化/) | 训练 | ✅ 吞吐提升近4倍 | ✅ demo |
| 03 | [Qwen3.5-35B 强化学习适配(verl-Megatron-GRPO)](./大模型训练与推理优化/03-qwen3.5-35b-强化学习适配/) | 训练/RL | ✅verl+grpo | ✅ verl-das |
| 04 | [Qwen3.6-27B 多模态推理优化(缓存命中 + 量化选型)](./大模型训练与推理优化/04-qwen3.6-27b-多模态推理优化/) | 推理 | ✅ 性能提升3倍以上 | ✅ demo |
| 05 | [Qwen3-VL-Embedding 推理优化(高并发 GEMM)](./大模型训练与推理优化/05-qwen3-vl-embedding-推理优化/) | 推理 | ✅性能恢复 | ✅ demo |
| 06 | [Qwen3-30B 训练优化(Grouped GEMM 融合,性能翻倍)](./大模型训练与推理优化/06-qwen3-30b-训练优化/) | 训练 | ✅ 性能翻倍 | ✅trace分析 |
| 07 | [DeepSeek-V4-Flash 推理优化(算子级 Prof 瓶颈排查)](./大模型训练与推理优化/07-deepseek-v4-flash-推理优化/) | 推理 | ✅ 三大瓶颈定位 | ✅ 压测脚本 |
| 08 | [DeepSeek-V4-Pro / GLM5.2 推理部署(IFB / PD 分离)](./大模型训练与推理优化/08-deepseek-v4-pro-glm5.2-推理/) | 推理 | ✅ 测试数据 | ✅ 指定缓存命中率 |
| 09 | [讯飞自研模型适配及优化(IFB / PD / Mooncake)](./大模型训练与推理优化/09-讯飞自研模型适配优化/) | 推理 | ✅ 适配+优化 |

## 二、具身智能模型优化

| # | 项目 | 内容 | 文档 | 代码 |
| --- | --- | --- | --- | --- |
| 10 | [DreamZero-DCU 适配优化(视频生成 RL)](./具身智能模型优化/dreamzero-dcu/) | Wan2.1-I2V-14B 基座部署适配 | ✅ 性能提升32% | ✅ 完整仓+patch |
| 11 | [Isaac-GR00T-DCU 适配优化(人形机器人基础模型)](./具身智能模型优化/isaac-gr00t-dcu/) | GR00T-N1 微调适配 | ✅ 性能提升42% | ✅ 完整仓+patch |
| 12 | [LeRobot-ACT-DCU 适配优化(动作策略 ACT)](./具身智能模型优化/lerobot-act-dcu/) | ACT 动作策略训练适配 | ✅ 性能提升30% | ✅ 完整仓+patch |
| 13 | [LingBot-VA-DCU 适配优化(因果世界模型)](./具身智能模型优化/lingbot-va-dcu/) | VA 世界模型部署适配 | ✅ 性能提升28% | ✅ 完整仓+patch |
| 14 | [LingBot-VLA-DCU 部署与性能优化(重点专项)](./具身智能模型优化/lingbot-vla-dcu/) | pi0.5 VLA 部署 | ✅ 性能提升52% | ✅ 完整仓+patch |

---

## 能力标签

- **训练优化**:Megatron / DeepSpeed / FSDP / verl, prof 抓取与算子级分析,ZeRO 与 Attention 选型, 模型框架层行为优化
- **推理优化**:vLLM / SGLang,PD 分离 / IFB / Mooncake 部署,模型量化与固定缓存命中压测,GPU 与国产加速卡(DCU / 沐曦)性能对标
- **具身智能**:VLA(pi0.5 / LingBot-VLA)、视频生成 RL(DreamZero)、人形基础模型(GR00T)、动作策略(ACT / VA)的 DCU 适配、patch 化交付与性能优化

欢迎提出任何宝贵意见与交流 
