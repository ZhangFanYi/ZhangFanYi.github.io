# ZhangFanYi(Yifan Zhang)

你好,我是张一帆,海光信息客户应用优化部成员,专注 **DCU/国产芯片生态下的大模型训练与推理性能优化**。这个站点是我的工作成果归档:每个项目包含优化思路、性能数据与代码。

---

## 项目列表(按时间线)

| # | 项目 | 类型 | 文档 | 代码 |
| --- | --- | --- | --- | --- |
| 01 | [DeepSeek-V3 671B 千卡集群训练及线性度验证](./01-deepseek-v3-671b-线性度验证/) | 训练/集群 | ✅ 线性度 97.1% | — |
| 02 | [Qwen3.5-0.8B 训练优化(ZeRO-0 + Flash Linear Attention)](./02-qwen3.5-0.8b-训练优化/) | 训练 | ✅ 吞吐大幅提升 | ✅ demo |
| 03 | [Qwen3.5-35B 强化学习适配(verl-Megatron-GRPO)](./03-qwen3.5-35b-强化学习适配/) | 训练/RL | — | ✅ verl-das |
| 04 | [Qwen3.6-27B 多模态推理优化(缓存命中 + 量化选型)](./04-qwen3.6-27b-多模态推理优化/) | 推理 | ✅ 达 A800 的 133%~335% | ✅ 压测脚本 |
| 05 | [Qwen3-VL-Embedding 推理优化(高并发 GEMM)](./05-qwen3-vl-embedding-推理优化/) | 推理 | — | ✅ 完整源码 |
| 06 | [Qwen3-30B 训练优化(Grouped GEMM 融合,性能翻倍)](./06-qwen3-30b-训练优化/) | 训练 | ✅ 性能翻倍 | — |
| 07 | [DeepSeek-V4-Flash 推理优化(算子级 Prof 瓶颈排查)](./07-deepseek-v4-flash-推理优化/) | 推理 | ✅ 三大瓶颈定位 | ✅ 压测脚本 |
| 08 | [DeepSeek-V4-Pro / GLM5.2 推理部署(IFB / PD 分离)](./08-deepseek-v4-pro-glm5.2-推理/) | 推理 | ✅ 测试数据 | — |
| 09 | [讯飞自研模型适配及优化(IFB / PD / Mooncake)](./09-讯飞自研模型适配优化/) | 推理 | 整理中 | — |

---

## 能力标签

- **训练优化**:Megatron / DeepSpeed / FSDP / verl(GRPO、PPO、蒸馏),prof 抓取与算子级分析,ZeRO 与 Attention 实现选型
- **推理优化**:vLLM / SGLang,PD 分离 / IFB / Mooncake 部署,量化模型选型与缓存命中,GPU 与国产加速卡(DCU / 沐曦)性能对标
- **性能工具链**:torch profiler trace 分析、SQL 算子级聚合、bench 压测脚本开发

## 关于本项目归档

- 文档以每个项目目录下的 `README.md` 精编呈现,原始 PDF / xlsx 报告随目录保留
- 未产出文档或代码的项目标记为"整理中",后续补充
