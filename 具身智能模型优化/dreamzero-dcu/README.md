<img width="879" height="117" alt="截屏2026-08-15 15 37 04" src="https://github.com/user-attachments/assets/dd06f14c-5f7e-48b0-bc5a-365597a862fe" /># DreamZero-DCU 适配优化

## 概述

面向具身智能的世界模型 **DreamZero**在海光 DCU 上的部署适配与优化

**DreamZero**模型结构
DreamZero (VLA)
├── 文本编码器 (T5)
├── 图像编码器 (CLIP)
├── 视频VAE
└── Action Head (WANPolicyHead)
    └── 视频DiT骨干网络 ← Wan2.1 / Wan2.2 
    
## 技术要点

- 模型:[DreamZero](https://github.com/dreamzero0/dreamzero)+ google/umt5-xxl 文本编码器
    hf download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./checkpoints/Wan2.1-I2V-14B-480P
    hf download google/umt5-xxl --local-dir ./checkpoints/umt5-xxl
- 数据集:DreamZero-DROID-Data(具身操作轨迹数据)
    huggingface-cli download GEAR-Dreams/DreamZero-DROID-Data --repo-type dataset --local-dir ./data/droid_lerobot
- DCU 适配:删除 NVIDIA 相关依赖,替换为 DCU 生态包(pypi.sourcefind.cn 对应版本)
- 视频解码组件 torchcodec 按海光文档单独适配
- 基础镜像:`harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.9.0-ubuntu22.04-dtk26.04-py3.10`
- 代码仓可直接拉取



## 优化记录

官方提供两种DiT主干网络 使用Wan2.1时 BW1000开箱性能为H20的113%
<img width="873" height="118" alt="image" src="https://github.com/user-attachments/assets/282c7a0f-52b1-4c76-85b9-c46711eb2578" />

使用Wan2.2-TI2V-5B时 开箱性能为H20的82.6%
使用如下手段进行优化：
# 1. 优化器更换 AdamW 更换为 AdamW_fused 性能提升2.6%
# 2. Channel Last 3d 性能提升4.1%
# 3. 消除fa的重复缓存 性能提升 21%
