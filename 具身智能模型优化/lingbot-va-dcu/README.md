# LingBot-VA-DCU 适配优化

## 概述

Robbyant 团队 **LingBot-VA**(Causal World Modeling for Robot Control,因果世界模型)在海光 DCU 上的部署适配与优化

## 技术要点

- 模型:[LingBot-VA](https://technology.robbyant.com/lingbot-va)(arXiv:2601.21998), 模型和数据集下载和官网保持一致

  ​```bash
  huggingface-cli download --repo-type dataset robbyant/robotwin-clean-and-aug-lerobot
  modelscope download --model Robbyant/lingbot-va-base
  ​```
- DCU 适配: 删除 NVIDIA 相关依赖,替换为 DCU 生态包(pypi.sourcefind.cn 对应版本)
- 完整镜像: wget https://hygon-torch-third-party-1251001002.cos.ap-shanghai.myqcloud.com/external/haiguang/image/va_image.tar.gz

## 优化记录
1. 修改FSDP粒度 实现overlap
2. 提取三个热点算子为单独函数并分别做融合
   
详情请见code/patches
优化后性能达到H20的96% Loss下降水平一致
