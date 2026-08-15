# LeRobot-ACT-DCU 适配优化

## 概述

LeRobot **ACT**动作策略在海光 DCU 上的训练适配与优化

## 技术要点

- 模型 数据集下载请参考code/datadown.sh
- 环境构建请参考code/dockerfile
- DCU 适配:`patches/` 内含 ACT on Hygon DCU 的优化 patch,镜像构建时通过 `ENABLE_OPTIMIZATION=1` 开关启用


## 优化记录
1. Channel Last
2. 开启TF32加速

<img width="1042" height="146" alt="截屏2026-08-15 16 27 21" src="https://github.com/user-attachments/assets/415e42fd-fbad-4b21-a951-44d707e218f0" />
<img width="1280" height="468" alt="image" src="https://github.com/user-attachments/assets/e9b4d161-c40e-4d0d-9537-cc8574ebee8c" />
从单卡到8卡的拓展比>95%
