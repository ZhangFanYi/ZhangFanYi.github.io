# 项目 15: Cosmos3-Nano DCU 适配优化(世界模型 / 具身策略微调)

## 概述

英伟达 **Cosmos**(cosmos-framework,Cosmos3-Nano 系列:Vision Nano SFT、Reasoner VideoPhy2-Nano SFT、Action Policy LIBERO / DROID-Nano 微调)在海光 DCU 上的微调适配与优化。官方代码仓为 [NVIDIA/cosmos-framework](https://github.com/NVIDIA/cosmos-framework),本页代码为其 DCU 适配版本。

## 技术要点

- 模型:Cosmos3-Nano 系列(vision nano / reasoner videophy2 nano / action policy libero·droid nano SFT)
- 官方基线:与官方仓一致,适配基于官方提交 `6d84808`(Support reasoner video input #25),全部修改以补丁形式存放
- DCU 适配:依赖替换为 DCU 生态版本;模型、训练、推理、启动脚本的修改集中在 `code/patches/` 文件夹(共 6 个补丁)
- 基础镜像:`harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.9.0-ubuntu22.04-dtk26.04-py3.10`
- 启动命令:`examples/launch_sft_*_hcu.sh`(HCU 版)与 `examples/launch_inference_generator_vision_nano_hcu.sh`(推理)

## 优化记录

### 第 0 项:FlashAttention max_seqlen 分桶,避免融合后重编译(45s/it → 35s/it)

开启融合(`torch.compile`)后,编译层会按 Python 整数的精确值做守卫,每个 batch 的序列长度不同就会触发一次重编译(报错/性能劣化)。因此将 FlashAttention 的 Python `max_seqlen` 启动参数按桶对齐(`1024 / 2048 / 4096 / 8192 / 16384 / 32768 / 65536`),精确长度仍通过 `max_*_len` 提供,只影响 FlashAttention 的启动参数。开启后训练性能由 **45s/it 提升到 35s/it**。

核心修改(`cosmos_framework/data/generator/sequence_packing/runtime.py`):

```python
FLASH_ATTN_MAX_SEQLEN_BUCKETS = (1024, 2048, 4096, 8192, 16384, 32768, 65536)


def _bucket_flash_attn_max_seqlen(n: int) -> int:
    """将 FlashAttention 的 max_seqlen 按桶对齐,避免 torch.compile 按精确值
    逐 batch 触发重编译;精确长度仍由 max_*_len 提供。"""
    if n < 0:
        raise ValueError(f"FlashAttention max_seqlen cannot be negative, got {n}")
    if n == 0:
        return 0
    for bucket in FLASH_ATTN_MAX_SEQLEN_BUCKETS:
        if n <= bucket:
            return bucket
    raise ValueError(
        f"sequence length {n} exceeds FlashAttention max_seqlen buckets "
        f"{FLASH_ATTN_MAX_SEQLEN_BUCKETS}"
    )
```

并在 `init_sequence_pack` 中传给 FlashAttention:

```python
fa_max_sample_len=_bucket_flash_attn_max_seqlen(_max_sample_len),
fa_max_causal_len=_bucket_flash_attn_max_seqlen(_max_causal_len),
fa_max_full_len=_bucket_flash_attn_max_seqlen(_max_full_len),
```

## 代码

[code](./code/) 为完整适配仓,与官方仓的差异以补丁形式保存在 `code/patches/`:

```
code/
├── cosmos_framework/  # 源码(HCU 适配)
├── examples/          # HCU / NVIDIA 本地版 SFT 与推理启动脚本
├── patches/           # 相对官方基线 6d84808 的 6 个补丁
│   ├── 0001-feat-hcu-preserve-Cosmos3-training-adaptations.patch   # 训练适配(FA max_seqlen 分桶等)
│   ├── 0002-feat-training-add-HCU-and-NVIDIA-SFT-launchers.patch   # 新增 HCU / NVIDIA SFT 启动脚本
│   ├── 0003-fix-hcu-return-frame-timestamps-from-PyAV-decoder.patch
│   ├── 0004-align-vision-SFT-configs-with-official-9726697.patch
│   ├── 0005-enable-offline-W-B-logging-for-vision-nano.patch
│   └── 0006-fix-inference-make-local-generator-launchers-self-co.patch  # 推理启动自包含
└── README.md        # 官方仓说明
```

应用补丁:`git format-patch` 生成的补丁,可用 `git apply patches/*.patch` 直接打到官方仓 `6d84808` 上。