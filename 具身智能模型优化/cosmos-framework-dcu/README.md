#  Cosmos3-Nano DCU 适配优化

## 概述

英伟达 **Cosmos**(cosmos-framework,Cosmos3-Nano 在海光 DCU 上的微调适配与优化
官方代码仓 [NVIDIA/cosmos-framework](https://github.com/NVIDIA/cosmos-framework)

## 技术要点

- 模型:Cosmos3-Nano
- 数据集 模型 ckpt获取及转换均参考官方方案
- DCU 适配:依赖替换为 DCU 生态版本 镜像获取
  ```
  wget https://hygon-torch-third-party-1251001002.cos.ap-shanghai.myqcloud.com/external/haiguang/image/cosmos_210.tar.gz
  ```
- 启动命令: bash examples/launch_sft_vision_nano.sh

## 优化记录

### FlashAttention max_seqlen 分桶 避免融合后重编译(性能从45s/it提升到35s/it)

开启融合(`torch.compile`)后,编译层会按 Python 整数的精确值做守卫,每个 batch 的序列长度不同就会触发一次重编译(报错)。因此将 FlashAttention 的 Python `max_seqlen` 启动参数按桶对齐(`1024 / 2048 / 4096 / 8192 / 16384 / 32768 / 65536`),精确长度仍通过 `max_*_len` 提供,只影响 FlashAttention 的启动参数。开启后训练性能由 **45s/it 提升到 35s/it**。

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
详情请见patches
