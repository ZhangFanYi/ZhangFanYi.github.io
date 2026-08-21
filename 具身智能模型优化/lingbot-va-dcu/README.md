# LingBot-VA-DCU 适配优化

## 概述

Robbyant 团队 **LingBot-VA**(Causal World Modeling for Robot Control,因果世界模型)在海光 DCU 上的部署适配与优化

## 技术要点

- 模型:[LingBot-VA](https://technology.robbyant.com/lingbot-va)(arXiv:2601.21998), 模型和数据集下载和官网保持一致

  ​```
  huggingface-cli download --repo-type dataset robbyant/robotwin-clean-and-aug-lerobot
  ​```
  ​```
  modelscope download --model Robbyant/lingbot-va-base
  ​```
- DCU 适配: 删除 NVIDIA 相关依赖,替换为 DCU 生态包(pypi.sourcefind.cn 对应版本)
- 完整镜像: wget https://hygon-torch-third-party-1251001002.cos.ap-shanghai.myqcloud.com/external/haiguang/image/va_image.tar.gz

## 优化记录
1. 修改FSDP粒度 实现overlap
2. 提取三个热点算子为单独函数并分别做融合
   
详情请见code/patches
优化后性能达到H20的96% Loss下降水平一致

## 优化分析

### 实现通算并行
官方代码仓：
```
for block in model.blocks:
    fully_shard(block.attn1, **fsdp_config)   # 独立 FSDP 单元 1
    fully_shard(block.attn2, **fsdp_config)   # 独立 FSDP 单元 2
    fully_shard(block.ffn,   **fsdp_config)   # 独立 FSDP 单元 3
    fully_shard(block,       **fsdp_config)   # 父单元 4
fully_shard(model, ...)                       
```
修改后
```
for block in model.blocks:
    fully_shard(block, **fsdp_config)          # 只剩 1 个单元 
fully_shard(model, ...)
```

FSDP里设置的是reshard_after_forward=True 
这表明着每个单元在forward前基于allgather做unshared把分片参数凑齐 forward结束后reshared到各卡上 backward前同样unshared 基于all gather拼接每张卡上的参数 结束后reduce scatter   
细粒度时: attn1刚all_gather完 attn2的all_gather马上就要发 中间只隔了一个attn1的forward compute窗口太短 通信藏不进去 而且collective启动频率高(每 layer 多 3 倍) launcher和RCCL的启动开销叠加 
粗粒度时: block N的forward一开始就一次性all_gather整个block的参数 然后block N要跑完attn1+attn2+ffn三段计算 这个大窗口足够让FSDP在后台预取block N+1的all_gather 实现block N计算和block N+1通信的 overlap  


### 热点算子提取
原本的实现中 一堆if条件会判断有没有kv_cache rotary cache_name等 训练图/推理图/不同cache_name的变体组合起来 guard失败就重新trace一次 组合体变多就会撞上cache_size_limit  抛出recompilation limit reached报错 

因此选择性编译 把block forward中热点算子提取陈独立方法 再分别torch.compile attention本身不碰 提取了
_attn1_input：融合norm1 + *(1+scale) + shift + cast 这串 elementwise
_attn2_input：走lightop.RMSNorm原生 kernel + 周边cast 融合
_ffn_residual：float cast + * c_gate_msa + + hidden_states + type_as 四个elementwise 融成一个 kernel
WanAttention里的_apply_rotary_emb：to(float64) + 复数乘 freqs

```
if os.getenv("COMPILE_BLOCK_OPS", "0") == "1":
    self._apply_rotary_emb = torch.compile(self._apply_rotary_emb, dynamic=True)
# block 里:
    self._attn1_input  = torch.compile(self._attn1_input,  dynamic=True)
    self._attn2_input  = torch.compile(self._attn2_input,  dynamic=True)
    self._ffn_residual = torch.compile(self._ffn_residual, dynamic=True)
```
dynamic=True把具体的shape值换成符号变量 guard变成结构性断言 shape抖动不再失败 提取出去后把会变的Python值（if kv_cache/cache_name 这类特化源）移出编译边界 控制流留在eager forward 不进Dynamo
