# Qwen3.5-VLM SFT 训练 Demo

使用 1 条数据验证 Qwen3.5-0.8B VLM SFT 训练流程的正确性。

## 运行

```bash
# 前置：需先在当前目录安装环境（bash setup_env.sh）
# 设置模型路径后启动
MODEL_PATH=/path/to/Qwen3.5-0.8B bash run.sh
```

## 预期输出

```
Step 1/10 Loss: ~1.6
Step 2/10 Loss: ~1.0
Step 3/10 Loss: ~0.3
...
Step 10/10 Loss: ~0.001
Training completed!
```

Loss 持续下降即验证训练正确。
