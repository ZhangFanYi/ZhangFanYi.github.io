import torch
import time
from PIL import Image
import numpy as np
from src.models.qwen3_vl_embedding import Qwen3VLEmbedder

# 1. 初始化模型
model = Qwen3VLEmbedder(
    model_name_or_path="/zhangyifan/qwen_vl/modl/",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)

# 优化点：将模型转换为通道最后的内存格式（NDHWC）
model.model.visual.to(memory_format=torch.channels_last_3d)

# 2. 准备测试数据
# width, height = 1792, 992
width, height = 1905, 1071
fake_img = Image.fromarray(
    np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
)
inputs = [{"image": fake_img}]

# 3. 预热（5次）
print("预热中...")
for _ in range(5):
    _ = model.process(inputs)
torch.cuda.synchronize()

# 4. 正式测试：连续执行50次
print(f"开始连续测试 50 次推理 ({width}x{height})...")

start_time = time.perf_counter()

with torch.no_grad():
    for _ in range(50):
        _ = model.process(inputs)

# 5. 同步，确保所有操作完成
torch.cuda.synchronize()
end_time = time.perf_counter()

# 6. 计算结果
total_seconds = end_time - start_time
avg_seconds = total_seconds / 50

print("\n" + "="*30)
print(f"总计耗时: {total_seconds:.4f} 秒")
print(f"单次平均耗时: {avg_seconds:.4f} 秒")
print(f"每秒处理图片数 (FPS): {1 / avg_seconds:.2f}")
print("="*30)
