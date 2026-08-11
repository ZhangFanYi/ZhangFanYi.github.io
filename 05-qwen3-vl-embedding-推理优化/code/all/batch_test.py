import torch
import time
import argparse
import os
from PIL import Image
import numpy as np
from src.models.qwen3_vl_embedding import Qwen3VLEmbedder

# ===== 解析命令行参数 =====
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=None, help="batch size (overrides env var)")
parser.add_argument("--num_iter", type=int, default=None, help="number of iterations")
args = parser.parse_args()

# ===== 优先使用命令行参数，其次使用环境变量，最后使用默认值 =====
BATCH_SIZE = args.batch_size if args.batch_size is not None else int(os.environ.get("BATCH_SIZE", 1))
NUM_ITER = args.num_iter if args.num_iter is not None else int(os.environ.get("NUM_ITER", 50))

# 1. 初始化
model = Qwen3VLEmbedder(
    model_name_or_path="/zhangyifan/qwen_vl/modl/",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)

# 优化：通道最后格式
model.model.visual.to(memory_format=torch.channels_last_3d)

# 2. 准备数据（batch）
width, height = 1905, 1071
inputs = []
for _ in range(BATCH_SIZE):
    fake_img = Image.fromarray(
        np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    )
    inputs.append({"image": fake_img})

print(f"Batch size: {BATCH_SIZE}, 迭代次数: {NUM_ITER}")

# 3. 预热
for _ in range(5):
    _ = model.process(inputs)
torch.cuda.synchronize()

# 4. 正式测试
print(f"开始测试 (batch_size={BATCH_SIZE})...")
start_time = time.perf_counter()

with torch.no_grad():
    for _ in range(NUM_ITER):
        _ = model.process(inputs)

torch.cuda.synchronize()
end_time = time.perf_counter()

# 5. 计算
total_seconds = end_time - start_time
avg_seconds = total_seconds / NUM_ITER
total_images = BATCH_SIZE * NUM_ITER
fps = total_images / total_seconds

print("\n" + "="*30)
print(f"Batch size: {BATCH_SIZE}")
print(f"迭代次数: {NUM_ITER}")
print(f"处理图片总数: {total_images}")
print(f"总计耗时: {total_seconds:.4f} 秒")
print(f"单次迭代平均耗时: {avg_seconds:.4f} 秒")
print(f"每秒处理图片数 (FPS): {fps:.2f}")
print("="*30)
