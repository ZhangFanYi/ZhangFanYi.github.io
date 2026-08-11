#!/bin/bash
# Qwen3.5-VLM SFT 最小训练 Demo
# Usage: bash run.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

# . "${SCRIPT_DIR}/.venv/bin/activate"

MODEL_PATH="${MODEL_PATH:-/public/home/yuhui1/models/Qwen3.5-0.8B}"
DATA_PATH="${DATA_PATH:-./data.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-./output}"
EPOCHS="${EPOCHS:-10}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
# ATTN_IMPL="flash_attention_3"

mkdir -p "${OUTPUT_DIR}"

accelerate launch \
  --mixed_precision bf16 \
  --use_deepspeed \
  --deepspeed_config_file ds_config.json \
  --num_machines 1 \
  --dynamo_backend no \
  --num_processes 1 \
  train.py \
    --model_path "${MODEL_PATH}" \
    --data_path "${DATA_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --epochs "${EPOCHS}" \
    --attn_implementation "${ATTN_IMPL}" \
    --save_checkpoint