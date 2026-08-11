#!/bin/bash
# BW150 Baseline：新仓库运行，复用旧环境中的只读模型资产。
export DREAMZERO_ROOT="/zhangyifan/dz/dreamzero-xyz"
export DATA_ROOT="/zhangyifan/dz/all"
export PYTHONPATH="${DREAMZERO_ROOT}:${PYTHONPATH:-}"
export DROID_DATA_ROOT="${DATA_ROOT}"
# 保留四卡 Baseline 默认值，同时允许单卡实验在启动命令中显式覆盖。
export OUTPUT_DIR="${OUTPUT_DIR:-${DREAMZERO_ROOT}/checkpoints/dz_droid_wan22_bf16_100_xyz_baseline}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NUM_GPUS="${NUM_GPUS:-8}"
export WAN22_CKPT_DIR="/zhangyifan/dz/dreamzero/Wan2.2-TI2V-5B"
export IMAGE_ENCODER_DIR="/zhangyifan/dz/all/dreamzero/model"
export TEXT_ENCODER_PATH="${DATA_ROOT}/dreamzero/model/models_t5_umt5-xxl-enc-bf16.pth"
export TOKENIZER_DIR="/zhangyifan/dz/dreamzero/checkpoints/umt5-xxl"
export MIOPEN_FIND_MODE=3
export MIOPEN_PRECISION_FP32_FP32_FP32_TF32_FP32=1
export PYTORCH_MIOPEN_SUGGEST_NHWC=1
