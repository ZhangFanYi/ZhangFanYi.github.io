#!/usr/bin/env bash
# NVIDIA/H20 local-asset mirror of the HCU VideoPhy-2 Nano performance run.

set -euo pipefail

: "${TOML_FILE:=examples/toml/sft_config/videophy2_sft_nano.toml}"
: "${VIDEOPHYSICS_ROOT:=/data/datasets/videophysics_iter2_repro}"
: "${VLM_MODEL_NAME:=/public/opendas/DL_DATA/llm-models/Qwen3-VL-8B-Instruct}"
: "${VLM_SAFETENSORS_PATH:=/public/opendas/DL_DATA/llm-models/Cosmos3/checkpoints/Cosmos3-Nano-VLM}"
: "${RUN_NAME:=reasoner_videophy2_nano_nvidia_h20_perf_8card}"
: "${OUTPUT_ROOT:=/data/outputs/nvidia_training}"
: "${LOG_FILENAME:=${RUN_NAME}.log}"
: "${MASTER_PORT:=30711}"
: "${MAX_ITER:=70}"
: "${CHECKPOINT_SAVE_ITER:=1000}"

export VIDEOPHYSICS_ROOT
source "$(dirname "${BASH_SOURCE[0]}")/_sft_nvidia_local_common.sh"

EXTRA_DATASET_CHECK='[[ -f "$VIDEOPHYSICS_ROOT/videophy2_train/meta.json" ]] || { echo "ERROR: missing $VIDEOPHYSICS_ROOT/videophy2_train/meta.json" >&2; exit 1; }'
TAIL_OVERRIDES=(
    "job.wandb_mode=disabled"
    "job.name=${RUN_NAME}"
    "model.config.policy.backbone.model_name=${VLM_MODEL_NAME}"
    "model.config.policy.backbone.safetensors_path=${VLM_SAFETENSORS_PATH}"
    "model.config.parallelism.data_parallel_shard_degree=${NPROC_PER_NODE}"
    "trainer.seed=42"
    "trainer.max_iter=${MAX_ITER}"
    "scheduler.cycle_lengths=[${MAX_ITER}]"
    "trainer.callbacks.log_tensor_shape.num_log=-1"
    "checkpoint.save_iter=${CHECKPOINT_SAVE_ITER}"
)
TAIL_OVERRIDES+=("$@")

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
