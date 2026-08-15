#!/usr/bin/env bash
# NVIDIA/H20 local-model mirror of the HCU LLaVA-OneVision Nano run.

set -euo pipefail

: "${TOML_FILE:=examples/toml/sft_config/llava_ov.toml}"
: "${VLM_MODEL_NAME:=/public/opendas/DL_DATA/llm-models/Qwen3-VL-8B-Instruct}"
: "${VLM_SAFETENSORS_PATH:=/public/opendas/DL_DATA/llm-models/Cosmos3/checkpoints/Cosmos3-Nano-VLM}"
: "${RUN_NAME:=reasoner_llava_ov_nvidia_h20_perf_8card}"
: "${OUTPUT_ROOT:=/data/outputs/nvidia_training}"
: "${LOG_FILENAME:=${RUN_NAME}.log}"
: "${MASTER_PORT:=30715}"
: "${MAX_ITER:=70}"
: "${MAX_TOKENS:=16000}"
: "${CHECKPOINT_SAVE_ITER:=1000}"
: "${DCP_ASYNC_MODE_ENABLED:=false}"
: "${HF_HUB_OFFLINE:=0}"

source "$(dirname "${BASH_SOURCE[0]}")/_sft_nvidia_local_common.sh"

EXTRA_DATASET_CHECK='[[ -d "$VLM_MODEL_NAME" ]] || { echo "ERROR: VLM_MODEL_NAME not found: $VLM_MODEL_NAME" >&2; exit 1; }
[[ -d "$VLM_SAFETENSORS_PATH" ]] || { echo "ERROR: VLM_SAFETENSORS_PATH not found: $VLM_SAFETENSORS_PATH" >&2; exit 1; }
compgen -G "$VLM_SAFETENSORS_PATH/*.safetensors" >/dev/null || { echo "ERROR: no *.safetensors files under $VLM_SAFETENSORS_PATH" >&2; exit 1; }'
TAIL_OVERRIDES=(
    "job.wandb_mode=disabled"
    "job.name=${RUN_NAME}"
    "model.config.policy.backbone.model_name=${VLM_MODEL_NAME}"
    "model.config.policy.backbone.safetensors_path=${VLM_SAFETENSORS_PATH}"
    "model.config.parallelism.data_parallel_shard_degree=${NPROC_PER_NODE}"
    "data_setting.max_tokens=${MAX_TOKENS}"
    "trainer.seed=42"
    "trainer.max_iter=${MAX_ITER}"
    "scheduler.cycle_lengths=[${MAX_ITER}]"
    "trainer.callbacks.log_tensor_shape.num_log=-1"
    "checkpoint.save_iter=${CHECKPOINT_SAVE_ITER}"
    "checkpoint.dcp_async_mode_enabled=${DCP_ASYNC_MODE_ENABLED}"
)
TAIL_OVERRIDES+=("$@")

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
