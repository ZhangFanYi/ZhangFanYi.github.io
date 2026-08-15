#!/usr/bin/env bash
# HCU wrapper for the official LLaVA-OneVision Reasoner SFT recipe.
# The dataset is streamed from Hugging Face; no local dataset or DCP checkpoint
# is required. Defaults can be overridden with environment variables, and
# additional Hydra overrides can be passed as positional arguments.

set -euo pipefail

: "${TOML_FILE:=examples/toml/sft_config/llava_ov.toml}"
: "${VLM_MODEL_NAME:=/public/opendas/DL_DATA/llm-models/qwen3/Qwen3-VL-8B-Instruct}"
: "${VLM_SAFETENSORS_PATH:=/data/Cosmos/checkpoints/Cosmos3-Nano-VLM}"
: "${RUN_NAME:=reasoner_llava_ov_hcu_perf_8card}"
: "${OUTPUT_ROOT:=/data/Cosmos/cosmos/outputs/hcu_training}"
: "${LOG_FILENAME:=${RUN_NAME}.log}"
: "${HIP_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"
: "${MASTER_PORT:=29715}"
: "${MAX_ITER:=70}"
: "${MAX_TOKENS:=16000}"
: "${CHECKPOINT_SAVE_ITER:=1000}"
: "${DCP_ASYNC_MODE_ENABLED:=false}"
: "${PERF_RECORD_ENABLED:=true}"
: "${PERF_WARMUP_ITERS:=10}"

HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES//[[:space:]]/}"
if [[ -z "$HIP_VISIBLE_DEVICES" || "$HIP_VISIBLE_DEVICES" == ,* || "$HIP_VISIBLE_DEVICES" == *, || "$HIP_VISIBLE_DEVICES" == *,,* ]]; then
    echo "ERROR: HIP_VISIBLE_DEVICES must be a comma-separated list without empty entries: $HIP_VISIBLE_DEVICES" >&2
    exit 2
fi
IFS=',' read -r -a HCU_DEVICE_IDS <<< "$HIP_VISIBLE_DEVICES"
NPROC_PER_NODE=${#HCU_DEVICE_IDS[@]}

export COSMOS_TRAINING="${COSMOS_TRAINING:-true}"
# This recipe streams its dataset from Hugging Face, so offline mode must not
# default to 1 even though model/tokenizer files are supplied locally.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export HIP_VISIBLE_DEVICES
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PERF_RECORD_ENABLED PERF_WARMUP_ITERS
PERF_PLATFORM="hcu"
PERF_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES"

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

echo ">>> HCU devices: ${HIP_VISIBLE_DEVICES}"
echo ">>> Processes per node: ${NPROC_PER_NODE}"
echo ">>> Max iterations / scheduler cycle: ${MAX_ITER}"
echo ">>> Max packed tokens: ${MAX_TOKENS}"
echo ">>> Async DCP save: ${DCP_ASYNC_MODE_ENABLED}"
echo ">>> Dataset: lmms-lab/LLaVA-OneVision-Data (Hugging Face streaming)"

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
