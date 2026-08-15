#!/usr/bin/env bash
# HCU wrapper for the official VideoPhy-2 Nano SFT recipe.
# Keep model/training parameters in the official TOML; this file only supplies
# HCU topology and local data/checkpoint paths. Script-level defaults can be
# overridden with environment variables; recipe overrides can be passed as
# positional arguments after the script name.

set -euo pipefail

: "${TOML_FILE:=examples/toml/sft_config/videophy2_sft_nano.toml}"
: "${VIDEOPHYSICS_ROOT:=/data/Cosmos/datasets/videophysics_iter2_repro}"
: "${VLM_MODEL_NAME:=/public/opendas/DL_DATA/llm-models/qwen3/Qwen3-VL-8B-Instruct}"
: "${VLM_SAFETENSORS_PATH:=/data/Cosmos/checkpoints/Cosmos3-Nano-VLM}"
: "${RUN_NAME:=reasoner_videophy2_nano_hcu_perf_8card}"
: "${OUTPUT_ROOT:=/data/Cosmos/cosmos/outputs/hcu_training}"
: "${LOG_FILENAME:=${RUN_NAME}.log}"
: "${HIP_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"
: "${MASTER_PORT:=29711}"
: "${MAX_ITER:=70}"
: "${CHECKPOINT_SAVE_ITER:=1000}"
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
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HIP_VISIBLE_DEVICES
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export VIDEOPHYSICS_ROOT
export PERF_RECORD_ENABLED PERF_WARMUP_ITERS
PERF_PLATFORM="hcu"
PERF_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES"

EXTRA_DATASET_CHECK='[[ -f "$VIDEOPHYSICS_ROOT/videophy2_train/meta.json" ]] || { echo "ERROR: missing $VIDEOPHYSICS_ROOT/videophy2_train/meta.json" >&2; exit 1; }'

TAIL_OVERRIDES=(
    "job.wandb_mode=disabled"
    "job.name=${RUN_NAME}"
    "model.config.policy.backbone.model_name=${VLM_MODEL_NAME}"
    "model.config.policy.backbone.safetensors_path=${VLM_SAFETENSORS_PATH}"
    "model.config.parallelism.data_parallel_shard_degree=${NPROC_PER_NODE}"
    "trainer.seed=42"
    "trainer.max_iter=${MAX_ITER}"
    # Keep the cosine schedule valid when the HCU performance run extends the
    # official 50-iteration recipe. Otherwise scheduler step 51 has no cycle.
    "scheduler.cycle_lengths=[${MAX_ITER}]"
    "trainer.callbacks.log_tensor_shape.num_log=-1"
    "checkpoint.save_iter=${CHECKPOINT_SAVE_ITER}"
)
TAIL_OVERRIDES+=("$@")

echo ">>> HCU devices: ${HIP_VISIBLE_DEVICES}"
echo ">>> Processes per node: ${NPROC_PER_NODE}"
echo ">>> Max iterations / scheduler cycle: ${MAX_ITER}"
echo ">>> HCU FSDP communication ordering: framework"

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
