#!/usr/bin/env bash
# NVIDIA/H20 local-asset mirror of the HCU Generator Vision Super LoRA run.

set -euo pipefail

: "${TOML_FILE:=examples/toml/sft_config/vision_sft_super.toml}"
: "${DATASET_PATH:=/data/datasets/BridgeData2-Subset-Synthetic-Captions/sft_dataset_bridge}"
: "${BASE_CHECKPOINT_PATH:=/public/opendas/DL_DATA/llm-models/Cosmos3/checkpoints/Cosmos3-Super-DCP}"
: "${WAN_VAE_PATH:=/public/opendas/DL_DATA/llm-models/Cosmos3/checkpoints/wan22_vae/Wan2.2_VAE.pth}"
: "${RUN_NAME:=generator_vision_sft_super_nvidia_h20_perf_8card}"
: "${OUTPUT_ROOT:=/data/outputs/nvidia_training}"
: "${LOG_FILENAME:=${RUN_NAME}.log}"
: "${MASTER_PORT:=30714}"
: "${MAX_ITER:=70}"
: "${CHECKPOINT_SAVE_ITER:=1000}"
: "${DCP_ASYNC_MODE_ENABLED:=false}"

source "$(dirname "${BASH_SOURCE[0]}")/_sft_nvidia_local_common.sh"
if (( NPROC_PER_NODE < 2 || NPROC_PER_NODE % 2 != 0 )); then
    echo "ERROR: Vision Super requires an even GPU count >= 2 for context parallelism=2; got ${NPROC_PER_NODE}" >&2
    exit 2
fi
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

EXTRA_DATASET_CHECK='[[ -f "$DATASET_PATH/train/video_dataset_file.jsonl" ]] || { echo "ERROR: missing $DATASET_PATH/train/video_dataset_file.jsonl" >&2; exit 1; }
[[ -f "$BASE_CHECKPOINT_PATH/checkpoint.json" ]] || { echo "ERROR: missing $BASE_CHECKPOINT_PATH/checkpoint.json" >&2; exit 1; }
compgen -G "$BASE_CHECKPOINT_PATH/model/*.distcp" >/dev/null || { echo "ERROR: no DCP shards under $BASE_CHECKPOINT_PATH/model" >&2; exit 1; }
'
TAIL_OVERRIDES=(
    "job.wandb_mode=disabled"
    "job.name=${RUN_NAME}"
    "model.config.vlm_config.tokenizer.pretrained_model_name=${VLM_TOKENIZER_PATH}"
    "trainer.seed=42"
    "trainer.max_iter=${MAX_ITER}"
    "checkpoint.save_iter=${CHECKPOINT_SAVE_ITER}"
    "checkpoint.dcp_async_mode_enabled=${DCP_ASYNC_MODE_ENABLED}"
)
TAIL_OVERRIDES+=("$@")

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
