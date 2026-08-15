#!/usr/bin/env bash
# HCU wrapper for the official Generator Vision Nano CosmosDataLoader recipe.
# This is the only launcher for both short validation and longer runs: override
# MAX_ITER/CHECKPOINT_SAVE_ITER or append Hydra overrides on the command line.

set -euo pipefail

: "${TOML_FILE:=examples/toml/sft_config/vision_sft_nano_mapstyle_dataloader.toml}"
: "${DATASET_PATH:=/data/Cosmos/datasets/BridgeData2-Subset-Synthetic-Captions/sft_dataset_bridge}"
: "${BASE_CHECKPOINT_PATH:=/public4/opendas/DL_DATA/llm-models/Cosmos3/checkpoints/Cosmos3-Nano-DCP}"
: "${WAN_VAE_PATH:=/public4/opendas/DL_DATA/llm-models/Cosmos3/checkpoints/wan22_vae/Wan2.2_VAE.pth}"
: "${LOCAL_PROCESSOR_DIR:=/public4/opendas/DL_DATA/llm-models/Cosmos3/Cosmos3-Nano}"
: "${RUN_NAME:=generator_vision_sft_nano_cosmosdataloader_hcu_perf_8card}"
: "${OUTPUT_ROOT:=/data/Cosmos/cosmos/outputs/hcu_training}"
: "${LOG_FILENAME:=${RUN_NAME}.log}"
: "${HIP_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"
: "${MASTER_PORT:=29713}"
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
export PERF_RECORD_ENABLED PERF_WARMUP_ITERS
PERF_PLATFORM="hcu"
PERF_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES"

EXTRA_DATASET_CHECK='[[ -f "$DATASET_PATH/train/video_dataset_file.jsonl" ]] || { echo "ERROR: missing $DATASET_PATH/train/video_dataset_file.jsonl" >&2; exit 1; }
[[ -f "$BASE_CHECKPOINT_PATH/checkpoint.json" ]] || { echo "ERROR: missing $BASE_CHECKPOINT_PATH/checkpoint.json" >&2; exit 1; }
compgen -G "$BASE_CHECKPOINT_PATH/model/*.distcp" >/dev/null || { echo "ERROR: no DCP shards under $BASE_CHECKPOINT_PATH/model" >&2; exit 1; }'

TAIL_OVERRIDES=(
    "job.name=${RUN_NAME}"
    "job.wandb_mode=offline"
    "model.config.vlm_config.tokenizer.local_processor_dir=${LOCAL_PROCESSOR_DIR}"
    "trainer.max_iter=${MAX_ITER}"
    "checkpoint.save_iter=${CHECKPOINT_SAVE_ITER}"
)
TAIL_OVERRIDES+=("$@")

echo ">>> HCU devices: ${HIP_VISIBLE_DEVICES}"
echo ">>> Processes per node: ${NPROC_PER_NODE}"
echo ">>> Max iterations: ${MAX_ITER}"

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
