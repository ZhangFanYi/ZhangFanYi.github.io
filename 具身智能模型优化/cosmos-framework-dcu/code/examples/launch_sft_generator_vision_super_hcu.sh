#!/usr/bin/env bash
# HCU wrapper for the official Generator Vision Super LoRA SFT recipe.
# One launcher covers short validation and longer runs through env overrides.

set -euo pipefail

: "${TOML_FILE:=examples/toml/sft_config/vision_sft_super.toml}"
: "${DATASET_PATH:=/data/Cosmos/datasets/BridgeData2-Subset-Synthetic-Captions/sft_dataset_bridge}"
: "${BASE_CHECKPOINT_PATH:=/public4/opendas/DL_DATA/llm-models/Cosmos3/Cosmos3-Super-DCP}"
: "${WAN_VAE_PATH:=/public4/opendas/DL_DATA/llm-models/Cosmos3/checkpoints/wan22_vae/Wan2.2_VAE.pth}"
: "${LOCAL_PROCESSOR_DIR:=/public4/opendas/DL_DATA/llm-models/Cosmos3/Cosmos3-Super-Text2Image}"
: "${RUN_NAME:=generator_vision_sft_super_hcu_perf_8card}"
: "${OUTPUT_ROOT:=/root/Cosmos3/outputs/hcu_training}"
: "${LOG_FILENAME:=${RUN_NAME}.log}"
: "${HIP_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"
: "${MASTER_PORT:=29714}"
: "${MAX_ITER:=70}"
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

# Vision Super fixes context parallelism at two ranks, so WORLD_SIZE must be
# divisible by two. The remaining ranks form the auto-sized FSDP shard mesh.
if (( NPROC_PER_NODE < 2 || NPROC_PER_NODE % 2 != 0 )); then
    echo "ERROR: Vision Super requires an even HCU count >= 2 for context parallelism=2; got ${NPROC_PER_NODE}" >&2
    exit 2
fi

export COSMOS_TRAINING="${COSMOS_TRAINING:-true}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HIP_VISIBLE_DEVICES
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PERF_RECORD_ENABLED PERF_WARMUP_ITERS
PERF_PLATFORM="hcu"
PERF_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES"

EXTRA_DATASET_CHECK='[[ -f "$DATASET_PATH/train/video_dataset_file.jsonl" ]] || { echo "ERROR: missing $DATASET_PATH/train/video_dataset_file.jsonl" >&2; exit 1; }
[[ -f "$BASE_CHECKPOINT_PATH/checkpoint.json" ]] || { echo "ERROR: missing $BASE_CHECKPOINT_PATH/checkpoint.json" >&2; exit 1; }
compgen -G "$BASE_CHECKPOINT_PATH/model/*.distcp" >/dev/null || { echo "ERROR: no DCP shards under $BASE_CHECKPOINT_PATH/model" >&2; exit 1; }
[[ -d "$LOCAL_PROCESSOR_DIR" ]] || { echo "ERROR: missing local processor directory $LOCAL_PROCESSOR_DIR" >&2; exit 1; }'

TAIL_OVERRIDES=(
    "job.wandb_mode=disabled"
    "job.name=${RUN_NAME}"
    "model.config.vlm_config.tokenizer.local_processor_dir=${LOCAL_PROCESSOR_DIR}"
    "trainer.seed=42"
    "trainer.max_iter=${MAX_ITER}"
    "checkpoint.save_iter=${CHECKPOINT_SAVE_ITER}"
    "checkpoint.dcp_async_mode_enabled=${DCP_ASYNC_MODE_ENABLED}"
)
TAIL_OVERRIDES+=("$@")

echo ">>> HCU devices: ${HIP_VISIBLE_DEVICES}"
echo ">>> Processes per node: ${NPROC_PER_NODE}"
echo ">>> Context parallel ranks: 2"
echo ">>> Max iterations: ${MAX_ITER}"
echo ">>> Async DCP save: ${DCP_ASYNC_MODE_ENABLED}"

# Do not clear LD_LIBRARY_PATH here: the HCU/DTK runtime depends on its login
# shell dynamic-library paths. The CUDA launcher clears it for a different
# host-library isolation requirement.
source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
