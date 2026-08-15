#!/usr/bin/env bash
# HCU wrapper for the official Cosmos3-Nano LIBERO-10 Action Policy SFT recipe.

set -euo pipefail

: "${TOML_FILE:=examples/toml/sft_config/action_policy_libero_10_nano.toml}"
: "${LIBERO_ROOT:=/root/Cosmos3/datasets/LIBERO_LeRobot_v3/libero_10}"
: "${BASE_CHECKPOINT_PATH:=/public4/opendas/DL_DATA/llm-models/Cosmos3/checkpoints/Cosmos3-Nano-DCP}"
: "${WAN_VAE_PATH:=/public4/opendas/DL_DATA/llm-models/Cosmos3/checkpoints/wan22_vae/Wan2.2_VAE.pth}"
: "${LOCAL_PROCESSOR_DIR:=/public4/opendas/DL_DATA/llm-models/Cosmos3/Cosmos3-Nano}"
: "${RUN_NAME:=action_policy_libero_10_nano_hcu_perf_8card}"
: "${OUTPUT_ROOT:=/data/Cosmos/cosmos/outputs/hcu_training}"
: "${LOG_FILENAME:=${RUN_NAME}.log}"
: "${HIP_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"
: "${MASTER_PORT:=29717}"
: "${MAX_ITER:=70}"
: "${MAX_SAMPLES_PER_BATCH:=32}"
: "${NUM_WORKERS:=4}"
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
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HIP_VISIBLE_DEVICES
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export LIBERO_ROOT
export PERF_RECORD_ENABLED PERF_WARMUP_ITERS
PERF_PLATFORM="hcu"
PERF_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES"

EXTRA_DATASET_CHECK='[[ -f "$LIBERO_ROOT/meta/info.json" ]] || { echo "ERROR: missing $LIBERO_ROOT/meta/info.json; download the libero_10 subset of nvidia/LIBERO_LeRobot_v3" >&2; exit 1; }
[[ -f "$BASE_CHECKPOINT_PATH/checkpoint.json" ]] || { echo "ERROR: missing $BASE_CHECKPOINT_PATH/checkpoint.json" >&2; exit 1; }
compgen -G "$BASE_CHECKPOINT_PATH/model/*.distcp" >/dev/null || { echo "ERROR: no DCP shards under $BASE_CHECKPOINT_PATH/model" >&2; exit 1; }
[[ -d "$LOCAL_PROCESSOR_DIR" ]] || { echo "ERROR: missing local processor directory $LOCAL_PROCESSOR_DIR" >&2; exit 1; }'

TAIL_OVERRIDES=(
    "job.wandb_mode=disabled"
    "job.name=${RUN_NAME}"
    "model.config.vlm_config.tokenizer.local_processor_dir=${LOCAL_PROCESSOR_DIR}"
    "model.config.parallelism.data_parallel_shard_degree=${NPROC_PER_NODE}"
    "model.config.parallelism.data_parallel_replicate_degree=1"
    "dataloader_train.max_samples_per_batch=${MAX_SAMPLES_PER_BATCH}"
    "dataloader_train.dataloader.num_workers=${NUM_WORKERS}"
    "trainer.seed=42"
    "trainer.max_iter=${MAX_ITER}"
    "checkpoint.save_iter=${CHECKPOINT_SAVE_ITER}"
    "checkpoint.dcp_async_mode_enabled=${DCP_ASYNC_MODE_ENABLED}"
)
TAIL_OVERRIDES+=("$@")

echo ">>> HCU devices: ${HIP_VISIBLE_DEVICES}"
echo ">>> Processes per node / FSDP shard degree: ${NPROC_PER_NODE}"
echo ">>> Max iterations: ${MAX_ITER}"
echo ">>> Max samples per packed batch: ${MAX_SAMPLES_PER_BATCH}"
echo ">>> Async DCP save: ${DCP_ASYNC_MODE_ENABLED}"

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
