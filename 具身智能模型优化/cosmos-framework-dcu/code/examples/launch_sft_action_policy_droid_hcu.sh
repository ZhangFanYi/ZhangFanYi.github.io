#!/usr/bin/env bash
# HCU wrapper for the official Cosmos3-Nano DROID Action Policy SFT recipe.
# One launcher covers short validation and longer runs through env overrides.

set -euo pipefail

: "${TOML_FILE:=examples/toml/sft_config/action_policy_droid_nano.toml}"
: "${DROID_ROOT:=/root/Cosmos3/datasets/Cosmos3-DROID}"
: "${BASE_CHECKPOINT_PATH:=/public4/opendas/DL_DATA/llm-models/Cosmos3/checkpoints/Cosmos3-Nano-DCP}"
: "${WAN_VAE_PATH:=/public4/opendas/DL_DATA/llm-models/Cosmos3/checkpoints/wan22_vae/Wan2.2_VAE.pth}"
: "${LOCAL_PROCESSOR_DIR:=/public4/opendas/DL_DATA/llm-models/Cosmos3/Cosmos3-Nano}"
: "${RUN_NAME:=action_policy_droid_nano_hcu_perf_8card}"
: "${OUTPUT_ROOT:=/data/Cosmos/cosmos/outputs/hcu_training}"
: "${LOG_FILENAME:=${RUN_NAME}.log}"
: "${HIP_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"
: "${MASTER_PORT:=29716}"
: "${MAX_ITER:=70}"
: "${MAX_SAMPLES_PER_BATCH:=32}"
: "${NUM_WORKERS:=16}"
: "${COMPILE_TOKENIZER_ENABLED:=true}"
: "${CHECKPOINT_SAVE_ITER:=1000}"
: "${DCP_ASYNC_MODE_ENABLED:=false}"
: "${KEEP_RANGES_PATH:=}"
: "${VIDEO_BACKEND:=pyav}"
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
export DROID_ROOT
export PERF_RECORD_ENABLED PERF_WARMUP_ITERS
PERF_PLATFORM="hcu"
PERF_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES"

EXTRA_DATASET_CHECK='[[ -f "$DROID_ROOT/success/meta/info.json" ]] || { echo "ERROR: missing $DROID_ROOT/success/meta/info.json; prepare the Cosmos3-DROID success split first" >&2; exit 1; }
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
    "model.config.rectified_flow_training_config.loss_scale=10.0"
    "dataloader_train.dataloader.datasets.droid.dataset.video_backend=${VIDEO_BACKEND}"
    "trainer.seed=42"
    "trainer.max_iter=${MAX_ITER}"
    "trainer.callbacks.compile_tokenizer.enabled=${COMPILE_TOKENIZER_ENABLED}"
    "checkpoint.save_iter=${CHECKPOINT_SAVE_ITER}"
    "checkpoint.dcp_async_mode_enabled=${DCP_ASYNC_MODE_ENABLED}"
)
if [[ -n "$KEEP_RANGES_PATH" ]]; then
    [[ -f "$KEEP_RANGES_PATH" ]] || { echo "ERROR: KEEP_RANGES_PATH not found: $KEEP_RANGES_PATH" >&2; exit 1; }
    TAIL_OVERRIDES+=(
        "dataloader_train.dataloader.datasets.droid.dataset.use_filter_dict=true"
        "dataloader_train.dataloader.datasets.droid.dataset.filter_dict_path=${KEEP_RANGES_PATH}"
    )
fi
TAIL_OVERRIDES+=("$@")

echo ">>> HCU devices: ${HIP_VISIBLE_DEVICES}"
echo ">>> Processes per node / FSDP shard degree: ${NPROC_PER_NODE}"
echo ">>> Max iterations: ${MAX_ITER}"
echo ">>> Max samples per packed batch: ${MAX_SAMPLES_PER_BATCH}"
echo ">>> Async DCP save: ${DCP_ASYNC_MODE_ENABLED}"
echo ">>> Video backend: ${VIDEO_BACKEND}"
if [[ -n "$KEEP_RANGES_PATH" ]]; then
    echo ">>> DROID keep-ranges filter: ${KEEP_RANGES_PATH}"
fi

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
