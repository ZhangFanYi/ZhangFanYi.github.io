#!/usr/bin/env bash
# NVIDIA/H20 local-asset mirror of the HCU DROID Action Policy run.

set -euo pipefail

: "${TOML_FILE:=examples/toml/sft_config/action_policy_droid_nano.toml}"
: "${DROID_ROOT:=/data/datasets/droid_plus_lerobot_640x360_20260412}"
: "${BASE_CHECKPOINT_PATH:=/public/opendas/DL_DATA/llm-models/Cosmos3/checkpoints/Cosmos3-Nano-DCP}"
: "${WAN_VAE_PATH:=/public/opendas/DL_DATA/llm-models/Cosmos3/checkpoints/wan22_vae/Wan2.2_VAE.pth}"
: "${RUN_NAME:=action_policy_droid_nano_nvidia_h20_perf_8card}"
: "${OUTPUT_ROOT:=/data/outputs/nvidia_training}"
: "${LOG_FILENAME:=${RUN_NAME}.log}"
: "${MASTER_PORT:=30716}"
: "${MAX_ITER:=70}"
: "${MAX_SAMPLES_PER_BATCH:=32}"
: "${NUM_WORKERS:=16}"
: "${COMPILE_TOKENIZER_ENABLED:=true}"
: "${CHECKPOINT_SAVE_ITER:=1000}"
: "${DCP_ASYNC_MODE_ENABLED:=false}"
: "${KEEP_RANGES_PATH:=}"
: "${VIDEO_BACKEND:=torchcodec}"

export DROID_ROOT
source "$(dirname "${BASH_SOURCE[0]}")/_sft_nvidia_local_common.sh"
EXTRA_DATASET_CHECK='[[ -f "$DROID_ROOT/success/meta/info.json" ]] || { echo "ERROR: missing $DROID_ROOT/success/meta/info.json" >&2; exit 1; }
[[ -f "$BASE_CHECKPOINT_PATH/checkpoint.json" ]] || { echo "ERROR: missing $BASE_CHECKPOINT_PATH/checkpoint.json" >&2; exit 1; }
compgen -G "$BASE_CHECKPOINT_PATH/model/*.distcp" >/dev/null || { echo "ERROR: no DCP shards under $BASE_CHECKPOINT_PATH/model" >&2; exit 1; }
'
TAIL_OVERRIDES=(
    "job.wandb_mode=disabled"
    "job.name=${RUN_NAME}"
    "model.config.vlm_config.tokenizer.pretrained_model_name=${VLM_TOKENIZER_PATH}"
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

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
