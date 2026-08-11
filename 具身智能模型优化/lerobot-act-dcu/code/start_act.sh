#!/bin/bash
set -euo pipefail

# Usage:
#   DEVICES=7 bash start_act.sh [data_root] [output_dir]
#   DEVICES=0,1,2,3,4,5,6,7 BATCH_SIZE=11 bash start_act.sh

DEVICES=${DEVICES:-7}
DATA_ROOT=${1:-/data/dk_test/lerobot-act-dcu-data}
RUN_ID=${RUN_ID:-act_$(date +%Y%m%d_%H%M%S)}
OUTPUT_DIR=${2:-/data/dk_test/lerobot-act-dcu-runs/${RUN_ID}}

BATCH_SIZE=${BATCH_SIZE:-86}
STEPS=${STEPS:-1000}
SAVE_FREQ=${SAVE_FREQ:-500}
LOG_FREQ=${LOG_FREQ:-20}
NUM_WORKERS=${NUM_WORKERS:-4}
VIDEO_BACKEND=${VIDEO_BACKEND:-torchcodec}
DATASET_NAME=${DATASET_NAME:-aloha_mobile_cabinet}
DATASET_REPO_ID=${DATASET_REPO_ID:-local/aloha_mobile_cabinet}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29501}

DATASET_PATH="${DATA_ROOT}/datasets/${DATASET_NAME}"
TORCH_HOME_DIR="${DATA_ROOT}/models/torch"

case "${DATA_ROOT}" in
  /data/dk_test/*) ;;
  *) echo "[ERROR] data_root must be under /data/dk_test: ${DATA_ROOT}" >&2; exit 2 ;;
esac
case "${OUTPUT_DIR}" in
  /data/dk_test/*) ;;
  *) echo "[ERROR] output_dir must be under /data/dk_test: ${OUTPUT_DIR}" >&2; exit 2 ;;
esac
[[ "${DEVICES}" =~ ^[0-9]+(,[0-9]+)*$ ]] || {
  echo "[ERROR] DEVICES must look like 7 or 0,1,2,3" >&2
  exit 2
}

IFS=',' read -r -a DEVICE_LIST <<< "${DEVICES}"
WORLD_SIZE=${#DEVICE_LIST[@]}
GLOBAL_BATCH=$((BATCH_SIZE * WORLD_SIZE))

[[ -d "${DATASET_PATH}" ]] || {
  echo "[ERROR] missing dataset: ${DATASET_PATH}; run datadown.sh first" >&2
  exit 2
}
[[ -s "${TORCH_HOME_DIR}/hub/checkpoints/resnet18-f37072fd.pth" ]] || {
  echo "[ERROR] missing ResNet18 checkpoint under ${TORCH_HOME_DIR}; run datadown.sh first" >&2
  exit 2
}
[[ ! -e "${OUTPUT_DIR}" ]] || {
  echo "[ERROR] output already exists: ${OUTPUT_DIR}" >&2
  exit 2
}
mkdir -p "${OUTPUT_DIR}"

export HIP_VISIBLE_DEVICES="${DEVICES}"
export CUDA_VISIBLE_DEVICES="${DEVICES}"
export TORCH_HOME="${TORCH_HOME_DIR}"
export TOKENIZERS_PARALLELISM=false

export MIOPEN_FIND_MODE="${MIOPEN_FIND_MODE:-1}"
export MIOPEN_PRECISION_FP32_FP32_FP32_TF32_FP32="${MIOPEN_PRECISION_FP32_FP32_FP32_TF32_FP32:-1}"
export PYTORCH_MIOPEN_SUGGEST_NHWC="${PYTORCH_MIOPEN_SUGGEST_NHWC:-1}"

ACCELERATE_ARGS=(
  accelerate launch
  --num_machines=1
  --num_processes="${WORLD_SIZE}"
  --main_process_port="${MAIN_PROCESS_PORT}"
)
[[ "${WORLD_SIZE}" -eq 1 ]] || ACCELERATE_ARGS+=(--multi_gpu)

TRAIN_ARGS=(
  -m lerobot.scripts.lerobot_train
  --policy.type=act
  --policy.device=cuda
  --policy.push_to_hub=false
  --wandb.enable=false
  --dataset.repo_id="${DATASET_REPO_ID}"
  --dataset.root="${DATASET_PATH}"
  --dataset.video_backend="${VIDEO_BACKEND}"
  --output_dir="${OUTPUT_DIR}/training_output"
  --batch_size="${BATCH_SIZE}"
  --steps="${STEPS}"
  --save_freq="${SAVE_FREQ}"
  --log_freq="${LOG_FREQ}"
  --num_workers="${NUM_WORKERS}"
)

{
  echo "devices=${DEVICES}"
  echo "world_size=${WORLD_SIZE}"
  echo "batch_size_per_device=${BATCH_SIZE}"
  echo "global_batch=${GLOBAL_BATCH}"
  echo "steps=${STEPS}"
  echo "dataset_path=${DATASET_PATH}"
  echo "video_backend=${VIDEO_BACKEND}"
  echo "torch_home=${TORCH_HOME}"
} > "${OUTPUT_DIR}/run-manifest.txt"

echo "[START] ACT devices=${DEVICES} global_batch=${GLOBAL_BATCH}"
echo "[LOG] ${OUTPUT_DIR}/train.log"
set +e
"${ACCELERATE_ARGS[@]}" "${TRAIN_ARGS[@]}" 2>&1 | tee "${OUTPUT_DIR}/train.log"
TRAIN_RC=${PIPESTATUS[0]}
set -e
echo "training_exit_status=${TRAIN_RC}" > "${OUTPUT_DIR}/run-status.txt"
[[ "${TRAIN_RC}" -eq 0 ]] || exit "${TRAIN_RC}"

EXPECTED_BATCH="Effective batch size: ${BATCH_SIZE} x ${WORLD_SIZE} = ${GLOBAL_BATCH}"
grep -Fq "${EXPECTED_BATCH}" "${OUTPUT_DIR}/train.log" || {
  echo "summary_status=FAILED_BATCH_ACCOUNTING" >> "${OUTPUT_DIR}/run-status.txt"
  echo "[ERROR] expected log line not found: ${EXPECTED_BATCH}" >&2
  exit 2
}
echo "summary_status=PASS" >> "${OUTPUT_DIR}/run-status.txt"
