#!/bin/bash
set -euo pipefail

# Usage:
#   bash datadown.sh [data_root]
#   LOCAL_DATASET_SRC=/path/to/aloha_mobile_cabinet bash datadown.sh /data/dk_test/lerobot-act-dcu-data
#   HF_DATASET_REPO=org/repo bash datadown.sh /data/dk_test/lerobot-act-dcu-data

DATA_ROOT=${1:-/data/dk_test/lerobot-act-dcu-data}
MODEL_ROOT="${DATA_ROOT}/models"
DATASET_ROOT="${DATA_ROOT}/datasets"
TORCH_HOME_DIR="${MODEL_ROOT}/torch"
TORCH_CKPT_DIR="${TORCH_HOME_DIR}/hub/checkpoints"
DATASET_NAME=${DATASET_NAME:-aloha_mobile_cabinet}
DATASET_DEST="${DATASET_ROOT}/${DATASET_NAME}"

RESNET18_URL=${RESNET18_URL:-https://download.pytorch.org/models/resnet18-f37072fd.pth}
RESNET18_SHA256=${RESNET18_SHA256:-f37072fd}
HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

case "${DATA_ROOT}" in
  /data/dk_test/*) ;;
  *) echo "[ERROR] data_root must be under /data/dk_test: ${DATA_ROOT}" >&2; exit 2 ;;
esac

mkdir -p "${MODEL_ROOT}" "${DATASET_ROOT}" "${TORCH_CKPT_DIR}"

download_resnet18() {
  local dst="${TORCH_CKPT_DIR}/resnet18-f37072fd.pth"
  if [[ -s "${dst}" ]]; then
    echo "[SKIP] ResNet18 checkpoint exists: ${dst}"
    return
  fi
  echo "[DOWNLOAD] ResNet18 backbone checkpoint"
  wget -O "${dst}.partial" --tries=3 --timeout=30 "${RESNET18_URL}"
  if [[ "${RESNET18_SHA256}" != "skip" ]]; then
    sha256sum "${dst}.partial" | grep -q "${RESNET18_SHA256}" || {
      echo "[ERROR] ResNet18 sha256 prefix mismatch" >&2
      rm -f "${dst}.partial"
      exit 2
    }
  fi
  mv "${dst}.partial" "${dst}"
  echo "[READY] ${dst}"
}

prepare_dataset() {
  if [[ -d "${DATASET_DEST}" && -f "${DATASET_DEST}/meta/info.json" ]]; then
    echo "[SKIP] Dataset exists: ${DATASET_DEST}"
    return
  fi

  if [[ -n "${LOCAL_DATASET_SRC:-}" ]]; then
    [[ -d "${LOCAL_DATASET_SRC}" ]] || {
      echo "[ERROR] LOCAL_DATASET_SRC does not exist: ${LOCAL_DATASET_SRC}" >&2
      exit 2
    }
    echo "[COPY] ${LOCAL_DATASET_SRC} -> ${DATASET_DEST}"
    mkdir -p "${DATASET_ROOT}"
    cp -a "${LOCAL_DATASET_SRC}" "${DATASET_DEST}.partial"
    mv "${DATASET_DEST}.partial" "${DATASET_DEST}"
    echo "[READY] ${DATASET_DEST}"
    return
  fi

  if [[ -n "${HF_DATASET_REPO:-}" ]]; then
    command -v hf >/dev/null 2>&1 || {
      echo "[ERROR] hf CLI is missing; install huggingface_hub or run inside the image" >&2
      exit 2
    }
    echo "[DOWNLOAD] dataset ${HF_DATASET_REPO}"
    HF_ENDPOINT="${HF_ENDPOINT}" hf download "${HF_DATASET_REPO}" \
      --repo-type dataset \
      --local-dir "${DATASET_DEST}.partial"
    mv "${DATASET_DEST}.partial" "${DATASET_DEST}"
    echo "[READY] ${DATASET_DEST}"
    return
  fi

  echo "[WARN] Dataset is not prepared."
  echo "       Set LOCAL_DATASET_SRC=/path/to/aloha_mobile_cabinet or HF_DATASET_REPO=org/repo."
}

download_resnet18
prepare_dataset

cat > "${DATA_ROOT}/env.sh" <<EOF
export TORCH_HOME=${TORCH_HOME_DIR}
export HF_ENDPOINT=${HF_ENDPOINT}
EOF

echo "[DONE] artifacts root: ${DATA_ROOT}"
echo "[INFO] source ${DATA_ROOT}/env.sh before training, or use start_act.sh which sets TORCH_HOME automatically."
