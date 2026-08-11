#!/bin/bash
set -euo pipefail

# Usage:
#   bash datadown.sh [data_root]
# Example:
#   bash datadown.sh /data/dk_test/lingbot-vla-dcu-data

DATA_ROOT=${1:-/data/dk_test/lingbot-vla-dcu-data}
MODEL_ROOT="${DATA_ROOT}/models"
DATASET_ROOT="${DATA_ROOT}/datasets"
HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

PI05_REPO=lerobot/pi05_base
PI05_REVISION=7de663972b7817d2c4cf2d84c821153dfea772e9
TOKENIZER_REPO=leo009/paligemma-3b-pt-224
TOKENIZER_REVISION=39996beb6fb17c5d16a50d3ef8f7a96ad9d03986
DATASET_REPO=lerobot/pusht
DATASET_REVISION=7628202a2180972f291ba1bc6723834921e72c19

case "${DATA_ROOT}" in
  /data/dk_test/*) ;;
  *) echo "[ERROR] data_root must be under /data/dk_test: ${DATA_ROOT}" >&2; exit 2 ;;
esac

command -v hf >/dev/null 2>&1 || {
  echo "[ERROR] hf CLI is missing; run this script inside the delivered image" >&2
  exit 2
}

artifact_ready() {
  local kind=$1
  local path=$2
  case "${kind}" in
    model)
      [[ -f "${path}/config.json" ]] \
        && find "${path}" -type f -name '*.safetensors' -print -quit | grep -q .
      ;;
    tokenizer)
      [[ -f "${path}/config.json" && -f "${path}/tokenizer_config.json" ]]
      ;;
    dataset)
      { [[ -f "${path}/meta/info.json" ]] || [[ -f "${path}/meta_data/info.json" ]]; } \
        && find "${path}" -type f \( -name '*.parquet' -o -name '*.mp4' \) -print -quit | grep -q .
      ;;
  esac
}

download_one() {
  local label=$1
  local kind=$2
  local repo_type=$3
  local repo=$4
  local revision=$5
  local destination=$6
  local partial="${destination}.partial"

  if artifact_ready "${kind}" "${destination}"; then
    echo "[SKIP] ${label} already exists: ${destination}"
    return
  fi
  [[ ! -e "${destination}" ]] || {
    echo "[ERROR] ${label} path exists but is incomplete: ${destination}" >&2
    exit 2
  }
  [[ ! -e "${partial}" ]] || {
    echo "[ERROR] incomplete staging directory exists: ${partial}" >&2
    exit 2
  }

  mkdir -p "$(dirname "${destination}")" "${partial}"
  echo "[DOWNLOAD] ${repo}@${revision}"
  args=(
    hf download "${repo}"
    --repo-type "${repo_type}"
    --revision "${revision}"
    --local-dir "${partial}"
  )
  if [[ "${kind}" == tokenizer ]]; then
    args+=(--include '*.json' '*.txt' '*.model' '*.tiktoken')
  fi
  HF_ENDPOINT="${HF_ENDPOINT}" "${args[@]}"

  artifact_ready "${kind}" "${partial}" || {
    echo "[ERROR] ${label} download is incomplete: ${partial}" >&2
    exit 2
  }
  {
    echo "repo_type=${repo_type}"
    echo "repo=${repo}"
    echo "revision=${revision}"
  } > "${partial}/.download-complete"
  mv "${partial}" "${destination}"
  echo "[READY] ${label}: ${destination}"
}

mkdir -p "${MODEL_ROOT}" "${DATASET_ROOT}"

download_one pi05 model model \
  "${PI05_REPO}" "${PI05_REVISION}" "${MODEL_ROOT}/pi05_base"
download_one tokenizer tokenizer model \
  "${TOKENIZER_REPO}" "${TOKENIZER_REVISION}" "${MODEL_ROOT}/paligemma-3b-pt-224"
download_one pusht dataset dataset \
  "${DATASET_REPO}" "${DATASET_REVISION}" "${DATASET_ROOT}/pusht"

echo "[DONE] pi0.5 model, tokenizer and dataset are ready under ${DATA_ROOT}"
