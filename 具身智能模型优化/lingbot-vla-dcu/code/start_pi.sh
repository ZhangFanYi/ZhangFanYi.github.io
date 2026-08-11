#!/bin/bash
set -euo pipefail

# Usage:
#   DEVICES=7 bash start_pi.sh [data_root] [output_dir]
#   DEVICES=0,1,2,3,4,5,6,7 bash start_pi.sh

DEVICES=${DEVICES:-7}
DATA_ROOT=${1:-/data/dk_test/lingbot-vla-dcu-data}
RUN_ID=${RUN_ID:-pi05_$(date +%Y%m%d_%H%M%S)}
OUTPUT_DIR=${2:-/data/dk_test/lingbot-vla-dcu-runs/${RUN_ID}}

BATCH_SIZE=${BATCH_SIZE:-32}
STEPS=${STEPS:-20000}
LOG_FREQ=${LOG_FREQ:-10}
WARMUP_STEPS=${WARMUP_STEPS:-20}
NUM_WORKERS=${NUM_WORKERS:-4}
OPTIMIZER_LR=${OPTIMIZER_LR:-1e-4}
SAVE_CHECKPOINT=${SAVE_CHECKPOINT:-true}
SAVE_FREQ=${SAVE_FREQ:-10000}
TOLERANCE_S=${TOLERANCE_S:-5e-4}

MODEL_PATH="${DATA_ROOT}/models/pi05_base"
TOKENIZER_PATH="${DATA_ROOT}/models/paligemma-3b-pt-224"
DATASET_PATH="${DATA_ROOT}/datasets/pusht"

case "${DATA_ROOT}" in
  /data/dk_test/*) ;;
  *) echo "[ERROR] data_root must be under /data/dk_test" >&2; exit 2 ;;
esac
case "${OUTPUT_DIR}" in
  /data/dk_test/*) ;;
  *) echo "[ERROR] output_dir must be under /data/dk_test" >&2; exit 2 ;;
esac
[[ "${DEVICES}" =~ ^[0-9]+(,[0-9]+)*$ ]] || {
  echo "[ERROR] DEVICES must look like 7 or 0,1,2,3" >&2
  exit 2
}
IFS=',' read -r -a DEVICE_LIST <<< "${DEVICES}"
WORLD_SIZE=${#DEVICE_LIST[@]}
GLOBAL_BATCH=$((BATCH_SIZE * WORLD_SIZE))

for path in "${MODEL_PATH}" "${TOKENIZER_PATH}" "${DATASET_PATH}"; do
  [[ -d "${path}" ]] || {
    echo "[ERROR] missing input: ${path}; run datadown.sh first" >&2
    exit 2
  }
done
[[ ! -e "${OUTPUT_DIR}" ]] || {
  echo "[ERROR] output already exists: ${OUTPUT_DIR}" >&2
  exit 2
}
mkdir -p "${OUTPUT_DIR}/model-view"

# Keep checkpoint files read-only. Rewrite only processor JSON so the tokenizer
# resolves to the local fixed-revision directory.
for file in "${MODEL_PATH}"/*; do
  name=$(basename "${file}")
  case "${name}" in
    policy_preprocessor.json|policy_postprocessor.json) continue ;;
  esac
  ln -s "${file}" "${OUTPUT_DIR}/model-view/${name}"
done

python - \
  "${MODEL_PATH}/policy_preprocessor.json" \
  "${OUTPUT_DIR}/model-view/policy_preprocessor.json" \
  "${MODEL_PATH}/policy_postprocessor.json" \
  "${OUTPUT_DIR}/model-view/policy_postprocessor.json" \
  "${TOKENIZER_PATH}" <<'PY'
import json
import sys
from pathlib import Path

pre_src, pre_dst, post_src, post_dst, tokenizer = map(Path, sys.argv[1:])

def rewrite(source, destination, expected_tokenizers):
    payload = json.loads(source.read_text())
    steps = []
    found = 0
    for step in payload.get("steps", []):
        name = step.get("registry_name")
        if name in {"relative_actions_processor", "absolute_actions_processor"}:
            if step.get("config", {}).get("enabled", False):
                raise SystemExit(f"unsupported enabled processor: {name}")
            continue
        if name == "tokenizer_processor":
            step.setdefault("config", {})["tokenizer_name"] = str(tokenizer)
            found += 1
        steps.append(step)
    if found != expected_tokenizers:
        raise SystemExit(f"{source}: expected {expected_tokenizers} tokenizer step, found {found}")
    payload["steps"] = steps
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

rewrite(pre_src, pre_dst, 1)
rewrite(post_src, post_dst, 0)
PY

export HIP_VISIBLE_DEVICES="${DEVICES}"
export CUDA_VISIBLE_DEVICES="${DEVICES}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

ACCELERATE_ARGS=(
  accelerate launch
  --num_machines=1
  --num_processes="${WORLD_SIZE}"
  --gpu_ids="${DEVICES}"
  --mixed_precision=no
  --dynamo_backend=no
)
[[ "${WORLD_SIZE}" -eq 1 ]] || ACCELERATE_ARGS+=(--multi_gpu)

TRAIN_ARGS=(
  --dataset.repo_id=lerobot/pusht
  --dataset.root="${DATASET_PATH}"
  --dataset.tolerance_s="${TOLERANCE_S}"
  --rename_map='{}'
  --policy.optimizer_lr="${OPTIMIZER_LR}"
  --policy.push_to_hub=false
  --wandb.enable=false
  --batch_size="${BATCH_SIZE}"
  --log_freq="${LOG_FREQ}"
  --num_workers="${NUM_WORKERS}"
  --policy.device=cuda
  --steps="${STEPS}"
  --policy.type=pi05
  --policy.pretrained_path="${OUTPUT_DIR}/model-view"
  --policy.gradient_checkpointing=true
  --policy.dtype=bfloat16
  --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}'
  --save_checkpoint="${SAVE_CHECKPOINT}"
  --save_freq="${SAVE_FREQ}"
  --eval_freq=0
  --output_dir="${OUTPUT_DIR}/training_output"
)

{
  echo "devices=${DEVICES}"
  echo "world_size=${WORLD_SIZE}"
  echo "batch_size_per_device=${BATCH_SIZE}"
  echo "global_batch=${GLOBAL_BATCH}"
  echo "steps=${STEPS}"
  echo "warmup_steps=${WARMUP_STEPS}"
  echo "tolerance_s=${TOLERANCE_S}"
} > "${OUTPUT_DIR}/run-manifest.txt"

echo "[START] pi0.5 devices=${DEVICES} global_batch=${GLOBAL_BATCH}"
echo "[LOG] ${OUTPUT_DIR}/train.log"
set +e
"${ACCELERATE_ARGS[@]}" "$(command -v lerobot-train)" "${TRAIN_ARGS[@]}" \
  2>&1 | tee "${OUTPUT_DIR}/train.log"
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

# Throughput only uses logged windows whose ending step is after warmup.
python - "${OUTPUT_DIR}/train.log" "${OUTPUT_DIR}/summary.json" \
  "${GLOBAL_BATCH}" "${WARMUP_STEPS}" <<'PY'
import json
import re
import statistics
import sys
from pathlib import Path

log_path, output_path = map(Path, sys.argv[1:3])
global_batch, warmup = map(int, sys.argv[3:])
pattern = re.compile(
    r"\bstep:(?P<step>\d+)\b.*?\bupdt_s:(?P<update>\d+(?:\.\d+)?)\b.*?"
    r"\bdata_s:(?P<data>\d+(?:\.\d+)?)\b"
)
windows = [
    (int(m.group("step")), float(m.group("update")) + float(m.group("data")))
    for m in pattern.finditer(log_path.read_text(errors="ignore"))
]
stable = [(step, seconds) for step, seconds in windows if step > warmup]
payload = {
    "selection": "logged windows with ending step > warmup_steps",
    "warmup_steps": warmup,
    "parsed_windows": len(windows),
    "stable_windows": len(stable),
    "global_batch": global_batch,
}
if stable:
    times = [seconds for _, seconds in stable]
    median = statistics.median(times)
    payload.update({
        "status": "PASS",
        "stable_step_range": [stable[0][0], stable[-1][0]],
        "median_step_s": median,
        "stable_median_samples_per_s": global_batch / median,
    })
else:
    payload["status"] = "NO_STABLE_WINDOWS"
output_path.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload))
PY

SUMMARY_STATUS=$(python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
  "${OUTPUT_DIR}/summary.json")
echo "summary_status=${SUMMARY_STATUS}" >> "${OUTPUT_DIR}/run-status.txt"
