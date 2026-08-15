# Shared NVIDIA single-node defaults for local-asset performance launchers.

: "${CUDA_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"
: "${VLM_TOKENIZER_PATH:=/public/opendas/DL_DATA/llm-models/Qwen3-VL-8B-Instruct}"
: "${PERF_RECORD_ENABLED:=true}"
: "${PERF_WARMUP_ITERS:=10}"

[[ -f "$VLM_TOKENIZER_PATH/vocab.json" ]] || { echo "ERROR: missing $VLM_TOKENIZER_PATH/vocab.json" >&2; exit 1; }
[[ -f "$VLM_TOKENIZER_PATH/merges.txt" ]] || { echo "ERROR: missing $VLM_TOKENIZER_PATH/merges.txt" >&2; exit 1; }

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES//[[:space:]]/}"
if [[ -z "$CUDA_VISIBLE_DEVICES" || "$CUDA_VISIBLE_DEVICES" == ,* || "$CUDA_VISIBLE_DEVICES" == *, || "$CUDA_VISIBLE_DEVICES" == *,,* ]]; then
    echo "ERROR: CUDA_VISIBLE_DEVICES must be a comma-separated list without empty entries: $CUDA_VISIBLE_DEVICES" >&2
    exit 2
fi
IFS=',' read -r -a NVIDIA_DEVICE_IDS <<< "$CUDA_VISIBLE_DEVICES"
NPROC_PER_NODE=${#NVIDIA_DEVICE_IDS[@]}

export CUDA_VISIBLE_DEVICES
export COSMOS_TRAINING="${COSMOS_TRAINING:-true}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PERF_RECORD_ENABLED PERF_WARMUP_ITERS
# The official CUDA environment recommends clearing host CUDA/NCCL paths.
export LD_LIBRARY_PATH="${NVIDIA_LD_LIBRARY_PATH:-}"

PERF_PLATFORM="${PERF_PLATFORM:-nvidia_h20}"
PERF_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"

echo ">>> NVIDIA devices: ${CUDA_VISIBLE_DEVICES}"
echo ">>> Processes per node: ${NPROC_PER_NODE}"
