#!/usr/bin/env bash
# NVIDIA local wrapper around the official Cosmos Framework Generator inference CLI.
# Defaults are a one-image smoke. ``official-t2v`` preserves the Framework
# runtime defaults for a full official T2V request; paths remain local overrides.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$WORKDIR"

: "${CUDA_VISIBLE_DEVICES:=0}"
: "${MASTER_PORT:=30730}"
: "${INFERENCE_PROFILE:=smoke}"
case "$INFERENCE_PROFILE" in
    smoke)
        DEFAULT_INPUT_FILE=inputs/omni/t2i.json
        DEFAULT_OUTPUT_DIR=outputs/inference_generator_vision_nano_nvidia_smoke
        DEFAULT_PARALLELISM_PRESET=latency
        DEFAULT_RESOLUTION=256
        DEFAULT_NUM_FRAMES=1
        DEFAULT_GUARDRAILS=false
        ;;
    quality-t2v)
        DEFAULT_INPUT_FILE=inputs/omni/t2v.json
        DEFAULT_OUTPUT_DIR=outputs/inference_generator_vision_nano_nvidia_quality_t2v
        DEFAULT_PARALLELISM_PRESET=throughput
        DEFAULT_RESOLUTION=720
        DEFAULT_NUM_FRAMES=189
        DEFAULT_GUARDRAILS=false
        ;;
    official-t2v)
        DEFAULT_INPUT_FILE=inputs/omni/t2v.json
        DEFAULT_OUTPUT_DIR=outputs/inference_generator_vision_nano_nvidia_official_t2v
        DEFAULT_PARALLELISM_PRESET=latency
        DEFAULT_RESOLUTION=720
        DEFAULT_NUM_FRAMES=189
        DEFAULT_GUARDRAILS=true
        ;;
    official-i2v)
        DEFAULT_INPUT_FILE=inputs/omni/i2v_local_nvidia.json
        DEFAULT_OUTPUT_DIR=outputs/inference_generator_vision_nano_nvidia_official_i2v
        DEFAULT_PARALLELISM_PRESET=latency
        DEFAULT_RESOLUTION=720
        DEFAULT_NUM_FRAMES=189
        DEFAULT_GUARDRAILS=true
        ;;
    *)
        echo "ERROR: INFERENCE_PROFILE must be smoke, quality-t2v, official-t2v, or official-i2v: $INFERENCE_PROFILE" >&2
        exit 2
        ;;
esac
: "${INPUT_FILE:=$DEFAULT_INPUT_FILE}"
: "${OUTPUT_DIR:=$DEFAULT_OUTPUT_DIR}"
: "${CHECKPOINT_PATH:=/public/opendas/DL_DATA/llm-models/Cosmos3/checkpoints/Cosmos3-Nano-DCP}"
: "${WAN_VAE_PATH:=/public/opendas/DL_DATA/llm-models/Cosmos3/checkpoints/wan22_vae/Wan2.2_VAE.pth}"
: "${PROCESSOR_DIR:=/public/opendas/DL_DATA/llm-models/Cosmos3/Cosmos-3-Nano}"
: "${PARALLELISM_PRESET:=$DEFAULT_PARALLELISM_PRESET}"
: "${SEED:=0}"
: "${RESOLUTION:=$DEFAULT_RESOLUTION}"
: "${NUM_FRAMES:=$DEFAULT_NUM_FRAMES}"
: "${GUARDRAILS:=$DEFAULT_GUARDRAILS}"
: "${BENCHMARK:=true}"
: "${HF_HUB_OFFLINE:=1}"
: "${LOG_FILE:=${OUTPUT_DIR%/}/inference.log}"
: "${LOCAL_ASSET_LAUNCHER:=$WORKDIR/tools/run_framework_inference_local_assets.py}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES//[[:space:]]/}"
if [[ -z "$CUDA_VISIBLE_DEVICES" || "$CUDA_VISIBLE_DEVICES" == ,* || "$CUDA_VISIBLE_DEVICES" == *, || "$CUDA_VISIBLE_DEVICES" == *,,* ]]; then
    echo "ERROR: CUDA_VISIBLE_DEVICES must be a comma-separated list: $CUDA_VISIBLE_DEVICES" >&2
    exit 2
fi
IFS=',' read -r -a NVIDIA_DEVICE_IDS <<< "$CUDA_VISIBLE_DEVICES"
NPROC_PER_NODE=${#NVIDIA_DEVICE_IDS[@]}

[[ -f "$INPUT_FILE" ]] || { echo "ERROR: input file not found: $INPUT_FILE" >&2; exit 1; }
[[ -f "$WAN_VAE_PATH" ]] || { echo "ERROR: VAE file not found: $WAN_VAE_PATH" >&2; exit 1; }
[[ -d "$PROCESSOR_DIR" ]] || { echo "ERROR: processor directory not found: $PROCESSOR_DIR" >&2; exit 1; }
if [[ -d "$CHECKPOINT_PATH/model" ]]; then
    compgen -G "$CHECKPOINT_PATH/model/*.distcp" >/dev/null || { echo "ERROR: no DCP shards under $CHECKPOINT_PATH/model" >&2; exit 1; }
elif compgen -G "$CHECKPOINT_PATH/*.distcp" >/dev/null; then
    :
elif [[ -f "$CHECKPOINT_PATH/config.json" ]]; then
    :
else
    echo "ERROR: checkpoint must be a local DCP directory or a local HF directory with config.json: $CHECKPOINT_PATH" >&2
    exit 1
fi
OFFLINE_MODE=false
case "${HF_HUB_OFFLINE,,}" in
    1|true|yes) OFFLINE_MODE=true ;;
esac
if [[ "$GUARDRAILS" == "true" && "$OFFLINE_MODE" == "true" ]]; then
    QWEN3GUARD_LOCAL_PATH="${COSMOS_QWEN3GUARD_PATH:-${COSMOS_HCU_QWEN3GUARD_DIR:-}}"
    [[ -n "${COSMOS_GUARDRAIL1_PATH:-}" ]] || {
        echo "ERROR: GUARDRAILS=true with HF_HUB_OFFLINE=$HF_HUB_OFFLINE requires COSMOS_GUARDRAIL1_PATH" >&2
        exit 2
    }
    [[ -d "$COSMOS_GUARDRAIL1_PATH" ]] || {
        echo "ERROR: COSMOS_GUARDRAIL1_PATH is not a directory: $COSMOS_GUARDRAIL1_PATH" >&2
        exit 2
    }
    [[ -n "$QWEN3GUARD_LOCAL_PATH" ]] || {
        echo "ERROR: GUARDRAILS=true with HF_HUB_OFFLINE=$HF_HUB_OFFLINE requires COSMOS_QWEN3GUARD_PATH" >&2
        exit 2
    }
    [[ -d "$QWEN3GUARD_LOCAL_PATH" ]] || {
        echo "ERROR: COSMOS_QWEN3GUARD_PATH is not a directory: $QWEN3GUARD_LOCAL_PATH" >&2
        exit 2
    }
fi
export COSMOS_TRAINING="${COSMOS_TRAINING:-false}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export CUDA_VISIBLE_DEVICES
export WAN_VAE_PATH PROCESSOR_DIR
if [[ -z "${COSMOS_SMOKE+x}" ]]; then
    if [[ "$INFERENCE_PROFILE" == "smoke" ]]; then COSMOS_SMOKE=1; else COSMOS_SMOKE=0; fi
fi
export COSMOS_SMOKE
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
# Prefer the current PyTorch allocator variable while accepting a caller's
# deprecated spelling as input for this process.
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"
unset PYTORCH_CUDA_ALLOC_CONF
export PYTHONPATH="$WORKDIR/examples/inference_py310_compat:$WORKDIR${PYTHONPATH:+:$PYTHONPATH}"
if [[ -n "${NVIDIA_LD_LIBRARY_PATH:-}" ]]; then
    export LD_LIBRARY_PATH="$NVIDIA_LD_LIBRARY_PATH"
fi
if [[ "$GUARDRAILS" == "true" ]]; then
    if ! python -c 'import better_profanity, nltk, sentencepiece, google.protobuf; from retinaface.data import cfg_re50'; then
        echo "ERROR: GUARDRAILS=true requires the Framework guardrail Python extras in the interpreter used by torchrun: better-profanity, nltk, protobuf, retinaface-py, sentencepiece. retinaface-py does not require TensorFlow." >&2
        exit 2
    fi
fi

INFERENCE_ARGS=(
    -i "$INPUT_FILE"
    -o "$OUTPUT_DIR"
    --checkpoint-path "$CHECKPOINT_PATH"
    --parallelism-preset="$PARALLELISM_PRESET"
    --seed="$SEED"
    --resolution="$RESOLUTION"
    --num-frames="$NUM_FRAMES"
)
if [[ "$GUARDRAILS" == "true" ]]; then INFERENCE_ARGS+=(--guardrails); else INFERENCE_ARGS+=(--no-guardrails); fi
# Keep the official Omni parallelism planner in control.  In current Framework
# builds, FSDP sharding and CP/CFGP are overlay meshes: a multi-GPU latency run
# must retain its default dp_shard_size=WORLD_SIZE instead of replicating the
# model with an explicit dp_shard_size=1.
[[ "$BENCHMARK" == "true" ]] && INFERENCE_ARGS+=(--benchmark)
INFERENCE_ARGS+=("$@")

[[ -f "$LOCAL_ASSET_LAUNCHER" ]] || { echo "ERROR: local asset launcher not found: $LOCAL_ASSET_LAUNCHER" >&2; exit 1; }
INFERENCE_ENTRY=("$LOCAL_ASSET_LAUNCHER")

mkdir -p "$(dirname -- "$LOG_FILE")"
echo ">>> workdir: $WORKDIR"
echo ">>> NVIDIA devices: $CUDA_VISIBLE_DEVICES"
echo ">>> profile: $INFERENCE_PROFILE"
echo ">>> parallelism: $PARALLELISM_PRESET"
echo ">>> guardrails: $GUARDRAILS | torch_compile: official CLI default | weights: regular checkpoint"
echo ">>> HF_HUB_OFFLINE: $HF_HUB_OFFLINE"
echo ">>> local asset shim: $LOCAL_ASSET_LAUNCHER"
[[ -n "${COSMOS_GUARDRAIL1_PATH:-}" ]] && echo ">>> Guardrail1 path: $COSMOS_GUARDRAIL1_PATH"
[[ -n "${COSMOS_QWEN3GUARD_PATH:-}" ]] && echo ">>> Qwen3Guard path: $COSMOS_QWEN3GUARD_PATH"
echo ">>> checkpoint: $CHECKPOINT_PATH"
echo ">>> input: $INPUT_FILE"
echo ">>> output: $OUTPUT_DIR"
echo ">>> log: $LOG_FILE"

set +e
torchrun --nproc_per_node="$NPROC_PER_NODE" --master_port="$MASTER_PORT" "${INFERENCE_ENTRY[@]}" "${INFERENCE_ARGS[@]}" 2>&1 | tee "$LOG_FILE"
EXIT_CODE=${PIPESTATUS[0]}
set -e
echo ">>> inference exit: $EXIT_CODE"
exit "$EXIT_CODE"
