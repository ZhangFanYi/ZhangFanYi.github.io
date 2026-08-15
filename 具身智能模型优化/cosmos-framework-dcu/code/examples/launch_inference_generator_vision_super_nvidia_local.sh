#!/usr/bin/env bash
# NVIDIA local wrapper for the official Generator Super Framework inference CLI.
# Resource resolution and inference logic are delegated to the Nano wrapper;
# only the model/processor defaults differ.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${INFERENCE_PROFILE:=smoke}"
case "$INFERENCE_PROFILE" in
    smoke)
        : "${INPUT_FILE:=inputs/omni/t2i.json}"
        : "${OUTPUT_DIR:=outputs/inference_generator_vision_super_nvidia_smoke}"
        ;;
    quality-t2v)
        : "${INPUT_FILE:=inputs/omni/t2v.json}"
        : "${OUTPUT_DIR:=outputs/inference_generator_vision_super_nvidia_quality_t2v}"
        ;;
    official-t2v)
        : "${INPUT_FILE:=inputs/omni/t2v.json}"
        : "${OUTPUT_DIR:=outputs/inference_generator_vision_super_nvidia_official_t2v}"
        ;;
    official-i2v)
        : "${INPUT_FILE:=inputs/omni/i2v_local_nvidia.json}"
        : "${OUTPUT_DIR:=outputs/inference_generator_vision_super_nvidia_official_i2v}"
        ;;
    *)
        echo "ERROR: INFERENCE_PROFILE must be smoke, quality-t2v, official-t2v, or official-i2v: $INFERENCE_PROFILE" >&2
        exit 2
        ;;
esac

: "${CHECKPOINT_PATH:=/public/opendas/DL_DATA/llm-models/Cosmos3/Cosmos-3-Super}"
: "${PROCESSOR_DIR:=/public/opendas/DL_DATA/llm-models/Cosmos3/Cosmos-3-Super}"
: "${WAN_VAE_PATH:=/public/opendas/DL_DATA/llm-models/Cosmos3/checkpoints/wan22_vae/Wan2.2_VAE.pth}"
: "${COSMOS_SMOKE:=$([[ "$INFERENCE_PROFILE" == smoke ]] && echo 1 || echo 0)}"

export INFERENCE_PROFILE INPUT_FILE OUTPUT_DIR CHECKPOINT_PATH PROCESSOR_DIR WAN_VAE_PATH COSMOS_SMOKE
exec bash "$SCRIPT_DIR/launch_inference_generator_vision_nano_nvidia_local.sh" "$@"
