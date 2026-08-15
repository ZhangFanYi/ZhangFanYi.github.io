#!/usr/bin/env bash
# Backward-compatible alias for the aligned NVIDIA-local performance launcher.

set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/launch_sft_generator_vision_nano_nvidia_local.sh" "$@"
