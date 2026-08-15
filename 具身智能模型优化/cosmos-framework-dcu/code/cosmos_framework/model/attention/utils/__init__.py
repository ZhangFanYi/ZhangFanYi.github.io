# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
Imaginaire4 Attention Subpackage:
Unified implementation for all Attention implementations.

Utilities: compute capability detection, helpers, and more.
"""

from typing import Any

import torch

from cosmos_framework.model.attention.utils.determinism import torch_deterministic_mode
from cosmos_framework.model.attention.utils.environment import is_torch_compiling
from cosmos_framework.model.attention.utils.safe_ops import log


def get_arch_tag(device: torch.device | None = None) -> int:
    """
    Returns the compute capability of a given torch CUDA/HIP device, otherwise returns 0.

    HCU uses the torch.cuda API on top of HIP, so torch.version.cuda is None even
    when a CUDA-like accelerator is available. Keep the NVIDIA path unchanged and
    allow HIP-backed devices to report an architecture tag for backend selection.
    """
    is_cuda_like = torch.cuda.is_available() and (torch.version.cuda or getattr(torch.version, "hip", None))
    if is_cuda_like and (device is None or device.type == "cuda"):
        try:
            major, minor = torch.cuda.get_device_capability(device)
            arch_tag = major * 10 + minor
            if arch_tag > 0:
                return arch_tag
        except Exception as e:
            log.debug(f"Could not query CUDA/HIP device capability: {e}")

        if getattr(torch.version, "hip", None):
            return 90

    return 0


def log_or_raise_error(msg: str, raise_error: bool = False, exception: Any = RuntimeError):
    if raise_error:
        raise exception(msg)
    else:
        log.debug(msg)


def is_full(dtype: torch.dtype) -> bool:
    return dtype == torch.float32


def is_half(dtype: torch.dtype) -> bool:
    return dtype in [torch.float16, torch.bfloat16]


def is_fp8(dtype: torch.dtype) -> bool:
    return dtype in [torch.float8_e5m2, torch.float8_e4m3fn]


def is_hopper(device: torch.device | None = None) -> bool:
    return get_arch_tag(device) == 90


def is_blackwell_dc(device: torch.device | None = None) -> bool:
    return get_arch_tag(device) in [100, 103]


__all__ = [
    "get_arch_tag",
    "log_or_raise_error",
    "is_full",
    "is_half",
    "is_fp8",
    "is_hopper",
    "is_blackwell_dc",
    "is_torch_compiling",
    "torch_deterministic_mode",
]
