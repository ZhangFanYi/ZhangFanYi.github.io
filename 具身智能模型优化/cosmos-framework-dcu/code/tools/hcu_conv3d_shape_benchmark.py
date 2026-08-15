#!/usr/bin/env python3
"""Benchmark the Conv3D path observed in the Cosmos3 HCU profiler trace.

The representative trace call has:

* raw causal input: [1, 160, 68, 128, 128]
* steady-state cache: [1, 160, 2, 128, 128]
* padded convolution input: [1, 160, 70, 130, 130]
* weight: [160, 160, 3, 3, 3]
* bias: [160]
* dtype: BF16

The three default modes separate the backend convolution cost from the
causal padding and cache concatenation cost.  Run on one idle HCU GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F


MODES = ("backend_conv", "causal_no_cache", "causal_cache")
RAW_SHAPE = (1, 160, 68, 128, 128)
CACHE_SHAPE = (1, 160, 2, 128, 128)
PADDED_SHAPE = (1, 160, 70, 130, 130)
WEIGHT_SHAPE = (160, 160, 3, 3, 3)
BIAS_SHAPE = (160,)
CAUSAL_PAD = (1, 1, 1, 1, 2, 0)
SPATIAL_PAD = (1, 1, 1, 1, 0, 0)


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _metadata(device: torch.device) -> dict[str, Any]:
    props = torch.cuda.get_device_properties(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "device_index": device.index,
        "device_name": torch.cuda.get_device_name(device),
        "gcn_arch_name": getattr(props, "gcnArchName", None),
        "total_memory_bytes": int(total_bytes),
        "free_memory_bytes_at_start": int(free_bytes),
        "torch_version": torch.__version__,
        "torch_git_version": getattr(torch.version, "git_version", None),
        "hip_version": getattr(torch.version, "hip", None),
        "cuda_version": getattr(torch.version, "cuda", None),
        "python_version": platform.python_version(),
        "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "git_revision": _git_revision(),
    }


def _make_tensors(device: torch.device) -> dict[str, torch.Tensor]:
    dtype = torch.bfloat16
    return {
        "x": torch.randn(RAW_SHAPE, device=device, dtype=dtype),
        "cache": torch.randn(CACHE_SHAPE, device=device, dtype=dtype),
        "weight": torch.randn(WEIGHT_SHAPE, device=device, dtype=dtype),
        "bias": torch.randn(BIAS_SHAPE, device=device, dtype=dtype),
    }


def _make_call(mode: str, tensors: dict[str, torch.Tensor]) -> tuple[Callable[[], torch.Tensor], dict[str, Any]]:
    x = tensors["x"]
    cache = tensors["cache"]
    weight = tensors["weight"]
    bias = tensors["bias"]

    if mode == "backend_conv":
        padded = F.pad(x, CAUSAL_PAD)
        assert tuple(padded.shape) == PADDED_SHAPE

        def call() -> torch.Tensor:
            return F.conv3d(padded, weight, bias, stride=1, padding=0, dilation=1, groups=1)

        details = {
            "description": "F.conv3d on the already padded input; excludes pad/cat cost",
            "conv_input_shape": list(padded.shape),
            "conv_input_stride": list(padded.stride()),
            "padding_in_call": [0, 0, 0],
        }
    elif mode == "causal_no_cache":

        def call() -> torch.Tensor:
            padded = F.pad(x, CAUSAL_PAD)
            return F.conv3d(padded, weight, bias, stride=1, padding=0, dilation=1, groups=1)

        details = {
            "description": "CausalConv3d-like path with temporal/spatial F.pad and no cache",
            "raw_input_shape": list(x.shape),
            "raw_input_stride": list(x.stride()),
            "pad": list(CAUSAL_PAD),
        }
    elif mode == "causal_cache":

        def call() -> torch.Tensor:
            cached_input = torch.cat((cache, x), dim=2)
            padded = F.pad(cached_input, SPATIAL_PAD)
            return F.conv3d(padded, weight, bias, stride=1, padding=0, dilation=1, groups=1)

        details = {
            "description": "Steady-state CausalConv3d-like path with cache cat and spatial F.pad",
            "raw_input_shape": list(x.shape),
            "cache_shape": list(cache.shape),
            "raw_input_stride": list(x.stride()),
            "cache_stride": list(cache.stride()),
            "pad_after_cache_cat": list(SPATIAL_PAD),
        }
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    details.update(
        {
            "mode": mode,
            "dtype": str(weight.dtype),
            "weight_shape": list(weight.shape),
            "weight_stride": list(weight.stride()),
            "bias_shape": list(bias.shape),
            "output_shape": [1, 160, 68, 128, 128],
        }
    )
    return call, details


def _measure(
    call: Callable[[], torch.Tensor],
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
    runs: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        output = call()
    torch.cuda.synchronize(device)

    gpu_times: list[float] = []
    host_times: list[float] = []
    checksum: float | None = None
    timing_mode = "cuda_event"

    try:
        for _ in range(runs):
            torch.cuda.synchronize(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            host_start = time.perf_counter()
            start.record()
            for _ in range(repeats):
                output = call()
            end.record()
            host_end = time.perf_counter()
            end.synchronize()
            gpu_times.append(start.elapsed_time(end) / repeats)
            host_times.append((host_end - host_start) * 1000.0 / repeats)
            checksum = float(output.float().mean().item())
    except RuntimeError as exc:
        if "Event" not in str(exc) and "event" not in str(exc):
            raise
        timing_mode = "synchronized_host_timer"
        gpu_times.clear()
        host_times.clear()
        for _ in range(runs):
            torch.cuda.synchronize(device)
            host_start = time.perf_counter()
            for _ in range(repeats):
                output = call()
            torch.cuda.synchronize(device)
            host_end = time.perf_counter()
            elapsed = (host_end - host_start) * 1000.0 / repeats
            gpu_times.append(elapsed)
            host_times.append(elapsed)
            checksum = float(output.float().mean().item())

    return {
        "timing_mode": timing_mode,
        "gpu_ms_per_run": gpu_times,
        "host_ms_per_run": host_times,
        "gpu_ms_mean": sum(gpu_times) / len(gpu_times),
        "gpu_ms_min": min(gpu_times),
        "gpu_ms_max": max(gpu_times),
        "gpu_ms_median": sorted(gpu_times)[len(gpu_times) // 2],
        "host_ms_mean": sum(host_times) / len(host_times),
        "checksum": checksum,
    }


def _run_mode(
    mode: str,
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
    runs: int,
) -> dict[str, Any]:
    tensors = _make_tensors(device)
    call, details = _make_call(mode, tensors)
    result = _measure(
        call,
        device=device,
        warmup=warmup,
        repeats=repeats,
        runs=runs,
    )
    details.update(
        {
            "warmup": warmup,
            "repeats_per_run": repeats,
            "runs": runs,
            "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }
    )
    details.update(result)
    del tensors, call
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    return details


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("hcu_conv3d_shape_benchmark.json"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: torch.cuda is not available; run inside the HCU/DTK container.", file=sys.stderr)
        return 2

    torch.set_grad_enabled(False)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    metadata = _metadata(device)
    metadata.update(
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "modes": args.modes,
            "raw_shape": list(RAW_SHAPE),
            "cache_shape": list(CACHE_SHAPE),
            "padded_shape": list(PADDED_SHAPE),
            "weight_shape": list(WEIGHT_SHAPE),
            "bias_shape": list(BIAS_SHAPE),
        }
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))

    records: list[dict[str, Any]] = []
    for mode in args.modes:
        print(f"\nRUN mode={mode}", flush=True)
        torch.cuda.reset_peak_memory_stats(device)
        try:
            record = _run_mode(
                mode,
                device=device,
                warmup=args.warmup,
                repeats=args.repeats,
                runs=args.runs,
            )
            records.append(record)
            print(
                "RESULT "
                f"mode={mode} "
                f"gpu_ms_mean={record['gpu_ms_mean']:.4f} "
                f"gpu_ms_median={record['gpu_ms_median']:.4f} "
                f"gpu_ms_min={record['gpu_ms_min']:.4f} "
                f"gpu_ms_max={record['gpu_ms_max']:.4f} "
                f"peak_reserved_gib={record['peak_memory_reserved_bytes'] / 2**30:.3f}",
                flush=True,
            )
        except Exception as exc:  # keep the failed mode in the output
            error = {"mode": mode, "error_type": type(exc).__name__, "error": str(exc)}
            records.append(error)
            print(f"ERROR mode={mode}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            torch.cuda.empty_cache()

    payload = {"metadata": metadata, "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nWROTE {args.output}")
    return 0 if any("gpu_ms_mean" in record for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
