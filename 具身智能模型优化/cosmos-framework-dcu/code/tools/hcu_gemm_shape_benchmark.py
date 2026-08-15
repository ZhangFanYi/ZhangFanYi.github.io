#!/usr/bin/env python3
"""Benchmark the BF16 GEMM shapes observed in a Cosmos3 HCU trace.

The default cases reproduce the layouts seen in rank0_trace.json.gz:

* mm_up:     [M, 4096] @ [4096, 12288]
* mm_down:   [M, 12288] @ [12288, 4096]
* addmm_down: bias + [M, 12288] @ [12288, 4096]

Run this as a standalone, single-GPU test.  It does not initialize
torch.distributed and does not load the Cosmos3 model.
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


DEFAULT_M_VALUES = (17335, 18470, 25344, 27648)
CASE_NAMES = ("mm_up", "mm_down", "addmm_down")


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _device_metadata(device: torch.device) -> dict[str, Any]:
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


def _make_case(
    case_name: str,
    m: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[tuple[torch.Tensor, ...], Callable[..., torch.Tensor], dict[str, Any]]:
    if case_name == "mm_up":
        k, n = 4096, 12288
        a = torch.randn((m, k), device=device, dtype=dtype)
        # Construct [K, N] with stride [1, K], matching the trace's weight layout.
        b = torch.randn((n, k), device=device, dtype=dtype).transpose(0, 1)
        tensors = (a, b)
        op = torch.mm
        metadata = {
            "operation": "torch.mm",
            "input_dims": [[m, k], [k, n]],
            "output_dims": [m, n],
        }
    elif case_name == "mm_down":
        k, n = 12288, 4096
        a = torch.randn((m, k), device=device, dtype=dtype)
        b = torch.randn((n, k), device=device, dtype=dtype).transpose(0, 1)
        tensors = (a, b)
        op = torch.mm
        metadata = {
            "operation": "torch.mm",
            "input_dims": [[m, k], [k, n]],
            "output_dims": [m, n],
        }
    elif case_name == "addmm_down":
        k, n = 12288, 4096
        bias = torch.randn((m, n), device=device, dtype=dtype)
        a = torch.randn((m, k), device=device, dtype=dtype)
        b = torch.randn((n, k), device=device, dtype=dtype).transpose(0, 1)
        tensors = (bias, a, b)
        op = torch.addmm
        metadata = {
            "operation": "torch.addmm",
            "input_dims": [[m, n], [m, k], [k, n]],
            "output_dims": [m, n],
        }
    else:
        raise ValueError(f"Unsupported case: {case_name}")

    metadata.update(
        {
            "case": case_name,
            "m": m,
            "dtype": str(dtype),
            "input_strides": [list(t.stride()) for t in tensors],
            "input_numel": [t.numel() for t in tensors],
        }
    )
    return tensors, op, metadata


def _call_factory(
    op: Callable[..., torch.Tensor],
    tensors: tuple[torch.Tensor, ...],
    *,
    use_compile: bool,
) -> tuple[Callable[[], torch.Tensor], str]:
    def eager_call() -> torch.Tensor:
        return op(*tensors)

    if not use_compile:
        return eager_call, "eager"

    def compiled_fn(*args: torch.Tensor) -> torch.Tensor:
        return op(*args)

    compiled = torch.compile(compiled_fn, dynamic=False)

    def compiled_call() -> torch.Tensor:
        return compiled(*tensors)

    return compiled_call, "torch.compile(dynamic=False)"


def _measure(
    call: Callable[[], torch.Tensor],
    *,
    warmup: int,
    repeats: int,
    runs: int,
    device: torch.device,
) -> dict[str, Any]:
    for _ in range(warmup):
        output = call()
    torch.cuda.synchronize(device)

    gpu_ms_per_op: list[float] = []
    host_ms_per_op: list[float] = []
    checksum: float | None = None
    timing_mode = "cuda_event"

    try:
        for _ in range(runs):
            torch.cuda.synchronize(device)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            host_start = time.perf_counter()
            start_event.record()
            for _ in range(repeats):
                output = call()
            end_event.record()
            host_end = time.perf_counter()
            end_event.synchronize()
            gpu_ms_per_op.append(start_event.elapsed_time(end_event) / repeats)
            host_ms_per_op.append((host_end - host_start) * 1000.0 / repeats)
            checksum = float(output.float().mean().item())
    except RuntimeError as exc:
        if "Event" not in str(exc) and "event" not in str(exc):
            raise
        timing_mode = "synchronized_host_timer"
        gpu_ms_per_op.clear()
        host_ms_per_op.clear()
        for _ in range(runs):
            torch.cuda.synchronize(device)
            host_start = time.perf_counter()
            for _ in range(repeats):
                output = call()
            torch.cuda.synchronize(device)
            host_end = time.perf_counter()
            elapsed = (host_end - host_start) * 1000.0 / repeats
            gpu_ms_per_op.append(elapsed)
            host_ms_per_op.append(elapsed)
            checksum = float(output.float().mean().item())

    return {
        "timing_mode": timing_mode,
        "gpu_ms_per_op": gpu_ms_per_op,
        "host_ms_per_op": host_ms_per_op,
        "gpu_ms_mean": sum(gpu_ms_per_op) / len(gpu_ms_per_op),
        "gpu_ms_min": min(gpu_ms_per_op),
        "gpu_ms_max": max(gpu_ms_per_op),
        "gpu_ms_median": sorted(gpu_ms_per_op)[len(gpu_ms_per_op) // 2],
        "host_ms_mean": sum(host_ms_per_op) / len(host_ms_per_op),
        "checksum": checksum,
    }


def _run_one(
    case_name: str,
    m: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    warmup: int,
    repeats: int,
    runs: int,
    use_compile: bool,
) -> dict[str, Any]:
    tensors, op, metadata = _make_case(case_name, m, device=device, dtype=dtype)
    call, execution_mode = _call_factory(op, tensors, use_compile=use_compile)

    # Compile/warm up outside the measured runs.  This also makes the first
    # measured result representative of steady state rather than compilation.
    result = _measure(
        call,
        warmup=warmup,
        repeats=repeats,
        runs=runs,
        device=device,
    )
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    metadata.update(
        {
            "execution_mode": execution_mode,
            "warmup": warmup,
            "repeats_per_run": repeats,
            "runs": runs,
            "peak_memory_allocated_bytes": peak_allocated,
            "peak_memory_reserved_bytes": peak_reserved,
        }
    )
    metadata.update(result)

    del tensors, call, op
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    return metadata


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=CASE_NAMES, default=list(CASE_NAMES))
    parser.add_argument("--m-values", nargs="+", type=int, default=list(DEFAULT_M_VALUES))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--compile", action="store_true", help="also benchmark torch.compile(dynamic=False)")
    parser.add_argument("--output", type=Path, default=Path("hcu_gemm_shape_benchmark.json"))
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not torch.cuda.is_available():
        print("ERROR: torch.cuda is not available; run this inside the HCU/DTK container.", file=sys.stderr)
        return 2

    torch.set_grad_enabled(False)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    dtype = torch.bfloat16
    metadata = _device_metadata(device)
    metadata.update(
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "cases": args.cases,
            "m_values": args.m_values,
            "dtype": str(dtype),
        }
    )

    print(json.dumps(metadata, indent=2, sort_keys=True))
    records: list[dict[str, Any]] = []
    for case_name in args.cases:
        for m in args.m_values:
            print(f"\nRUN case={case_name} M={m} compile={args.compile}", flush=True)
            torch.cuda.reset_peak_memory_stats(device)
            try:
                record = _run_one(
                    case_name,
                    m,
                    device=device,
                    dtype=dtype,
                    warmup=args.warmup,
                    repeats=args.repeats,
                    runs=args.runs,
                    use_compile=args.compile,
                )
                records.append(record)
                print(
                    "RESULT "
                    f"case={case_name} M={m} "
                    f"gpu_ms_mean={record['gpu_ms_mean']:.4f} "
                    f"gpu_ms_median={record['gpu_ms_median']:.4f} "
                    f"gpu_ms_min={record['gpu_ms_min']:.4f} "
                    f"gpu_ms_max={record['gpu_ms_max']:.4f} "
                    f"peak_reserved_gib={record['peak_memory_reserved_bytes'] / 2**30:.3f}",
                    flush=True,
                )
            except Exception as exc:  # keep the failed shape visible in the JSON output
                error = {
                    "case": case_name,
                    "m": m,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                records.append(error)
                print(f"ERROR case={case_name} M={m}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
                torch.cuda.empty_cache()

    payload = {"metadata": metadata, "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nWROTE {args.output}")
    return 0 if any("gpu_ms_mean" in record for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
