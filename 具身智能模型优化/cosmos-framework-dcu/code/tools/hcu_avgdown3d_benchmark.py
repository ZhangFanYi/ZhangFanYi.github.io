#!/usr/bin/env python3
"""Benchmark an AvgDown3D reshape/reduce against a pixel-unshuffle variant.

The candidate is intentionally kept in this standalone benchmark first.  It
must pass the output-error check and show a stable GPU-time improvement before
it is considered for the training VAE path.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone

import torch
import torch.nn as nn
import torch.nn.functional as F

from cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16 import AvgDown3D


class PixelUnshuffleAvgDown3D(nn.Module):
    """Use a 2-D pixel-unshuffle primitive for spatial rearrangement."""

    def __init__(self, in_channels: int, out_channels: int, factor_t: int, factor_s: int = 1):
        super().__init__()
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = factor_t * factor_s * factor_s
        if in_channels * self.factor % out_channels:
            raise ValueError("in_channels * factor must be divisible by out_channels")
        self.group_size = in_channels * self.factor // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_t = (self.factor_t - x.shape[2] % self.factor_t) % self.factor_t
        if pad_t:
            x = F.pad(x, (0, 0, 0, 0, pad_t, 0))
        b, c, t, h, w = x.shape
        if self.factor_s > 1:
            # pixel_unshuffle is a native 4-D rearrangement; run it over all
            # frames at once, then restore the 5-D VAE layout.
            x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            x = F.pixel_unshuffle(x, self.factor_s)
            x = x.view(b, t, c * self.factor_s * self.factor_s, h // self.factor_s, w // self.factor_s)
            x = x.permute(0, 2, 1, 3, 4)
        b, c_with_space, t, h, w = x.shape
        if self.factor_t > 1:
            x = x.reshape(
                b,
                c_with_space // (self.factor_s * self.factor_s),
                self.factor_s * self.factor_s,
                t // self.factor_t,
                self.factor_t,
                h,
                w,
            )
            x = x.permute(0, 1, 4, 2, 3, 5, 6).reshape(
                b, c_with_space * self.factor_t, t // self.factor_t, h, w
            )
        return x.view(b, self.out_channels, self.group_size, x.shape[2], h, w).mean(dim=2)


def _sync() -> None:
    torch.cuda.synchronize()


def _measure(module: nn.Module, x: torch.Tensor, warmup: int, repeat: int) -> float:
    with torch.no_grad():
        for _ in range(warmup):
            module(x)
        _sync()
        start = time.perf_counter()
        for _ in range(repeat):
            module(x)
        _sync()
        return (time.perf_counter() - start) * 1000.0 / repeat


def _device_name() -> str:
    try:
        return torch.cuda.get_device_name(torch.cuda.current_device())
    except Exception:
        return "unknown"


def _run_case(
    device: torch.device,
    shape: tuple[int, int, int, int, int],
    out_channels: int,
    factor_t: int,
    factor_s: int,
    dtype: torch.dtype,
    warmup: int,
    repeat: int,
    channels_last: bool,
) -> dict[str, object]:
    x = torch.randn(shape, device=device, dtype=dtype)
    if channels_last:
        x = x.to(memory_format=torch.channels_last_3d)
    eager = AvgDown3D(shape[1], out_channels, factor_t=factor_t, factor_s=factor_s).to(device)
    candidate = PixelUnshuffleAvgDown3D(
        shape[1], out_channels, factor_t=factor_t, factor_s=factor_s
    ).to(device)
    with torch.no_grad():
        eager_out = eager(x)
        candidate_out = candidate(x)
    diff = (eager_out.float() - candidate_out.float()).abs()
    reference = eager_out.float().abs().mean().item()
    eager_ms = _measure(eager, x, warmup, repeat)
    candidate_ms = _measure(candidate, x, warmup, repeat)
    return {
        "shape": list(shape),
        "out_channels": out_channels,
        "factor_t": factor_t,
        "factor_s": factor_s,
        "channels_last_3d": channels_last,
        "eager_ms": eager_ms,
        "candidate_ms": candidate_ms,
        "speedup_percent": (eager_ms / candidate_ms - 1.0) * 100.0,
        "max_abs_diff": diff.max().item(),
        "mean_abs_diff": diff.mean().item(),
        "relative_mean_abs_diff": diff.mean().item() / max(reference, 1e-12),
        "output_shape": list(eager_out.shape),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/tmp/hcu_avgdown3d_benchmark.json")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA/HIP device is required")
    device = torch.device("cuda", torch.cuda.current_device())
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    # These cases cover the three AvgDown3D transitions in the Wan VAE:
    # [160 -> 320, spatial], [320 -> 640, temporal + spatial], and
    # [640 -> 640, temporal].
    cases = [
        ((1, 160, 68, 128, 128), 320, 1, 2),
        ((1, 320, 34, 64, 64), 640, 2, 2),
        ((1, 640, 17, 32, 32), 640, 2, 1),
    ]
    results = []
    for shape, out_channels, factor_t, factor_s in cases:
        result = _run_case(
            device,
            shape,
            out_channels,
            factor_t,
            factor_s,
            dtype,
            args.warmup,
            args.repeat,
            args.channels_last,
        )
        results.append(result)
        print(
            "RESULT"
            f" shape={shape} out={out_channels} ft={factor_t} fs={factor_s}"
            f" eager_ms={result['eager_ms']:.4f}"
            f" candidate_ms={result['candidate_ms']:.4f}"
            f" speedup_percent={result['speedup_percent']:.2f}"
            f" max_abs_diff={result['max_abs_diff']:.6g}"
            f" rel_mean_abs_diff={result['relative_mean_abs_diff']:.6g}"
        )

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "device_name": _device_name(),
        "device_index": torch.cuda.current_device(),
        "dtype": str(dtype),
        "channels_last_3d": args.channels_last,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "cases": results,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"WROTE {args.output}")


if __name__ == "__main__":
    main()
