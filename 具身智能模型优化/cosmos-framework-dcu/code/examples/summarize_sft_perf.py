#!/usr/bin/env python3
"""Extract a compact, machine-readable SFT performance summary from a log."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


INITIAL_ITER_RE = re.compile(r"Iteration\s+(\d+)(?:/\d+)?.*?Time:\s*([0-9.]+)s")
STEADY_ITER_RE = re.compile(r"(?:^|\])\s*(\d+)\s*:\s*iter_speed\s+([0-9.]+)\s+seconds per iteration")
LOSS_RE = re.compile(r"train/loss_avg:\s*([-+0-9.eE]+)\s*\(iteration\s+(\d+)\)")
ACTION_LOSS_RE = re.compile(r"Iteration\s+(\d+):.*?Loss:\s*([-+0-9.eE]+)")
GRAD_RE = re.compile(r"clip_grad_norm/(?:[^:\s]+/)*global:\s*([-+0-9.eE]+)\s*\(iteration\s+(\d+)\)")
TOKENS_RE = re.compile(
    r"(\d+)\s*:\s*tokens_per_sec_per_gpu\s+([0-9.]+)\s*\|\s*"
    r"useful_tokens_per_sec_per_gpu\s+([0-9.]+)\s*\|\s*"
    r"useful_supervised_tokens_per_sec_per_gpu\s+([0-9.]+)\s*\|\s*"
    r"tokens_per_step\s+([0-9.]+)\s*\|\s*samples_per_step\s+([0-9.]+)"
)
PEAK_MEM_RE = re.compile(r"peak_mem_gb\s+([0-9.]+)")
RANK_RE = re.compile(r"\[RANK\s+(\d+)\]")


def _stats(values: list[float]) -> dict[str, float | int] | None:
    values = [value for value in values if math.isfinite(value)]
    if not values:
        return None
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def _rank0_or_median(values: list[tuple[int | None, float]]) -> float:
    rank0 = [value for rank, value in values if rank == 0]
    return rank0[-1] if rank0 else statistics.median(value for _, value in values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--nproc", required=True, type=int)
    parser.add_argument("--warmup-iters", default=10, type=int)
    parser.add_argument("--start-epoch", required=True, type=int)
    parser.add_argument("--end-epoch", required=True, type=int)
    parser.add_argument("--exit-code", required=True, type=int)
    parser.add_argument("--toml", required=True)
    parser.add_argument("--max-iter", type=int)
    parser.add_argument("--git-commit", default="unknown")
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = args.log.read_text(encoding="utf-8", errors="replace")
    iteration_values: dict[int, list[tuple[int | None, float]]] = defaultdict(list)
    steady_values: dict[int, list[tuple[int | None, float]]] = defaultdict(list)
    losses: dict[int, list[float]] = defaultdict(list)
    grad_norms: dict[int, list[tuple[int | None, float]]] = defaultdict(list)
    throughput: dict[int, dict[str, float]] = {}

    for line in text.splitlines():
        rank_match = RANK_RE.search(line)
        rank = int(rank_match.group(1)) if rank_match else None
        if match := INITIAL_ITER_RE.search(line):
            iteration_values[int(match.group(1))].append((rank, float(match.group(2))))
        if match := STEADY_ITER_RE.search(line):
            steady_values[int(match.group(1))].append((rank, float(match.group(2))))
        if match := LOSS_RE.search(line):
            losses[int(match.group(2))].append(float(match.group(1)))
        elif match := ACTION_LOSS_RE.search(line):
            losses[int(match.group(1))].append(float(match.group(2)))
        if match := GRAD_RE.search(line):
            grad_norms[int(match.group(2))].append((rank, float(match.group(1))))
        if match := TOKENS_RE.search(line):
            iteration = int(match.group(1))
            values = {
                "tokens_per_sec_per_gpu": float(match.group(2)),
                "useful_tokens_per_sec_per_gpu": float(match.group(3)),
                "useful_supervised_tokens_per_sec_per_gpu": float(match.group(4)),
                "tokens_per_step": float(match.group(5)),
                "samples_per_step": float(match.group(6)),
            }
            if peak_match := PEAK_MEM_RE.search(line):
                values["peak_mem_gb"] = float(peak_match.group(1))
            throughput[iteration] = values

    # The callback's steady-window metric is preferable when available. Initial
    # hit-counter timings remain useful for short runs and warm-up inspection.
    merged_times = {
        iteration: _rank0_or_median(values)
        for iteration, values in iteration_values.items()
    }
    merged_times.update(
        {iteration: _rank0_or_median(values) for iteration, values in steady_values.items()}
    )
    stable_times = [
        value for iteration, value in sorted(merged_times.items()) if iteration > args.warmup_iters
    ]
    all_times = [value for _, value in sorted(merged_times.items())]
    stable_throughput = [
        values
        for iteration, values in sorted(throughput.items())
        if iteration > args.warmup_iters
    ]

    throughput_summary = {}
    for key in (
        "tokens_per_sec_per_gpu",
        "useful_tokens_per_sec_per_gpu",
        "useful_supervised_tokens_per_sec_per_gpu",
        "tokens_per_step",
        "samples_per_step",
        "peak_mem_gb",
    ):
        if result := _stats([row[key] for row in stable_throughput if key in row]):
            throughput_summary[key] = result
    useful = throughput_summary.get("useful_tokens_per_sec_per_gpu")
    if useful:
        throughput_summary["useful_tokens_per_sec_global_estimate"] = {
            key: value * args.nproc if key != "count" else value
            for key, value in useful.items()
        }

    loss_series = [
        {
            "iteration": iteration,
            "mean": statistics.fmean(values),
            "min": min(values),
            "max": max(values),
        }
        for iteration, values in sorted(losses.items())
    ]
    grad_series = [
        {"iteration": iteration, "global": _rank0_or_median(values)}
        for iteration, values in sorted(grad_norms.items())
    ]
    steady_time_stats = _stats(stable_times)
    iteration_throughput = None
    if steady_time_stats and steady_time_stats["mean"] > 0:
        iteration_throughput = 1.0 / steady_time_stats["mean"]
    samples_per_step = throughput_summary.get("samples_per_step")
    if samples_per_step and steady_time_stats and steady_time_stats["mean"] > 0:
        throughput_summary["samples_per_sec_global_estimate"] = (
            samples_per_step["mean"] * args.nproc / steady_time_stats["mean"]
        )

    summary = {
        "schema_version": 1,
        "run": {
            "name": args.run_name,
            "platform": args.platform,
            "devices": args.devices,
            "nproc_per_node": args.nproc,
            "toml": args.toml,
            "max_iter": args.max_iter,
            "warmup_iters_excluded": args.warmup_iters,
            "git_commit": args.git_commit,
            "overrides": args.override,
            "start_time": datetime.fromtimestamp(args.start_epoch, timezone.utc).isoformat(),
            "end_time": datetime.fromtimestamp(args.end_epoch, timezone.utc).isoformat(),
            "wall_time_sec": args.end_epoch - args.start_epoch,
            "exit_code": args.exit_code,
            "log_file": str(args.log),
        },
        "iteration_time_sec": {
            "all": _stats(all_times),
            "steady": steady_time_stats,
            "steady_iterations_per_sec": iteration_throughput,
            "series": [
                {"iteration": iteration, "seconds": seconds}
                for iteration, seconds in sorted(merged_times.items())
            ],
        },
        "throughput": throughput_summary,
        "loss": {
            "finite": all(math.isfinite(row["mean"]) for row in loss_series),
            "series": loss_series,
        },
        "grad_norm": {
            "finite": all(math.isfinite(row["global"]) for row in grad_series),
            "series": grad_series,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    steady = summary["iteration_time_sec"]["steady"]
    md_path = args.output.with_suffix(".md")
    lines = [
        f"# {args.run_name}",
        "",
        f"- Platform: `{args.platform}`",
        f"- Devices / processes: `{args.devices}` / `{args.nproc}`",
        f"- Git commit: `{args.git_commit}`",
        f"- Exit code: `{args.exit_code}`",
        f"- Wall time: `{args.end_epoch - args.start_epoch}s`",
        f"- Warm-up iterations excluded: `{args.warmup_iters}`",
    ]
    if steady:
        lines.extend(
            [
                f"- Steady iterations: `{steady['count']}`",
                f"- Steady iteration time mean/median: `{steady['mean']:.4f}s` / `{steady['median']:.4f}s`",
                f"- Steady iteration time min/max: `{steady['min']:.4f}s` / `{steady['max']:.4f}s`",
                f"- Steady iteration throughput: `{1.0 / steady['mean']:.6f} iteration/s`",
            ]
        )
    else:
        lines.append("- Steady iteration time: unavailable (run is shorter than warm-up window or log format did not match)")
    if useful:
        lines.append(
            f"- Useful tokens/s/GPU mean: `{useful['mean']:.2f}`; global estimate: `{useful['mean'] * args.nproc:.2f}`"
        )
    lines.extend(["", f"Raw log: `{args.log}`", f"JSON: `{args.output}`", ""])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f">>> Performance summary: {args.output}")
    print(f">>> Performance markdown: {md_path}")


if __name__ == "__main__":
    main()
