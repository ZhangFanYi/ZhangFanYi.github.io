#!/usr/bin/env python3
"""Materialize a small, link-free Cosmos3-DROID performance subset."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pyarrow.parquet as pq


CAMERAS = (
    "observation.image.exterior_image_1_left",
    "observation.image.exterior_image_2_left",
    "observation.image.wrist_image_left",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-success",
        type=Path,
        required=True,
        help="Source LeRobot success directory containing meta/, data/ and videos/.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="New DROID_ROOT. The script creates output-root/success.",
    )
    parser.add_argument("--episodes", type=int, default=64)
    return parser.parse_args()


def payload_paths(root: Path, row: dict) -> set[Path]:
    paths = {
        root
        / "data"
        / f"chunk-{row['data/chunk_index']:03d}"
        / f"file-{row['data/file_index']:03d}.parquet"
    }
    for camera in CAMERAS:
        paths.add(
            root
            / "videos"
            / camera
            / f"chunk-{row[f'videos/{camera}/chunk_index']:03d}"
            / f"file-{row[f'videos/{camera}/file_index']:03d}.mp4"
        )
    return paths


def main() -> None:
    args = parse_args()
    source = args.source_success.resolve(strict=True)
    output_root = args.output_root.absolute()
    output = output_root / "success"

    if args.episodes < 8:
        raise ValueError("--episodes must be at least 8 for an 8-rank performance run")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_root}")

    episode_files = sorted((source / "meta" / "episodes").rglob("*.parquet"))
    if not episode_files:
        raise FileNotFoundError(f"no episode metadata found under {source}")

    columns = ["episode_index", "episode_id", "length", "data/chunk_index", "data/file_index"]
    for camera in CAMERAS:
        columns.extend((f"videos/{camera}/chunk_index", f"videos/{camera}/file_index"))

    rows: list[dict] = []
    for episode_file in episode_files:
        rows.extend(pq.read_table(episode_file, columns=columns).to_pylist())
    rows.sort(key=lambda row: int(row["episode_index"]))
    selected = rows[: args.episodes]
    if len(selected) != args.episodes:
        raise ValueError(f"requested {args.episodes} episodes, only found {len(selected)}")

    payloads = set().union(*(payload_paths(source, row) for row in selected))
    missing = sorted(path for path in payloads if not path.is_file())
    if missing:
        preview = "\n".join(str(path) for path in missing[:10])
        raise FileNotFoundError(f"{len(missing)} selected payload files are missing:\n{preview}")

    output.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source / "meta", output / "meta", symlinks=False)
    for source_path in sorted(payloads):
        destination = output / source_path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination, follow_symlinks=True)

    keep_ranges = {}
    for row in selected:
        episode_id = row["episode_id"]
        key = (
            f"gs://xembodiment_data/r2d2/r2d2-data-full/{episode_id}/recordings/MP4"
            f"--gs://xembodiment_data/r2d2/r2d2-data-full/{episode_id}/trajectory.h5"
        )
        keep_ranges[key] = [[0, int(row["length"])]]

    keep_path = output_root / f"keep_ranges_perf{args.episodes}.json"
    keep_path.write_text(json.dumps(keep_ranges, separators=(",", ":")))

    links = [path for path in output_root.rglob("*") if path.is_symlink()]
    if links:
        raise RuntimeError(f"materialized subset unexpectedly contains {len(links)} symbolic links")

    payload_bytes = sum(path.stat().st_size for path in payloads)
    train_windows = sum(max(0, int(row["length"]) - 16) for row in selected)
    print(f"DROID_ROOT={output_root}")
    print(f"KEEP_RANGES_PATH={keep_path}")
    print(f"episodes={len(selected)} train_windows={train_windows}")
    print(f"payload_files={len(payloads)} payload_GiB={payload_bytes / 2**30:.3f}")
    print("symbolic_links=0")


if __name__ == "__main__":
    main()
