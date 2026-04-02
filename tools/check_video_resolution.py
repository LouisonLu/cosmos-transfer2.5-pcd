#!/usr/bin/env python3
"""Check video resolution with ffprobe.

Examples:
  python tools/check_video_resolution.py path/to/video.mp4
  python tools/check_video_resolution.py "assets/**/*.mp4"
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Tuple


def _run_ffprobe(video_path: Path) -> Tuple[int, int, str]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate",
        "-of",
        "csv=p=0",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffprobe failed")

    parts = result.stdout.strip().split(",")
    if len(parts) < 3:
        raise RuntimeError(f"Unexpected ffprobe output: {result.stdout!r}")

    width = int(parts[0])
    height = int(parts[1])
    fps = parts[2]
    return width, height, fps


def _expand_inputs(inputs: Iterable[str]) -> List[Path]:
    paths: List[Path] = []
    for item in inputs:
        if any(ch in item for ch in "*?["):
            paths.extend(Path().glob(item))
        else:
            paths.append(Path(item))
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Check video resolution with ffprobe.")
    parser.add_argument("inputs", nargs="+", help="Video file(s) or glob pattern(s).")
    args = parser.parse_args()

    if shutil.which("ffprobe") is None:
        print("error: ffprobe not found in PATH.", file=sys.stderr)
        return 2

    paths = _expand_inputs(args.inputs)
    if not paths:
        print("error: no files matched the inputs.", file=sys.stderr)
        return 2

    exit_code = 0
    for path in paths:
        if not path.exists():
            print(f"missing: {path}", file=sys.stderr)
            exit_code = 1
            continue
        try:
            width, height, fps = _run_ffprobe(path)
        except Exception as exc:  # noqa: BLE001
            print(f"error: {path}: {exc}", file=sys.stderr)
            exit_code = 1
            continue
        print(f"{path}: {width}x{height} @ {fps} fps")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
