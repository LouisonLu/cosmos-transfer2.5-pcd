#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate paired masked RGB, PCD, prompt, and target-loss-mask files."""

import argparse
import json
from pathlib import Path

import numpy as np
from decord import VideoReader, cpu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Split root containing rgb_videos/, pcd_videos/, prompts/, mask/",
    )
    parser.add_argument("--pcd-suffix", default="_erp_pose")
    parser.add_argument("--prompt-suffix", default="_prompt")
    parser.add_argument("--mask-suffix", default="_mask")
    parser.add_argument("--required-frames", type=int, default=93)
    parser.add_argument("--mask-threshold", type=int, default=128)
    parser.add_argument("--black-threshold", type=int, default=24)
    parser.add_argument("--max-invalid-nonblack-ratio", type=float, default=0.05)
    parser.add_argument(
        "--frame-indices",
        default="0,46,92",
        help="Comma-separated frames used for decoded pixel checks; use 'all' for every required frame",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="0 checks every RGB video")
    parser.add_argument("--report", type=Path, default=None, help="Optional JSON report path")
    return parser.parse_args()


def read_prompt(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, str):
        return data.strip()
    if isinstance(data, dict):
        value = data.get("caption", data.get("text", data.get("prompt", "")))
        return value.strip() if isinstance(value, str) else ""
    return ""


def select_frame_indices(spec: str, required_frames: int) -> list[int]:
    if spec.strip().lower() == "all":
        return list(range(required_frames))
    indices = sorted({int(value.strip()) for value in spec.split(",") if value.strip()})
    if not indices or indices[0] < 0 or indices[-1] >= required_frames:
        raise ValueError(f"frame indices must be within [0, {required_frames - 1}], got {indices}")
    return indices


def video_metadata(reader: VideoReader) -> tuple[int, int, int]:
    first = reader[0].asnumpy()
    return len(reader), int(first.shape[0]), int(first.shape[1])


def check_sample(
    rgb_path: Path,
    pcd_path: Path,
    prompt_path: Path,
    mask_path: Path,
    args: argparse.Namespace,
    frame_indices: list[int],
) -> dict:
    result = {"stem": rgb_path.stem, "errors": []}
    for label, path in (("PCD", pcd_path), ("prompt", prompt_path), ("mask", mask_path)):
        if not path.is_file():
            result["errors"].append(f"missing {label}: {path}")
    if result["errors"]:
        return result

    if not read_prompt(prompt_path):
        result["errors"].append(f"empty prompt: {prompt_path}")
        return result

    rgb_reader = VideoReader(str(rgb_path), ctx=cpu(0), num_threads=2)
    pcd_reader = VideoReader(str(pcd_path), ctx=cpu(0), num_threads=2)
    mask_reader = VideoReader(str(mask_path), ctx=cpu(0), num_threads=2)
    rgb_meta = video_metadata(rgb_reader)
    pcd_meta = video_metadata(pcd_reader)
    mask_meta = video_metadata(mask_reader)
    result["rgb_metadata"] = rgb_meta
    result["pcd_metadata"] = pcd_meta
    result["mask_metadata"] = mask_meta

    for label, metadata in (("RGB", rgb_meta), ("PCD", pcd_meta), ("mask", mask_meta)):
        if metadata[0] < args.required_frames:
            result["errors"].append(
                f"{label} has {metadata[0]} frames; required={args.required_frames}"
            )
    if rgb_meta[1:] != mask_meta[1:]:
        result["errors"].append(f"RGB/mask resolution mismatch: RGB={rgb_meta[1:]}, mask={mask_meta[1:]}")
    if rgb_meta[1:] != pcd_meta[1:]:
        result["errors"].append(f"RGB/PCD resolution mismatch: RGB={rgb_meta[1:]}, PCD={pcd_meta[1:]}")
    if result["errors"]:
        return result

    rgb = rgb_reader.get_batch(frame_indices).asnumpy()
    pcd = pcd_reader.get_batch(frame_indices).asnumpy()
    mask = mask_reader.get_batch(frame_indices).asnumpy().astype(np.float32).mean(axis=-1)
    valid = mask >= args.mask_threshold
    invalid = ~valid
    invalid_count = int(invalid.sum())
    result["valid_ratio"] = float(valid.mean())

    if invalid_count:
        rgb_nonblack = np.max(rgb, axis=-1) > args.black_threshold
        pcd_nonblack = np.max(pcd, axis=-1) > args.black_threshold
        result["rgb_invalid_nonblack_ratio"] = float((rgb_nonblack & invalid).sum() / invalid_count)
        result["pcd_invalid_nonblack_ratio"] = float((pcd_nonblack & invalid).sum() / invalid_count)
        if result["rgb_invalid_nonblack_ratio"] > args.max_invalid_nonblack_ratio:
            result["errors"].append(
                "RGB invalid region is not black enough: "
                f"ratio={result['rgb_invalid_nonblack_ratio']:.6f}"
            )
        if result["pcd_invalid_nonblack_ratio"] > args.max_invalid_nonblack_ratio:
            result["errors"].append(
                "PCD invalid region is not black enough: "
                f"ratio={result['pcd_invalid_nonblack_ratio']:.6f}"
            )
    else:
        result["rgb_invalid_nonblack_ratio"] = 0.0
        result["pcd_invalid_nonblack_ratio"] = 0.0
    return result


def main() -> int:
    args = parse_args()
    frame_indices = select_frame_indices(args.frame_indices, args.required_frames)
    rgb_root = args.dataset_dir / "rgb_videos"
    pcd_root = args.dataset_dir / "pcd_videos"
    prompt_root = args.dataset_dir / "prompts"
    mask_root = args.dataset_dir / "mask"

    rgb_paths = sorted(rgb_root.glob("*.mp4"))
    if args.max_samples > 0:
        rgb_paths = rgb_paths[: args.max_samples]
    if not rgb_paths:
        raise FileNotFoundError(f"No RGB MP4 files found in {rgb_root}")

    results = []
    for rgb_path in rgb_paths:
        stem = rgb_path.stem
        result = check_sample(
            rgb_path,
            pcd_root / f"{stem}{args.pcd_suffix}.mp4",
            prompt_root / f"{stem}{args.prompt_suffix}.json",
            mask_root / f"{stem}{args.mask_suffix}.mp4",
            args,
            frame_indices,
        )
        results.append(result)
        if result["errors"]:
            print(f"FAIL {stem}: {'; '.join(result['errors'])}")

    failures = [result for result in results if result["errors"]]
    summary = {
        "dataset_dir": str(args.dataset_dir),
        "checked": len(results),
        "passed": len(results) - len(failures),
        "failed": len(failures),
        "frame_indices": frame_indices,
        "results": results,
    }
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2))
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Report: {args.report}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
