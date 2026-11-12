# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wan2.1 VAE reconstruction with zero vs circular width padding on 360° videos (inference only, no training).

Each video is read and resized exactly like the PwP inference pipeline (--frame-size), encoded and decoded once
with the original VAE and once with the circular-width VAE. Reported per video and on average:

- psnr_full / psnr_edges / psnr_interior: reconstruction PSNR on the whole frame, on the left+right border bands
  (--edge-px wide each) and on the rest;
- seam_gap: mean |x[..., W-1] - x[..., 0]|, the step across the 360° seam;
- neighbor_gap: mean |x[..., c+1] - x[..., c]| over all other adjacent columns, the typical step;
- seam_ratio = seam_gap / neighbor_gap (about 1 when the seam looks like any other column boundary).

The input video rows give the reference values. Crops around the seam (frame rolled by W/2, input / zero /
circular from top to bottom) are written as PNG files.

Usage (GPU machine, repository root):
    torchrun --nproc_per_node=1 scripts/pwp_geometry/vae_seam_test.py --video-dir DIR --output-dir OUT
"""

import argparse
import csv
import importlib
import json
from pathlib import Path

import torch
import torchvision

from cosmos_transfer2._src.imaginaire.lazy_config import instantiate
from cosmos_transfer2._src.imaginaire.utils import distributed
from cosmos_transfer2._src.imaginaire.utils.config_helper import override
from cosmos_transfer2._src.transfer2.inference.utils import read_and_process_video
from cosmos_transfer2._src.transfer2.pwp_geometry.frame_size import resolution_for_frame_size
from cosmos_transfer2._src.transfer2.pwp_geometry.method import FRAME_SIZE_CHOICES
from cosmos_transfer2._src.transfer2.pwp_geometry.vae_circular import convert_wan_tokenizer_to_circular

METRICS = ("psnr_full", "psnr_edges", "psnr_interior", "seam_gap", "neighbor_gap", "seam_ratio")


def build_tokenizer():
    """The VAE exactly as the PwP experiments build it (same config node and weights)."""
    config = importlib.import_module("cosmos_transfer2.pwp_geometry_config").make_config()
    config = override(
        config,
        [
            "--",
            "experiment=pwp_geometry_stage1",
            "model.config.pwp_area_loss=off",
            "model.config.pwp_vae_w_padding=zero",
            "model.config.pwp_sphere_pos_emb=off",
        ],
    )
    return instantiate(config.model.config.tokenizer)


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = torch.mean((a - b) ** 2).item()
    return float("inf") if mse == 0 else 10.0 * torch.log10(torch.tensor(1.0 / mse)).item()


def seam_metrics(video_0_1: torch.Tensor) -> dict[str, float]:
    seam_gap = (video_0_1[..., -1] - video_0_1[..., 0]).abs().mean().item()
    neighbor_gap = (video_0_1[..., 1:] - video_0_1[..., :-1]).abs().mean().item()
    return {"seam_gap": seam_gap, "neighbor_gap": neighbor_gap, "seam_ratio": seam_gap / max(neighbor_gap, 1e-12)}


def reconstruction_metrics(recon_0_1: torch.Tensor, input_0_1: torch.Tensor, edge_px: int) -> dict[str, float]:
    width = input_0_1.shape[-1]
    edges = torch.cat([torch.arange(edge_px), torch.arange(width - edge_px, width)])
    interior = torch.arange(edge_px, width - edge_px)
    return {
        "psnr_full": psnr(recon_0_1, input_0_1),
        "psnr_edges": psnr(recon_0_1[..., edges], input_0_1[..., edges]),
        "psnr_interior": psnr(recon_0_1[..., interior], input_0_1[..., interior]),
        **seam_metrics(recon_0_1),
    }


def load_video(path: Path, frames: int, frame_size: str) -> torch.Tensor:
    """(1, C, T, H, W) in [-1, 1], bf16 on GPU, preprocessed like the inference pipeline."""
    video = read_and_process_video(str(path), resolution=resolution_for_frame_size(frame_size), max_frames=frames)[0]
    if video.ndim == 4:
        video = video[None]
    if video.shape[2] != frames:
        raise ValueError(f"{path.name}: {video.shape[2]} frames read, expected {frames}")
    return (video.float() / 127.5 - 1.0).to(device="cuda", dtype=torch.bfloat16)


def save_seam_crop(videos_0_1: list[torch.Tensor], path: Path, crop_px: int) -> None:
    rows = []
    for video in videos_0_1:
        frame = video[0, :, video.shape[2] // 2].float().cpu()
        width = frame.shape[-1]
        frame = torch.roll(frame, shifts=width // 2, dims=-1)
        rows.append(frame[..., width // 2 - crop_px : width // 2 + crop_px])
    torchvision.utils.save_image(torch.cat(rows, dim=-2), path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video-dir", type=Path, required=True, help="Folder with 360° ERP .mp4 videos")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=93, help="Frames per video (1 + 4k)")
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--frame-size", choices=FRAME_SIZE_CHOICES, required=True, help="Same as --pwp-frame-size")
    parser.add_argument("--edge-px", type=int, default=32, help="Width of each border band for psnr_edges")
    parser.add_argument("--crop-px", type=int, default=128, help="Half width of the seam crops")
    args = parser.parse_args()
    if (args.frames - 1) % 4 != 0:
        raise SystemExit("--frames must be 1 + 4k")

    distributed.init()
    videos = sorted(args.video_dir.glob("*.mp4"))[: args.max_videos]
    if not videos:
        raise SystemExit(f"no .mp4 in {args.video_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "seam_crops").mkdir(exist_ok=True)

    tokenizers = {"zero": build_tokenizer(), "circular": build_tokenizer()}
    counts = convert_wan_tokenizer_to_circular(tokenizers["circular"])
    print(f"circular conversion: {counts}")

    rows = []
    for index, path in enumerate(videos, 1):
        video = load_video(path, args.frames, args.frame_size)
        input_0_1 = (video.float() + 1) / 2
        results = {"input": seam_metrics(input_0_1)}
        recons = {}
        with torch.no_grad():
            for name, tokenizer in tokenizers.items():
                recon = tokenizer.decode(tokenizer.encode(video)).float().clamp(-1, 1)
                recons[name] = (recon + 1) / 2
                results[name] = reconstruction_metrics(recons[name], input_0_1, args.edge_px)
        for variant, metrics in results.items():
            rows.append({"video": path.name, "variant": variant, **metrics})
        save_seam_crop(
            [input_0_1, recons["zero"], recons["circular"]],
            args.output_dir / "seam_crops" / f"{path.stem}.png",
            args.crop_px,
        )
        print(
            f"[{index}/{len(videos)}] {path.name}: seam_ratio input {results['input']['seam_ratio']:.3f} | "
            f"zero {results['zero']['seam_ratio']:.3f} | circular {results['circular']['seam_ratio']:.3f}; "
            f"psnr_edges zero {results['zero']['psnr_edges']:.2f} | circular {results['circular']['psnr_edges']:.2f}"
        )

    with (args.output_dir / "per_video.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video", "variant", *METRICS])
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for variant in ("input", "zero", "circular"):
        variant_rows = [row for row in rows if row["variant"] == variant]
        summary[variant] = {
            metric: sum(row[metric] for row in variant_rows) / len(variant_rows)
            for metric in METRICS
            if metric in variant_rows[0]
        }
    summary["num_videos"] = len(videos)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
