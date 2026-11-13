# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check whether the black outside the mask leaks into the hardlocked latent (report only, no generation).

The no-video-path hardlock locks frame 0 of the latent of the *masked* first frame inside the eroded mask. The VAE
encoder sees the black around the visible region, so latent cells near the mask border can differ from the latent of
the real frame. For every prepared request this script encodes the masked frame (exactly as inference does: the
surrogate video, read and resized by the pipeline) and the unmasked first frame the same way, and reports, per
``--erode-px``, how far the locked latent cells are from the clean latent, plus the error by distance to the mask
border and the share of near-black pixels inside the eroded mask.

Prepare the requests with run_cosmos_inference_batch.py ``--prepare-only`` (same data and mask settings as the
inference run); it writes <work root>/<name>/<name>.json, rgb_frame0.png and masked_rgb_frame0.png. Then:

    python scripts/pwp_geometry/check_hardlock_black_leak.py --work-root <WORK_ROOT> -o <OUT_DIR> \\
      --checkpoint-path <CKPT>.pt --experiment pwp_geometry_inference_hardlock \\
      --config-file cosmos_transfer2/pwp_geometry_config.py --disable-guardrails \\
      --pwp-vae-w-padding zero --pwp-sphere-pos-emb off

The error of a latent cell is the channel L2 distance between the two latents divided by the median channel norm of
the clean latent. Cells far from the border give the floor (should be ~0).
"""

import json
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import pydantic
import torch
import tyro
from cosmos_oss.init import cleanup_environment, init_environment

from cosmos_transfer2.config import SetupArguments, handle_tyro_exception, is_rank0

DISTANCE_BINS_PX = (0, 8, 16, 24, 32, 48, 64, 96, 10**9)
NEAR_BLACK = 20  # uint8, every channel below this


class Args(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")

    work_root: Path
    """Work root of run_cosmos_inference_batch.py --prepare-only (one folder per sample)."""

    pwp_vae_w_padding: Literal["zero", "circular"]
    pwp_sphere_pos_emb: Literal["off", "xyz"]
    pwp_frame_size: Literal["1280x640", "legacy_720"] = "1280x640"

    erode_px: list[int] = [0, 8, 16, 24, 32, 48]
    """Mask erosions to compare (the inference default is 8)."""

    limit: int | None = None
    """Check only the first N requests."""

    setup: SetupArguments


def _requests(work_root: Path, limit: int | None) -> list[dict]:
    paths = sorted(path for path in work_root.glob("*/*.json") if path.stem == path.parent.name)
    if not paths:
        raise ValueError(f"No prepared requests under {work_root}; run run_cosmos_inference_batch.py --prepare-only")
    requests = [json.loads(path.read_text()) for path in paths[:limit]]
    for request in requests:
        if "guided_generation_mask" not in request:
            raise ValueError(f"{request['name']}: prepared without --guided-hardlock, there is no hardlock to check")
    return requests


def _first_latent(pipeline, image_path: str, request: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Latent frame 0 of the surrogate video built from ``image_path`` (as in inference) and its frame 0 (uint8)."""
    from cosmos_transfer2._src.transfer2.inference.utils import uint8_to_normalized_float

    frames, _, _, _, _ = pipeline._build_surrogate_input_video(
        image_context_path=image_path,
        input_control_video_paths={"depth": request["depth"]["control_path"]},
        hint_key=["depth"],
        resolution=pipeline.pwp_resolution,
        max_frames=request.get("max_frames"),
    )
    chunk = request.get("num_video_frames_per_chunk", 93)
    frames = pipeline._pad_input_frames(frames[:, :chunk], frames[:, :chunk].shape[1], chunk)
    video = uint8_to_normalized_float(frames, dtype=torch.bfloat16)[None].cuda()
    latent = pipeline.model.encode(video).contiguous().float()
    return latent[0, :, 0], frames[:, 0]


def _distance_to_unknown(known_H_W: np.ndarray, wrap: bool) -> np.ndarray:
    """Pixel distance of every known pixel to the nearest unknown one (0 outside); wraps left/right if ``wrap``."""
    known = known_H_W.astype(np.uint8)
    width = known.shape[1]
    if wrap:
        known = np.concatenate([known, known, known], axis=1)
    distance = cv2.distanceTransform(known, cv2.DIST_L2, 5)
    return distance[:, width : 2 * width] if wrap else distance


def _check_sample(pipeline, request: dict, erode_px: list[int], wrap: bool, out_dir: Path) -> dict:
    masked_path = request["image_context_path"]
    rgb_path = str(Path(masked_path).with_name("rgb_frame0.png"))
    masked_latent, masked_frame = _first_latent(pipeline, masked_path, request)
    clean_latent, _ = _first_latent(pipeline, rgb_path, request)
    latent_h, latent_w = masked_latent.shape[-2:]
    pixel_h, pixel_w = masked_frame.shape[-2:]

    error = (masked_latent - clean_latent).norm(dim=0)
    error = (error / clean_latent.norm(dim=0).median()).cpu().numpy()

    mask = pipeline._read_guided_generation_mask(
        request["guided_generation_mask"],
        h=pixel_h,
        w=pixel_w,
        resolution=pipeline.pwp_resolution,
        max_frames=request.get("max_frames"),
    ).squeeze(0)[:, :1]  # [3, 1, H, W]: frame 0, the only hardlocked frame
    known = mask[0, 0].numpy() >= 0.5
    cell_rows = np.arange(latent_h) * pixel_h // latent_h + pixel_h // latent_h // 2
    cell_cols = np.arange(latent_w) * pixel_w // latent_w + pixel_w // latent_w // 2
    cell_distance = _distance_to_unknown(known, wrap)[np.ix_(cell_rows, cell_cols)]
    near_black = (masked_frame.numpy() < NEAR_BLACK).all(axis=0)

    by_erode = {}
    for px in erode_px:
        # The same erosion (wrapping when circular) and latent weight map as the inference hardlock, frame 0 only.
        eroded = pipeline._erode_guided_generation_mask(mask.clone(), px)
        weight = pipeline.construct_latent_weight_map(eroded[None], h=latent_h, w=latent_w, c=1)
        weight = weight[0, 0, 0].cpu().numpy()
        locked = weight >= 0.5
        eroded_known = eroded[0, 0].numpy() >= 0.5
        by_erode[px] = {
            "locked_cells": int(locked.sum()),
            "locked_weighted_mean_error": float((weight * error).sum() / max(weight.sum(), 1e-8)),
            "locked_p95_error": float(np.percentile(error[locked], 95)) if locked.any() else None,
            "locked_max_error": float(error[locked].max()) if locked.any() else None,
            "locked_share_error_over_0.1": float((error[locked] > 0.1).mean()) if locked.any() else None,
            "near_black_share_inside_eroded_mask": float(near_black[eroded_known].mean()) if eroded_known.any() else None,
        }

    by_distance = {}
    for low, high in zip(DISTANCE_BINS_PX[:-1], DISTANCE_BINS_PX[1:]):
        cells = (cell_distance > low) & (cell_distance <= high) if low else (cell_distance > 0) & (cell_distance <= high)
        label = f"{low}-{high}px" if high < 10**9 else f">{low}px"
        by_distance[label] = {
            "cells": int(cells.sum()),
            "mean_error": float(error[cells].mean()) if cells.any() else None,
            "p95_error": float(np.percentile(error[cells], 95)) if cells.any() else None,
        }

    heat = cv2.applyColorMap((np.clip(error / 0.5, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    heat = cv2.resize(heat, (pixel_w, pixel_h), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(known.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(heat, contours, -1, (255, 255, 255), 1)
    cv2.imwrite(str(out_dir / f"{request['name']}_latent_error.png"), heat)
    return {"name": request["name"], "by_erode_px": by_erode, "by_distance_to_unknown": by_distance}


def _mean(values: list) -> float | None:
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def main(args: Args) -> None:
    from cosmos_transfer2.inference_pwp_hardlock_no_video_path import Control2WorldInferencePwPHardlockNoVideoPath

    requests = _requests(args.work_root, args.limit)
    args.setup.output_dir.mkdir(parents=True, exist_ok=True)
    inference = Control2WorldInferencePwPHardlockNoVideoPath(
        args.setup,
        batch_hint_keys=["depth"],
        pwp_vae_w_padding=args.pwp_vae_w_padding,
        pwp_sphere_pos_emb=args.pwp_sphere_pos_emb,
        pwp_frame_size=args.pwp_frame_size,
    )
    pipeline = inference.inference_pipeline
    wrap = args.pwp_vae_w_padding == "circular"
    with torch.no_grad():
        samples = [_check_sample(pipeline, request, args.erode_px, wrap, args.setup.output_dir) for request in requests]

    summary = {
        "vae_w_padding": args.pwp_vae_w_padding,
        "frame_size": args.pwp_frame_size,
        "samples": len(samples),
        "mean_by_erode_px": {
            px: {key: _mean([s["by_erode_px"][px][key] for s in samples]) for key in samples[0]["by_erode_px"][px]}
            for px in args.erode_px
        },
        "mean_by_distance_to_unknown": {
            label: {key: _mean([s["by_distance_to_unknown"][label][key] for s in samples]) for key in ("mean_error", "p95_error")}
            for label in samples[0]["by_distance_to_unknown"]
        },
        "per_sample": samples,
    }
    path = args.setup.output_dir / "hardlock_black_leak.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\nHardlock black-leak check, {len(samples)} sample(s), VAE padding {args.pwp_vae_w_padding}: {path}")
    print("erode_px  locked_cells  weighted_mean_err  p95_err  max_err  share_err>0.1  near_black_inside")
    for px, row in summary["mean_by_erode_px"].items():
        values = [row["locked_cells"], row["locked_weighted_mean_error"], row["locked_p95_error"], row["locked_max_error"],
                  row["locked_share_error_over_0.1"], row["near_black_share_inside_eroded_mask"]]
        print(f"{px:>8}  " + "  ".join("     -" if v is None else f"{v:>12.4f}" for v in values))
    print("distance_to_unknown  mean_err  p95_err")
    for label, row in summary["mean_by_distance_to_unknown"].items():
        print(f"{label:>19}  " + "  ".join("     -" if v is None else f"{v:>8.4f}" for v in row.values()))


if __name__ == "__main__":
    init_environment()
    try:
        args = tyro.cli(Args, description=__doc__, console_outputs=is_rank0(), config=(tyro.conf.OmitArgPrefixes,))
    except Exception as e:
        handle_tyro_exception(e)
    main(args)
    cleanup_environment()
