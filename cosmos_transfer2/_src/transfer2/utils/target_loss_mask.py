# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for aligning target-validity masks and computing masked latent losses."""

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


def align_valid_mask_to_augmented_video(
    mask_T_H_W: Tensor,
    image_size: Tensor | Sequence[int | float],
    *,
    source_aspect_ratio_tolerance: float = 0.01,
) -> Tensor:
    """Resize a binary validity mask to the augmented video content and zero-pad it.

    ``image_size`` follows the existing Transfer2 augmentation convention:
    ``(target_h, target_w, resized_content_h, resized_content_w)``. The RGB video
    uses reflection padding, but the validity mask must use zero padding so padded
    pixels never contribute to the target loss.

    Args:
        mask_T_H_W: Binary mask with shape ``(T, H, W)`` and 1 for valid pixels.
        image_size: Final and resized-content dimensions from the video augmentor.
        source_aspect_ratio_tolerance: Maximum relative aspect-ratio mismatch.

    Returns:
        Float mask with shape ``(1, T, target_h, target_w)``.
    """
    if mask_T_H_W.ndim != 3:
        raise ValueError(f"Expected mask shape (T, H, W), got {tuple(mask_T_H_W.shape)}")

    dimensions = torch.as_tensor(image_size).flatten().tolist()
    if len(dimensions) != 4:
        raise ValueError(f"Expected image_size with four values, got {dimensions}")
    target_h, target_w, content_h, content_w = (int(round(value)) for value in dimensions)
    if min(target_h, target_w, content_h, content_w) <= 0:
        raise ValueError(f"image_size values must be positive, got {dimensions}")
    if content_h > target_h or content_w > target_w:
        raise ValueError(
            "Resized content cannot be larger than the augmented video: "
            f"content={(content_h, content_w)}, target={(target_h, target_w)}"
        )

    source_h, source_w = mask_T_H_W.shape[-2:]
    source_ratio = source_w / source_h
    content_ratio = content_w / content_h
    relative_ratio_error = abs(source_ratio - content_ratio) / max(content_ratio, 1e-12)
    if relative_ratio_error > source_aspect_ratio_tolerance:
        raise ValueError(
            "Mask and RGB content aspect ratios do not match: "
            f"mask={source_w}x{source_h}, resized_rgb={content_w}x{content_h}, "
            f"relative_error={relative_ratio_error:.6f}"
        )

    mask = (mask_T_H_W > 0.5).to(dtype=torch.float32).unsqueeze(1)
    mask = F.interpolate(mask, size=(content_h, content_w), mode="nearest")

    pad_left = (target_w - content_w) // 2
    pad_right = target_w - content_w - pad_left
    pad_top = (target_h - content_h) // 2
    pad_bottom = target_h - content_h - pad_top
    mask = F.pad(mask, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0.0)
    return mask.permute(1, 0, 2, 3).contiguous()


def build_safe_latent_valid_mask(
    pixel_valid_mask_B_1_T_H_W: Tensor,
    latent_shape_T_H_W: Sequence[int],
    *,
    spatial_guard_px: int = 8,
    temporal_guard_frames: int = 1,
    min_valid_ratio: float = 0.01,
) -> Tensor:
    """Conservatively map a pixel validity mask to the VAE latent grid.

    Invalid pixels are dilated before adaptive max pooling. Therefore a latent
    cell is valid only when its entire contributing bin, including the configured
    guard region, is valid.
    """
    if pixel_valid_mask_B_1_T_H_W.ndim != 5 or pixel_valid_mask_B_1_T_H_W.shape[1] != 1:
        raise ValueError(
            "Expected target_loss_mask shape (B, 1, T, H, W), got "
            f"{tuple(pixel_valid_mask_B_1_T_H_W.shape)}"
        )
    if len(latent_shape_T_H_W) != 3:
        raise ValueError(f"Expected latent shape (T, H, W), got {tuple(latent_shape_T_H_W)}")
    if spatial_guard_px < 0 or temporal_guard_frames < 0:
        raise ValueError("Loss-mask guard sizes must be non-negative")
    if not 0.0 <= min_valid_ratio <= 1.0:
        raise ValueError("min_valid_ratio must be in [0, 1]")

    latent_shape = tuple(int(value) for value in latent_shape_T_H_W)
    if min(latent_shape) <= 0:
        raise ValueError(f"Latent dimensions must be positive, got {latent_shape}")

    valid = (pixel_valid_mask_B_1_T_H_W > 0.5).to(dtype=torch.float32)
    invalid = 1.0 - valid

    if spatial_guard_px > 0 or temporal_guard_frames > 0:
        kernel = (
            2 * temporal_guard_frames + 1,
            2 * spatial_guard_px + 1,
            2 * spatial_guard_px + 1,
        )
        padding = (temporal_guard_frames, spatial_guard_px, spatial_guard_px)
        invalid = F.max_pool3d(invalid, kernel_size=kernel, stride=1, padding=padding)

    invalid_latent = F.adaptive_max_pool3d(invalid, output_size=latent_shape)
    latent_valid = (invalid_latent < 0.5).to(dtype=torch.float32)

    valid_ratios = latent_valid.flatten(1).mean(dim=1)
    bad_samples = torch.nonzero(valid_ratios < min_valid_ratio, as_tuple=False).flatten()
    if bad_samples.numel() > 0:
        ratios = ", ".join(f"{float(valid_ratios[index]):.6f}" for index in bad_samples)
        raise ValueError(
            "Target loss mask has too little valid latent coverage after guarding: "
            f"sample_indices={bad_samples.tolist()}, ratios=[{ratios}], minimum={min_valid_ratio:.6f}"
        )
    return latent_valid


def masked_mse_sums_per_sample(
    prediction_B_C_T_H_W: Tensor,
    target_B_C_T_H_W: Tensor,
    valid_mask_B_1_T_H_W: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return per-sample masked squared-error sums and valid element counts."""
    if prediction_B_C_T_H_W.shape != target_B_C_T_H_W.shape:
        raise ValueError(
            "Prediction and target shapes must match: "
            f"prediction={tuple(prediction_B_C_T_H_W.shape)}, target={tuple(target_B_C_T_H_W.shape)}"
        )
    expected_mask_shape = (prediction_B_C_T_H_W.shape[0], 1, *prediction_B_C_T_H_W.shape[2:])
    if valid_mask_B_1_T_H_W.shape != expected_mask_shape:
        raise ValueError(
            f"Expected valid mask shape {expected_mask_shape}, got {tuple(valid_mask_B_1_T_H_W.shape)}"
        )

    squared_error = (prediction_B_C_T_H_W - target_B_C_T_H_W).float().square()
    expanded_mask = valid_mask_B_1_T_H_W.to(device=squared_error.device, dtype=squared_error.dtype).expand_as(
        squared_error
    )
    numerator = (squared_error * expanded_mask).flatten(1).sum(dim=1)
    denominator = expanded_mask.flatten(1).sum(dim=1)
    return numerator, denominator
