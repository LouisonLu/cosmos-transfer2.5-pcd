# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mask guards that follow the VAE width padding.

With circular VAE padding a latent next to the left border also sees the rightmost pixels, so the guard that
keeps invalid pixels out of the Stage 2 loss and the erosion of the inference hardlock mask must wrap around the
left/right border too. With zero padding the original functions are used unchanged.
"""

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from cosmos_transfer2._src.transfer2.pwp_geometry.vae_circular import circular_pad_width
from cosmos_transfer2._src.transfer2.utils.target_loss_mask import build_safe_latent_valid_mask


def build_safe_latent_valid_mask_for_vae(
    pixel_valid_mask_B_1_T_H_W: Tensor,
    latent_shape_T_H_W: Sequence[int],
    *,
    circular_width: bool,
    spatial_guard_px: int = 8,
    temporal_guard_frames: int = 1,
    min_valid_ratio: float = 0.01,
) -> Tensor:
    """``build_safe_latent_valid_mask`` whose width guard wraps around when the VAE padding is circular."""
    if not circular_width:
        return build_safe_latent_valid_mask(
            pixel_valid_mask_B_1_T_H_W,
            latent_shape_T_H_W,
            spatial_guard_px=spatial_guard_px,
            temporal_guard_frames=temporal_guard_frames,
            min_valid_ratio=min_valid_ratio,
        )
    if pixel_valid_mask_B_1_T_H_W.ndim != 5 or pixel_valid_mask_B_1_T_H_W.shape[1] != 1:
        raise ValueError(
            "Expected target_loss_mask shape (B, 1, T, H, W), got " f"{tuple(pixel_valid_mask_B_1_T_H_W.shape)}"
        )
    if spatial_guard_px < 0 or temporal_guard_frames < 0:
        raise ValueError("Loss-mask guard sizes must be non-negative")

    invalid = 1.0 - (pixel_valid_mask_B_1_T_H_W > 0.5).to(dtype=torch.float32)
    if spatial_guard_px > 0 or temporal_guard_frames > 0:
        # Same dilation as the original, except that the width neighbourhood wraps around. Padding T and H
        # with the pooling's implicit -inf is equivalent to the original because the invalid map is >= 0.
        kernel = (2 * temporal_guard_frames + 1, 2 * spatial_guard_px + 1, 2 * spatial_guard_px + 1)
        invalid = F.max_pool3d(
            circular_pad_width(invalid, spatial_guard_px),
            kernel_size=kernel,
            stride=1,
            padding=(temporal_guard_frames, spatial_guard_px, 0),
        )
    # The guard is already applied, so the original function only pools to the latent grid and checks coverage.
    return build_safe_latent_valid_mask(
        1.0 - invalid,
        latent_shape_T_H_W,
        spatial_guard_px=0,
        temporal_guard_frames=0,
        min_valid_ratio=min_valid_ratio,
    )


def erode_guided_generation_mask_circular_width(mask_C_T_H_W: Tensor, erode_px: int) -> Tensor:
    """Hardlock mask erosion of the no-video-path pipeline with the width neighbourhood wrapping around."""
    if erode_px == 0:
        return mask_C_T_H_W

    binary_T_1_H_W = (mask_C_T_H_W[:1] >= 0.5).float().permute(1, 0, 2, 3)
    kernel_size = 2 * erode_px + 1
    invalid_T_1_H_W = 1.0 - binary_T_1_H_W
    eroded_T_1_H_W = 1.0 - F.max_pool2d(
        circular_pad_width(invalid_T_1_H_W, erode_px),
        kernel_size=kernel_size,
        stride=1,
        padding=(erode_px, 0),
    )
    return eroded_T_1_H_W.permute(1, 0, 2, 3).repeat(mask_C_T_H_W.shape[0], 1, 1, 1)


__all__ = ["build_safe_latent_valid_mask_for_vae", "erode_guided_generation_mask_circular_width"]
