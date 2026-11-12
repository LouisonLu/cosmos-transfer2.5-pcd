# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Spherical XYZ position embedding for ERP DiT tokens.

Each token gets the unit vector of its (longitude λ, latitude φ) center,
``(cos φ cos λ, cos φ sin λ, sin φ)``, so the two ends of the 360° seam and all longitudes near a pole are close.
A small MLP maps it to the model width; the output layer is zero-initialized, so a new embedding leaves the
pretrained network's output unchanged until it is trained. The existing 3D RoPE is kept as is.
"""

import math

import torch
import torch.nn as nn
from torch import Tensor

from cosmos_transfer2._src.transfer2.pwp_geometry.area_weight import check_full_width_content, erp_content_rows


def token_latitude_longitude(padding_mask_B_1_H_W: Tensor, token_h: int, token_w: int) -> tuple[Tensor, Tensor]:
    """Latitude ``(B, token_h)`` and longitude ``(token_w,)`` of the token centers, in radians.

    The ERP content rows come from ``padding_mask`` (1 = padded). Rows of the reflection padding above/below the
    content mirror the content, so they get the mirrored latitude.
    """
    if padding_mask_B_1_H_W.ndim != 4 or padding_mask_B_1_H_W.shape[1] != 1:
        raise ValueError(f"Expected padding mask shape (B, 1, H, W), got {tuple(padding_mask_B_1_H_W.shape)}")
    pixel_h, pixel_w = padding_mask_B_1_H_W.shape[-2:]
    if pixel_h * token_w != pixel_w * token_h:
        raise ValueError(
            f"Token grid {token_h}x{token_w} is not a uniform downsampling of the padding mask {pixel_h}x{pixel_w}. "
            "A spatial context-parallel split is not supported with the spherical position embedding."
        )
    scale = pixel_h / token_h
    device = padding_mask_B_1_H_W.device

    row_centers = (torch.arange(token_h, dtype=torch.float64, device=device) + 0.5) * scale
    latitudes = []
    for mask_H_W in padding_mask_B_1_H_W[:, 0]:
        check_full_width_content(mask_H_W)
        top, bottom = erp_content_rows(mask_H_W)
        latitude = math.pi * (0.5 - (row_centers - top) / (bottom - top))
        latitude = torch.where(latitude > math.pi / 2, math.pi - latitude, latitude)
        latitude = torch.where(latitude < -math.pi / 2, -math.pi - latitude, latitude)
        latitudes.append(latitude)
    longitude = 2 * math.pi * (torch.arange(token_w, dtype=torch.float64, device=device) + 0.5) / token_w - math.pi
    return torch.stack(latitudes), longitude


def sphere_xyz(latitude_B_H: Tensor, longitude_W: Tensor) -> Tensor:
    """Unit vectors with shape ``(B, H, W, 3)``."""
    lat = latitude_B_H[:, :, None]
    lon = longitude_W[None, None, :]
    return torch.stack(
        torch.broadcast_tensors(torch.cos(lat) * torch.cos(lon), torch.cos(lat) * torch.sin(lon), torch.sin(lat)),
        dim=-1,
    )


class SphereXYZPositionalEmbedding(nn.Module):
    """XYZ unit vector -> Linear -> SiLU -> Linear(zero init), added to the patch-embedded tokens."""

    def __init__(self, model_channels: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.proj_in = nn.Linear(3, hidden_dim)
        self.act = nn.SiLU()
        self.proj_out = nn.Linear(hidden_dim, model_channels)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.proj_in.reset_parameters()
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, padding_mask_B_1_H_W: Tensor | None, token_h: int, token_w: int) -> Tensor:
        """Embedding with shape ``(B, 1, token_h, token_w, D)``, broadcast over time."""
        if padding_mask_B_1_H_W is None:
            padding_mask_B_1_H_W = torch.zeros(1, 1, token_h, token_w, device=self.proj_in.weight.device)
        latitude, longitude = token_latitude_longitude(padding_mask_B_1_H_W, token_h, token_w)
        xyz = sphere_xyz(latitude, longitude).to(dtype=self.proj_in.weight.dtype)
        return self.proj_out(self.act(self.proj_in(xyz)))[:, None]


__all__ = ["SphereXYZPositionalEmbedding", "sphere_xyz", "token_latitude_longitude"]
