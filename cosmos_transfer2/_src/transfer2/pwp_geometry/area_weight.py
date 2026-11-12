# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Spherical-area weights for equirectangular (ERP) latent rectified-flow losses.

Every ERP latent row spans the same latitude interval Δ, so the sphere area of a row centered at latitude φ is
``2π · (sin(φ + Δ/2) − sin(φ − Δ/2)) = 4π · sin(Δ/2) · cos φ``, i.e. proportional to ``cos φ``. After the loss
is normalized by the weight sum, ``cos φ`` at the row center is therefore the exact per-row sphere area. The
weight multiplies the squared error once (``cos φ``, not the ``cos² φ`` of 360DVD's latitude-aware loss).
"""

import math

import torch
from torch import Tensor


def erp_content_rows(padding_mask_H_W: Tensor) -> tuple[int, int]:
    """Return the ``[top, bottom)`` rows that hold ERP content.

    ``padding_mask`` follows the Transfer2 augmentor convention: 1 for padded pixels and 0 for content. A row is
    content if any of its pixels is content.
    """
    if padding_mask_H_W.ndim != 2:
        raise ValueError(f"Expected padding mask shape (H, W), got {tuple(padding_mask_H_W.shape)}")

    content_rows = (padding_mask_H_W < 0.5).any(dim=1)
    rows = torch.nonzero(content_rows).flatten()
    if rows.numel() == 0:
        raise ValueError("Padding mask marks every row as padding; no ERP content rows found.")
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    if not bool(content_rows[top:bottom].all()):
        raise ValueError(f"ERP content rows are not contiguous within [{top}, {bottom}).")
    return top, bottom


def check_full_width_content(padding_mask_H_W: Tensor) -> None:
    """ERP content must span the full width; horizontal padding would break the 360° wrap-around."""
    content_cols = (padding_mask_H_W < 0.5).any(dim=0)
    if not bool(content_cols.all()):
        padded = int((~content_cols).sum())
        raise ValueError(f"{padded} padded columns found; the PwP geometry options need full-width ERP content.")


def build_sphere_area_weights(padding_mask_B_1_H_W: Tensor, latent_h: int) -> Tensor:
    """Return per-row latent weights with shape ``(B, 1, 1, latent_h, 1)``.

    ERP latent rows get ``w(h) = cos(π((h + 0.5) / H − 0.5))``, where ``h`` is the row index inside the ERP
    content and ``H`` is the number of ERP latent rows. Reflection-padded rows get weight 0.
    """
    if padding_mask_B_1_H_W.ndim != 4 or padding_mask_B_1_H_W.shape[1] != 1:
        raise ValueError(f"Expected padding mask shape (B, 1, H, W), got {tuple(padding_mask_B_1_H_W.shape)}")

    batch_size, _, pixel_h, _ = padding_mask_B_1_H_W.shape
    if pixel_h % latent_h != 0:
        raise ValueError(f"Padding mask height {pixel_h} is not a multiple of latent height {latent_h}.")
    stride = pixel_h // latent_h

    weights_B_H = torch.zeros(batch_size, latent_h, dtype=torch.float32, device=padding_mask_B_1_H_W.device)
    for b in range(batch_size):
        top, bottom = erp_content_rows(padding_mask_B_1_H_W[b, 0])
        if top % stride != 0 or bottom % stride != 0:
            raise ValueError(
                f"ERP content rows [{top}, {bottom}) are not aligned to the latent stride {stride}; "
                "latitude rows would be ambiguous."
            )
        h0, h1 = top // stride, bottom // stride
        h = torch.arange(h1 - h0, dtype=torch.float64, device=padding_mask_B_1_H_W.device)
        weights_B_H[b, h0:h1] = torch.cos(math.pi * ((h + 0.5) / (h1 - h0) - 0.5)).to(torch.float32)

    return weights_B_H[:, None, None, :, None]


__all__ = ["build_sphere_area_weights", "check_full_width_content", "erp_content_rows"]
