# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""No-video-path hardlock inference with optional ERP blended decoding.

This module deliberately leaves the established no-video-path pipeline untouched.
It reuses its complete sampling path and substitutes only the final VAE decode for
the current sample when ``blended_decoding`` is enabled.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from cosmos_transfer2._src.imaginaire.utils import log
from cosmos_transfer2._src.transfer2.inference.inference_pipeline_partial_hardlock_no_video_path import (
    ControlVideo2WorldInferencePartialHardlockNoVideoPath,
)


class _BlendedDecodeModelProxy:
    """Delegate every model operation except decode to the loaded model instance."""

    def __init__(self, model: Any) -> None:
        self._model = model

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return ControlVideo2WorldInferencePartialHardlockNoVideoPathBlendedDecode.decode_blended_latent(
            self._model.decode,
            latent,
        )


class ControlVideo2WorldInferencePartialHardlockNoVideoPathBlendedDecode(
    ControlVideo2WorldInferencePartialHardlockNoVideoPath
):
    """Original hardlock pipeline with Beyond-the-Frame blended VAE decoding."""

    default_blended_decoding = True

    @staticmethod
    def decode_blended_latent(
        decoder: Callable[[torch.Tensor], torch.Tensor],
        latent: torch.Tensor,
    ) -> torch.Tensor:
        """Decode the same ERP latent at 0 and 180 degrees, then blend them.

        The second decode sees the original ERP seam at the center of its input.
        After rolling it back, it contributes most near the original left/right
        boundary, while the unrotated decode contributes most near the center.
        """
        if latent.ndim != 5:
            raise ValueError(f"Expected latent [B, C, T, H, W], got {tuple(latent.shape)}")
        if latent.shape[-1] % 2 != 0:
            raise ValueError(
                "Blended decoding requires an even latent width for an exact 180-degree ERP roll; "
                f"got width={latent.shape[-1]}"
            )

        primary = decoder(latent)
        if primary.ndim != 5:
            raise ValueError(f"Expected decoded video [B, C, T, H, W], got {tuple(primary.shape)}")
        if primary.shape[-1] < 2 or primary.shape[-1] % 2 != 0:
            raise ValueError(
                "Blended decoding requires an even decoded width of at least 2 for an exact 180-degree ERP roll; "
                f"got width={primary.shape[-1]}"
            )

        rotated_latent = torch.roll(latent, shifts=latent.shape[-1] // 2, dims=-1)
        rotated = decoder(rotated_latent)
        rotated_aligned = torch.roll(rotated, shifts=-(rotated.shape[-1] // 2), dims=-1)

        width = primary.shape[-1]
        x = torch.linspace(0, 1, width, device=primary.device, dtype=primary.dtype)
        primary_weight = 1 - torch.abs(2 * x - 1)
        primary_weight = primary_weight.view(*([1] * (primary.ndim - 1)), width)
        return primary * primary_weight + rotated_aligned * (1 - primary_weight)

    def generate_img2world_no_video_path(self, *args: Any, blended_decoding: bool | None = None, **kwargs: Any):
        enabled = self.default_blended_decoding if blended_decoding is None else blended_decoding
        if not enabled:
            log.info("Blended decoding disabled; using the original single VAE decode.")
            return super().generate_img2world_no_video_path(*args, **kwargs)

        original_model = self.model
        self.model = _BlendedDecodeModelProxy(original_model)
        log.info("Using Beyond-the-Frame blended decoding: 0-degree + 180-degree latent decode.")
        try:
            return super().generate_img2world_no_video_path(*args, **kwargs)
        finally:
            self.model = original_model
