# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""No-video-path partial-hardlock pipeline with the PwP switches."""

import torch

from cosmos_transfer2._src.imaginaire.utils import log
from cosmos_transfer2._src.transfer2.inference.inference_pipeline_partial_hardlock_no_video_path import (
    ControlVideo2WorldInferencePartialHardlockNoVideoPath,
)
from cosmos_transfer2._src.transfer2.pwp_geometry.frame_size import (
    NATIVE_RESOLUTION,
    check_native_erp_input,
    resolution_for_frame_size,
)
from cosmos_transfer2._src.transfer2.pwp_geometry.masks import erode_guided_generation_mask_circular_width
from cosmos_transfer2._src.transfer2.pwp_geometry.method import check_choice


class PwPGeometryInferencePartialHardlockNoVideoPath(ControlVideo2WorldInferencePartialHardlockNoVideoPath):
    """Adds the PwP switches to the experiment overrides, checks them against the checkpoint stamp after loading,
    reads the inputs at the PwP frame size, and erodes the hardlock mask with wrap-around when the VAE padding is
    circular."""

    def __init__(
        self,
        *args,
        pwp_vae_w_padding: str,
        pwp_sphere_pos_emb: str,
        pwp_frame_size: str,
        exp_override_opts=None,
        **kwargs,
    ):
        self.pwp_vae_w_padding = check_choice("vae_w_padding", pwp_vae_w_padding)
        self.pwp_sphere_pos_emb = check_choice("sphere_pos_emb", pwp_sphere_pos_emb)
        self.pwp_frame_size = check_choice("frame_size", pwp_frame_size)
        self.pwp_resolution = resolution_for_frame_size(pwp_frame_size)
        exp_override_opts = list(exp_override_opts or []) + [
            f"model.config.pwp_vae_w_padding={pwp_vae_w_padding}",
            f"model.config.pwp_sphere_pos_emb={pwp_sphere_pos_emb}",
        ]
        super().__init__(*args, exp_override_opts=exp_override_opts, **kwargs)
        self.pwp_checkpoint_method = self.model.pwp_check_checkpoint()

    def generate_img2world_no_video_path(self, *args, resolution: str = "720", **kwargs):
        """The frame size replaces the per-sample ``resolution`` (the JSON's "720" is ignored)."""
        log.info(f"PwP: frame size {self.pwp_frame_size} (resolution key {self.pwp_resolution})")
        return super().generate_img2world_no_video_path(*args, resolution=self.pwp_resolution, **kwargs)

    def _build_surrogate_input_video(self, *args, **kwargs):
        outputs = super()._build_surrogate_input_video(*args, **kwargs)
        if kwargs.get("resolution") == NATIVE_RESOLUTION:
            original_h, original_w = outputs[-1]
            check_native_erp_input(original_h, original_w, source="The control video")
        return outputs

    def _erode_guided_generation_mask(self, mask_C_T_H_W: torch.Tensor, erode_px: int) -> torch.Tensor:
        if self.pwp_vae_w_padding == "circular":
            if erode_px > 0:
                log.info(f"PwP: eroding the hardlock mask by {erode_px}px with horizontal wrap-around")
            return erode_guided_generation_mask_circular_width(mask_C_T_H_W, erode_px)
        return ControlVideo2WorldInferencePartialHardlockNoVideoPath._erode_guided_generation_mask(
            mask_C_T_H_W, erode_px
        )


__all__ = ["PwPGeometryInferencePartialHardlockNoVideoPath"]
