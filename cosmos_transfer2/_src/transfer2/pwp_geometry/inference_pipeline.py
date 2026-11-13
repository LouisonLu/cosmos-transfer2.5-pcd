# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""No-video-path partial-hardlock pipeline with the PwP switches and the optional ERP inference techniques."""

import torch

from cosmos_transfer2._src.imaginaire.utils import log
from cosmos_transfer2._src.transfer2.inference.inference_pipeline_partial_hardlock_no_video_path import (
    ControlVideo2WorldInferencePartialHardlockNoVideoPath,
)
from cosmos_transfer2._src.transfer2.inference.inference_pipeline_partial_hardlock_no_video_path_blended_decode import (
    ControlVideo2WorldInferencePartialHardlockNoVideoPathBlendedDecode,
)
from cosmos_transfer2._src.transfer2.pwp_geometry.frame_size import (
    NATIVE_RESOLUTION,
    check_native_erp_input,
    resolution_for_frame_size,
)
from cosmos_transfer2._src.transfer2.pwp_geometry.masks import erode_guided_generation_mask_circular_width
from cosmos_transfer2._src.transfer2.pwp_geometry.method import NATIVE_FRAME_SIZE, check_choice


class PwPGeometryInferencePartialHardlockNoVideoPath(ControlVideo2WorldInferencePartialHardlockNoVideoPath):
    """Adds the PwP switches to the experiment overrides, checks them against the checkpoint stamp after loading,
    reads the inputs at the PwP frame size, and erodes the hardlock mask with wrap-around when the VAE padding is
    circular. The ERP inference techniques are off unless enabled: dual-orientation blended decoding, latent
    rotation during denoising and latitude-aware noise (``pwp_latitude_noise_strength`` > 0)."""

    def __init__(
        self,
        *args,
        pwp_vae_w_padding: str,
        pwp_sphere_pos_emb: str,
        pwp_frame_size: str = NATIVE_FRAME_SIZE,
        pwp_blended_decoding: bool = False,
        pwp_latent_rotation: bool = False,
        pwp_latitude_noise_strength: float = 0.0,
        exp_override_opts=None,
        **kwargs,
    ):
        if not 0.0 <= pwp_latitude_noise_strength <= 1.0:
            raise ValueError(f"pwp_latitude_noise_strength must be in [0, 1], got {pwp_latitude_noise_strength}")
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
        self.pwp_blended_decoding = pwp_blended_decoding
        self.model.pwp_latent_rotation = pwp_latent_rotation
        self.model.pwp_latitude_noise_strength = float(pwp_latitude_noise_strength)

    def pwp_techniques(self) -> dict:
        return {
            "blended_decoding": self.pwp_blended_decoding,
            "latent_rotation": self.model.pwp_latent_rotation,
            "latitude_noise_strength": self.model.pwp_latitude_noise_strength,
        }

    def generate_img2world_no_video_path(self, *args, resolution: str = "720", **kwargs):
        """The frame size replaces the per-sample ``resolution`` (the JSON's "720" is ignored)."""
        log.info(f"PwP: frame size {self.pwp_frame_size} (resolution key {self.pwp_resolution})")
        log.info(f"PwP inference techniques: {self.pwp_techniques()}")
        if not self.pwp_blended_decoding:
            return super().generate_img2world_no_video_path(*args, resolution=self.pwp_resolution, **kwargs)
        # Decode the final latent at 0 and 180 degrees and blend them (the existing blended-decode function).
        original_decode = self.model.decode
        self.model.decode = lambda latent: (
            ControlVideo2WorldInferencePartialHardlockNoVideoPathBlendedDecode.decode_blended_latent(
                original_decode, latent
            )
        )
        try:
            return super().generate_img2world_no_video_path(*args, resolution=self.pwp_resolution, **kwargs)
        finally:
            del self.model.decode

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
