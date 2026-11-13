# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""No-video-path partial-hardlock pipeline with the PwP switches and the optional ERP inference techniques."""

import contextlib
import sys

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
from cosmos_transfer2._src.transfer2.pwp_geometry.inference_techniques import mask_control
from cosmos_transfer2._src.transfer2.pwp_geometry.masks import erode_guided_generation_mask_circular_width
from cosmos_transfer2._src.transfer2.pwp_geometry.method import NATIVE_FRAME_SIZE, check_choice


class PwPGeometryInferencePartialHardlockNoVideoPath(ControlVideo2WorldInferencePartialHardlockNoVideoPath):
    """Adds the PwP switches to the experiment overrides, checks them against the checkpoint stamp after loading,
    reads the inputs at the PwP frame size, and erodes the hardlock mask with wrap-around when the VAE padding is
    circular. Off unless enabled: masking the PCD control with the hardlock mask as in Stage 1 training, and the
    ERP inference techniques (dual-orientation blended decoding, latent rotation during denoising, latitude-aware
    noise with ``pwp_latitude_noise_strength`` > 0)."""

    def __init__(
        self,
        *args,
        pwp_vae_w_padding: str,
        pwp_sphere_pos_emb: str,
        pwp_frame_size: str = NATIVE_FRAME_SIZE,
        pwp_mask_pcd: bool = False,
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
        self.pwp_mask_pcd = pwp_mask_pcd
        self.pwp_blended_decoding = pwp_blended_decoding
        self.model.pwp_latent_rotation = pwp_latent_rotation
        self.model.pwp_latitude_noise_strength = float(pwp_latitude_noise_strength)

    def pwp_techniques(self) -> dict:
        return {
            "mask_pcd": self.pwp_mask_pcd,
            "blended_decoding": self.pwp_blended_decoding,
            "latent_rotation": self.model.pwp_latent_rotation,
            "latitude_noise_strength": self.model.pwp_latitude_noise_strength,
        }

    def generate_img2world_no_video_path(self, *args, resolution: str = "720", **kwargs):
        """The frame size replaces the per-sample ``resolution`` (the JSON's "720" is ignored)."""
        log.info(f"PwP: frame size {self.pwp_frame_size} (resolution key {self.pwp_resolution})")
        log.info(f"PwP inference techniques: {self.pwp_techniques()}")
        with contextlib.ExitStack() as stack:
            if self.pwp_mask_pcd:
                stack.enter_context(
                    self._masked_control_reader(kwargs.get("guided_generation_mask"), kwargs.get("max_frames"))
                )
            if self.pwp_blended_decoding:
                # Decode the final latent at 0 and 180 degrees and blend them (the existing blended-decode function).
                original_decode = self.model.decode
                self.model.decode = lambda latent: (
                    ControlVideo2WorldInferencePartialHardlockNoVideoPathBlendedDecode.decode_blended_latent(
                        original_decode, latent
                    )
                )
                stack.callback(delattr, self.model, "decode")
            return super().generate_img2world_no_video_path(*args, resolution=self.pwp_resolution, **kwargs)

    @contextlib.contextmanager
    def _masked_control_reader(self, mask_path, max_frames):
        """The original pipeline reads the control videos with its module's ``read_and_process_control_input``; for
        this call only, it returns them black outside frame 0 of the hardlock mask (not eroded) in every frame."""
        if mask_path is None:
            raise ValueError("pwp_mask_pcd needs the hardlock mask (guided_generation_mask) of the sample")
        module = sys.modules[ControlVideo2WorldInferencePartialHardlockNoVideoPath.__module__]
        original = module.read_and_process_control_input

        def read_masked(*args, **kwargs):
            control_input_dict, mask_video_dict = original(*args, **kwargs)
            for key, control in control_input_dict.items():
                mask = self._read_guided_generation_mask(
                    mask_path,
                    h=control.shape[-2],
                    w=control.shape[-1],
                    resolution=kwargs.get("resolution", self.pwp_resolution),
                    max_frames=max_frames,
                )
                control_input_dict[key] = mask_control(control, mask)
                log.info(f"PwP: {key} is black outside the frame-0 hardlock mask in every frame (as in training)")
            return control_input_dict, mask_video_dict

        module.read_and_process_control_input = read_masked
        try:
            yield
        finally:
            module.read_and_process_control_input = original

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
