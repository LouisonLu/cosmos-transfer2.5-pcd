# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Partial-hardlock inference model with the PwP DiT, optional circular VAE width padding and the optional ERP
inference techniques (latent rotation, latitude-aware noise; see inference_techniques.py)."""

import contextlib
import sys
from typing import Dict, Optional

import attrs
import torch

from cosmos_transfer2._src.imaginaire.utils import log
from cosmos_transfer2._src.transfer2.models.vid2vid_model_control_vace_rectified_flow_partial_hardlock import (
    ControlVideo2WorldModelRectifiedFlowPartialHardlock,
    ControlVideo2WorldRectifiedFlowPartialHardlockConfig,
)
from cosmos_transfer2._src.transfer2.pwp_geometry.inference_techniques import (
    RotationSchedule,
    mix_latitude_noise,
    rotated_denoise,
)
from cosmos_transfer2._src.transfer2.pwp_geometry.method import UNSET, check_choice
from cosmos_transfer2._src.transfer2.pwp_geometry.model import PwPGeometryModelMixin


@attrs.define(slots=False)
class PwPGeometryPartialHardlockConfig(ControlVideo2WorldRectifiedFlowPartialHardlockConfig):
    """The two switches that change the network or the VAE; the area loss only affects training."""

    pwp_vae_w_padding: str = UNSET
    pwp_sphere_pos_emb: str = UNSET
    pwp_sphere_pos_emb_hidden_dim: int = 256


class _LatitudeNoiseMisc:
    """``misc`` of the hardlock model module, whose ``arch_invariant_rand`` mixes in the latitude-aware noise."""

    def __init__(self, misc, strength: float) -> None:
        self._misc = misc
        self._strength = strength

    def __getattr__(self, name: str):
        return getattr(self._misc, name)

    def arch_invariant_rand(self, shape, dtype, device, seed=None):
        noise = self._misc.arch_invariant_rand(shape, dtype, device, seed)
        return mix_latitude_noise(noise, seed, self._strength)


@contextlib.contextmanager
def _latitude_noise_in_sampler(strength: float):
    """The original sampling loop draws its initial noise from its module's ``misc``; swap it for this call only."""
    module = sys.modules[ControlVideo2WorldModelRectifiedFlowPartialHardlock.__module__]
    original = module.misc
    module.misc = _LatitudeNoiseMisc(original, strength)
    try:
        yield
    finally:
        module.misc = original


class PwPGeometryPartialHardlockModel(PwPGeometryModelMixin, ControlVideo2WorldModelRectifiedFlowPartialHardlock):
    config: PwPGeometryPartialHardlockConfig

    # Inference techniques, set by the PwP inference pipeline; all off by default.
    pwp_latent_rotation: bool = False
    pwp_latitude_noise_strength: float = 0.0
    _pwp_rotation: Optional[RotationSchedule] = None
    _pwp_latent_roll: int = 0

    def __init__(self, config: PwPGeometryPartialHardlockConfig, *args, **kwargs):
        check_choice("vae_w_padding", config.pwp_vae_w_padding)
        check_choice("sphere_pos_emb", config.pwp_sphere_pos_emb)
        super().__init__(config, *args, **kwargs)
        self._pwp_setup_vae()

    def _pwp_expected_settings(self) -> dict[str, str]:
        return {
            "vae_w_padding": self.config.pwp_vae_w_padding,
            "sphere_pos_emb": self.config.pwp_sphere_pos_emb,
        }

    def _pwp_networks(self) -> list[torch.nn.Module]:
        # Inference loads the EMA weights into the regular network; only that network is used.
        return [self.net]

    @torch.no_grad()
    def generate_samples_from_batch(self, data_batch: Dict, guidance: float = 1.5, seed: int = 1, **kwargs):
        """The original sampling loop; the enabled techniques only wrap its initial noise and its network calls."""
        strength = self.pwp_latitude_noise_strength
        if not self.pwp_latent_rotation and strength == 0.0:
            return super().generate_samples_from_batch(data_batch, guidance=guidance, seed=seed, **kwargs)
        if self.net.is_context_parallel_enabled:
            raise ValueError("Latent rotation and latitude-aware noise need the whole latent on one GPU (no context parallelism).")
        if self.pwp_latent_rotation and not hasattr(self.net, "pwp_token_roll"):
            raise TypeError(f"Latent rotation needs the PwP network, got {type(self.net).__name__}")
        rotation = RotationSchedule(seed, self.net.patch_spatial) if self.pwp_latent_rotation else None
        with contextlib.ExitStack() as stack:
            if strength > 0.0:
                stack.enter_context(_latitude_noise_in_sampler(strength))
            if rotation is not None:
                self._pwp_rotation = rotation
                stack.callback(setattr, self, "_pwp_rotation", None)
            samples = super().generate_samples_from_batch(data_batch, guidance=guidance, seed=seed, **kwargs)
        if rotation is not None:
            log.info(f"PwP latent rotation: roll per step in latent columns {rotation.shifts}")
        return samples

    def get_velocity_fn_from_batch(self, data_batch: Dict, guidance: float = 1.5, is_negative_prompt: bool = False):
        velocity_fn = super().get_velocity_fn_from_batch(data_batch, guidance, is_negative_prompt=is_negative_prompt)
        rotation = self._pwp_rotation
        if rotation is None:
            return velocity_fn

        def rotated_velocity_fn(noise: torch.Tensor, noise_x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
            # One roll per sampler step, shared by the conditional and unconditional calls.
            self._pwp_latent_roll = rotation.next(noise_x.shape[-1])
            try:
                return velocity_fn(noise, noise_x, timestep)
            finally:
                self._pwp_latent_roll = 0

        return rotated_velocity_fn

    def denoise(self, noise: torch.Tensor, xt_B_C_T_H_W: torch.Tensor, timesteps_B_T: torch.Tensor, condition):
        shift = self._pwp_latent_roll
        if shift == 0:
            return super().denoise(noise, xt_B_C_T_H_W, timesteps_B_T, condition)
        patch = self.net.patch_spatial
        self.net.pwp_token_roll = shift // patch
        try:
            return rotated_denoise(super().denoise, shift, patch, noise, xt_B_C_T_H_W, timesteps_B_T, condition)
        finally:
            self.net.pwp_token_roll = 0


__all__ = ["PwPGeometryPartialHardlockConfig", "PwPGeometryPartialHardlockModel"]
