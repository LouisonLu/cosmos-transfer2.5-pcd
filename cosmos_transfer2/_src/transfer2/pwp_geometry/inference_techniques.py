# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional ERP inference techniques of the PwP hardlock pipeline (all off by default).

- Latent rotation during denoising: every network call sees the latent and everything aligned with it rolled
  horizontally by a random multiple of the patch width; the velocity is rolled back before the sampler step. The
  sampler state therefore always stays in the original orientation (the UniPC sampler keeps earlier outputs), and
  the only change is where the left/right array border, i.e. the 360° seam, falls for the network.
- Latitude-aware noise: the initial noise of each latent row at latitude φ holds ``max(1, round(W cos φ))``
  independent values spread around the circle, so rows near the poles are not filled with far more independent
  noise than their sphere area; every element stays N(0, 1). ``strength`` mixes it with the usual i.i.d. noise as
  ``sqrt(1 - strength) * iid + sqrt(strength) * latitude``.

Dual-orientation blended decoding reuses the existing 0°/180° decode of the blended-decode pipeline.
"""

import math
from dataclasses import fields

import numpy as np
import torch
from torch import Tensor

# Random streams of the techniques; the i.i.d. noise of the original sampler uses the plain seed.
_LATITUDE_NOISE_STREAM = 1
_ROTATION_STREAM = 2

# Condition fields that are not laid out on the latent or pixel grid (text and image-context tokens, scalars).
NON_SPATIAL_CONDITION_FIELDS = frozenset(
    {"crossattn_emb", "img_context_emb", "fps", "num_conditional_frames_B"}
)


def latitude_aware_noise(shape, seed: int) -> Tensor:
    """N(0, 1) noise ``[..., H, W]`` whose row at latitude φ is ``round(W cos φ)`` (at least 1) independent values
    spread around the circle (nearest neighbour, random phase per row), as float32 on the CPU."""
    *batch, height, width = shape
    rng = np.random.default_rng([seed, _LATITUDE_NOISE_STREAM])
    noise = np.empty((*batch, height, width), dtype=np.float32)
    columns = (np.arange(width) + 0.5) / width
    for row in range(height):
        latitude = math.pi * (0.5 - (row + 0.5) / height)
        count = max(1, min(width, round(width * math.cos(latitude))))
        values = rng.standard_normal((*batch, count)).astype(np.float32)
        index = np.floor((columns + rng.random()) * count).astype(np.int64) % count
        noise[..., row, :] = values[..., index]
    return torch.from_numpy(noise)


def mix_latitude_noise(iid_noise: Tensor, seed: int, strength: float) -> Tensor:
    """``sqrt(1 - strength) * iid + sqrt(strength) * latitude`` (unit variance); ``iid_noise`` itself at 0."""
    if not 0.0 <= strength <= 1.0:
        raise ValueError(f"Latitude-aware noise strength must be in [0, 1], got {strength}")
    if strength == 0.0:
        return iid_noise
    latitude = latitude_aware_noise(iid_noise.shape, seed).to(device=iid_noise.device, dtype=iid_noise.dtype)
    return math.sqrt(1.0 - strength) * iid_noise + math.sqrt(strength) * latitude


class RotationSchedule:
    """Seeded random roll per network call, a multiple of the patch width in ``[0, latent width)``."""

    def __init__(self, seed: int, patch: int) -> None:
        self.patch = patch
        self.rng = np.random.default_rng([seed, _ROTATION_STREAM])
        self.shifts: list[int] = []

    def next(self, latent_width: int) -> int:
        if latent_width % self.patch:
            raise ValueError(f"Latent width {latent_width} is not a multiple of the patch width {self.patch}")
        shift = int(self.rng.integers(latent_width // self.patch)) * self.patch
        self.shifts.append(shift)
        return shift


def roll_width(x: Tensor, shift: int) -> Tensor:
    return x if shift == 0 else torch.roll(x, shifts=shift, dims=-1)


def roll_condition(condition, shift: int, latent_hw: tuple[int, int], patch: int):
    """The condition with every field on the latent or pixel grid rolled with the latent.

    A tensor field whose last two dims are ``k`` times the latent ``(H, W)`` (``k = 1``: latent, ``k = 8``: pixels)
    is rolled by ``k * shift``; a spatio-temporal control weight ``[M, B, T, H/patch, W/patch, 1]`` by
    ``shift / patch``. Any other tensor field must be listed in ``NON_SPATIAL_CONDITION_FIELDS``; an unknown one
    raises, so no grid-aligned input is silently left unrolled.
    """
    height, width = latent_hw
    if shift % patch:
        raise ValueError(f"Roll {shift} is not a multiple of the patch width {patch}")
    kwargs = {field.name: getattr(condition, field.name) for field in fields(condition)}
    for name, value in kwargs.items():
        if not isinstance(value, Tensor) or value.ndim < 2:
            continue
        if name == "control_context_scale" and value.ndim == 6:
            if tuple(value.shape[-3:-1]) != (height // patch, width // patch):
                raise ValueError(f"Control weight map {tuple(value.shape)} does not match the latent {latent_hw}")
            kwargs[name] = torch.roll(value, shifts=shift // patch, dims=-2)
            continue
        scale = value.shape[-1] // width
        if scale >= 1 and tuple(value.shape[-2:]) == (scale * height, scale * width):
            kwargs[name] = torch.roll(value, shifts=scale * shift, dims=-1)
        elif name not in NON_SPATIAL_CONDITION_FIELDS:
            raise ValueError(
                f"Latent rotation does not know how to roll condition field {name!r} with shape {tuple(value.shape)}"
            )
    return type(condition)(**kwargs)


def rotated_denoise(denoise_fn, shift: int, patch: int, noise: Tensor, xt: Tensor, timesteps: Tensor, condition):
    """``denoise_fn`` on the inputs rolled by ``shift`` latent columns; the prediction is rolled back."""
    if shift == 0:
        return denoise_fn(noise, xt, timesteps, condition)
    condition = roll_condition(condition, shift, tuple(xt.shape[-2:]), patch)
    prediction = denoise_fn(roll_width(noise, shift), roll_width(xt, shift), timesteps, condition)
    return roll_width(prediction, -shift)


__all__ = [
    "NON_SPATIAL_CONDITION_FIELDS",
    "RotationSchedule",
    "latitude_aware_noise",
    "mix_latitude_noise",
    "roll_condition",
    "roll_width",
    "rotated_denoise",
]
