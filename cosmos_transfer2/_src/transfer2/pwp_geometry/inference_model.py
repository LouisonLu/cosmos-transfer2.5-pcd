# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Partial-hardlock inference model with the PwP DiT and optional circular VAE width padding."""

import attrs
import torch

from cosmos_transfer2._src.transfer2.models.vid2vid_model_control_vace_rectified_flow_partial_hardlock import (
    ControlVideo2WorldModelRectifiedFlowPartialHardlock,
    ControlVideo2WorldRectifiedFlowPartialHardlockConfig,
)
from cosmos_transfer2._src.transfer2.pwp_geometry.method import UNSET, check_choice
from cosmos_transfer2._src.transfer2.pwp_geometry.model import PwPGeometryModelMixin


@attrs.define(slots=False)
class PwPGeometryPartialHardlockConfig(ControlVideo2WorldRectifiedFlowPartialHardlockConfig):
    """The two switches that change the network or the VAE; the area loss only affects training."""

    pwp_vae_w_padding: str = UNSET
    pwp_sphere_pos_emb: str = UNSET
    pwp_sphere_pos_emb_hidden_dim: int = 256


class PwPGeometryPartialHardlockModel(PwPGeometryModelMixin, ControlVideo2WorldModelRectifiedFlowPartialHardlock):
    config: PwPGeometryPartialHardlockConfig

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


__all__ = ["PwPGeometryPartialHardlockConfig", "PwPGeometryPartialHardlockModel"]
