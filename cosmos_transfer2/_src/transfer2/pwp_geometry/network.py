# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transfer2 control DiT with an optional spherical XYZ position embedding and the PwP method stamp."""

from typing import Optional

import torch

from cosmos_transfer2._src.transfer2.networks.minimal_v4_lvg_dit_control_vace import MinimalV4LVGControlVaceDiT
from cosmos_transfer2._src.transfer2.pwp_geometry.method import STAMP_UNSET, check_choice
from cosmos_transfer2._src.transfer2.pwp_geometry.sphere_pos_emb import SphereXYZPositionalEmbedding

PWP_NET_TARGET = "cosmos_transfer2._src.transfer2.pwp_geometry.network.PwPSphereControlVaceDiT"


class PwPSphereControlVaceDiT(MinimalV4LVGControlVaceDiT):
    """``MinimalV4LVGControlVaceDiT`` plus:

    - ``pwp_method_stamp``: a persistent buffer that records the PwP switches the weights were trained with, so
      the record travels with every checkpoint format (DCP, consolidated .pt, EMA conversion);
    - with ``pwp_sphere_pos_emb="xyz"``: a spherical XYZ embedding added to every patch-embedded token stream
      (main branch and control branch). The 3D RoPE is unchanged.

    With ``pwp_sphere_pos_emb="off"`` the forward pass is exactly the parent's.
    """

    def __init__(self, *args, pwp_sphere_pos_emb: str = "off", pwp_sphere_pos_emb_hidden_dim: int = 256, **kwargs):
        check_choice("sphere_pos_emb", pwp_sphere_pos_emb)
        super().__init__(*args, **kwargs)
        self.pwp_sphere_pos_emb = pwp_sphere_pos_emb
        self.register_buffer("pwp_method_stamp", torch.tensor(STAMP_UNSET, dtype=torch.long), persistent=True)
        if pwp_sphere_pos_emb == "xyz":
            self.pwp_sphere_xyz = SphereXYZPositionalEmbedding(self.model_channels, pwp_sphere_pos_emb_hidden_dim)
        else:
            self.pwp_sphere_xyz = None
        self._init_pwp_states()

    def _init_pwp_states(self) -> None:
        self.pwp_method_stamp.fill_(-1)
        if self.pwp_sphere_xyz is not None:
            self.pwp_sphere_xyz.reset_parameters()

    def init_weights(self):
        super().init_weights()
        # The parent constructor calls init_weights before the PwP states exist; build_net calls it again after
        # materializing the meta-device network, which is when the stamp and the XYZ layers are initialized.
        if hasattr(self, "pwp_method_stamp"):
            self._init_pwp_states()

    def prepare_embedded_sequence(
        self,
        x_B_C_T_H_W: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        embedder=None,
    ):
        x_B_T_H_W_D, rope_emb, extra_pos_emb = super().prepare_embedded_sequence(
            x_B_C_T_H_W, fps=fps, padding_mask=padding_mask, embedder=embedder
        )
        if self.pwp_sphere_xyz is not None:
            _, _, token_h, token_w, _ = x_B_T_H_W_D.shape
            sphere_emb = self.pwp_sphere_xyz(padding_mask, token_h, token_w)
            x_B_T_H_W_D = x_B_T_H_W_D + sphere_emb.to(dtype=x_B_T_H_W_D.dtype)
        return x_B_T_H_W_D, rope_emb, extra_pos_emb


__all__ = ["PWP_NET_TARGET", "PwPSphereControlVaceDiT"]
