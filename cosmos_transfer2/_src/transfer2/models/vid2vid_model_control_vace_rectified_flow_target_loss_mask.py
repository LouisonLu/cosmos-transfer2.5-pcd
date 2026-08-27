# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Isolated rectified-flow model with a target-validity masked training loss."""

import attrs
import torch
import torch.distributed as dist
from einops import rearrange

from cosmos_transfer2._src.transfer2.models.vid2vid_model_control_vace_rectified_flow import (
    ControlVideo2WorldModelRectifiedFlow,
    ControlVideo2WorldRectifiedFlowConfig,
)
from cosmos_transfer2._src.transfer2.utils.target_loss_mask import (
    build_safe_latent_valid_mask,
    masked_mse_sums_per_sample,
)


@attrs.define(slots=False)
class ControlVideo2WorldTargetLossMaskRectifiedFlowConfig(ControlVideo2WorldRectifiedFlowConfig):
    """Configuration for conservative target-validity loss masking."""

    target_loss_mask_spatial_guard_px: int = 8
    target_loss_mask_temporal_guard_frames: int = 1
    target_loss_mask_min_valid_ratio: float = 0.01


class ControlVideo2WorldModelTargetLossMaskRectifiedFlow(ControlVideo2WorldModelRectifiedFlow):
    """Apply rectified-flow MSE only where paired target RGB is valid."""

    config: ControlVideo2WorldTargetLossMaskRectifiedFlowConfig

    def _context_parallel_masked_mean(
        self,
        local_numerator_B: torch.Tensor,
        local_denominator_B: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize by global valid elements while retaining local gradients."""
        cp_group = self.get_context_parallel_group()
        cp_size = 1 if cp_group is None else cp_group.size()
        if cp_size == 1:
            return local_numerator_B / local_denominator_B.clamp_min(1.0)

        global_denominator_B = local_denominator_B.detach().clone()
        dist.all_reduce(global_denominator_B, op=dist.ReduceOp.SUM, group=cp_group)

        # Gradients are averaged across context-parallel ranks. Multiplication by
        # cp_size makes that average equal the globally normalized masked sum.
        return local_numerator_B * cp_size / global_denominator_B.clamp_min(1.0)

    def forward(self, data_batch: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        if "target_loss_mask" not in data_batch:
            raise KeyError(
                "target_loss_mask is required. Use SingleViewTransferDatasetTargetLossMask "
                "with this model."
            )

        if self.config.text_encoder_config is not None and self.config.text_encoder_config.compute_online:
            text_embeddings = self.text_encoder.compute_text_embeddings_online(data_batch, self.input_caption_key)
            data_batch["t5_text_embeddings"] = text_embeddings
            data_batch["t5_text_mask"] = torch.ones(
                text_embeddings.shape[0], text_embeddings.shape[1], device="cuda"
            )

        _, x0_B_C_T_H_W, condition = self.get_data_and_condition(data_batch)
        latent_channels = x0_B_C_T_H_W.shape[1]

        target_loss_mask = data_batch["target_loss_mask"].to(device=x0_B_C_T_H_W.device)
        latent_valid_mask = build_safe_latent_valid_mask(
            target_loss_mask,
            x0_B_C_T_H_W.shape[2:],
            spatial_guard_px=self.config.target_loss_mask_spatial_guard_px,
            temporal_guard_frames=self.config.target_loss_mask_temporal_guard_frames,
            min_valid_ratio=self.config.target_loss_mask_min_valid_ratio,
        ).to(dtype=x0_B_C_T_H_W.dtype)
        full_latent_valid_ratio_B = latent_valid_mask.float().flatten(1).mean(dim=1)

        epsilon_B_C_T_H_W = torch.randn(x0_B_C_T_H_W.size(), **self.tensor_kwargs_fp32)
        batch_size = x0_B_C_T_H_W.size(0)
        t_B = self.rectified_flow.sample_train_time(batch_size).to(**self.tensor_kwargs_fp32)
        t_B = rearrange(t_B, "b -> b 1")

        # Pack the mask with x0 so the existing context-parallel split applies
        # exactly the same temporal/spatial partition to both tensors.
        x0_and_mask = torch.cat([x0_B_C_T_H_W, latent_valid_mask], dim=1)
        x0_and_mask, condition, epsilon_B_C_T_H_W, t_B = self.broadcast_split_for_model_parallelsim(
            x0_and_mask, condition, epsilon_B_C_T_H_W, t_B
        )
        x0_B_C_T_H_W = x0_and_mask[:, :latent_channels]
        latent_valid_mask = x0_and_mask[:, latent_channels : latent_channels + 1]

        timesteps = self.rectified_flow.get_discrete_timestamp(t_B, self.tensor_kwargs_fp32)
        if self.config.use_high_sigma_strategy:
            raise NotImplementedError("High sigma strategy is buggy when using CP")

        sigmas = self.rectified_flow.get_sigmas(timesteps, self.tensor_kwargs_fp32)
        timesteps = rearrange(timesteps, "b -> b 1")
        sigmas = rearrange(sigmas, "b -> b 1")
        xt_B_C_T_H_W, vt_B_C_T_H_W = self.rectified_flow.get_interpolation(
            epsilon_B_C_T_H_W, x0_B_C_T_H_W, sigmas
        )

        vt_pred_B_C_T_H_W = self.denoise(
            noise=epsilon_B_C_T_H_W,
            xt_B_C_T_H_W=xt_B_C_T_H_W.to(**self.tensor_kwargs),
            timesteps_B_T=timesteps,
            condition=condition,
        )

        local_numerator_B, local_denominator_B = masked_mse_sums_per_sample(
            vt_pred_B_C_T_H_W,
            vt_B_C_T_H_W,
            latent_valid_mask,
        )
        per_instance_loss = self._context_parallel_masked_mean(local_numerator_B, local_denominator_B)

        time_weights_B = self.rectified_flow.train_time_weight(timesteps, self.tensor_kwargs_fp32).reshape(-1)
        loss = torch.mean(time_weights_B * per_instance_loss)
        output_batch = {
            "x0": x0_B_C_T_H_W,
            "xt": xt_B_C_T_H_W,
            "sigma": sigmas,
            "condition": condition,
            "model_pred": vt_pred_B_C_T_H_W,
            "edm_loss": loss,
            "timesteps": timesteps,
            "per_instance_loss": per_instance_loss,
            "n_cond_frames": condition.num_conditional_frames_B,
            "target_loss_mask_valid_ratio": full_latent_valid_ratio_B,
            "target_loss_mask_local_elements": local_denominator_B.detach(),
        }
        return output_batch, loss


__all__ = [
    "ControlVideo2WorldModelTargetLossMaskRectifiedFlow",
    "ControlVideo2WorldTargetLossMaskRectifiedFlowConfig",
]
