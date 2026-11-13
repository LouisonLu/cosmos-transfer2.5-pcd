# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PwP geometry training model: one rectified-flow model whose loss and geometry are chosen by config switches.

- ``pwp_area_loss``: ``off`` | ``sphere_area`` (cos-latitude weight on the squared error)
- ``pwp_vae_w_padding``: ``zero`` | ``circular`` (horizontal wrap-around in every VAE convolution)
- ``pwp_sphere_pos_emb``: ``off`` | ``xyz`` (spherical XYZ embedding on the DiT tokens)
- ``pwp_frame_size``: ``1280x640`` | ``legacy_720`` (data setting: the dataset resolution follows it)
- ``pwp_target_loss_mask``: Stage 2 only, set by the stage 2 experiment (loss on valid target pixels only)

The loss weight is ``target_valid_mask × area_weight``. With the area loss and the target mask both off, the
parent's ``forward`` runs unchanged.
"""

import copy
import os
import sys
import time
from pathlib import Path
from typing import Optional

import attrs
import torch
import torch.distributed as dist
from einops import rearrange
from omegaconf import DictConfig, open_dict
from torch.distributed._tensor import DTensor

from cosmos_transfer2._src.imaginaire.lazy_config.registry import locate
from cosmos_transfer2._src.imaginaire.utils import log
from cosmos_transfer2._src.transfer2.models.vid2vid_model_control_vace_rectified_flow import (
    ControlVideo2WorldModelRectifiedFlow,
    ControlVideo2WorldRectifiedFlowConfig,
)
from cosmos_transfer2._src.transfer2.networks.minimal_v4_lvg_dit_control_vace import MinimalV4LVGControlVaceDiT
from cosmos_transfer2._src.transfer2.pwp_geometry.area_weight import build_sphere_area_weights
from cosmos_transfer2._src.transfer2.pwp_geometry.masks import build_safe_latent_valid_mask_for_vae
from cosmos_transfer2._src.transfer2.pwp_geometry.method import (
    STAMP_UNSET,
    UNSET,
    PwPMethod,
    append_jsonl,
    check_choice,
    checkpoint_mismatches,
    git_state,
)
from cosmos_transfer2._src.transfer2.pwp_geometry.network import PWP_NET_TARGET
from cosmos_transfer2._src.transfer2.pwp_geometry.vae_circular import convert_wan_tokenizer_to_circular
from cosmos_transfer2._src.transfer2.utils.target_loss_mask import masked_mse_sums_per_sample

REPO_DIR = Path(__file__).resolve().parents[4]


def _full_weight_norm(param: torch.Tensor) -> float:
    """L2 norm of a (possibly FSDP-sharded) parameter. Collective under FSDP: every rank must call it."""
    value = param.detach()
    if isinstance(value, DTensor):
        value = value.full_tensor()
    return value.float().norm().item()


def pwp_net_config(net_config, sphere_pos_emb: str, hidden_dim: int):
    """Copy of the DiT config that builds ``PwPSphereControlVaceDiT`` with the same arguments."""
    target = net_config["_target_"]
    target_cls = locate(target) if isinstance(target, str) else target
    if target_cls is not MinimalV4LVGControlVaceDiT:
        raise TypeError(f"The PwP geometry model expects a MinimalV4LVGControlVaceDiT network, got {target}")
    new_config = copy.deepcopy(net_config)
    if isinstance(new_config, DictConfig):
        with open_dict(new_config):
            new_config["_target_"] = PWP_NET_TARGET
            new_config["pwp_sphere_pos_emb"] = sphere_pos_emb
            new_config["pwp_sphere_pos_emb_hidden_dim"] = hidden_dim
    else:
        new_config.update(
            _target_=PWP_NET_TARGET, pwp_sphere_pos_emb=sphere_pos_emb, pwp_sphere_pos_emb_hidden_dim=hidden_dim
        )
    return new_config


class PwPGeometryModelMixin:
    """Shared by the training and inference models: PwP DiT, circular VAE, and the checkpoint stamp check.

    Must precede the Cosmos model class in the bases so that ``build_net`` is intercepted.
    """

    _pwp_checked: bool = False
    _pwp_loaded_method: Optional[PwPMethod] = None
    # Training may start any method from unstamped (official) weights; inference must match them.
    _pwp_any_method_from_unstamped: bool = False

    def _pwp_expected_settings(self) -> dict[str, str]:
        """Switches that must match the stamp of the loaded checkpoint."""
        raise NotImplementedError

    def build_net(self, keep_on_cpu: bool = False):
        original = self.config.net
        self.config.net = pwp_net_config(
            original, self.config.pwp_sphere_pos_emb, self.config.pwp_sphere_pos_emb_hidden_dim
        )
        try:
            return super().build_net(keep_on_cpu=keep_on_cpu)
        finally:
            self.config.net = original

    def _pwp_setup_vae(self) -> None:
        if self.config.pwp_vae_w_padding == "circular":
            counts = convert_wan_tokenizer_to_circular(self.tokenizer)
            log.critical(f"PwP: VAE width padding converted to circular {counts}", rank0_only=False)

    def _pwp_networks(self) -> list[torch.nn.Module]:
        networks = [self.net]
        if getattr(self, "net_ema", None) is not None:
            networks.append(self.net_ema)
        return networks

    def pwp_checkpoint_method(self) -> Optional[PwPMethod]:
        """Decode the stamp of the loaded weights (``None``: no stamp, i.e. official or original-pipeline)."""
        stamps = {tuple(net.pwp_method_stamp.tolist()) for net in self._pwp_networks()}
        # A consolidated .pt holds one set of weights, so the other network may still be unstamped.
        stamps.discard(STAMP_UNSET)
        if len(stamps) > 1:
            raise ValueError(f"The regular and EMA networks carry different PwP stamps: {sorted(stamps)}")
        return PwPMethod.from_stamp(stamps.pop() if stamps else STAMP_UNSET)

    def pwp_check_checkpoint(self) -> Optional[PwPMethod]:
        """Raise if the loaded weights were trained with different switches; run once after loading."""
        if self._pwp_checked:
            return self._pwp_loaded_method
        loaded = self.pwp_checkpoint_method()
        if loaded is None and self._pwp_any_method_from_unstamped:
            mismatches = []
        else:
            mismatches = checkpoint_mismatches(loaded, self._pwp_expected_settings())
        if mismatches:
            raise ValueError(
                "PwP settings do not match the loaded checkpoint:\n  - "
                + "\n  - ".join(mismatches)
                + "\nUse the checkpoint of the same method, or change the settings on the command line."
            )
        self._pwp_loaded_method = loaded
        self._pwp_checked = True
        log.critical(
            "PwP: loaded checkpoint stamp = " + (loaded.run_name() if loaded else "none (official/original weights)"),
            rank0_only=True,
        )
        return loaded


@attrs.define(slots=False)
class PwPGeometryRectifiedFlowConfig(ControlVideo2WorldRectifiedFlowConfig):
    """Training config. The three method switches and the frame size have no default; set them on the command line."""

    pwp_area_loss: str = UNSET
    pwp_vae_w_padding: str = UNSET
    pwp_sphere_pos_emb: str = UNSET
    pwp_frame_size: str = UNSET
    pwp_sphere_pos_emb_hidden_dim: int = 256
    pwp_target_loss_mask: bool = False
    # Filled by the experiment (interpolations); only written to the run record.
    pwp_record_job_path: str = ""
    pwp_record_load_path: Optional[str] = None
    pwp_record_dataset_dir: Optional[str] = None
    target_loss_mask_spatial_guard_px: int = 8
    target_loss_mask_temporal_guard_frames: int = 1
    target_loss_mask_min_valid_ratio: float = 0.01


class PwPGeometryRectifiedFlowModel(PwPGeometryModelMixin, ControlVideo2WorldModelRectifiedFlow):
    config: PwPGeometryRectifiedFlowConfig
    # Every method's Stage 1 starts from the official weights; resuming or Stage 2 still needs a matching stamp.
    _pwp_any_method_from_unstamped = True

    def __init__(self, config: PwPGeometryRectifiedFlowConfig, *args, **kwargs):
        self.pwp_method = PwPMethod(
            area_loss=config.pwp_area_loss,
            vae_w_padding=config.pwp_vae_w_padding,
            sphere_pos_emb=config.pwp_sphere_pos_emb,
        )
        check_choice("frame_size", config.pwp_frame_size)
        super().__init__(config, *args, **kwargs)
        self._pwp_setup_vae()
        log.critical(
            f"PwP method: {self.pwp_method.as_dict()}, frame_size={config.pwp_frame_size}, "
            f"target_loss_mask={config.pwp_target_loss_mask}",
            rank0_only=True,
        )

    def freeze_base_model(self):
        """The XYZ embedding is a new layer, so it stays trainable like the control branch."""
        super().freeze_base_model()
        if self.net.pwp_sphere_xyz is not None:
            for param in self.net.pwp_sphere_xyz.parameters():
                param.requires_grad = True

    def training_step(
        self, data_batch: dict[str, torch.Tensor], iteration: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        if self.net.pwp_sphere_xyz is not None and (iteration <= 5 or iteration % 50 == 0):
            # Record only: the zero-initialized output layer starts at 0 and must grow once training updates it.
            out_norm = _full_weight_norm(self.net.pwp_sphere_xyz.proj_out.weight)
            in_norm = _full_weight_norm(self.net.pwp_sphere_xyz.proj_in.weight)
            log.info(
                f"PwP XYZ iteration {iteration}: proj_out weight norm {out_norm:.3e}, proj_in weight norm {in_norm:.3e}"
            )
        return super().training_step(data_batch, iteration)

    def _pwp_expected_settings(self) -> dict[str, str]:
        return {
            "area_loss": self.pwp_method.area_loss,
            "vae_w_padding": self.pwp_method.vae_w_padding,
            "sphere_pos_emb": self.pwp_method.sphere_pos_emb,
        }

    def pwp_check_checkpoint(self) -> Optional[PwPMethod]:
        """Check the loaded stamp, then stamp both networks with this run's method so checkpoints carry it."""
        already_checked = self._pwp_checked
        loaded = super().pwp_check_checkpoint()
        if not already_checked:
            stamp = torch.tensor(self.pwp_method.to_stamp(), dtype=torch.long)
            for net in self._pwp_networks():
                net.pwp_method_stamp.copy_(stamp.to(net.pwp_method_stamp.device))
            if not dist.is_initialized() or dist.get_rank() == 0:
                self._pwp_write_run_record(loaded)
        return loaded

    def _pwp_write_run_record(self, loaded: Optional[PwPMethod]) -> None:
        """Append one line per (re)start to ``<job dir>/pwp_run_records.jsonl`` and to W&B config. Never raises."""
        record = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "method": self.pwp_method.as_dict(),
            "frame_size": self.config.pwp_frame_size,
            "target_loss_mask": bool(self.config.pwp_target_loss_mask),
            "init_checkpoint_method": loaded.as_dict() if loaded else None,
            "load_path": self.config.pwp_record_load_path,
            "dataset_dir": self.config.pwp_record_dataset_dir,
            "world_size": dist.get_world_size() if dist.is_initialized() else 1,
            "command": " ".join(sys.argv),
            "git": git_state(REPO_DIR),
        }
        try:
            local_root = os.environ.get("IMAGINAIRE_OUTPUT_ROOT", "/tmp/imaginaire4-output")
            record_path = Path(local_root) / self.config.pwp_record_job_path / "pwp_run_records.jsonl"
            append_jsonl(record_path, record)
            log.critical(f"PwP run record appended to {record_path}")
            import wandb

            if wandb.run:
                wandb.run.config.update(
                    {
                        **{f"pwp/{key}": value for key, value in record["method"].items()},
                        "pwp/frame_size": record["frame_size"],
                        "pwp/target_loss_mask": record["target_loss_mask"],
                        "pwp/init_checkpoint_method": loaded.run_name() if loaded else "none",
                        "pwp/git_commit": record["git"].get("commit"),
                        "pwp/git_dirty": record["git"].get("dirty"),
                    },
                    allow_val_change=True,
                )
        except Exception as error:  # the record must never stop training
            log.warning(f"PwP run record could not be written: {error}")

    def _context_parallel_weighted_mean(
        self,
        local_numerator_B: torch.Tensor,
        local_denominator_B: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize by the global weight sum while retaining local gradients."""
        cp_group = self.get_context_parallel_group()
        cp_size = 1 if cp_group is None else cp_group.size()
        if cp_size == 1:
            return local_numerator_B / local_denominator_B.clamp_min(1e-6)

        global_denominator_B = local_denominator_B.detach().clone()
        dist.all_reduce(global_denominator_B, op=dist.ReduceOp.SUM, group=cp_group)

        # Gradients are averaged across context-parallel ranks. Multiplication by
        # cp_size makes that average equal the globally normalized weighted sum.
        return local_numerator_B * cp_size / global_denominator_B.clamp_min(1e-6)

    def forward(self, data_batch: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        self.pwp_check_checkpoint()
        if not self.config.pwp_target_loss_mask and not self.pwp_method.uses_area_loss:
            return super().forward(data_batch)

        if self.config.pwp_target_loss_mask and "target_loss_mask" not in data_batch:
            raise KeyError(
                "target_loss_mask is required. Use SingleViewTransferDatasetTargetLossMask with the stage 2 experiment."
            )
        if self.pwp_method.uses_area_loss and "padding_mask" not in data_batch:
            raise KeyError("padding_mask is required to locate ERP rows for the sphere-area loss.")
        # Read before any in-place batch processing.
        padding_mask = data_batch["padding_mask"].detach().clone() if self.pwp_method.uses_area_loss else None

        if self.config.text_encoder_config is not None and self.config.text_encoder_config.compute_online:
            text_embeddings = self.text_encoder.compute_text_embeddings_online(data_batch, self.input_caption_key)
            data_batch["t5_text_embeddings"] = text_embeddings
            data_batch["t5_text_mask"] = torch.ones(
                text_embeddings.shape[0], text_embeddings.shape[1], device="cuda"
            )

        _, x0_B_C_T_H_W, condition = self.get_data_and_condition(data_batch)
        latent_channels = x0_B_C_T_H_W.shape[1]
        x0_dtype = x0_B_C_T_H_W.dtype
        _, _, latent_t, latent_h, latent_w = x0_B_C_T_H_W.shape

        loss_weight = torch.ones(
            x0_B_C_T_H_W.shape[0], 1, latent_t, latent_h, latent_w, dtype=torch.float32, device=x0_B_C_T_H_W.device
        )
        output_extras = {}
        if self.config.pwp_target_loss_mask:
            latent_valid_mask = build_safe_latent_valid_mask_for_vae(
                data_batch["target_loss_mask"].to(device=x0_B_C_T_H_W.device),
                x0_B_C_T_H_W.shape[2:],
                circular_width=self.pwp_method.uses_circular_vae,
                spatial_guard_px=self.config.target_loss_mask_spatial_guard_px,
                temporal_guard_frames=self.config.target_loss_mask_temporal_guard_frames,
                min_valid_ratio=self.config.target_loss_mask_min_valid_ratio,
            )
            loss_weight = loss_weight * latent_valid_mask
            output_extras["target_loss_mask_valid_ratio"] = latent_valid_mask.float().flatten(1).mean(dim=1)
        if self.pwp_method.uses_area_loss:
            area_weight = build_sphere_area_weights(padding_mask.to(device=x0_B_C_T_H_W.device), latent_h)
            loss_weight = loss_weight * area_weight

        epsilon_B_C_T_H_W = torch.randn(x0_B_C_T_H_W.size(), **self.tensor_kwargs_fp32)
        batch_size = x0_B_C_T_H_W.size(0)
        t_B = self.rectified_flow.sample_train_time(batch_size).to(**self.tensor_kwargs_fp32)
        t_B = rearrange(t_B, "b -> b 1")  # add a dimension for T, all frames share the same sigma

        # Pack the weight with x0 (in fp32, so bf16 latents round-trip exactly) so the existing
        # context-parallel split applies the same partition to both tensors.
        x0_and_weight = torch.cat([x0_B_C_T_H_W.float(), loss_weight], dim=1)
        x0_and_weight, condition, epsilon_B_C_T_H_W, t_B = self.broadcast_split_for_model_parallelsim(
            x0_and_weight, condition, epsilon_B_C_T_H_W, t_B
        )
        x0_B_C_T_H_W = x0_and_weight[:, :latent_channels].to(x0_dtype)
        loss_weight = x0_and_weight[:, latent_channels : latent_channels + 1]

        timesteps = self.rectified_flow.get_discrete_timestamp(t_B, self.tensor_kwargs_fp32)
        if self.config.use_high_sigma_strategy:
            raise NotImplementedError("High sigma strategy is buggy when using CP")

        sigmas = self.rectified_flow.get_sigmas(timesteps, self.tensor_kwargs_fp32)
        timesteps = rearrange(timesteps, "b -> b 1")
        sigmas = rearrange(sigmas, "b -> b 1")
        xt_B_C_T_H_W, vt_B_C_T_H_W = self.rectified_flow.get_interpolation(epsilon_B_C_T_H_W, x0_B_C_T_H_W, sigmas)

        vt_pred_B_C_T_H_W = self.denoise(
            noise=epsilon_B_C_T_H_W,
            xt_B_C_T_H_W=xt_B_C_T_H_W.to(**self.tensor_kwargs),
            timesteps_B_T=timesteps,
            condition=condition,
        )

        local_numerator_B, local_denominator_B = masked_mse_sums_per_sample(
            vt_pred_B_C_T_H_W,
            vt_B_C_T_H_W,
            loss_weight,
        )
        per_instance_loss = self._context_parallel_weighted_mean(local_numerator_B, local_denominator_B)

        time_weights_B = self.rectified_flow.train_time_weight(timesteps, self.tensor_kwargs_fp32).reshape(-1)
        loss = torch.mean(time_weights_B * per_instance_loss)
        output_batch = {
            "x0": x0_B_C_T_H_W,
            "xt": xt_B_C_T_H_W,
            "sigma": sigmas,
            "condition": condition,
            "model_pred": vt_pred_B_C_T_H_W,
            "edm_loss": loss,
            # For per-sample loss logging
            "timesteps": timesteps,
            "per_instance_loss": per_instance_loss,
            "n_cond_frames": condition.num_conditional_frames_B,
            "pwp_loss_weight_local_sum": local_denominator_B.detach(),
            **output_extras,
        }
        return output_batch, loss


__all__ = [
    "PwPGeometryModelMixin",
    "PwPGeometryRectifiedFlowConfig",
    "PwPGeometryRectifiedFlowModel",
    "pwp_net_config",
]
