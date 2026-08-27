# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Isolated model config for target-validity masked rectified-flow training."""

from hydra.core.config_store import ConfigStore

from cosmos_transfer2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_transfer2._src.transfer2.models.vid2vid_model_control_vace_rectified_flow_target_loss_mask import (
    ControlVideo2WorldModelTargetLossMaskRectifiedFlow,
    ControlVideo2WorldTargetLossMaskRectifiedFlowConfig,
)

FSDP_CONFIG_CONTROL_VACE_RECTIFIED_FLOW_TARGET_LOSS_MASK = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(ControlVideo2WorldModelTargetLossMaskRectifiedFlow)(
        config=ControlVideo2WorldTargetLossMaskRectifiedFlowConfig(
            fsdp_shard_size=8,
            target_loss_mask_spatial_guard_px=8,
            target_loss_mask_temporal_guard_frames=1,
            target_loss_mask_min_valid_ratio=0.01,
        ),
        _recursive_=False,
    ),
)


def register_model_target_loss_mask() -> None:
    cs = ConfigStore.instance()
    cs.store(
        group="model",
        package="_global_",
        name="fsdp_control_vace_rectified_flow_target_loss_mask",
        node=FSDP_CONFIG_CONTROL_VACE_RECTIFIED_FLOW_TARGET_LOSS_MASK,
    )


__all__ = ["register_model_target_loss_mask"]
