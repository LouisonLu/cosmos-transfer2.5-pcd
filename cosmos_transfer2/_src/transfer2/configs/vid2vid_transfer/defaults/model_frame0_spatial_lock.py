# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Isolated model config for frame0 spatial-lock inference experiments."""

from hydra.core.config_store import ConfigStore

from cosmos_transfer2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_transfer2._src.transfer2.models.vid2vid_model_control_vace_rectified_flow_frame0_spatial_lock import (
    ControlVideo2WorldModelRectifiedFlowFrame0SpatialLock,
    ControlVideo2WorldRectifiedFlowFrame0SpatialLockConfig,
)


FSDP_CONFIG_CONTROL_VACE_RECTIFIED_FLOW_FRAME0_SPATIAL_LOCK = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(ControlVideo2WorldModelRectifiedFlowFrame0SpatialLock)(
        config=ControlVideo2WorldRectifiedFlowFrame0SpatialLockConfig(
            fsdp_shard_size=8,
        ),
        _recursive_=False,
    ),
)


def register_model_frame0_spatial_lock() -> None:
    cs = ConfigStore.instance()
    cs.store(
        group="model",
        package="_global_",
        name="fsdp_control_vace_rectified_flow_frame0_spatial_lock",
        node=FSDP_CONFIG_CONTROL_VACE_RECTIFIED_FLOW_FRAME0_SPATIAL_LOCK,
    )
