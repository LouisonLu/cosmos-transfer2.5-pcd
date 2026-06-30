# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Isolated model config for visible-region-preservation experiments."""

from hydra.core.config_store import ConfigStore

from cosmos_transfer2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_transfer2._src.transfer2.models.vid2vid_model_control_vace_rectified_flow_visible_preserve import (
    ControlVideo2WorldModelRectifiedFlowVisiblePreserve,
    ControlVideo2WorldRectifiedFlowVisiblePreserveConfig,
)


FSDP_CONFIG_CONTROL_VACE_RECTIFIED_FLOW_VISIBLE_PRESERVE = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(ControlVideo2WorldModelRectifiedFlowVisiblePreserve)(
        config=ControlVideo2WorldRectifiedFlowVisiblePreserveConfig(
            fsdp_shard_size=8,
        ),
        _recursive_=False,
    ),
)


def register_model_visible_preserve() -> None:
    cs = ConfigStore.instance()
    cs.store(
        group="model",
        package="_global_",
        name="fsdp_control_vace_rectified_flow_visible_preserve",
        node=FSDP_CONFIG_CONTROL_VACE_RECTIFIED_FLOW_VISIBLE_PRESERVE,
    )
