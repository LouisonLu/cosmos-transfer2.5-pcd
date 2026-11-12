# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Isolated model configs for the PwP geometry ablation (training and partial-hardlock inference)."""

from hydra.core.config_store import ConfigStore

from cosmos_transfer2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_transfer2._src.transfer2.pwp_geometry.inference_model import (
    PwPGeometryPartialHardlockConfig,
    PwPGeometryPartialHardlockModel,
)
from cosmos_transfer2._src.transfer2.pwp_geometry.model import (
    PwPGeometryRectifiedFlowConfig,
    PwPGeometryRectifiedFlowModel,
)

# Mirror FSDP_CONFIG_CONTROL_VACE_RECTIFIED_FLOW(_PARTIAL_HARDLOCK); the PwP switches have no default.
FSDP_CONFIG_CONTROL_VACE_RECTIFIED_FLOW_PWP_GEOMETRY = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(PwPGeometryRectifiedFlowModel)(
        config=PwPGeometryRectifiedFlowConfig(
            fsdp_shard_size=8,
        ),
        _recursive_=False,
    ),
)

FSDP_CONFIG_CONTROL_VACE_RECTIFIED_FLOW_PWP_GEOMETRY_PARTIAL_HARDLOCK = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(PwPGeometryPartialHardlockModel)(
        config=PwPGeometryPartialHardlockConfig(
            fsdp_shard_size=8,
        ),
        _recursive_=False,
    ),
)


def register_model_pwp_geometry() -> None:
    cs = ConfigStore.instance()
    cs.store(
        group="model",
        package="_global_",
        name="fsdp_control_vace_rectified_flow_pwp_geometry",
        node=FSDP_CONFIG_CONTROL_VACE_RECTIFIED_FLOW_PWP_GEOMETRY,
    )
    cs.store(
        group="model",
        package="_global_",
        name="fsdp_control_vace_rectified_flow_pwp_geometry_partial_hardlock",
        node=FSDP_CONFIG_CONTROL_VACE_RECTIFIED_FLOW_PWP_GEOMETRY_PARTIAL_HARDLOCK,
    )


__all__ = ["register_model_pwp_geometry"]
