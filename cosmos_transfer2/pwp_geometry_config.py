# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config wrapper for the PwP geometry ablation (stage 1, stage 2 and partial-hardlock inference)."""

from omegaconf import OmegaConf

# Importing these modules registers the experiment configs with Hydra.
import cosmos_transfer2.experiments.singleview.pwp_geometry  # noqa: F401
from cosmos_transfer2._src.transfer2.configs.vid2vid_transfer.config import make_config as _make_config
from cosmos_transfer2._src.transfer2.configs.vid2vid_transfer.defaults.dataloader_local_mask import (
    register_dataloader_local_mask,
)
from cosmos_transfer2._src.transfer2.configs.vid2vid_transfer.defaults.dataloader_local_target_loss_mask import (
    register_dataloader_local_target_loss_mask,
)
from cosmos_transfer2._src.transfer2.configs.vid2vid_transfer.defaults.model_pwp_geometry import (
    register_model_pwp_geometry,
)
from cosmos_transfer2._src.transfer2.pwp_geometry.frame_size import resolution_for_frame_size
from cosmos_transfer2._src.transfer2.pwp_geometry.method import run_name_resolver


def make_config():
    if not OmegaConf.has_resolver("pwp_run_name"):
        OmegaConf.register_new_resolver("pwp_run_name", run_name_resolver)
    if not OmegaConf.has_resolver("pwp_dataset_resolution"):
        OmegaConf.register_new_resolver("pwp_dataset_resolution", resolution_for_frame_size)
    config = _make_config()
    register_dataloader_local_mask()
    register_dataloader_local_target_loss_mask()
    register_model_pwp_geometry()
    return config


__all__ = ["make_config"]
