# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config wrapper for visible-region-preserving masked first-frame training."""

# Importing this module registers the V2 single-view experiment configs with Hydra.
import cosmos_transfer2.experiments.singleview.cosmos_singleview_visible_preserve  # noqa: F401

from cosmos_transfer2._src.transfer2.configs.vid2vid_transfer.config import make_config as _make_config
from cosmos_transfer2._src.transfer2.configs.vid2vid_transfer.defaults.dataloader_local_visible_preserve import (
    register_dataloader_local_visible_preserve,
)
from cosmos_transfer2._src.transfer2.configs.vid2vid_transfer.defaults.model_visible_preserve import (
    register_model_visible_preserve,
)


def make_config():
    config = _make_config()
    register_dataloader_local_visible_preserve()
    register_model_visible_preserve()
    return config


__all__ = ["make_config"]
