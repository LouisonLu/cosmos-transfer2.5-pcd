# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config wrapper for isolated masked-image-context single-view post-training."""

from cosmos_transfer2._src.transfer2.configs.vid2vid_transfer.config import make_config as _make_config
from cosmos_transfer2._src.transfer2.configs.vid2vid_transfer.defaults.dataloader_local_mask import (
    register_dataloader_local_mask,
)


def make_config():
    config = _make_config()
    register_dataloader_local_mask()
    return config


__all__ = ["make_config"]
