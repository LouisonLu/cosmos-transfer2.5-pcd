# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Isolated local dataloader configs for visible-region-preservation experiments."""

import torch.distributed as dist
from hydra.core.config_store import ConfigStore

from cosmos_transfer2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_transfer2._src.predict2.datasets.local_datasets.dataset_video import get_generic_dataloader, get_sampler
from cosmos_transfer2._src.transfer2.datasets.local_datasets.singleview_dataset_visible_preserve import (
    SingleViewTransferDatasetVisiblePreserve,
)


def register_dataloader_local_visible_preserve() -> None:
    """Register V2 dataloader configs without touching V1/default ones."""
    cs = ConfigStore()

    dataset_depth_visible_preserve = L(SingleViewTransferDatasetVisiblePreserve)(
        dataset_dir="PLACEHOLDER_UPDATE_DATASET_PATH",
        num_frames=93,
        video_size=(704, 1280),
        resolution="720",
        hint_key="control_input_depth",
        is_train=True,
        caption_type="t2w_qwen2p5_7b",
        use_image_context=True,
        image_context_from_rgb_first_frame=True,
        mask_image_context=True,
        mask_mode="waymo",
    )

    cs.store(
        group="data_train",
        package="dataloader_train",
        name="example_singleview_train_data_depth_visible_preserve",
        node=L(get_generic_dataloader)(
            dataset=dataset_depth_visible_preserve,
            sampler=L(get_sampler)(dataset=dataset_depth_visible_preserve) if dist.is_initialized() else None,
            batch_size=1,
            drop_last=True,
            num_workers=4,
            pin_memory=True,
        ),
    )
