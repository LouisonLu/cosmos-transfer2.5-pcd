# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Isolated local dataloader for paired masked-target RGB/PCD training."""

import torch.distributed as dist
from hydra.core.config_store import ConfigStore

from cosmos_transfer2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_transfer2._src.predict2.datasets.local_datasets.dataset_video import get_generic_dataloader, get_sampler
from cosmos_transfer2._src.transfer2.datasets.local_datasets.singleview_dataset_target_loss_mask import (
    SingleViewTransferDatasetTargetLossMask,
)


def register_dataloader_local_target_loss_mask() -> None:
    cs = ConfigStore.instance()

    dataset = L(SingleViewTransferDatasetTargetLossMask)(
        dataset_dir="PLACEHOLDER_UPDATE_DATASET_PATH",
        num_frames=93,
        video_size=(704, 1280),
        resolution="720",
        hint_key="control_input_depth",
        is_train=True,
        caption_type="t2w_qwen2p5_7b",
        target_video_dir="rgb_videos",
        input_video_dir=None,
        control_video_dir_override="pcd_videos",
        control_video_suffix="_erp_pose",
        use_image_context=True,
        image_context_from_rgb_first_frame=True,
        mask_image_context=False,
        mask_depth_control=False,
        mask_mode="waymo",
        target_loss_mask_dir="mask",
        target_loss_mask_suffix="_mask",
        target_loss_mask_threshold=128,
        target_loss_mask_white_is_valid=True,
        target_loss_mask_aspect_ratio_tolerance=0.01,
        prompt_dir="prompts",
        prompt_suffix="_prompt",
        enforce_mask_on_target=True,
        enforce_mask_on_condition=False,
        use_control_input_as_video_condition=True,
        strict_masked_input_validation=True,
        strict_masked_target_validation=True,
        strict_masked_control_validation=False,
        masked_input_black_threshold=24,
        max_invalid_nonblack_ratio=0.05,
        validate_file_pairs=True,
    )

    cs.store(
        group="data_train",
        package="dataloader_train",
        name="example_singleview_train_data_depth_target_loss_mask",
        node=L(get_generic_dataloader)(
            dataset=dataset,
            sampler=L(get_sampler)(dataset=dataset) if dist.is_initialized() else None,
            batch_size=1,
            drop_last=True,
            num_workers=4,
            pin_memory=True,
        ),
    )


__all__ = ["register_dataloader_local_target_loss_mask"]
