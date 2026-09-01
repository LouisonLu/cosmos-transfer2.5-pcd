# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-view post-training with paired masked RGB/PCD and target-validity loss masks."""

import os

from hydra.core.config_store import ConfigStore

from cosmos_transfer2.config import DEFAULT_BASE_EXPERIMENT, MODEL_CHECKPOINTS, ModelKey, ModelVariant

DEFAULT_DEPTH_MODEL_KEY = ModelKey(variant=ModelVariant.DEPTH)
DEPTH_CHECKPOINT = MODEL_CHECKPOINTS[DEFAULT_DEPTH_MODEL_KEY]


transfer2_singleview_posttrain_pcd_masked_rgb_target_loss_mask = dict(
    defaults=[
        DEFAULT_BASE_EXPERIMENT,
        {"override /model": "fsdp_control_vace_rectified_flow_target_loss_mask"},
        {"override /data_train": "example_singleview_train_data_depth_target_loss_mask"},
        {"override /conditioner": "video_prediction_control_conditioner_image_context"},
    ],
    job=dict(
        project="cosmos_transfer2_posttrain",
        group="local_single_view_target_loss_mask",
        name="transfer2_singleview_posttrain_pcd_masked_rgb_target_loss_mask",
    ),
    checkpoint=dict(
        save_iter=250,
        load_path=DEPTH_CHECKPOINT.s3.uri,
        load_training_state=False,
        strict_resume=False,
        load_from_object_store=dict(enabled=False),
        save_to_object_store=dict(enabled=False),
    ),
    model=dict(
        config=dict(
            hint_keys="depth",
            base_load_from=None,
            freeze_base_model=False,
            use_reference_image=True,
            min_num_conditional_frames=1,
            max_num_conditional_frames=1,
            target_loss_mask_spatial_guard_px=8,
            target_loss_mask_temporal_guard_frames=1,
            target_loss_mask_min_valid_ratio=0.01,
            net=dict(
                extra_image_context_dim=1152,
                share_q_in_i2v_cross_attn=True,
                img_context_deep_proj=False,
            ),
        ),
    ),
    dataloader_train=dict(
        dataset=dict(
            control_input_type="depth",
            target_video_dir="rgb_videos",
            input_video_dir=None,
            control_video_dir_override="pcd_videos",
            control_video_suffix="_erp_pose",
            use_image_context=True,
            image_context_from_rgb_first_frame=True,
            mask_image_context=False,
            mask_depth_control=False,
            target_loss_mask_dir="mask",
            target_loss_mask_suffix="_mask",
            prompt_dir="prompts",
            prompt_suffix="_prompt",
            enforce_mask_on_target=True,
            enforce_mask_on_condition=True,
            use_control_input_as_video_condition=True,
        ),
    ),
    trainer=dict(
        max_iter=5000,
        straggler_detection=dict(enabled=False),
        callbacks=dict(
            heart_beat=dict(save_s3=False),
            iter_speed=dict(save_s3=False),
            device_monitor=dict(save_s3=False),
            wandb=dict(save_s3=False),
            wandb_10x=dict(save_s3=False),
            dataloader_speed=dict(save_s3=False),
            frame_loss_log=dict(save_s3=False),
        ),
    ),
    scheduler=dict(
        cycle_lengths=[5000],
    ),
    model_parallel=dict(
        context_parallel_size=int(os.environ.get("WORLD_SIZE", "1")),
    ),
)


cs = ConfigStore.instance()
cs.store(
    group="experiment",
    package="_global_",
    name=transfer2_singleview_posttrain_pcd_masked_rgb_target_loss_mask["job"]["name"],
    node=transfer2_singleview_posttrain_pcd_masked_rgb_target_loss_mask,
)


__all__ = ["transfer2_singleview_posttrain_pcd_masked_rgb_target_loss_mask"]
