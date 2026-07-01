# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-view config for first-frame visible-region partial hardlock inference."""

import os

from hydra.core.config_store import ConfigStore

from cosmos_transfer2.config import DEFAULT_BASE_EXPERIMENT, MODEL_CHECKPOINTS, ModelKey, ModelVariant

DEFAULT_DEPTH_MODEL_KEY = ModelKey(variant=ModelVariant.DEPTH)
DEPTH_CHECKPOINT = MODEL_CHECKPOINTS[DEFAULT_DEPTH_MODEL_KEY]


transfer2_singleview_partical_hardlock_pcd_rgb_image_context_example = dict(
    defaults=[
        DEFAULT_BASE_EXPERIMENT,
        {"override /model": "fsdp_control_vace_rectified_flow_partical_hardlock"},
        {"override /data_train": "example_singleview_train_data_depth_mask"},
        {"override /conditioner": "video_prediction_control_conditioner_image_context"},
    ],
    job=dict(
        project="cosmos_transfer2_posttrain",
        group="local_single_view",
        name="transfer2_singleview_partical_hardlock_pcd_rgb_image_context_example",
    ),
    checkpoint=dict(
        save_iter=1000,
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
            use_reference_image=True,
            min_num_conditional_frames=0,
            max_num_conditional_frames=0,
            denoise_replace_gt_frames=False,
            freeze_base_model=False,
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
            use_image_context=True,
            image_context_from_rgb_first_frame=True,
            mask_image_context=True,
            mask_mode="waymo",
        ),
    ),
    trainer=dict(
        max_iter=5000,
        straggler_detection=dict(enabled=False),
        callbacks=dict(
            heart_beat=dict(save_s3=False),
            iter_speed=dict(save_s3=False),
            device_monitor=dict(save_s3=False),
            every_n_sample_reg=dict(save_s3=False, every_n=200),
            every_n_sample_ema=dict(save_s3=False, every_n=200),
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

for _item in [transfer2_singleview_partical_hardlock_pcd_rgb_image_context_example]:
    _name: str = _item["job"]["name"]  # pyrefly: ignore
    cs.store(
        group="experiment",
        package="_global_",
        name=_name,
        node=_item,
    )
