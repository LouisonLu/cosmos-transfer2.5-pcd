# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PwP geometry ablation experiments.

Each experiment is a deep copy of the original one, so data, optimizer, scheduler, checkpoint and network
settings stay identical. Only the model class (and, for stage 1, the mask dataset), the dataset resolution, the
sampler (new video order every epoch; stage 1 masks drawn per (epoch, video), see pwp_geometry/data.py), the
job group/name and the run-record fields differ. The three method switches and the frame size have no default;
set them on the command line:

    model.config.pwp_area_loss=off|sphere_area
    model.config.pwp_vae_w_padding=zero|circular
    model.config.pwp_sphere_pos_emb=off|xyz
    model.config.pwp_frame_size=1280x640|legacy_720

``dataloader_train.dataset.resolution`` follows the frame size (``pwp_erp_1280x640`` or the original ``720``).
``job.name`` is derived from them, e.g. ``stage1_baseline`` or ``stage2_baseline_arealoss_seam``
(``_legacy720`` is appended for the old frame size).
"""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_transfer2.experiments.singleview.cosmos_singleview_example import (
    transfer2_singleview_posttrain_pcd_rgb_image_context_example,
)
from cosmos_transfer2.experiments.singleview.cosmos_singleview_partial_hardlock import (
    transfer2_singleview_partial_hardlock_pcd_rgb_image_context_example,
)
from cosmos_transfer2.experiments.singleview.cosmos_singleview_target_loss_mask import (
    transfer2_singleview_posttrain_pcd_masked_rgb_target_loss_mask,
)

PWP_JOB_GROUP = "pwp_geometry"
_PWP_DATA = "cosmos_transfer2._src.transfer2.pwp_geometry.data"
_METHOD_ARGS = (
    "${model.config.pwp_area_loss},${model.config.pwp_vae_w_padding},${model.config.pwp_sphere_pos_emb},"
    "${model.config.pwp_frame_size}"
)


def _set_default(defaults: list, group: str, name: str) -> None:
    """Replace the ``override /<group>`` entry of a defaults list (or append it)."""
    key = f"override /{group}"
    for index, entry in enumerate(defaults):
        if isinstance(entry, dict) and key in entry:
            defaults[index] = {key: name}
            return
    defaults.append({key: name})


def _training_experiment(base: dict, stage: str, target_loss_mask: bool, data_train: str | None) -> dict:
    experiment = copy.deepcopy(base)
    _set_default(experiment["defaults"], "model", "fsdp_control_vace_rectified_flow_pwp_geometry")
    if data_train is not None:
        _set_default(experiment["defaults"], "data_train", data_train)
    experiment["job"]["group"] = PWP_JOB_GROUP
    experiment["job"]["name"] = f"${{pwp_run_name:{stage},{_METHOD_ARGS}}}"
    experiment["dataloader_train"]["dataset"]["resolution"] = "${pwp_dataset_resolution:${model.config.pwp_frame_size}}"
    # New video order every epoch; stage 1 also draws its mask from (seed, epoch, video). See pwp_geometry/data.py.
    if stage == "stage1":
        experiment["dataloader_train"]["dataset"]["_target_"] = _PWP_DATA + ".PwPSingleViewTransferDatasetMask"
    experiment["dataloader_train"]["sampler"] = {
        "_target_": _PWP_DATA + ".PwPEpochShuffleSampler",
        "dataset": "${dataloader_train.dataset}",
        "seed": "${trainer.seed}",
        "yield_epoch": stage == "stage1",
    }
    experiment["model"]["config"].update(
        pwp_target_loss_mask=target_loss_mask,
        # Only written to the run record (<job dir>/pwp_run_records.jsonl).
        pwp_record_job_path="${job.project}/${job.group}/${job.name}",
        pwp_record_load_path="${checkpoint.load_path}",
        pwp_record_dataset_dir="${dataloader_train.dataset.dataset_dir}",
    )
    return experiment


# Stage 1: general-domain 360° videos, mask-pool first frame + masked PCD, full RGB target.
pwp_geometry_stage1 = _training_experiment(
    transfer2_singleview_posttrain_pcd_rgb_image_context_example,
    stage="stage1",
    target_loss_mask=False,
    data_train="example_singleview_train_data_depth_mask",
)

# Stage 2: target-domain perspective-derived partial panoramas, loss on valid target pixels only.
pwp_geometry_stage2 = _training_experiment(
    transfer2_singleview_posttrain_pcd_masked_rgb_target_loss_mask,
    stage="stage2",
    target_loss_mask=True,
    data_train=None,
)

# Inference: partial hardlock (no-video-path pipeline) with the PwP DiT and VAE. The two switches that matter at
# inference are set by examples/inference_pwp_hardlock_no_video_path.py from its --pwp-* arguments.
pwp_geometry_inference_hardlock = copy.deepcopy(transfer2_singleview_partial_hardlock_pcd_rgb_image_context_example)
_set_default(
    pwp_geometry_inference_hardlock["defaults"],
    "model",
    "fsdp_control_vace_rectified_flow_pwp_geometry_partial_hardlock",
)
pwp_geometry_inference_hardlock["job"]["group"] = PWP_JOB_GROUP
pwp_geometry_inference_hardlock["job"]["name"] = "pwp_geometry_inference_hardlock"


cs = ConfigStore.instance()
for _name, _experiment in (
    ("pwp_geometry_stage1", pwp_geometry_stage1),
    ("pwp_geometry_stage2", pwp_geometry_stage2),
    ("pwp_geometry_inference_hardlock", pwp_geometry_inference_hardlock),
):
    cs.store(group="experiment", package="_global_", name=_name, node=_experiment)


__all__ = ["pwp_geometry_inference_hardlock", "pwp_geometry_stage1", "pwp_geometry_stage2"]
