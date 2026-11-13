# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PwP geometry inference: the no-video-path partial-hardlock inference with the PwP pipeline."""

import contextlib
import functools
import json
import sys
import time
from pathlib import Path

import cosmos_transfer2.inference_partial_hardlock_no_video_path as hardlock_no_video_path
from cosmos_transfer2._src.imaginaire.utils import log
from cosmos_transfer2._src.transfer2.pwp_geometry.inference_pipeline import (
    PwPGeometryInferencePartialHardlockNoVideoPath,
)
from cosmos_transfer2._src.transfer2.pwp_geometry.method import check_choice, git_state
from cosmos_transfer2.config import SetupArguments

REPO_DIR = Path(__file__).resolve().parents[1]


@contextlib.contextmanager
def _pipeline_class(pipeline_factory):
    """Let the original inference class build the PwP pipeline instead of its own (only inside this process)."""
    name = "ControlVideo2WorldInferencePartialHardlockNoVideoPath"
    original = getattr(hardlock_no_video_path, name)
    setattr(hardlock_no_video_path, name, pipeline_factory)
    try:
        yield
    finally:
        setattr(hardlock_no_video_path, name, original)


class Control2WorldInferencePwPHardlockNoVideoPath(hardlock_no_video_path.Control2WorldInferencePartialHardlockNoVideoPath):
    def __init__(
        self,
        args: SetupArguments,
        batch_hint_keys: list[str],
        pwp_vae_w_padding: str,
        pwp_sphere_pos_emb: str,
        pwp_frame_size: str,
        pwp_mask_pcd: bool = False,
        pwp_blended_decoding: bool = False,
        pwp_latent_rotation: bool = False,
        pwp_latitude_noise_strength: float = 0.0,
    ) -> None:
        check_choice("vae_w_padding", pwp_vae_w_padding)
        check_choice("sphere_pos_emb", pwp_sphere_pos_emb)
        if pwp_vae_w_padding == "circular" and args.enable_parallel_tokenizer:
            raise ValueError("--enable-parallel-tokenizer splits the VAE width and cannot be used with circular padding.")
        factory = functools.partial(
            PwPGeometryInferencePartialHardlockNoVideoPath,
            pwp_vae_w_padding=pwp_vae_w_padding,
            pwp_sphere_pos_emb=pwp_sphere_pos_emb,
            pwp_frame_size=pwp_frame_size,
            pwp_mask_pcd=pwp_mask_pcd,
            pwp_blended_decoding=pwp_blended_decoding,
            pwp_latent_rotation=pwp_latent_rotation,
            pwp_latitude_noise_strength=pwp_latitude_noise_strength,
        )
        with _pipeline_class(factory):
            super().__init__(args, batch_hint_keys=batch_hint_keys)

        if self.device_rank == 0:
            loaded = self.inference_pipeline.pwp_checkpoint_method
            record = {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "vae_w_padding": pwp_vae_w_padding,
                "sphere_pos_emb": pwp_sphere_pos_emb,
                "frame_size": pwp_frame_size,
                "techniques": self.inference_pipeline.pwp_techniques(),
                "checkpoint": str(args.checkpoint_path),
                "checkpoint_method": loaded.as_dict() if loaded else None,
                "experiment": args.experiment,
                "command": " ".join(sys.argv),
                "git": git_state(REPO_DIR),
            }
            record_path = Path(args.output_dir) / "pwp_inference_record.json"
            record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False, default=str) + "\n")
            log.critical(f"PwP inference record written to {record_path}")


__all__ = ["Control2WorldInferencePwPHardlockNoVideoPath"]
