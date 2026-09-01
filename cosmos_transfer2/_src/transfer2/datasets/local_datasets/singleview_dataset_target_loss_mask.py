# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Isolated single-view dataset with a per-frame target-validity loss mask."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from cosmos_transfer2._src.imaginaire.utils import log
from cosmos_transfer2._src.transfer2.datasets.local_datasets.singleview_dataset_mask import (
    SingleViewTransferDatasetMask,
)
from cosmos_transfer2._src.transfer2.utils.target_loss_mask import align_valid_mask_to_augmented_video


class SingleViewTransferDatasetTargetLossMask(SingleViewTransferDatasetMask):
    """Load masked RGB/PCD conditions and a paired 93-frame validity mask.

    The parent dataset retains the established RGB, PCD, image-context, and text
    pipeline. This subclass adds a strict one-to-one target mask and prompt
    mapping, aligns the mask using the augmentation result, and exposes it as
    ``target_loss_mask`` with shape ``(1, T, H, W)``.
    """

    def __init__(
        self,
        *args,
        target_loss_mask_dir: str = "mask",
        target_loss_mask_suffix: str = "_mask",
        target_loss_mask_threshold: int = 128,
        target_loss_mask_white_is_valid: bool = True,
        target_loss_mask_aspect_ratio_tolerance: float = 0.01,
        prompt_dir: str = "prompts",
        prompt_suffix: str = "_prompt",
        enforce_mask_on_target: bool = True,
        enforce_mask_on_condition: bool = True,
        use_control_input_as_video_condition: bool = True,
        strict_masked_input_validation: bool = True,
        strict_masked_target_validation: bool = True,
        strict_masked_control_validation: bool = False,
        masked_input_black_threshold: int = 24,
        max_invalid_nonblack_ratio: float = 0.05,
        validate_file_pairs: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        if not 0 <= target_loss_mask_threshold <= 255:
            raise ValueError("target_loss_mask_threshold must be in [0, 255]")
        if not 0 <= masked_input_black_threshold <= 255:
            raise ValueError("masked_input_black_threshold must be in [0, 255]")
        if not 0.0 <= max_invalid_nonblack_ratio <= 1.0:
            raise ValueError("max_invalid_nonblack_ratio must be in [0, 1]")
        if target_loss_mask_aspect_ratio_tolerance < 0.0:
            raise ValueError("target_loss_mask_aspect_ratio_tolerance must be non-negative")

        self.target_loss_mask_threshold = int(target_loss_mask_threshold)
        self.target_loss_mask_white_is_valid = bool(target_loss_mask_white_is_valid)
        self.target_loss_mask_aspect_ratio_tolerance = float(target_loss_mask_aspect_ratio_tolerance)
        self.enforce_mask_on_target = bool(enforce_mask_on_target)
        self.enforce_mask_on_condition = bool(enforce_mask_on_condition)
        self.use_control_input_as_video_condition = bool(use_control_input_as_video_condition)
        self.strict_masked_input_validation = bool(strict_masked_input_validation)
        self.strict_masked_target_validation = bool(strict_masked_target_validation)
        self.strict_masked_control_validation = bool(strict_masked_control_validation)
        self.masked_input_black_threshold = int(masked_input_black_threshold)
        self.max_invalid_nonblack_ratio = float(max_invalid_nonblack_ratio)

        self.target_loss_mask_root = self._resolve_dataset_path(target_loss_mask_dir)
        self.prompt_root = self._resolve_dataset_path(prompt_dir)
        self.target_loss_mask_paths = {
            Path(video_path).stem: self.target_loss_mask_root / f"{Path(video_path).stem}{target_loss_mask_suffix}.mp4"
            for video_path in self.video_paths
        }
        self.prompt_paths = {
            Path(video_path).stem: self.prompt_root / f"{Path(video_path).stem}{prompt_suffix}.json"
            for video_path in self.video_paths
        }

        if validate_file_pairs:
            self._validate_required_file_pairs()

        log.info("Initialized isolated target-loss-mask dataset")
        log.info(f"  Target masks: {self.target_loss_mask_root} (*{target_loss_mask_suffix}.mp4)")
        log.info(f"  Prompts: {self.prompt_root} (*{prompt_suffix}.json)")
        mask_semantics = "white=valid" if self.target_loss_mask_white_is_valid else "black=valid"
        log.info(f"  Mask semantics: {mask_semantics}")
        log.info(
            "  Mask enforcement: "
            f"target={self.enforce_mask_on_target}, conditions={self.enforce_mask_on_condition}, "
            f"control_as_video_condition={self.use_control_input_as_video_condition}"
        )
        log.info(
            "  Strict masked validation: "
            f"target={self.strict_masked_input_validation and self.strict_masked_target_validation}, "
            f"control={self.strict_masked_input_validation and self.strict_masked_control_validation}"
        )

    def _resolve_dataset_path(self, path: str) -> Path:
        candidate = Path(path)
        return candidate if candidate.is_absolute() else Path(self.dataset_dir) / candidate

    def _validate_required_file_pairs(self) -> None:
        missing_masks = [str(path) for path in self.target_loss_mask_paths.values() if not path.is_file()]
        missing_prompts = [str(path) for path in self.prompt_paths.values() if not path.is_file()]
        invalid_prompts: list[str] = []
        if not missing_prompts:
            for video_name, prompt_path in self.prompt_paths.items():
                try:
                    self._load_caption(video_name)
                except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
                    invalid_prompts.append(f"{prompt_path}: {error}")

        missing_controls: list[str] = []
        if self.control_video_dir_override is None:
            missing_controls.append(
                "control_video_dir_override is not configured; a paired masked PCD video is required"
            )
        else:
            for video_path in self.video_paths:
                video_name = Path(video_path).stem
                control_path = Path(self.control_video_dir_override) / (
                    f"{video_name}{self.control_video_suffix}.{self.ctrl_config['format']}"
                )
                if not control_path.is_file():
                    missing_controls.append(str(control_path))

        failures = {
            "target mask": missing_masks,
            "prompt": missing_prompts,
            "invalid prompt": invalid_prompts,
            "PCD/control": missing_controls,
        }
        messages = []
        for label, paths in failures.items():
            if paths:
                preview = "\n    ".join(paths[:5])
                messages.append(f"{label}: missing {len(paths)}\n    {preview}")
        if messages:
            raise FileNotFoundError("Required training pairs are incomplete:\n" + "\n".join(messages))

    def _load_caption(self, video_name: str) -> str:
        prompt_path = self.prompt_paths.get(video_name)
        if prompt_path is None or not prompt_path.is_file():
            raise FileNotFoundError(f"Prompt not found for {video_name}: {prompt_path}")

        with prompt_path.open(encoding="utf-8") as file:
            data = json.load(file)
        if isinstance(data, str):
            prompt = data
        elif isinstance(data, dict):
            prompt = data.get("caption", data.get("text", data.get("prompt", "")))
        else:
            prompt = ""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Prompt file has no non-empty caption/text/prompt field: {prompt_path}")
        return prompt.strip()

    def _load_aligned_target_loss_mask(self, data: dict[str, Any]) -> Tensor:
        video_name = str(data["__key__"])
        mask_path = self.target_loss_mask_paths.get(video_name)
        if mask_path is None or not mask_path.is_file():
            raise FileNotFoundError(f"Target loss mask not found for {video_name}: {mask_path}")

        frame_ids = [int(frame_id) for frame_id in data["frame_indices"]]
        mask_frames, _, _ = self._load_video(str(mask_path), frame_ids=frame_ids)
        if mask_frames.ndim == 4:
            mask_gray = mask_frames.astype(np.float32).mean(axis=-1)
        elif mask_frames.ndim == 3:
            mask_gray = mask_frames.astype(np.float32)
        else:
            raise ValueError(f"Unexpected decoded mask shape for {mask_path}: {mask_frames.shape}")

        if self.target_loss_mask_white_is_valid:
            valid = mask_gray >= self.target_loss_mask_threshold
        else:
            valid = mask_gray < self.target_loss_mask_threshold
        valid_T_H_W = torch.from_numpy(valid.astype(np.float32, copy=False))
        aligned = align_valid_mask_to_augmented_video(
            valid_T_H_W,
            data["image_size"],
            source_aspect_ratio_tolerance=self.target_loss_mask_aspect_ratio_tolerance,
        )
        if aligned.shape[1:] != data["video"].shape[1:]:
            raise ValueError(
                f"Aligned mask/video shapes differ for {video_name}: "
                f"mask={tuple(aligned.shape)}, video={tuple(data['video'].shape)}"
            )
        return aligned

    @staticmethod
    def _padding_region_mask(data: dict[str, Any], target_loss_mask: Tensor) -> Tensor:
        """Reconstruct a 1-valued padding region from the augmentor image size."""
        target_h, target_w, content_h, content_w = (
            int(round(value)) for value in torch.as_tensor(data["image_size"]).flatten().tolist()
        )
        pad_left = (target_w - content_w) // 2
        pad_top = (target_h - content_h) // 2
        padding = torch.ones_like(target_loss_mask[:, :1])
        padding[:, :, pad_top : pad_top + content_h, pad_left : pad_left + content_w] = 0
        return padding

    def _condition_keep_mask(self, data: dict[str, Any], target_loss_mask: Tensor) -> Tensor:
        """Keep reflection-padded condition pixels while masking invalid content."""
        padding = self._padding_region_mask(data, target_loss_mask)
        return torch.clamp(target_loss_mask + padding, min=0.0, max=1.0)

    @staticmethod
    def _apply_video_mask(video_C_T_H_W: Tensor, keep_1_T_H_W: Tensor) -> Tensor:
        if video_C_T_H_W.shape[1:] != keep_1_T_H_W.shape[1:]:
            raise ValueError(
                f"Cannot apply mask: video={tuple(video_C_T_H_W.shape)}, mask={tuple(keep_1_T_H_W.shape)}"
            )
        masked = video_C_T_H_W.float() * keep_1_T_H_W.to(device=video_C_T_H_W.device)
        if video_C_T_H_W.dtype == torch.uint8:
            return masked.round().clamp(0, 255).to(dtype=torch.uint8)
        return masked.to(dtype=video_C_T_H_W.dtype)

    def _invalid_nonblack_ratio(self, video_C_T_H_W: Tensor, data: dict[str, Any], valid_mask: Tensor) -> float:
        content = 1.0 - self._padding_region_mask(data, valid_mask)
        invalid_content = (valid_mask < 0.5) & (content > 0.5)
        invalid_count = int(invalid_content.sum())
        if invalid_count == 0:
            return 0.0
        nonblack = video_C_T_H_W.float().amax(dim=0, keepdim=True) > self.masked_input_black_threshold
        return float((nonblack & invalid_content).sum() / invalid_content.sum())

    def _validate_masked_tensor(
        self,
        label: str,
        video_C_T_H_W: Tensor,
        data: dict[str, Any],
        valid_mask: Tensor,
        enabled: bool,
    ) -> None:
        if not self.strict_masked_input_validation or not enabled:
            return
        ratio = self._invalid_nonblack_ratio(video_C_T_H_W, data, valid_mask)
        if ratio > self.max_invalid_nonblack_ratio:
            raise ValueError(
                f"{label} is not masked consistently with target_loss_mask for {data['__key__']}: "
                f"invalid_nonblack_ratio={ratio:.6f}, allowed={self.max_invalid_nonblack_ratio:.6f}, "
                f"black_threshold={self.masked_input_black_threshold}"
            )

    def __getitem__(self, index: int) -> dict[str, Any]:
        data = super().__getitem__(index)
        target_loss_mask = self._load_aligned_target_loss_mask(data)
        condition_keep_mask = self._condition_keep_mask(data, target_loss_mask)

        self._validate_masked_tensor(
            "Target RGB video",
            data["video"],
            data,
            target_loss_mask,
            enabled=self.strict_masked_target_validation,
        )
        if self.enforce_mask_on_target:
            data["video"] = self._apply_video_mask(data["video"], condition_keep_mask)

        control_key = f"control_input_{self.ctrl_type}"
        if control_key not in data:
            raise KeyError(f"Required masked control input is missing: {control_key}")
        self._validate_masked_tensor(
            "PCD/control video",
            data[control_key],
            data,
            target_loss_mask,
            enabled=self.strict_masked_control_validation,
        )
        if self.enforce_mask_on_condition:
            data[control_key] = self._apply_video_mask(data[control_key], condition_keep_mask)

        if self.use_control_input_as_video_condition:
            data["input_video"] = data[control_key].clone()
        elif "input_video" in data and self.enforce_mask_on_condition:
            data["input_video"] = self._apply_video_mask(data["input_video"], condition_keep_mask)

        # Derive image context from the selected and augmented target clip rather
        # than reloading source frame 0. This remains correct for videos longer
        # than the 93-frame training window.
        image_context = data["video"][:, 0].clone()
        if self.enforce_mask_on_condition:
            image_context = self._apply_video_mask(
                image_context.unsqueeze(1), condition_keep_mask[:, :1]
            ).squeeze(1)
        # The image-context conditioner expects the same [-1, 1] range produced
        # by the parent's Normalize augmentor. Replacing the parent's context
        # after augmentation must preserve that contract.
        data["image_context"] = image_context.float().div(127.5).sub(1.0)

        data["target_loss_mask"] = target_loss_mask
        data["target_loss_mask_valid_ratio"] = target_loss_mask.mean()
        return data


__all__ = ["SingleViewTransferDatasetTargetLossMask"]
