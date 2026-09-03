# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import cached_property
from typing import Literal

import pydantic

from cosmos_transfer2._src.imaginaire.flags import SMOKE
from cosmos_transfer2._src.imaginaire.utils import log
from cosmos_transfer2.config import (
    CONTROL_KEYS,
    DEFAULT_NEGATIVE_PROMPT,
    BlurConfig,
    CommonInferenceArguments,
    DepthConfig,
    EdgeConfig,
    Guidance,
    ResolvedDirectoryPath,
    ResolvedFilePath,
    SegConfig,
    Threshold,
    get_overrides_cls,
    path_to_str,
)


class InferenceArgumentsNoVideoPath(CommonInferenceArguments):
    image_context_path: ResolvedFilePath
    """Required. Path to the image context. This is the only RGB appearance input used by the no-video-path inference."""

    max_frames: pydantic.PositiveInt | None = None
    """Optional cap on the number of frames loaded from the control video(s) and guided mask."""

    num_conditional_frames: Literal[0, 1, 2] = 1
    """Used for chunk-wise long video generation. Number of frames from the previously-generated chunk to condition the next chunk on."""

    resolution: str = "720"
    """Output video resolution (e.g., '720', '480')."""

    sigma_max: str | None = None
    """Range from 0 to 200 for how much noise is added to the surrogate RGB input."""

    num_video_frames_per_chunk: pydantic.PositiveInt = 93
    """Number of video frames per chunk in the chunk-wise long video generation."""

    num_steps: pydantic.PositiveInt = 1 if SMOKE else 35
    """Number of sampling steps in the diffusion process."""

    show_control_condition: bool = False
    """Concatenate control videos and masks to the output video. Controls are still stored separately in the output directory."""

    show_input: bool = False
    """Concatenate the internally-synthesized surrogate RGB input video to the output video."""

    keep_input_resolution: bool = True
    """Whether to resize the output video to the control video's original resolution."""

    edge: EdgeConfig | None = None
    depth: DepthConfig | None = None
    vis: BlurConfig | None = None
    seg: SegConfig | None = None

    seed: int = 2025
    "Seed for generation randomness."

    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    """Negative prompt - describing what you do not want in the generated video."""

    prompt: str
    """Text prompt describing generation."""

    guided_generation_mask: ResolvedFilePath | ResolvedDirectoryPath | None = None
    """Path to guided generation mask. If None, guided generation is not used."""

    guided_generation_mask_first_frame_only: bool = False
    """Use only frame 0 of the guided-generation mask; treat all later frames as black."""

    guided_generation_mask_erode_px: pydantic.NonNegativeInt = 0
    """Shrink every white guided-mask region inward by this many pixels before latent hardlocking, without altering image context."""

    guided_generation_step_threshold: int = 25
    """Step threshold for guided generation."""

    guided_generation_foreground_labels: list[int] | None = None
    """List of label indices to treat as foreground in the mask. If None, any non-zero value is treated as foreground."""

    @cached_property
    def hint_keys(self) -> list[str]:
        return [key for key in CONTROL_KEYS if getattr(self, key, None) is not None]

    def model_post_init(self, __context) -> None:
        if len(self.hint_keys) == 0:
            raise ValueError("No controls provided, please provide at least one control key (edge, blur, depth, seg)")

        if "vis" in self.hint_keys and self.image_context_path:
            raise ValueError(
                "vis control and image_context_path are both used to transfer style. Using these modes together leads to conflicts. Please only provide one"
            )

        if self.guided_generation_mask_erode_px > 0 and self.guided_generation_mask is None:
            raise ValueError("guided_generation_mask_erode_px requires guided_generation_mask")

        for key in self.hint_keys:
            control = getattr(self, key)
            if control.control_path is None:
                raise ValueError(
                    f"{key}.control_path is required when using no-video-path inference. "
                    "On-the-fly control generation is unavailable without video_path."
                )
            if control.mask_prompt is not None and control.mask_path is None:
                raise ValueError(
                    f"{key}.mask_prompt requires video_path and is unsupported in no-video-path inference. "
                    "Please provide a precomputed mask_path instead."
                )

    @cached_property
    def control_weight_dict(self) -> dict[str, str]:
        control_weight_dict = {}
        for key in self.hint_keys:
            control_weight_dict[key] = str(getattr(self, key).control_weight)
        return control_weight_dict

    @cached_property
    def control_modalities(self) -> dict[str, str | None]:
        control_modalities = {}
        for key in self.hint_keys:
            control_modalities[key] = path_to_str(getattr(self, key).control_path)
            control_modalities[f"{key}_mask"] = path_to_str(getattr(self, key).mask_path)
            control_modalities[f"{key}_mask_prompt"] = getattr(self, key).mask_prompt
        return control_modalities

    @cached_property
    def preset_edge_threshold(self) -> Threshold:
        if "edge" in self.hint_keys:
            return getattr(self, "edge").preset_edge_threshold
        return "medium"

    @cached_property
    def preset_blur_strength(self) -> Threshold:
        if "vis" in self.hint_keys:
            return getattr(self, "vis").preset_blur_strength
        return "medium"

    @cached_property
    def seg_control_prompt(self) -> str | None:
        if "seg" not in self.hint_keys or getattr(self, "seg").control_path is not None:
            return None
        if getattr(self, "seg").control_prompt is not None:
            return getattr(self, "seg").control_prompt
        default_prompt = " ".join(self.prompt.split()[:128])
        log.warning(
            'No "control_prompt" provided for on-the-fly segmentation, using the first 128 words of the input prompt'
        )
        return default_prompt

    @cached_property
    def not_keep_input_resolution(self) -> bool:
        return not self.keep_input_resolution


InferenceOverridesNoVideoPath = get_overrides_cls(
    InferenceArgumentsNoVideoPath,
    exclude=[
        "name",
        "edge",
        "depth",
        "vis",
        "seg",
    ],
)
