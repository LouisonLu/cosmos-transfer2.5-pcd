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

import random
import time
from typing import Optional

import torch

from cosmos_transfer2._src.imaginaire.utils import log
from cosmos_transfer2._src.predict2.datasets.utils import VIDEO_RES_SIZE_INFO
from cosmos_transfer2._src.predict2.models.video2world_model import NUM_CONDITIONAL_FRAMES_KEY
from cosmos_transfer2._src.transfer2.datasets.augmentors.control_input import get_augmentor_for_eval
from cosmos_transfer2._src.transfer2.inference.inference_pipeline_partial_hardlock import (
    ControlVideo2WorldInferencePartialHardlock,
    _maybe_get_timer,
)
from cosmos_transfer2._src.transfer2.inference.utils import (
    get_t5_from_prompt,
    normalized_float_to_uint8,
    read_and_process_control_input,
    read_and_process_image_context,
    read_and_process_video,
    reshape_output_video_to_input_resolution,
    uint8_to_normalized_float,
)


class ControlVideo2WorldInferencePartialHardlockNoVideoPath(ControlVideo2WorldInferencePartialHardlock):
    """
    A partial-hardlock inference pipeline that does not require an external RGB video_path.

    Instead, it synthesizes an internal surrogate RGB video whose first frame comes from
    `image_context_path` and whose remaining frames are black. This keeps the stable
    partial-hardlock implementation untouched while preventing leakage from a full RGB video.
    """

    last_surrogate_input_frames: torch.Tensor | None = None
    last_surrogate_fps: int | None = None

    def _get_reference_control_path(
        self,
        input_control_video_paths: dict[str, str | None] | None,
        hint_key: list[str],
    ) -> str:
        if input_control_video_paths is None:
            raise ValueError("input_control_video_paths is required for no-video-path inference")

        for key in hint_key:
            control_path = input_control_video_paths.get(key)
            if control_path is not None:
                return control_path

        raise ValueError(
            "No control_path found in input_control_video_paths. "
            "Please provide at least one precomputed control video for no-video-path inference."
        )

    def _build_surrogate_input_video(
        self,
        image_context_path: str,
        input_control_video_paths: dict[str, str | None] | None,
        hint_key: list[str],
        resolution: str,
        max_frames: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor, int, str, tuple[int, int]]:
        reference_control_path = self._get_reference_control_path(input_control_video_paths, hint_key)
        reference_frames, fps, aspect_ratio, original_hw = read_and_process_video(
            reference_control_path,
            resolution=resolution,
            max_frames=max_frames,
        )
        if reference_frames.shape[1] == 0:
            raise ValueError("Reference control video is empty")

        image_context = read_and_process_image_context(
            image_context_path,
            resolution=VIDEO_RES_SIZE_INFO[resolution][aspect_ratio],
            resize=True,
            context_frame_idx=None,
        )
        if image_context is None:
            raise ValueError("image_context_path is required for no-video-path inference")

        image_context_uint8 = normalized_float_to_uint8(image_context.squeeze(0)).unsqueeze(1).cpu()
        surrogate_frames = torch.zeros_like(reference_frames, dtype=torch.uint8)
        surrogate_frames[:, :1] = image_context_uint8[:, :1]

        self.last_surrogate_input_frames = surrogate_frames.clone()
        self.last_surrogate_fps = fps

        return surrogate_frames, image_context, fps, aspect_ratio, original_hw

    def generate_img2world_no_video_path(
        self,
        prompt: str,
        image_context_path: str,
        guidance: int = 7,
        seed: int = 1,
        resolution: str = "720",
        negative_prompt: str | None = None,
        max_frames: int | None = None,
        input_control_video_paths: dict[str, str | None] | None = None,
        control_weight: str = "1.0",
        sigma_max: float | None = None,
        hint_key: list[str] = ["edge"],
        show_control_condition: bool = False,
        seg_control_prompt: str | None = None,
        show_input: bool = False,
        keep_input_resolution: bool = True,
        preset_blur_strength: str = "medium",
        preset_edge_threshold: str = "medium",
        num_conditional_frames: int = 1,
        num_video_frames_per_chunk: int = 93,
        num_steps: int = 35,
        guided_generation_mask: str | None = None,
        guided_generation_mask_first_frame_only: bool = False,
        guided_generation_step_threshold: int = 25,
        guided_generation_foreground_labels: list[int] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor], int, tuple[int, int]]:
        log.info("Synthesizing surrogate RGB input from image_context_path (frame 0) + black future frames...")
        input_frames, image_context, fps, aspect_ratio, original_hw = self._build_surrogate_input_video(
            image_context_path=image_context_path,
            input_control_video_paths=input_control_video_paths,
            hint_key=hint_key,
            resolution=resolution,
            max_frames=max_frames,
        )

        if guided_generation_mask is not None:
            guided_generation_mask = self._read_guided_generation_mask(
                guided_generation_mask,
                h=input_frames.shape[2],
                w=input_frames.shape[3],
                foreground_labels=guided_generation_foreground_labels,
                resolution=resolution,
                max_frames=max_frames,
            ).squeeze(0)
            if guided_generation_mask_first_frame_only:
                guided_generation_mask[:, 1:] = 0
                log.info("Using guided-generation mask on frame 0 only; future mask frames are black.")

        log.info("Computing prompt text embeddings...")
        with _maybe_get_timer(self.benchmark_timer, "get_text_embeddings"):
            if self.text_encoder_class == "T5":
                text_embeddings = get_t5_from_prompt(prompt, text_encoder_class="T5", cache_dir=self.cache_dir)
            else:
                text_embeddings = self.model.text_encoder.compute_text_embeddings_online(
                    {"ai_caption": [prompt], "images": None}, input_caption_key="ai_caption"
                )
            if negative_prompt:
                log.info("Computing negative prompt text embeddings...")
                if self.text_encoder_class == "T5":
                    neg_text_embeddings = get_t5_from_prompt(
                        negative_prompt, text_encoder_class="T5", cache_dir=self.cache_dir
                    )
                else:
                    neg_text_embeddings = self.model.text_encoder.compute_text_embeddings_online(
                        {"ai_caption": [negative_prompt], "images": None}, input_caption_key="ai_caption"
                    )
                self.neg_t5_embeddings = neg_text_embeddings

        log.info("Loading control inputs...")
        with _maybe_get_timer(self.benchmark_timer, "preprocessing"):
            control_input_dict, mask_video_dict = read_and_process_control_input(
                video_path=None,
                input_control_paths=input_control_video_paths,
                hint_key=hint_key,
                resolution=resolution,
                seg_control_prompt=seg_control_prompt,
            )

            num_total_frames, num_chunks, num_frames_per_chunk = self._get_num_chunks(
                input_frames, num_video_frames_per_chunk, num_conditional_frames
            )
            input_frames = self._pad_input_frames(input_frames, num_total_frames, num_video_frames_per_chunk)
            if guided_generation_mask is not None:
                guided_generation_mask = self._pad_input_frames(
                    guided_generation_mask, num_total_frames, num_video_frames_per_chunk
                )

            all_chunks, time_per_chunk = [], []
            control_video_dict = {}
            all_control_chunks = {key: [] for key in hint_key}
            prev_output = input_frames[:, :num_video_frames_per_chunk].to(torch.uint8).cuda()[None]

        self.model.eval()
        for chunk_id in range(num_chunks):
            log.info(f"Generating chunk {chunk_id + 1}/{num_chunks}")
            with _maybe_get_timer(self.benchmark_timer, "generate_chunk"):
                start_time = time.perf_counter()

                chunk_start_frame = chunk_id * num_frames_per_chunk
                chunk_end_frame = min(chunk_start_frame + num_video_frames_per_chunk, input_frames.shape[1])

                x_sigma_max = None
                partial_hardlock_mask = None
                cur_input_frames = input_frames[:, chunk_start_frame:chunk_end_frame]
                cur_input_frames = self._pad_input_frames(
                    cur_input_frames, cur_input_frames.shape[1], num_video_frames_per_chunk
                )
                if sigma_max is not None or guided_generation_mask is not None:
                    x0 = uint8_to_normalized_float(cur_input_frames, dtype=torch.bfloat16)[None].cuda(
                        non_blocking=True
                    )
                    x0 = self.model.encode(x0).contiguous()
                    if sigma_max is not None:
                        x_sigma_max = self.model.get_x_from_clean(x0, sigma_max, seed=(seed + chunk_id))

                    if guided_generation_mask is not None:
                        _, _, _, H, W = x0.shape
                        cur_guided_generation_mask = guided_generation_mask[:, chunk_start_frame:chunk_end_frame]
                        cur_guided_generation_mask = self._pad_input_frames(
                            cur_guided_generation_mask,
                            cur_guided_generation_mask.shape[1],
                            num_video_frames_per_chunk,
                        )
                        partial_hardlock_mask = self.construct_latent_weight_map(
                            cur_guided_generation_mask.unsqueeze(0), h=H, w=W, c=1
                        ).cuda(non_blocking=True)

                if isinstance(text_embeddings, list):
                    text_emb_idx = min(chunk_id, len(text_embeddings) - 1)
                    text_embedding = text_embeddings[text_emb_idx]
                else:
                    text_embedding = text_embeddings

                data_batch = self._get_data_batch_input(
                    cur_input_frames,
                    prev_output,
                    text_embedding,
                    fps,
                    negative_prompt=negative_prompt,
                    control_weight=control_weight,
                    image_context=image_context,
                )

                for k, v in control_input_dict.items():
                    cur_control_input = v[:, chunk_start_frame:chunk_end_frame]
                    data_batch[k] = self._pad_input_frames(
                        cur_control_input, cur_control_input.shape[1], num_video_frames_per_chunk
                    )
                    if k == "control_input_inpaint_mask":
                        data_batch["control_input_inpaint"] = cur_input_frames

                data_batch = get_augmentor_for_eval(
                    data_dict=data_batch,
                    input_keys=["input_video"],
                    output_keys=hint_key,
                    preset_edge_threshold=preset_edge_threshold,
                    preset_blur_strength=preset_blur_strength,
                )

                if chunk_id == 0:
                    data_batch[NUM_CONDITIONAL_FRAMES_KEY] = 1
                else:
                    data_batch[NUM_CONDITIONAL_FRAMES_KEY] = 1 + (num_conditional_frames - 1) // 4
                if partial_hardlock_mask is not None:
                    data_batch["partial_hardlock_mask"] = partial_hardlock_mask

                random.seed(seed)
                seed = random.randint(0, 1000000)
                log.info(f"Seed: {seed}")

                sample = self.model.generate_samples_from_batch(
                    data_batch,
                    n_sample=1,
                    guidance=guidance,
                    seed=seed,
                    is_negative_prompt=negative_prompt is not None,
                    x_sigma_max=x_sigma_max,
                    sigma_max=sigma_max,
                    num_steps=num_steps,
                )
                video = self.model.decode(sample).cpu()

                video_cat = video
                conditions = []
                if show_input:
                    x0 = uint8_to_normalized_float(cur_input_frames, dtype=torch.bfloat16)[None]
                    x0 = x0.to(device=video_cat.device)
                    video_cat = torch.cat([x0, video_cat], dim=-1)

                for key in hint_key:
                    control_input = data_batch["control_input_" + key]
                    if f"control_input_{key}_mask" in data_batch:
                        mask = data_batch[f"control_input_{key}_mask"].to(device=control_input.device)
                        control_input = (control_input + 1) / 2 * mask * 2 - 1

                    if chunk_id == 0:
                        all_control_chunks[key].append(control_input)
                    else:
                        all_control_chunks[key].append(control_input[:, :, num_conditional_frames:, :, :])

                    if show_control_condition:
                        conditions += [control_input.to(device=video_cat.device)]

                if show_control_condition:
                    video_cat = torch.cat([*conditions, video_cat], dim=-1)

                if chunk_id == 0:
                    all_chunks.append(video_cat)
                else:
                    all_chunks.append(video_cat[:, :, num_conditional_frames:, :, :])

                if chunk_id < num_chunks - 1:
                    last_frames = video[:, :, video.shape[2] - num_conditional_frames :, :, :]
                    last_frames_uint8 = normalized_float_to_uint8(last_frames)
                    blank_frames = torch.zeros(
                        (
                            1,
                            3,
                            num_video_frames_per_chunk - num_conditional_frames,
                            video.shape[-2],
                            video.shape[-1],
                        ),
                        dtype=torch.uint8,
                        device=video.device,
                    )
                    prev_output = torch.cat([last_frames_uint8, blank_frames], dim=2)

                end_time = time.perf_counter()
                time_per_chunk.append(end_time - start_time)

        with _maybe_get_timer(self.benchmark_timer, "postprocessing"):
            full_video = torch.cat(all_chunks, dim=2)
            full_video = full_video[:, :, :num_total_frames, :, :]

            full_video = full_video.cpu()
            for key in hint_key:
                if all_control_chunks[key]:
                    control_video_dict[key] = torch.cat(all_control_chunks[key], dim=2)
                    control_video_dict[key] = control_video_dict[key][:, :, :num_total_frames, :, :]

            if keep_input_resolution:
                full_video = reshape_output_video_to_input_resolution(
                    full_video, hint_key, show_control_condition, show_input, original_hw
                )
                for key in hint_key:
                    if key in control_video_dict and control_video_dict[key] is not None:
                        control_video_dict[key] = reshape_output_video_to_input_resolution(
                            control_video_dict[key], [key], False, False, original_hw
                        )

        log.info(f"Average time per chunk: {sum(time_per_chunk) / len(time_per_chunk)}")

        if guided_generation_mask is not None:
            if guided_generation_mask.ndim == 3:
                guided_generation_mask = guided_generation_mask.unsqueeze(0)
            mask_video_dict["guided_generation"] = guided_generation_mask

        return full_video, control_video_dict, mask_video_dict, fps, original_hw
