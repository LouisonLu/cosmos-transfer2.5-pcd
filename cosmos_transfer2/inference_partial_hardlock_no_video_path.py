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

from pathlib import Path

import numpy as np
import torch

from cosmos_transfer2._src.imaginaire.auxiliary.guardrail.common import presets as guardrail_presets
from cosmos_transfer2._src.imaginaire.flags import SMOKE
from cosmos_transfer2._src.imaginaire.lazy_config.lazy import LazyConfig
from cosmos_transfer2._src.imaginaire.utils import distributed, log, misc
from cosmos_transfer2._src.imaginaire.visualize.video import save_img_or_video
from cosmos_transfer2._src.transfer2.configs.vid2vid_transfer.experiment.experiment_list import EXPERIMENTS
from cosmos_transfer2._src.transfer2.inference.inference_pipeline_partial_hardlock_no_video_path import (
    ControlVideo2WorldInferencePartialHardlockNoVideoPath,
)
from cosmos_transfer2._src.transfer2.inference.utils import compile_tokenizer_if_enabled
from cosmos_transfer2.config import (
    CONTROL_KEYS,
    MODEL_CHECKPOINTS,
    ModelKey,
    SetupArguments,
    is_rank0,
    path_to_str,
)
from cosmos_transfer2.config_no_video_path import InferenceArgumentsNoVideoPath


class Control2WorldInferencePartialHardlockNoVideoPath:
    def __init__(
        self,
        args: SetupArguments,
        batch_hint_keys: list[str],
    ) -> None:
        log.debug(f"{args.__class__.__name__}({args})({batch_hint_keys})")
        self.setup_args = args
        self.batch_hint_keys = batch_hint_keys
        self.is_distilled = args.model_key.distilled

        if len(self.batch_hint_keys) == 1:
            checkpoint = MODEL_CHECKPOINTS[ModelKey(variant=self.batch_hint_keys[0], distilled=self.is_distilled)]
            self.checkpoint_list = [checkpoint.s3.uri]
            self.experiment = checkpoint.experiment
            if args.has_checkpoint_override:
                self.checkpoint_list = [args.checkpoint_path]
                log.debug(f"Using checkpoint path override: {args.checkpoint_path}")
            if args.has_experiment_override:
                self.experiment = args.experiment
                log.debug(f"Using experiment override: {args.experiment}")
        else:
            self.checkpoint_list = [
                MODEL_CHECKPOINTS[ModelKey(variant=key, distilled=self.is_distilled)].s3.uri for key in CONTROL_KEYS
            ]
            self.experiment = "multibranch_720p_t24_spaced_layer4_cr1pt1_rectified_flow_inference"

        torch.enable_grad(False)
        self.device_rank = 0

        process_group = None
        if args.context_parallel_size > 1:
            from megatron.core import parallel_state

            distributed.init()
            parallel_state.initialize_model_parallel(context_parallel_size=args.context_parallel_size)
            process_group = parallel_state.get_context_parallel_group()
            self.device_rank = distributed.get_rank(process_group)

        if args.enable_guardrails and self.device_rank == 0:
            self.text_guardrail_runner = guardrail_presets.create_text_guardrail_runner(
                offload_model_to_cpu=args.offload_guardrail_models
            )
            self.video_guardrail_runner = guardrail_presets.create_video_guardrail_runner(
                offload_model_to_cpu=args.offload_guardrail_models
            )
        else:
            self.text_guardrail_runner = None
            self.video_guardrail_runner = None

        self.benchmark_timer = misc.TrainingTimer()

        if self.is_distilled:
            registered_exp_name = self.experiment
            exp_override_opts: list[str] = []
            exp_override_opts.append("model.config.load_teacher_weights=False")
        elif args.has_experiment_override:
            registered_exp_name = args.experiment
            exp_override_opts = []
        else:
            registered_exp_name = EXPERIMENTS[self.experiment].registered_exp_name
            exp_override_opts = EXPERIMENTS[self.experiment].command_args.copy()

        self.inference_pipeline = ControlVideo2WorldInferencePartialHardlockNoVideoPath(
            registered_exp_name=registered_exp_name,
            checkpoint_paths=self.checkpoint_list,
            s3_credential_path="",
            exp_override_opts=exp_override_opts,
            process_group=process_group,
            use_cp_wan=args.enable_parallel_tokenizer,
            wan_cp_grid=args.parallel_tokenizer_grid,
            benchmark_timer=self.benchmark_timer if args.benchmark else None,
            config_file=args.config_file,
        )

        if self.is_distilled:
            log.info("Setting net_fake_score to None for distilled model inference")
            self.inference_pipeline.model.net_fake_score = None

        compile_tokenizer_if_enabled(self.inference_pipeline, args.compile_tokenizer.value)

        if self.device_rank == 0:
            log.info(f"Found {len(self.batch_hint_keys)} hint keys across all samples")
            if len(self.batch_hint_keys) > 1:
                log.warning(
                    "Loading the multicontrol model. Multicontrol inference is not strictly equal to single control"
                )

            args.output_dir.mkdir(parents=True, exist_ok=True)
            config_path = args.output_dir / "config.yaml"
            LazyConfig.save_yaml(self.inference_pipeline.config, config_path)
            log.info(f"Saved config to {config_path}")

    def generate(self, samples: list[InferenceArgumentsNoVideoPath], output_dir: Path) -> list[str]:
        if SMOKE:
            samples = samples[:1]

        sample_names = [sample.name for sample in samples]
        log.info(f"Generating {len(samples)} samples: {sample_names}")

        output_paths: list[str] = []
        for i_sample, sample in enumerate(samples):
            log.info(f"[{i_sample + 1}/{len(samples)}] Processing sample {sample.name}")
            output_path = self._generate_sample(sample, output_dir, sample_id=i_sample)
            if output_path is not None:
                output_paths.append(output_path)

        if is_rank0() and self.setup_args.benchmark:
            log.info("=" * 50)
            log.info("BENCHMARK RESULTS")
            log.info("=" * 50)
            log.info("Benchmark runs:")
            for key, value in self.benchmark_timer.results.items():
                log.info(f"{key}: {value} seconds")
            log.info("Average times:")
            for key, value in self.benchmark_timer.compute_average_results().items():
                log.info(f"{key}: {value:.2f} seconds")
            log.info("=" * 50)
        return output_paths

    def _generate_sample(self, sample: InferenceArgumentsNoVideoPath, output_dir: Path, sample_id: int = 0) -> str | None:
        log.debug(f"{sample.__class__.__name__}({sample})")
        output_path = output_dir / sample.name

        prompt: str = sample.prompt
        negative_prompt = None if self.is_distilled else sample.negative_prompt

        guided_generation_mask = (
            str(sample.guided_generation_mask) if sample.guided_generation_mask is not None else None
        )
        guided_generation_step_threshold = sample.guided_generation_step_threshold
        guided_generation_foreground_labels = sample.guided_generation_foreground_labels
        guided_generation_mask_first_frame_only = sample.guided_generation_mask_first_frame_only
        guided_generation_mask_erode_px = sample.guided_generation_mask_erode_px

        if self.device_rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
            open(f"{output_path}.json", "w").write(sample.model_dump_json())
            log.info(f"Saved arguments to {output_path}.json")

            with self.benchmark_timer("text_guardrail"):
                if self.text_guardrail_runner is not None:
                    log.info("Running guardrail check on prompt...")

                    if not guardrail_presets.run_text_guardrail(prompt, self.text_guardrail_runner):
                        message = f"Guardrail blocked generation. Prompt: {prompt}"
                        log.critical(message)
                        if self.setup_args.keep_going:
                            return None
                        raise Exception(message)
                    log.success("Passed guardrail on prompt")

                    if negative_prompt is not None:
                        if not guardrail_presets.run_text_guardrail(negative_prompt, self.text_guardrail_runner):
                            message = f"Guardrail blocked generation. Negative prompt: {negative_prompt}"
                            log.critical(message)
                            if self.setup_args.keep_going:
                                return None
                            raise Exception(message)
                        log.success("Passed guardrail on negative prompt")
                elif self.text_guardrail_runner is None:
                    log.warning("Guardrail checks on prompt are disabled")

        input_control_video_paths = sample.control_modalities
        log.info(f"Processing the following paths: {input_control_video_paths}")

        sigma_max = None if sample.sigma_max is None else float(sample.sigma_max)

        control_weight = ""
        for key in self.batch_hint_keys:
            control_weight += sample.control_weight_dict.get(key, "0.0") + ","
        control_weight = control_weight[:-1]

        if self.setup_args.benchmark:
            torch.cuda.synchronize()

        with self.benchmark_timer("generate_img2world"):
            guidance = None if self.is_distilled else sample.guidance
            output_video, control_video_dict, mask_video_dict, fps, _ = (
                self.inference_pipeline.generate_img2world_no_video_path(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image_context_path=path_to_str(sample.image_context_path),
                    max_frames=sample.max_frames,
                    guidance=guidance,
                    seed=sample.seed,
                    resolution=sample.resolution,
                    control_weight=control_weight,
                    sigma_max=sigma_max,
                    hint_key=sample.hint_keys,
                    input_control_video_paths=input_control_video_paths,
                    show_control_condition=sample.show_control_condition,
                    seg_control_prompt=sample.seg_control_prompt,
                    show_input=sample.show_input,
                    keep_input_resolution=not sample.not_keep_input_resolution,
                    preset_blur_strength=sample.preset_blur_strength,
                    preset_edge_threshold=sample.preset_edge_threshold,
                    num_conditional_frames=sample.num_conditional_frames,
                    num_video_frames_per_chunk=sample.num_video_frames_per_chunk,
                    num_steps=sample.num_steps,
                    guided_generation_mask=guided_generation_mask,
                    guided_generation_mask_first_frame_only=guided_generation_mask_first_frame_only,
                    guided_generation_mask_erode_px=guided_generation_mask_erode_px,
                    guided_generation_step_threshold=guided_generation_step_threshold,
                    guided_generation_foreground_labels=guided_generation_foreground_labels,
                )
            )
            if self.setup_args.benchmark:
                torch.cuda.synchronize()

        ext = "jpg" if output_video.shape[2] == 1 else "mp4"

        if self.is_distilled and output_video.shape[2] > 93:
            log.warning(
                "Generated output has "
                f"{output_video.shape[2]} frames (> 93). "
                "The distilled Transfer 2.5 model is not trained to support auto-regressive generation"
            )

        if self.device_rank == 0:
            surrogate_input = self.inference_pipeline.last_surrogate_input_frames
            surrogate_fps = self.inference_pipeline.last_surrogate_fps or fps
            if surrogate_input is not None:
                surrogate_video = surrogate_input.float() / 255.0
                save_img_or_video(surrogate_video, f"{output_path}_input_surrogate", fps=surrogate_fps)
                save_img_or_video(surrogate_video[:, :1], f"{output_path}_input_surrogate_first_frame", fps=surrogate_fps)
                log.info(f"Saved surrogate input video to {output_path}_input_surrogate.mp4")

            output_video = (1.0 + output_video[0]) / 2
            for key in control_video_dict:
                control_video_dict[key] = (1.0 + control_video_dict[key][0]) / 2
                save_img_or_video(control_video_dict[key], f"{output_path}_control_{key}", fps=fps)
                log.info(f"{key} control video saved to {output_path}_control_{key}.{ext}")

            with self.benchmark_timer("video_guardrail"):
                for key in mask_video_dict:
                    save_img_or_video(mask_video_dict[key], f"{output_path}_mask_{key}", fps=fps)
                    log.info(f"Mask for {key} saved to {output_path}_mask_{key}.{ext}")

                if self.video_guardrail_runner is not None:
                    log.info("Running guardrail check on video...")
                    frames = (output_video * 255.0).clamp(0.0, 255.0).to(torch.uint8)
                    frames = frames.permute(1, 2, 3, 0).cpu().numpy().astype(np.uint8)
                    processed_frames = guardrail_presets.run_video_guardrail(frames, self.video_guardrail_runner)
                    if processed_frames is None:
                        if self.setup_args.keep_going:
                            return None
                        raise Exception("Guardrail blocked video2world generation.")
                    log.success("Passed guardrail on generated video")

                    processed_video = torch.from_numpy(processed_frames).float().permute(3, 0, 1, 2) / 255.0
                    output_video = processed_video.to(output_video.device, dtype=output_video.dtype)
                else:
                    log.warning("Guardrail checks on video are disabled")

            save_img_or_video(output_video, str(output_path), fps=fps)
            prompt_save_path = f"{output_path}.txt"
            with open(prompt_save_path, "w") as f:
                f.write(sample.prompt)
            log.success(f"Generated video saved to {output_path}.{ext}")

        if sample_id == 0 and self.setup_args.benchmark:
            self.benchmark_timer.reset()

        torch.cuda.empty_cache()
        return f"{output_path}.{ext}"
