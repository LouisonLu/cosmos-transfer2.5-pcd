# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Isolated orchestration for no-video-path hardlock blended decoding."""

from pathlib import Path

import torch

from cosmos_transfer2._src.imaginaire.auxiliary.guardrail.common import presets as guardrail_presets
from cosmos_transfer2._src.imaginaire.utils import distributed, log, misc
from cosmos_transfer2._src.transfer2.configs.vid2vid_transfer.experiment.experiment_list import EXPERIMENTS
from cosmos_transfer2._src.transfer2.inference.inference_pipeline_partial_hardlock_no_video_path_blended_decode import (
    ControlVideo2WorldInferencePartialHardlockNoVideoPathBlendedDecode,
)
from cosmos_transfer2._src.transfer2.inference.utils import compile_tokenizer_if_enabled
from cosmos_transfer2.config import CONTROL_KEYS, MODEL_CHECKPOINTS, ModelKey, SetupArguments
from cosmos_transfer2.inference_partial_hardlock_no_video_path import (
    Control2WorldInferencePartialHardlockNoVideoPath,
)


class Control2WorldInferencePartialHardlockNoVideoPathBlendedDecode(
    Control2WorldInferencePartialHardlockNoVideoPath
):
    """Original orchestration with an isolated blended-decode pipeline instance."""

    def __init__(self, args: SetupArguments, batch_hint_keys: list[str]) -> None:
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
            exp_override_opts = ["model.config.load_teacher_weights=False"]
        elif args.has_experiment_override:
            registered_exp_name = args.experiment
            exp_override_opts = []
        else:
            registered_exp_name = EXPERIMENTS[self.experiment].registered_exp_name
            exp_override_opts = EXPERIMENTS[self.experiment].command_args.copy()

        self.inference_pipeline = ControlVideo2WorldInferencePartialHardlockNoVideoPathBlendedDecode(
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
                log.warning("Loading the multicontrol model. Multicontrol inference is not strictly equal to single control")
            args.output_dir.mkdir(parents=True, exist_ok=True)

            from cosmos_transfer2._src.imaginaire.lazy_config.lazy import LazyConfig

            LazyConfig.save_yaml(self.inference_pipeline.config, args.output_dir / "config.yaml")
            log.info(f"Saved config to {args.output_dir / 'config.yaml'}")

    def _generate_sample(self, sample, output_dir: Path, sample_id: int = 0) -> str | None:
        self.inference_pipeline.default_blended_decoding = sample.blended_decoding
        return super()._generate_sample(sample, output_dir, sample_id)
