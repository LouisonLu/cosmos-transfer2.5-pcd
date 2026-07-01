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
from typing import Annotated, Union

import pydantic
import tyro
from cosmos_oss.init import cleanup_environment, init_environment, init_output_dir

from cosmos_transfer2._src.imaginaire.utils import log
from cosmos_transfer2.config import (
    BlurConfig,
    DepthConfig,
    EdgeConfig,
    InferenceArguments,
    InferenceOverrides,
    SegConfig,
    SetupArguments,
    handle_tyro_exception,
    is_rank0,
)

"""Control-conditioned inference with first-chunk hard lock disabled."""

ControlUnion = Annotated[
    Union[
        Annotated[EdgeConfig, tyro.conf.subcommand("edge")],
        Annotated[DepthConfig, tyro.conf.subcommand("depth")],
        Annotated[BlurConfig, tyro.conf.subcommand("vis")],
        Annotated[SegConfig, tyro.conf.subcommand("seg")],
    ],
    tyro.conf.ConsolidateSubcommandArgs,
]


class Args(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")

    input_files: Annotated[list[Path], tyro.conf.arg(aliases=("-i",))]
    """Path(s) to the inference parameter file(s).
    If multiple files are provided, run "batch" inference. The model will be loaded once and all samples run sequentially.
    If there are different hint keys across the batch, the multicontrol model will be used regardless of the each sample's hint keys.
    """
    setup: SetupArguments
    """Setup arguments. These can only be provided via CLI."""
    overrides: InferenceOverrides
    """Inference parameter overrides. These can either be provided in the input json file or via CLI. CLI overrides will overwrite the values in the input file."""

    control: ControlUnion = EdgeConfig()
    """Control help. Run control:edge --help for more information about edge etc."""
    frame0_visible_strength: float | None = None
    """Deprecated alias for frame0_core_strength."""
    frame0_core_strength: float = 1.0
    """Strength for preserving the interior of the known frame0 visible region."""
    frame0_boundary_strength: float = 0.8
    """Strength for preserving the transition band around the frame0 visible region."""
    frame0_boundary_width: int = 1
    """Boundary width in latent cells between the visible core and generated region."""
    frame0_blend_until_step: int = 20
    """Last denoising step index that applies frame0 visible-region blending."""


def main(
    args: Args,
):
    inference_samples, batch_hint_keys = InferenceArguments.from_files(args.input_files, overrides=args.overrides)
    if args.setup.benchmark:
        if len(inference_samples) == 1:
            inference_samples = inference_samples * 4
            log.info(f"Repeating inference sample 4 times for benchmarking.")
        # assert len(inference_samples) > 1, "Benchmarking must be run for more than 1 sample."
    init_output_dir(args.setup.output_dir, profile=args.setup.profile)

    from cosmos_transfer2.inference_frame0_spatial_lock import Control2WorldInferenceFrame0SpatialLock

    inference = Control2WorldInferenceFrame0SpatialLock(args.setup, batch_hint_keys=batch_hint_keys)
    inference.inference_pipeline.force_hardlock_first_chunk = False
    frame0_core_strength = (
        args.frame0_visible_strength if args.frame0_visible_strength is not None else args.frame0_core_strength
    )
    inference.inference_pipeline.frame0_core_strength = frame0_core_strength
    inference.inference_pipeline.frame0_boundary_strength = args.frame0_boundary_strength
    inference.inference_pipeline.frame0_boundary_width = args.frame0_boundary_width
    inference.inference_pipeline.frame0_blend_until_step = args.frame0_blend_until_step
    log.info("First-chunk hard lock is disabled and frame0 spatial lock is enabled for this inference entry point.")
    log.info(
        "Frame0 spatial lock settings: "
        f"core_strength={frame0_core_strength}, "
        f"boundary_strength={args.frame0_boundary_strength}, "
        f"boundary_width={args.frame0_boundary_width}, "
        f"blend_until_step={args.frame0_blend_until_step}"
    )
    inference.generate(inference_samples, output_dir=args.setup.output_dir)


if __name__ == "__main__":
    init_environment()

    try:
        args = tyro.cli(
            Args,
            description=__doc__,
            console_outputs=is_rank0(),
            config=(tyro.conf.OmitArgPrefixes,),
        )
    except Exception as e:
        handle_tyro_exception(e)
    # pyrefly: ignore  # unbound-name
    main(args)

    cleanup_environment()
