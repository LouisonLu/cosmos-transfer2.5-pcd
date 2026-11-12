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

"""PwP geometry inference: same arguments as inference_partial_hardlock_no_video_path.py plus

    --pwp-vae-w-padding zero|circular
    --pwp-sphere-pos-emb off|xyz
    --pwp-frame-size 1280x640|legacy_720

The first two must match the checkpoint (they are checked against its stamp). The frame size should match the
one the checkpoint was trained with; it replaces the "resolution" of the input JSON. Use with
--experiment pwp_geometry_inference_hardlock --config-file cosmos_transfer2/pwp_geometry_config.py.
"""

from pathlib import Path
from typing import Annotated, Literal, Union

import pydantic
import tyro
from cosmos_oss.init import cleanup_environment, init_environment, init_output_dir

from cosmos_transfer2.config import (
    BlurConfig,
    DepthConfig,
    EdgeConfig,
    SegConfig,
    SetupArguments,
    handle_tyro_exception,
    is_rank0,
)
from cosmos_transfer2.config_no_video_path import (
    InferenceArgumentsNoVideoPath,
    InferenceOverridesNoVideoPath,
)

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
    """Path(s) to the inference parameter file(s)."""

    pwp_vae_w_padding: Literal["zero", "circular"]
    """VAE width padding the checkpoint was trained with."""

    pwp_sphere_pos_emb: Literal["off", "xyz"]
    """Spherical XYZ position embedding the checkpoint was trained with."""

    pwp_frame_size: Literal["1280x640", "legacy_720"]
    """1280x640: read the 2:1 inputs at 1280x640 (no stretch). legacy_720: the original stretch to 1280x704."""

    setup: SetupArguments
    """Setup arguments. These can only be provided via CLI."""

    overrides: InferenceOverridesNoVideoPath
    """Inference parameter overrides. These can either be provided in the input json file or via CLI."""

    control: ControlUnion = EdgeConfig()
    """Control help. Run control:edge --help for more information about edge etc."""


def main(args: Args):
    inference_samples, batch_hint_keys = InferenceArgumentsNoVideoPath.from_files(
        args.input_files, overrides=args.overrides
    )
    init_output_dir(args.setup.output_dir, profile=args.setup.profile)

    from cosmos_transfer2.inference_pwp_hardlock_no_video_path import Control2WorldInferencePwPHardlockNoVideoPath

    inference = Control2WorldInferencePwPHardlockNoVideoPath(
        args.setup,
        batch_hint_keys=batch_hint_keys,
        pwp_vae_w_padding=args.pwp_vae_w_padding,
        pwp_sphere_pos_emb=args.pwp_sphere_pos_emb,
        pwp_frame_size=args.pwp_frame_size,
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

    main(args)
    cleanup_environment()
